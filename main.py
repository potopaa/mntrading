#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mntrading CLI with robust ingest and remote-only storage.

Highlights:
- Auto-build a large universe of spot USDT symbols via --symbols-auto (no external loader required).
- Paginated OHLCV ingest (1000 per request) until no progress or max_candles reached.
- REMOTE_ONLY flow: read inputs from MinIO and write outputs there.

ENV:
  REMOTE_ONLY=1
  MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  AWS_ACCESS_KEY_ID=admin
  AWS_SECRET_ACCESS_KEY=adminadmin
  AWS_DEFAULT_REGION=us-east-1
  MINIO_BUCKET=mlflow
  MINIO_PREFIX=mntrading/

All comments are in English by request.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess

# ---------------- storage config ----------------
REMOTE_ONLY = os.getenv("REMOTE_ONLY", "0").lower() in ("1", "true", "yes", "on")
MINIO_ENABLED = any(os.getenv(k) for k in ("MLFLOW_S3_ENDPOINT_URL", "MINIO_BUCKET"))

try:
    from utils.minio_io import MinioSink
    SINK = MinioSink.from_env(enabled=MINIO_ENABLED)
except Exception:
    SINK = None

def _remote_enabled() -> bool:
    return SINK is not None

def _maybe_rm(path: Path) -> None:
    """Delete local file/dir if REMOTE_ONLY."""
    if not REMOTE_ONLY:
        return
    if path.is_file():
        try: path.unlink()
        except Exception: pass
    elif path.is_dir():
        try: shutil.rmtree(path)
        except Exception: pass

def _upload_file(local: Path, key: str) -> None:
    if _remote_enabled() and local.exists():
        try: SINK.upload_file(local, key)  # type: ignore[arg-type]
        except Exception: pass
        _maybe_rm(local)

def _upload_dir(local_dir: Path, prefix: str) -> None:
    if _remote_enabled() and local_dir.exists():
        try: SINK.upload_dir(local_dir, prefix)  # type: ignore[arg-type]
        except Exception: pass
        _maybe_rm(local_dir)

def _ensure_local_file(key: str, local: Path) -> None:
    """Fetch single object from MinIO if local missing."""
    if local.exists():
        return
    if not _remote_enabled():
        return
    try:
        if SINK.exists(key):  # type: ignore[attr-defined]
            local.parent.mkdir(parents=True, exist_ok=True)
            SINK.download_file(key, local)  # type: ignore[arg-type]
    except Exception:
        pass

def _ensure_local_dir(prefix: str, local_dir: Path) -> None:
    """Fetch all objects under prefix into local_dir if dir empty."""
    if local_dir.exists() and any(local_dir.rglob("*")):
        return
    if not _remote_enabled():
        return
    try:
        local_dir.mkdir(parents=True, exist_ok=True)
        SINK.download_dir(prefix, local_dir)  # type: ignore[arg-type]
    except Exception:
        pass

# ---------------- canonical paths (local cache) ----------------
DATA = Path("data")
RAW_DIR = DATA / "raw"
PAIRS_DIR = DATA / "pairs"
FEATURES_DIR = DATA / "features" / "pairs"
DATASETS_DIR = DATA / "datasets" / "pairs"
MODELS_DIR = DATA / "models"
MODELS_PAIRS_DIR = MODELS_DIR / "pairs"
BACKTEST_DIR = DATA / "backtest_results"
SIGNALS_DIR = DATA / "signals"
PORTFOLIO_DIR = DATA / "portfolio"

for d in (RAW_DIR, PAIRS_DIR, FEATURES_DIR, DATASETS_DIR, MODELS_DIR, MODELS_PAIRS_DIR, BACKTEST_DIR, SIGNALS_DIR, PORTFOLIO_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- optional project modules ----------------
try:
    from features import spread as _spread
    HAS_SPREAD = True
except Exception:
    _spread = None
    HAS_SPREAD = False

try:
    from features.labels import DatasetBuildConfig, build_datasets_for_manifest
    HAS_LABELS = True
except Exception:
    DatasetBuildConfig = None
    build_datasets_for_manifest = None
    HAS_LABELS = False

try:
    from models.train import train_baseline
    HAS_TRAIN = True
except Exception:
    train_baseline = None
    HAS_TRAIN = False

try:
    from backtest.runner import run_backtest
    HAS_BT = True
except Exception:
    run_backtest = None
    HAS_BT = False

# ---------------- symbol universe (ccxt, no external loader) ----------------
STABLES = {"USDT", "BUSD", "USDC", "TUSD", "FDUSD", "DAI"}

def _ccxt_exchange(name: str):
    """Return a ccxt exchange instance by name (e.g., 'binance')."""
    try:
        import ccxt  # type: ignore
    except Exception:
        raise SystemExit("ccxt is required: pip install ccxt")
    name = (name or "binance").lower()
    if not hasattr(ccxt, name):
        raise SystemExit(f"Unknown exchange: {name}")
    ex = getattr(ccxt, name)()
    ex.load_markets()
    return ex

def _list_spot_symbols_ccxt(exchange: str = "binance", quote: str = "USDT", top: int = 200,
                            exclude_stables: bool = True, use_tickers_sort: bool = True) -> List[str]:
    """
    Build a large universe of spot symbols with given quote from ccxt.
    Sorted by 24h quote volume when possible.
    """
    ex = _ccxt_exchange(exchange)
    quote = quote.upper().strip()

    syms: List[str] = []
    for sym, m in ex.markets.items():
        if not m.get("active"):  # skip inactive
            continue
        if not m.get("spot"):    # spot only
            continue
        if m.get("quote") != quote:
            continue
        base = str(m.get("base", "")).upper().strip()
        if exclude_stables and base in STABLES:
            continue
        syms.append(sym)

    if not syms:
        return syms

    if use_tickers_sort:
        try:
            tickers = ex.fetch_tickers(syms)
            def _qv(t: dict) -> float:
                info = t.get("info") or {}
                for k in ("quoteVolume", "volumeQuote", "qv"):
                    v = info.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
                v = t.get("quoteVolume")
                try:
                    return float(v) if v is not None else 0.0
                except Exception:
                    return 0.0
            syms.sort(key=lambda s: _qv(tickers.get(s, {})), reverse=True)
        except Exception:
            syms.sort()

    if top and top > 0:
        syms = syms[: int(top)]
    return syms

# ---------------- ingest (paginated OHLCV) ----------------
def _fetch_ohlcv_all_ccxt(ex, symbol: str, timeframe: str, since_ms: Optional[int],
                          per_page: int, max_candles: Optional[int]) -> pd.DataFrame:
    """
    Paginated OHLCV fetch:
    - 1000 per page (Binance cap) or per_page provided
    - continue while there is forward progress
    - optional total cap per symbol via max_candles
    """
    rows_all = []
    last_ts = None
    # very high upper bound; loop breaks by progress/lack of rows
    for _ in range(100000):
        rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=per_page)
        if not rows:
            break
        if last_ts is not None and rows[-1][0] <= last_ts:
            break
        rows_all.extend(rows)
        last_ts = rows[-1][0]
        since_ms = last_ts + 1
        # respect rate limit
        rl = getattr(ex, "rateLimit", 500)
        time.sleep((rl or 500) / 1000.0)
        if max_candles and len(rows_all) >= max_candles:
            break

    if not rows_all:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume","symbol"])
    df = pd.DataFrame(rows_all, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df

def _ingest_ccxt(symbols: List[str], timeframe: str, since_utc: Optional[str],
                 limit: int, max_candles: Optional[int], exchange: str) -> pd.DataFrame:
    ex = _ccxt_exchange(exchange)
    per_page = min(int(limit) if limit else 1000, 1000)
    since_ms = ex.parse8601(since_utc) if since_utc else None

    frames = []
    for s in tqdm(symbols, desc=f"Ingest(ccxt:{timeframe})"):
        if s not in ex.markets:
            print(f"[warn] skip {s}: not in exchange")
            continue
        frames.append(_fetch_ohlcv_all_ccxt(ex, s, timeframe, since_ms, per_page, max_candles))
    if not frames:
        raise SystemExit("Nothing ingested (ccxt)")
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol","ts"]).reset_index(drop=True)
    return out

# ---------------- helpers ----------------
def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def _load_pairs_from_json(path: str) -> List[Tuple[str, str]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str]] = []
    if isinstance(obj, dict) and "pairs" in obj:
        obj = obj["pairs"]
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, (list, tuple)) and len(it) == 2:
                a, b = str(it[0]).strip(), str(it[1]).strip()
                if a and b: pairs.append((a, b))
            elif isinstance(it, dict):
                for ka, kb in (("a","b"),("A","B"),("x","y")):
                    if ka in it and kb in it:
                        a, b = str(it[ka]).strip(), str(it[kb]).strip()
                        if a and b: pairs.append((a, b))
                        break
            elif isinstance(it, str) and "|" in it:
                a, b = it.split("|", 1)
                a, b = a.strip(), b.strip()
                if a and b: pairs.append((a, b))
    return pairs

def _parse_symbols_arg(symbols_arg: Optional[str]) -> Tuple[List[str], List[Tuple[str, str]], Optional[str]]:
    if not symbols_arg:
        return [], [], None
    s = symbols_arg.strip()
    if s.endswith(".json") or "*" in s:
        files = sorted([Path(p) for p in glob.glob(s)])
        pairs: List[Tuple[str, str]] = []
        for f in files:
            try:
                pairs.extend(_load_pairs_from_json(str(f)))
            except Exception as e:
                print(f"[warn] failed to parse {f}: {e}")
        return [], pairs, str(files[-1]) if files else None
    return [x.strip() for x in s.split(",") if x.strip()], [], None

def _pairs_to_symbol_union(pairs: List[Tuple[str, str]]) -> List[str]:
    s = set()
    for a,b in pairs:
        s.add(a); s.add(b)
    return sorted(s)

# ---------------- steps ----------------
def step_ingest(symbols_arg: Optional[str],
                timeframe: str,
                since_utc: Optional[str],
                limit: int,
                symbols_auto: bool,
                exchange: str,
                quote: str,
                top: int,
                max_candles: Optional[int]):
    """
    Ingest OHLCV for either:
    - explicit CSV symbols via --symbols
    - or auto universe via --symbols-auto/--exchange/--quote/--top
    """
    symbols: List[str] = []
    if symbols_arg:
        sym_csv, _, _ = _parse_symbols_arg(symbols_arg)
        symbols = sym_csv

    if (not symbols) and symbols_auto:
        symbols = _list_spot_symbols_ccxt(exchange=exchange, quote=quote, top=top,
                                          exclude_stables=True, use_tickers_sort=True)

    if not symbols:
        raise SystemExit("No symbols resolved for ingest (use --symbols or --symbols-auto).")

    ohlcv = _ingest_ccxt(
        symbols=symbols,
        timeframe=timeframe,
        since_utc=since_utc,
        limit=limit,
        max_candles=max_candles,
        exchange=exchange,
    )
    if ohlcv.empty:
        raise SystemExit("Ingest produced empty dataframe")

    out_file = RAW_DIR / f"ohlcv_{timeframe}.parquet"
    _save_parquet(ohlcv, out_file)
    _upload_file(out_file, f"raw/{out_file.name}")

    latest = RAW_DIR / "ohlcv.parquet"
    _save_parquet(ohlcv, latest)
    _upload_file(latest, "raw/ohlcv.parquet")

    meta = {
        "exchange": exchange,
        "quote": quote,
        "timeframe": timeframe,
        "since_utc": since_utc,
        "symbols": sorted(ohlcv["symbol"].unique().tolist()),
        "rows": int(len(ohlcv)),
        "max_candles": int(max_candles or 0),
        "per_page": int(min(limit, 1000)),
    }
    meta_file_tf = RAW_DIR / f"ohlcv_{timeframe}_meta.json"
    meta_file = RAW_DIR / "ohlcv_meta.json"
    meta_file_tf.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _upload_file(meta_file_tf, f"raw/{meta_file_tf.name}")
    _upload_file(meta_file, "raw/ohlcv_meta.json")

    print(f"[ingest] -> s3://.../raw/{out_file.name} (symbols: {len(meta['symbols'])}, rows: {meta['rows']})")

def step_screen(symbols_arg: Optional[str], since_utc_5m: str, limit_5m: int):
    # Ensure 1h parquet present locally
    _ensure_local_file("raw/ohlcv_1h.parquet", RAW_DIR / "ohlcv_1h.parquet")
    raw_1h = RAW_DIR / "ohlcv_1h.parquet"
    if not raw_1h.exists():
        raise SystemExit(f"1h parquet missing: {raw_1h}. Run ingest 1h first.")

    script = Path("screen_pairs.py")
    if not script.exists():
        raise SystemExit("screen_pairs.py not found in project root.")
    cmd = [sys.executable, str(script), "--raw-parquet", str(raw_1h)]
    if symbols_arg:
        cmd += ["--symbols", symbols_arg]
    print("[screen] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"screen_pairs.py failed with code {rc}")

    # Upload pairs artifacts and (if REMOTE_ONLY) clear local
    _upload_dir(PAIRS_DIR, "pairs/")

    js = sorted(PAIRS_DIR.glob("screened_pairs_*.json"), key=lambda p: p.stat().st_mtime)
    if not js:
        if _remote_enabled():
            keys = [k for k in SINK.list_prefix("pairs/") if k.endswith(".json")]  # type: ignore[attr-defined]
            keys.sort()
            if not keys:
                raise SystemExit("screen produced no screened_pairs_*.json")
            last_key = keys[-1]
            tmp_json = PAIRS_DIR / "_last_screened_pairs.json"
            SINK.download_file(last_key, tmp_json)  # type: ignore[arg-type]
            pairs = _load_pairs_from_json(str(tmp_json))
            tmp_json.unlink(missing_ok=True)
        else:
            raise SystemExit("screen produced no screened_pairs_*.json")
    else:
        pairs = _load_pairs_from_json(str(js[-1]))

    if not pairs:
        raise SystemExit("parsed 0 pairs after screen")
    symbols = _pairs_to_symbol_union(pairs)
    print(f"[screen] selected pairs={len(pairs)} symbols={len(symbols)}")

    # Auto-ingest 5m for selected symbols (paginated)
    step_ingest(
        symbols_arg=",".join(symbols),
        timeframe="5m",
        since_utc=since_utc_5m,
        limit=1000,
        symbols_auto=False,
        exchange="binance",
        quote="USDT",
        top=0,
        max_candles=None,
    )
    print("[screen] 5m ready: s3://.../raw/ohlcv_5m.parquet")

def step_features(symbols_arg: str, beta_window: int, z_window: int):
    _ensure_local_file("raw/ohlcv_5m.parquet", RAW_DIR / "ohlcv_5m.parquet")
    p5 = RAW_DIR / "ohlcv_5m.parquet"
    if not p5.exists():
        raise SystemExit(f"{p5} not found; run screen first to auto-ingest 5m")
    ohlcv = pd.read_parquet(p5)

    symbols, pairs, _ = _parse_symbols_arg(symbols_arg)
    produced: List[str] = []

    if HAS_SPREAD and hasattr(_spread, "compute_features_for_pairs"):
        if pairs:
            res = _spread.compute_features_for_pairs(raw_df=ohlcv, pairs=pairs, beta_window=beta_window, z_window=z_window)
            for pk, df in res.items():
                pair_dir = FEATURES_DIR / pk.replace("/", "_")
                pair_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(pair_dir / "features.parquet", index=False)
                produced.append(pk)
        else:
            if not symbols:
                symbols = sorted(ohlcv["symbol"].dropna().unique().tolist())
            todo = [(symbols[i], symbols[j]) for i in range(len(symbols)) for j in range(i+1, len(symbols))]
            res = _spread.compute_features_for_pairs(raw_df=ohlcv, pairs=todo, beta_window=beta_window, z_window=z_window)
            for pk, df in res.items():
                pair_dir = FEATURES_DIR / pk.replace("/", "_")
                pair_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(pair_dir / "features.parquet", index=False)
                produced.append(pk)
    else:
        # Minimal fallback (should not be used if features/spread.py is present)
        piv = ohlcv.pivot(index="ts", columns="symbol", values="close").sort_index()

        def _rolling_beta_alpha(y: pd.Series, x: pd.Series, win: int):
            x_mean = x.rolling(win).mean(); y_mean = y.rolling(win).mean()
            cov = (x*y).rolling(win).mean() - x_mean*y_mean
            var = x.rolling(win).var()
            beta = cov / var.replace(0.0, np.nan)
            alpha = y_mean - beta * x_mean
            return beta, alpha

        if pairs:
            todo = pairs
        else:
            symbols = symbols or sorted(piv.columns.tolist())
            todo = [(symbols[i], symbols[j]) for i in range(len(symbols)) for j in range(i+1, len(symbols))]

        for a,b in tqdm(todo, desc="Features(minimal)"):
            if a not in piv.columns or b not in piv.columns: continue
            px = piv[[a,b]].dropna().sort_index()
            a_close, b_close = px[a].astype("float64"), px[b].astype("float64")
            beta, alpha = _rolling_beta_alpha(a_close, b_close, int(beta_window))
            spread = a_close - (beta*b_close + alpha)
            z = (spread - spread.rolling(z_window).mean()) / (spread.rolling(z_window).std().replace(0.0, np.nan))
            df = pd.DataFrame({"ts": px.index, "a_close": a_close, "b_close": b_close,
                               "beta": beta, "alpha": alpha, "spread": spread, "z": z}).dropna().reset_index(drop=True)
            pk = f"{a.replace('/','_')}__{b.replace('/','_')}"
            pdir = FEATURES_DIR / pk
            pdir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pdir / "features.parquet", index=False)
            produced.append(pk)

    items = []
    for pk in sorted(set(produced)):
        p = (FEATURES_DIR / pk.replace("/", "_") / "features.parquet").resolve()
        if p.exists():
            items.append({"pair": pk, "path": str(p), "features": ["a","b","beta","alpha","spread","z"]})
    manifest_path = FEATURES_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps({"items": items, "features_dir": str(FEATURES_DIR.resolve())}, indent=2), encoding="utf-8")
    print(f"[features] built {len(items)} pairs -> {FEATURES_DIR}")

    _upload_dir(FEATURES_DIR, "features/")
    _ensure_local_file("features/_manifest.json", FEATURES_DIR.parent / "_manifest.json")
    _upload_file(manifest_path, "features/_manifest.json")
    _upload_file(manifest_path, "features/pairs/_manifest.json")

def step_dataset(pairs_manifest: str, label_type: str, z_th: float, lag_features: int, horizon: int, out_dir: Optional[str] = None):
    _ensure_local_dir("features/", FEATURES_DIR.parent)
    local_pairs_manifest = FEATURES_DIR / "_manifest.json"
    local_root_manifest = FEATURES_DIR.parent / "_manifest.json"
    if not local_pairs_manifest.exists():
        _ensure_local_file("features/pairs/_manifest.json", local_pairs_manifest)
    use_manifest_path = None
    if local_pairs_manifest.exists():
        use_manifest_path = str(local_pairs_manifest)
        print("[dataset] using manifest:", use_manifest_path)
    else:
        _ensure_local_file("features/_manifest.json", local_root_manifest)
        if local_root_manifest.exists():
            use_manifest_path = str(local_root_manifest)
            print("[dataset] using manifest (root):", use_manifest_path)
        else:
            raise SystemExit("features manifest not found locally nor in MinIO (tried pairs/_manifest.json and _manifest.json)")

    if not HAS_LABELS:
        raise SystemExit("features.labels not available")
    out_dir_path = Path(out_dir) if out_dir else DATASETS_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)
    cfg = DatasetBuildConfig(label_type=label_type, zscore_threshold=float(z_th), lag_features=int(lag_features), horizon=int(horizon))
    man = build_datasets_for_manifest(features_manifest=use_manifest_path, out_dir=out_dir_path, cfg=cfg)
    print(f"[dataset] built {len(man.get('items', []))} datasets -> {out_dir_path}")
    _upload_dir(out_dir_path, "datasets/")
    parent_manifest = out_dir_path.parent / "_manifest.json"
    if parent_manifest.exists():
        _upload_file(parent_manifest, "datasets/_manifest.json")

def step_train(use_dataset: bool, n_splits: int, gap: int, max_train_size: int, early_stopping_rounds: int, proba_threshold: float):
    _ensure_local_dir("datasets/", DATASETS_DIR.parent)
    _ensure_local_dir("features/", FEATURES_DIR.parent)
    if not HAS_TRAIN: raise SystemExit("models.train not available")
    report = train_baseline(datasets_dir=str(DATASETS_DIR), features_dir=str(FEATURES_DIR), out_dir=str(MODELS_DIR),
                            use_dataset=use_dataset, n_splits=n_splits, gap=gap, max_train_size=max_train_size,
                            early_stopping_rounds=early_stopping_rounds, proba_threshold=proba_threshold)
    report_path = MODELS_DIR / "_train_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _upload_dir(MODELS_DIR, "models/")

    # ---- MLflow registration & challenger/champion promotion (injected) ----
    try:
        import os, json as _json, statistics as _stats
        import mlflow
        from mlflow.tracking import MlflowClient

        # Aggregate AUC over champions across pairs (fallback to accuracy if AUC missing)
        aucs = []
        accs = []
        for item in (report.get("items") or []):
            if item.get("skipped"):
                continue
            met = (item.get("metrics") or {}).get(item.get("champion") or "", {})
            if isinstance(met, dict):
                if "auc" in met and isinstance(met["auc"], (int, float)):
                    aucs.append(float(met["auc"]))
                if "accuracy" in met and isinstance(met["accuracy"], (int, float)):
                    accs.append(float(met["accuracy"]))
        agg_auc = float(_stats.fmean(aucs)) if aucs else None
        agg_acc = float(_stats.fmean(accs)) if accs else None

        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "mntrading")
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name="train_full", nested=True) as run:
            if agg_auc is not None:
                mlflow.log_metric("oof_auc_mean", agg_auc)
            if agg_acc is not None:
                mlflow.log_metric("oof_acc_mean", agg_acc)
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            # Log the whole models dir as a bundle
            mlflow.log_artifacts(str(MODELS_DIR), artifact_path="bundle")
            run_id = run.info.run_id

        client = MlflowClient()
        model_name = os.getenv("MLFLOW_MODEL_NAME", "mntrading_model")
        mv = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/bundle",
            run_id=run_id
        )

        # Simple challenger/champion: compare oof_auc_mean (or oof_acc_mean)
        def _get_metric(rid, key):
            r = client.get_run(rid)
            return r.data.metrics.get(key)

        # Find current Production, if any
        current_prod = None
        for v in client.search_model_versions(f"name='{model_name}'"):
            if v.current_stage == "Production":
                current_prod = v
                break

        new_auc = _get_metric(run_id, "oof_auc_mean")
        new_acc = _get_metric(run_id, "oof_acc_mean")

        better = False
        if current_prod is None:
            better = True
        else:
            old_auc = _get_metric(current_prod.run_id, "oof_auc_mean")
            old_acc = _get_metric(current_prod.run_id, "oof_acc_mean")
            # Prefer AUC if available, else accuracy
            if new_auc is not None and old_auc is not None:
                better = (new_auc >= old_auc)
            elif new_acc is not None and old_acc is not None:
                better = (new_acc >= old_acc)
            elif new_auc is not None and old_auc is None:
                better = True
            elif new_acc is not None and old_acc is None:
                better = True

        if better:
            if current_prod is not None:
                client.transition_model_version_stage(model_name, current_prod.version, stage="Archived")
            client.transition_model_version_stage(model_name, mv.version, stage="Production")
            print(f"[mlflow] Promoted {model_name} v{mv.version} to Production")
        else:
            client.transition_model_version_stage(model_name, mv.version, stage="Staging")
            print(f"[mlflow] Registered {model_name} v{mv.version} to Staging")
    except Exception as e:
        print(f"[mlflow] WARN: registration/promotion skipped: {e}")
    # ---- end MLflow block ----
    print(f"[train] report -> s3://.../models/_train_report.json")


def step_backtest(signals_from: str, proba_threshold: float, fee_rate: float):
    """Run backtest and persist a normalized summary JSON for the UI/report."""
    _ensure_local_dir("features/", FEATURES_DIR.parent)
    _ensure_local_dir("datasets/", DATASETS_DIR.parent)
    _ensure_local_dir("models/", MODELS_DIR)
    if not HAS_BT:
        raise SystemExit("backtest.runner not available")

    # run the actual backtest
    summary = run_backtest(
        features_dir=str(FEATURES_DIR),
        datasets_dir=str(DATASETS_DIR),
        models_dir=str(MODELS_PAIRS_DIR),
        out_dir=str(BACKTEST_DIR),
        signals_from=signals_from,
        proba_threshold=proba_threshold,
        fee_rate=fee_rate,
    )

    # Normalize keys for UI stability
    def _f(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    norm = {
        "start": summary.get("start"),
        "end": summary.get("end"),
        "n_trades": int(summary.get("n_trades", 0) or 0),
        "gross_return": _f(summary.get("gross_return", summary.get("return", 0.0)), 0.0),
        "net_return": _f(summary.get("net_return", summary.get("net", summary.get("return", 0.0))), 0.0),
        "max_dd": _f(summary.get("max_dd", summary.get("max_drawdown", 0.0)), 0.0),
        "sharpe": _f(summary.get("sharpe", 0.0), 0.0),
        "win_rate": _f(summary.get("win_rate", 0.0), 0.0),
        "threshold": float(proba_threshold),
        "signals_from": str(signals_from),
        "pairs": summary.get("pairs", {}),
    }

    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = BACKTEST_DIR / "_summary.json"
    summary_path.write_text(json.dumps(norm, indent=2), encoding="utf-8")
    _upload_dir(BACKTEST_DIR, "backtest/")
    print(f"[backtest] summary -> {summary_path}")


if __name__ == "__main__":
    main()
