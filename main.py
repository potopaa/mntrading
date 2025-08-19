#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mntrading CLI with remote-only storage option.

Key idea:
- Persist everything in MinIO (S3). Local files are only ephemeral caches.
- When REMOTE_ONLY=1: inputs are fetched from MinIO before a step; outputs are uploaded and local cache is removed.

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
    """Fetch object from MinIO if local missing."""
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

# ---------------- optional imports ----------------
HAS_LOADER = False
try:
    from data_loader import loader as _loader
    HAS_LOADER = True
except Exception:
    _loader = None
    HAS_LOADER = False

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

def _fallback_ccxt_ingest(symbols: List[str], timeframe: str, since_utc: Optional[str], limit: int) -> pd.DataFrame:
    """
    Robust CCXT ingest with pagination:
    - per-page capped at 1000 (Binance hard cap)
    - no premature break on short pages; we continue until no new candles
    """
    try:
        import ccxt  # type: ignore
    except Exception:
        raise SystemExit("ccxt not installed and loader not provided.")
    ex = ccxt.binance()
    ex.load_markets()

    per_page = min(int(limit), 1000) if limit else 1000

    def _fetch_all(sym: str) -> pd.DataFrame:
        since_ms = ex.parse8601(since_utc) if since_utc else None
        all_rows = []
        last_ts = None
        for _ in range(10000):  # large upper bound; loop breaks by progress
            rows = ex.fetch_ohlcv(sym, timeframe=timeframe, since=since_ms, limit=per_page)
            if not rows:
                break
            if last_ts is not None and rows[-1][0] <= last_ts:
                # no forward progress -> stop
                break
            all_rows.extend(rows)
            last_ts = rows[-1][0]
            since_ms = last_ts + 1
            # Respect rate limit
            time.sleep(ex.rateLimit / 1000.0)
            # DO NOT break on len(rows) < per_page — exchanges cap page size
        if not all_rows:
            return pd.DataFrame(columns=["ts","open","high","low","close","volume","symbol"])
        df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["symbol"] = sym
        return df

    frames = []
    for s in tqdm(symbols, desc=f"Ingest(ccxt:{timeframe})"):
        if s not in ex.markets:
            print(f"[warn] skip {s}: not in exchange")
            continue
        frames.append(_fetch_all(s))
    if not frames:
        raise SystemExit("Nothing ingested (ccxt)")
    return pd.concat(frames, ignore_index=True).sort_values(["symbol","ts"])

# ---------------- steps ----------------
def step_ingest(symbols_arg: str, timeframe: str, since_utc: Optional[str], limit: int):
    symbols, _, _ = _parse_symbols_arg(symbols_arg)
    if not symbols:
        raise SystemExit("No symbols for ingest")

    if HAS_LOADER and hasattr(_loader, "load_ohlcv_for_symbols"):
        ohlcv = _loader.load_ohlcv_for_symbols(symbols=symbols, timeframe=timeframe, since_utc=since_utc, limit=limit)
    else:
        ohlcv = _fallback_ccxt_ingest(symbols, timeframe=timeframe, since_utc=since_utc, limit=limit)
    if ohlcv.empty:
        raise SystemExit("Ingest produced empty dataframe")

    out_file = RAW_DIR / f"ohlcv_{timeframe}.parquet"
    _save_parquet(ohlcv, out_file)
    _upload_file(out_file, f"raw/{out_file.name}")

    # Also write "latest" alias and meta
    latest = RAW_DIR / "ohlcv.parquet"
    _save_parquet(ohlcv, latest)
    _upload_file(latest, "raw/ohlcv.parquet")

    meta = {
        "timeframe": timeframe,
        "since_utc": since_utc,
        "symbols": sorted(ohlcv["symbol"].unique().tolist()),
        "rows": int(len(ohlcv)),
    }
    meta_file_tf = RAW_DIR / f"ohlcv_{timeframe}_meta.json"
    meta_file = RAW_DIR / "ohlcv_meta.json"
    meta_file_tf.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _upload_file(meta_file_tf, f"raw/{meta_file_tf.name}")
    _upload_file(meta_file, "raw/ohlcv_meta.json")

    print(f"[ingest] -> s3://.../raw/{out_file.name}")

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

    js = sorted(PAIRS_DIR.glob("screened_pairs_*.json"), key=lambda p: p.stat().st_mtime)  # may be removed if REMOTE_ONLY
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

    # Auto-ingest 5m for selected symbols (now paginated)
    step_ingest(",".join(symbols), timeframe="5m", since_utc=since_utc_5m, limit=limit_5m)
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
    print(f"[train] report -> s3://.../models/_train_report.json")

def step_backtest(signals_from: str, proba_threshold: float, fee_rate: float):
    _ensure_local_dir("features/", FEATURES_DIR.parent)
    _ensure_local_dir("datasets/", DATASETS_DIR.parent)
    _ensure_local_dir("models/", MODELS_DIR)
    if not HAS_BT: raise SystemExit("backtest.runner not available")
    summary = run_backtest(features_dir=str(FEATURES_DIR), datasets_dir=str(DATASETS_DIR), models_dir=str(MODELS_PAIRS_DIR),
                           out_dir=str(BACKTEST_DIR), signals_from=signals_from, proba_threshold=proba_threshold, fee_rate=fee_rate)
    summary_path = BACKTEST_DIR / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _upload_dir(BACKTEST_DIR, "backtest/")
    print(f"[backtest] summary -> s3://.../backtest/_summary.json")

def step_select(summary_path: str, registry_out: str, sharpe_min: float, maxdd_max: float, top_k: int,
                require_oof: bool=False, min_auc: Optional[float]=None, min_rows: Optional[int]=None, max_per_symbol: Optional[int]=None):
    _ensure_local_dir("backtest/", BACKTEST_DIR)
    _ensure_local_file("models/_train_report.json", MODELS_DIR / "_train_report.json")
    try:
        from models.select import select_champions as _select
        _select(summary_path=summary_path, registry_out=registry_out, sharpe_min=sharpe_min, maxdd_max=maxdd_max, top_k=top_k,
                require_oof=require_oof, train_report_path=str(MODELS_DIR / "_train_report.json"),
                min_auc=min_auc, min_rows=min_rows, max_per_symbol=max_per_symbol)
    except Exception as e:
        print(f"[select] fallback due to: {e}")
        obj = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        pairs = obj.get("pairs", {})
        rows = []
        for pair, it in pairs.items():
            met = (it or {}).get("metrics") or {}
            try:
                sharpe = float(met.get("sharpe"))
                maxdd = float(met.get("maxdd"))
            except Exception:
                continue
            if np.isnan(sharpe) or np.isnan(maxdd): continue
            if sharpe >= float(sharpe_min) and maxdd <= float(maxdd_max):
                rows.append((pair, sharpe, maxdd))
        rows.sort(key=lambda t: t[1], reverse=True)
        rows = rows[: int(top_k)]
        reg = {"pairs": [{"pair": r[0], "rank": i+1, "metrics": {"sharpe": r[1], "maxdd": r[2]}} for i, r in enumerate(rows)]}
        Path(registry_out).parent.mkdir(parents=True, exist_ok=True)
        Path(registry_out).write_text(json.dumps(reg, indent=2), encoding="utf-8")
    _upload_file(Path(registry_out), f"models/{Path(registry_out).name}")
    print(f"[select] registry -> s3://.../models/{Path(registry_out).name}")

def step_promote(registry_in: str, production_map_out: str):
    _ensure_local_file(f"models/{Path(registry_in).name}", Path(registry_in))
    p = Path(registry_in)
    if not p.exists(): raise SystemExit(f"Registry not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    pairs = [it["pair"] for it in obj.get("pairs", [])]
    prod = {"pairs": pairs}
    out = Path(production_map_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prod, indent=2), encoding="utf-8")
    _upload_file(out, f"models/{out.name}")
    print(f"[promote] production map -> s3://.../models/{out.name}")

def step_inference(registry_in: str, pairs_manifest: str, signals_from: str, proba_threshold: float,
                   signals_out: Optional[str]=None, n_last: int=1, update: bool=True, skip_flat: bool=False):
    _ensure_local_file(f"models/{Path(registry_in).name}", Path(registry_in))
    _ensure_local_dir("features/", FEATURES_DIR.parent)
    script = Path("inference.py")
    if not script.exists(): raise SystemExit("inference.py not found.")
    out_dir = Path(signals_out) if signals_out else SIGNALS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script), "--registry", str(registry_in),
           "--pairs-manifest", str(pairs_manifest), "--signals-from", str(signals_from),
           "--proba-threshold", str(proba_threshold), "--model-dir", str(MODELS_PAIRS_DIR),
           "--n-last", str(int(n_last)), "--out", str(out_dir)]
    if update: cmd.append("--update")
    if skip_flat: cmd.append("--skip-flat")
    print("[inference] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0: raise SystemExit(f"inference.py failed with code {rc}")
    _upload_dir(out_dir, "signals/")

def step_aggregate(signals_dir: Optional[str], pairs_manifest: str, min_proba: float, top_k: int, out_dir: Optional[str]=None):
    _ensure_local_dir("signals/", SIGNALS_DIR)
    _ensure_local_dir("features/", FEATURES_DIR.parent)
    script = Path("portfolio") / "aggregate_signals.py"
    if not script.exists(): raise SystemExit("portfolio/aggregate_signals.py not found.")
    sig_dir = Path(signals_dir) if signals_dir else SIGNALS_DIR
    out_p = Path(out_dir) if out_dir else PORTFOLIO_DIR
    out_p.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(script),
           "--signals-dir", str(sig_dir),
           "--pairs-manifest", str(pairs_manifest),
           "--min-proba", str(min_proba),
           "--top-k", str(int(top_k)),
           "--out-dir", str(out_p)]
    print("[aggregate] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0: raise SystemExit(f"aggregate_signals.py failed with code {rc}")
    _upload_dir(out_p, "portfolio/")

def step_report(orders_json: Optional[str]=None, backtest_summary: Optional[str]=None, out_markdown: Optional[str]=None):
    _ensure_local_dir("portfolio/", PORTFOLIO_DIR)  # optional inputs
    _ensure_local_dir("backtest/", BACKTEST_DIR)
    script = Path("portfolio") / "report_latest.py"
    if not script.exists(): raise SystemExit("portfolio/report_latest.py not found.")
    cmd = [sys.executable, str(script)]
    if orders_json: cmd += ["--orders-json", str(orders_json)]
    if backtest_summary: cmd += ["--backtest-summary", str(backtest_summary)]
    if out_markdown: cmd += ["--out", str(out_markdown)]
    print("[report] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0: raise SystemExit(f"report_latest.py failed with code {rc}")
    _upload_dir(PORTFOLIO_DIR, "portfolio/")

# ---------------- CLI ----------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="mntrading CLI")
    ap.add_argument("--mode", required=True,
                    choices=["screen","ingest","features","dataset","train","backtest","select","promote","inference","aggregate","report"])

    # ingest
    ap.add_argument("--symbols", type=str, help="CSV of symbols OR path/glob to screened_pairs_*.json")
    ap.add_argument("--timeframe", type=str, default="5m")
    ap.add_argument("--since-utc", type=str, default=None)
    ap.add_argument("--limit", type=int, default=1000)

    # screen (auto 5m ingest control)
    ap.add_argument("--since-utc-5m", type=str, default="2025-01-01T00:00:00Z")
    ap.add_argument("--limit-5m", type=int, default=1000)

    # features
    ap.add_argument("--beta-window", type=int, default=300)
    ap.add_argument("--z-window", type=int, default=300)

    # dataset
    ap.add_argument("--pairs-manifest", type=str, default=str(FEATURES_DIR / "_manifest.json"))
    ap.add_argument("--label-type", type=str, default="z_threshold", choices=["z_threshold","revert_direction"])
    ap.add_argument("--zscore-threshold", type=float, default=1.5)
    ap.add_argument("--lag-features", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=None)

    # train/backtest
    ap.add_argument("--use-dataset", action="store_true")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=24)
    ap.add_argument("--max-train-size", type=int, default=0)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--proba-threshold", type=float, default=0.55)
    ap.add_argument("--signals-from", type=str, default="auto", choices=["auto","model","z"])
    ap.add_argument("--fee-rate", type=float, default=0.0005)

    # selection/promote
    ap.add_argument("--summary-path", type=str, default=str(BACKTEST_DIR / "_summary.json"))
    ap.add_argument("--registry-out", type=str, default=str(MODELS_DIR / "registry.json"))
    ap.add_argument("--sharpe-min", type=float, default=0.0)
    ap.add_argument("--maxdd-max", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--production-map-out", type=str, default=str(MODELS_DIR / "production_map.json"))
    ap.add_argument("--registry-in", type=str, default=str(MODELS_DIR / "registry.json"))

    # inference / aggregate / report
    ap.add_argument("--signals-dir", type=str, default=str(SIGNALS_DIR))
    ap.add_argument("--portfolio-dir", type=str, default=str(PORTFOLIO_DIR))
    ap.add_argument("--orders-json", type=str, default=str(PORTFOLIO_DIR / "latest_orders.json"))
    ap.add_argument("--report-out", type=str, default=str(PORTFOLIO_DIR / "_latest_report.md"))
    ap.add_argument("--n-last", type=int, default=1)
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--skip-flat", action="store_true")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()
    if args.mode == "ingest":
        step_ingest(args.symbols, args.timeframe, args.since_utc, args.limit)
    elif args.mode == "screen":
        step_screen(args.symbols, args.since_utc_5m, args.limit_5m)
    elif args.mode == "features":
        step_features(args.symbols, args.beta_window, args.z_window)
    elif args.mode == "dataset":
        step_dataset(args.pairs_manifest, args.label_type, args.zscore_threshold, args.lag_features, args.horizon, args.out_dir)
    elif args.mode == "train":
        step_train(args.use_dataset, args.n_splits, args.gap, args.max_train_size, args.early_stopping_rounds, args.proba_threshold)
    elif args.mode == "backtest":
        step_backtest(args.signals_from, args.proba_threshold, args.fee_rate)
    elif args.mode == "select":
        step_select(args.summary_path, args.registry_out, args.sharpe_min, args.maxdd_max, args.top_k,
                    require_oof=getattr(args, "require_oof", False),
                    min_auc=getattr(args, "min_auc", None),
                    min_rows=getattr(args, "min_rows", None),
                    max_per_symbol=getattr(args, "max_per_symbol", None))
    elif args.mode == "promote":
        step_promote(args.registry_in, args.production_map_out)
    elif args.mode == "inference":
        step_inference(args.registry_in, str(FEATURES_DIR / "_manifest.json"),
                       args.signals_from, args.proba_threshold,
                       signals_out=args.signals_dir, n_last=args.n_last, update=bool(args.update), skip_flat=bool(args.skip_flat))
    elif args.mode == "aggregate":
        step_aggregate(args.signals_dir, str(FEATURES_DIR / "_manifest.json"), args.proba_threshold, args.top_k, args.portfolio_dir)
    elif args.mode == "report":
        step_report(args.orders_json, args.summary_path, args.report_out)

if __name__ == "__main__":
    main()
