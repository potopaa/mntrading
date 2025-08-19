#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# --------------------------- Paths ---------------------------
DATA = Path("data")
RAW_DIR = DATA / "raw"
FEATURES_DIR = DATA / "features" / "pairs"
DATASETS_DIR = DATA / "datasets" / "pairs"
MODELS_DIR = DATA / "models"
MODELS_PAIRS_DIR = MODELS_DIR / "pairs"
BACKTEST_DIR = DATA / "backtest_results"

for d in (RAW_DIR, FEATURES_DIR, DATASETS_DIR, MODELS_DIR, MODELS_PAIRS_DIR, BACKTEST_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------- Optional imports ----------------------
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


# ------------------------- Helpers ---------------------------
def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _load_pairs_from_json(path: str) -> List[Tuple[str, str]]:
    """
    Robustly parse pairs JSON. Supported shapes:
      1) {"pairs": [["A","B"], ...]}
      2) {"pairs": [{"a":"A","b":"B"}, ...]}   (also allows keys "A","B" / "x","y")
      3) ["A|B", ...]
      4) [["A","B"], ...]
    """
    raw = Path(path).read_text(encoding="utf-8")
    obj = json.loads(raw)
    pairs: List[Tuple[str, str]] = []

    # Case 1/2: dict with "pairs"
    if isinstance(obj, dict) and "pairs" in obj:
        seq = obj["pairs"]
        if isinstance(seq, list):
            for it in seq:
                # list/tuple of 2
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    a, b = str(it[0]).strip(), str(it[1]).strip()
                    if a and b:
                        pairs.append((a, b))
                    continue
                # dict with keys
                if isinstance(it, dict):
                    for ka, kb in (("a", "b"), ("A", "B"), ("x", "y")):
                        if ka in it and kb in it:
                            a, b = str(it[ka]).strip(), str(it[kb]).strip()
                            if a and b:
                                pairs.append((a, b))
                            break
        return pairs

    # Case 3: list of "A|B" strings
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, str) and "|" in it:
                a, b = it.split("|", 1)
                a, b = a.strip(), b.strip()
                if a and b:
                    pairs.append((a, b))
            elif isinstance(it, (list, tuple)) and len(it) == 2:
                a, b = str(it[0]).strip(), str(it[1]).strip()
                if a and b:
                    pairs.append((a, b))
    return pairs


def _parse_symbols_arg(symbols_arg: Optional[str]) -> Tuple[List[str], List[Tuple[str, str]], Optional[str]]:
    """
    Parse --symbols which can be:
      - CSV of symbols: "BTC/USDT,ETH/USDT"
      - Path/glob to screened_pairs_*.json: "data/pairs/screened_pairs_*.json"
    Returns (symbols, pairs, source_json_used)
    """
    if not symbols_arg:
        return [], [], None
    s = symbols_arg.strip()
    if s.endswith(".json") or "*" in s:
        files = [Path(x) for x in glob.glob(s)]
        files.sort()
        if not files:
            raise SystemExit(f"No pairs json found by glob: {s}")
        pairs: List[Tuple[str, str]] = []
        for p in files:
            try:
                pairs.extend(_load_pairs_from_json(str(p)))
            except Exception as e:
                print(f"[warn] failed to parse {p}: {e}")
        return [], pairs, str(files[-1])
    else:
        symbols = [x.strip() for x in s.split(",") if x.strip()]
        return symbols, [], None


def _fallback_ccxt_ingest(symbols: List[str], timeframe: str, since_utc: Optional[str], limit: int) -> pd.DataFrame:
    """Very small ccxt fallback to fetch OHLCV when local loader is not available."""
    try:
        import ccxt  # type: ignore
    except Exception:
        raise SystemExit("ccxt not installed and loader not provided. Install ccxt or implement data_loader.loader.")
    ex = ccxt.binance()
    ex.load_markets()

    def _fetch_one(sym: str) -> pd.DataFrame:
        since_ms = ex.parse8601(since_utc) if since_utc else None
        rows = ex.fetch_ohlcv(sym, timeframe=timeframe, since=since_ms, limit=limit)
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["symbol"] = sym
        return df

    frames = []
    for s in tqdm(symbols, desc=f"Ingest(ccxt:{timeframe})"):
        if s not in ex.markets:
            print(f"[warn] skip {s}: not in exchange markets")
            continue
        frames.append(_fetch_one(s))
    if not frames:
        raise SystemExit("Nothing ingested (ccxt fallback)")
    ohlcv = pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"])
    return ohlcv


# ------------------------- Steps -----------------------------
def step_ingest(symbols_arg: str, timeframe: str, since_utc: Optional[str], limit: int):

    symbols, _, _ = _parse_symbols_arg(symbols_arg)
    if not symbols:
        raise SystemExit("No symbols provided/resolved for ingest")

    if HAS_LOADER and hasattr(_loader, "load_ohlcv_for_symbols"):
        ohlcv = _loader.load_ohlcv_for_symbols(symbols=symbols, timeframe=timeframe, since_utc=since_utc, limit=limit)
    else:
        ohlcv = _fallback_ccxt_ingest(symbols, timeframe, since_utc, limit)

    if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
        raise SystemExit("Ingest produced empty dataframe")

    tf_file = RAW_DIR / f"ohlcv_{timeframe}.parquet"
    _save_parquet(ohlcv, tf_file)
    _save_parquet(ohlcv, RAW_DIR / "ohlcv.parquet")

    meta = {
        "timeframe": timeframe,
        "since_utc": since_utc,
        "symbols": sorted(ohlcv["symbol"].unique().tolist()) if "symbol" in ohlcv.columns else [],
        "rows": int(len(ohlcv)),
    }
    (RAW_DIR / f"ohlcv_{timeframe}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (RAW_DIR / "ohlcv_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest] ohlcv -> {tf_file}")
    print(f"[ingest] ohlcv -> {RAW_DIR / 'ohlcv.parquet'}")


def step_features(symbols_arg: str, beta_window: int, z_window: int):
    # read raw parquet (combined)
    p_ohlcv = RAW_DIR / "ohlcv.parquet"
    if not p_ohlcv.exists():
        raise SystemExit(f"{p_ohlcv} not found; run ingest first")

    ohlcv = pd.read_parquet(p_ohlcv)
    syms, pairs, src_json = _parse_symbols_arg(symbols_arg)

    produced_pairs: List[str] = []
    if HAS_SPREAD and hasattr(_spread, "compute_features_for_pairs"):
        if pairs:
            res = _spread.compute_features_for_pairs(
                raw_df=ohlcv, pairs=pairs, beta_window=beta_window, z_window=z_window
            )
            for pk, df in res.items():
                pair_dir = (FEATURES_DIR / pk.replace("/", "_"))
                pair_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(pair_dir / "features.parquet", index=False)
                produced_pairs.append(pk)
        else:
            if not syms:
                syms = sorted(ohlcv["symbol"].dropna().unique().tolist())
            pairs_all = [(syms[i], syms[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))]
            res = _spread.compute_features_for_pairs(
                raw_df=ohlcv, pairs=pairs_all, beta_window=beta_window, z_window=z_window
            )
            for pk, df in res.items():
                pair_dir = (FEATURES_DIR / pk.replace("/", "_"))
                pair_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(pair_dir / "features.parquet", index=False)
                produced_pairs.append(pk)
    else:
        print("[features] WARNING: features.spread not available; using minimal internal implementation.")
        piv = ohlcv.pivot(index="ts", columns="symbol", values="close").sort_index()

        def _rolling_beta_alpha(y: pd.Series, x: pd.Series, win: int):
            x_mean = x.rolling(win).mean()
            y_mean = y.rolling(win).mean()
            cov = (x * y).rolling(win).mean() - x_mean * y_mean
            var = x.rolling(win).var()
            beta = cov / (var.replace(0.0, np.nan))
            alpha = y_mean - beta * x_mean
            return beta, alpha

        if pairs:
            todo = pairs
        else:
            syms = syms or sorted(piv.columns.tolist())
            todo = [(syms[i], syms[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))]

        for a, b in tqdm(todo, desc="Features(minimal)"):
            if a not in piv.columns or b not in piv.columns:
                continue
            px = piv[[a, b]].dropna().sort_index()
            a_close = px[a].astype("float64")
            b_close = px[b].astype("float64")
            beta, alpha = _rolling_beta_alpha(y=a_close, x=b_close, win=int(beta_window))
            spread = a_close - (beta * b_close + alpha)
            z = (spread - spread.rolling(z_window).mean()) / (spread.rolling(z_window).std().replace(0.0, np.nan))
            df = pd.DataFrame(
                {
                    "ts": px.index,
                    "a_close": a_close,
                    "b_close": b_close,
                    "beta": beta,
                    "alpha": alpha,
                    "spread": spread,
                    "z": z,
                }
            ).dropna().reset_index(drop=True)
            pk = f"{a.replace('/', '_')}__{b.replace('/', '_')}"
            pair_dir = FEATURES_DIR / pk
            pair_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pair_dir / "features.parquet", index=False)
            produced_pairs.append(f"{a}__{b}")

    # write features manifest
    items = []
    for pk in sorted(set(produced_pairs)):
        p = (FEATURES_DIR / pk.replace("/", "_") / "features.parquet").resolve()
        if p.exists():
            items.append({"pair": pk, "path": str(p), "features": ["a", "b", "beta", "alpha", "spread", "z"]})
    manifest = {"items": items, "features_dir": str(FEATURES_DIR.resolve())}
    (FEATURES_DIR / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[features] built {len(items)} pairs -> {FEATURES_DIR}")
    print(f"[features] manifest -> {FEATURES_DIR / '_manifest.json'}")


def step_dataset(
    pairs_manifest: str,
    label_type: str,
    z_th: float,
    lag_features: int,
    horizon: int,
    out_dir: Optional[str] = None,
):
    if not HAS_LABELS:
        raise SystemExit("features.labels not available. Add features/labels.py with build_datasets_for_manifest().")
    out_dir_path = Path(out_dir) if out_dir else DATASETS_DIR
    out_dir_path.mkdir(parents=True, exist_ok=True)

    cfg = DatasetBuildConfig(
        label_type=label_type,
        zscore_threshold=float(z_th),
        lag_features=int(lag_features),
        horizon=int(horizon),
    )
    man = build_datasets_for_manifest(
        features_manifest=pairs_manifest,
        out_dir=out_dir_path,
        cfg=cfg,
    )
    print(f"[dataset] built {len(man.get('items', []))} datasets -> {out_dir_path}")
    print(f"[dataset] manifest -> {out_dir_path.parent / '_manifest.json'}")


def step_train(use_dataset: bool, n_splits: int, gap: int, max_train_size: int,
               early_stopping_rounds: int, proba_threshold: float):
    if not HAS_TRAIN:
        raise SystemExit("models.train not available. Add models/train.py with train_baseline().")
    report = train_baseline(
        datasets_dir=str(DATASETS_DIR),
        features_dir=str(FEATURES_DIR),
        out_dir=str(MODELS_DIR),
        use_dataset=use_dataset,
        n_splits=n_splits,
        gap=gap,
        max_train_size=max_train_size,
        early_stopping_rounds=early_stopping_rounds,
        proba_threshold=proba_threshold,
    )
    (MODELS_DIR / "_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] report -> {MODELS_DIR / '_train_report.json'}")


def step_backtest(signals_from: str, proba_threshold: float, fee_rate: float):
    """Backtest with current models and features."""
    if not HAS_BT:
        raise SystemExit("backtest.runner not available. Add backtest/runner.py with run_backtest().")
    # IMPORTANT: do not pass unsupported kwargs like 'use_dataset'
    summary = run_backtest(
        features_dir=str(FEATURES_DIR),
        datasets_dir=str(DATASETS_DIR),      # keep if your runner supports it; otherwise it will be ignored if positional
        models_dir=str(MODELS_PAIRS_DIR),
        out_dir=str(BACKTEST_DIR),
        signals_from=signals_from,
        proba_threshold=proba_threshold,
        fee_rate=fee_rate,
    )
    (BACKTEST_DIR / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[backtest] summary -> {BACKTEST_DIR / '_summary.json'}")


def step_select(summary_path: str, registry_out: str, sharpe_min: float, maxdd_max: float, top_k: int,
                require_oof: bool = False, min_auc: Optional[float] = None,
                min_rows: Optional[int] = None, max_per_symbol: Optional[int] = None):
    try:
        from models.select import select_champions as _select
        _select(
            summary_path=summary_path,
            registry_out=registry_out,
            sharpe_min=sharpe_min,
            maxdd_max=maxdd_max,
            top_k=top_k,
            require_oof=require_oof,
            train_report_path=str(MODELS_DIR / "_train_report.json"),
            min_auc=min_auc,
            min_rows=min_rows,
            max_per_symbol=max_per_symbol,
        )
        print(f"[select] registry -> {registry_out}")
    except Exception as e:
        print(f"[select] WARNING: using minimal selector due to: {e}")
        obj = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        pairs = obj.get("pairs", {})
        rows = []
        for pair, it in pairs.items():
            met = (it or {}).get("metrics") or {}
            sharpe = float(met.get("sharpe") or np.nan)
            maxdd = float(met.get("maxdd") or np.nan)
            if np.isnan(sharpe) or np.isnan(maxdd):
                continue
            if sharpe >= float(sharpe_min) and maxdd <= float(maxdd_max):
                rows.append((pair, sharpe, maxdd))
        rows.sort(key=lambda t: t[1], reverse=True)
        rows = rows[: int(top_k)]
        reg = {"pairs": [{"pair": r[0], "rank": i + 1, "metrics": {"sharpe": r[1], "maxdd": r[2]}} for i, r in enumerate(rows)]}
        Path(registry_out).parent.mkdir(parents=True, exist_ok=True)
        Path(registry_out).write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[select] registry -> {registry_out}")


def step_promote(registry_in: str, production_map_out: str):
    p = Path(registry_in)
    if not p.exists():
        raise SystemExit(f"Registry not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    pairs = [it["pair"] for it in obj.get("pairs", [])]
    prod = {"pairs": pairs}
    Path(production_map_out).parent.mkdir(parents=True, exist_ok=True)
    Path(production_map_out).write_text(json.dumps(prod, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[promote] production map -> {production_map_out}")


# -------------------- Extra steps wrappers --------------------
def step_screen(symbols_arg: Optional[str] = None, raw_parquet: Optional[str] = None):
    """Run pairs screening to produce data/pairs/screened_pairs_*.json."""
    script = Path("screen_pairs.py")
    if not script.exists():
        raise SystemExit("screen_pairs.py not found. Please keep it in the project root.")
    raw_par = Path(raw_parquet) if raw_parquet else (RAW_DIR / "ohlcv.parquet")
    if not raw_par.exists():
        raise SystemExit(f"Raw parquet not found: {raw_par}. Run --mode ingest first.")
    cmd = [sys.executable, str(script), "--raw-parquet", str(raw_par)]
    if symbols_arg:
        cmd += ["--symbols", symbols_arg]
    print("[screen] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"screen_pairs.py failed with exit code {rc}")


def step_inference(registry_in: str,
                   pairs_manifest: str,
                   signals_from: str,
                   proba_threshold: float,
                   signals_out: Optional[str] = None,
                   n_last: int = 1,
                   update: bool = True,
                   skip_flat: bool = False):
    """Produce live signals JSONL files under data/signals using inference.py."""
    script = Path("inference.py")
    if not script.exists():
        raise SystemExit("inference.py not found.")
    out_dir = Path(signals_out) if signals_out else (DATA / "signals")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--registry", str(registry_in),
        "--pairs-manifest", str(pairs_manifest),
        "--signals-from", str(signals_from),
        "--proba-threshold", str(proba_threshold),
        "--model-dir", str(MODELS_PAIRS_DIR),
        "--n-last", str(int(n_last)),
        "--out", str(out_dir),
    ]
    if update:
        cmd.append("--update")
    if skip_flat:
        cmd.append("--skip-flat")
    print("[inference] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"inference.py failed with exit code {rc}")


def step_aggregate(signals_dir: Optional[str],
                   pairs_manifest: str,
                   min_proba: float,
                   top_k: int,
                   out_dir: Optional[str] = None):
    """Aggregate per-pair signals into portfolio orders."""
    script = Path("portfolio") / "aggregate_signals.py"
    if not script.exists():
        raise SystemExit("portfolio/aggregate_signals.py not found.")
    sig_dir = Path(signals_dir) if signals_dir else (DATA / "signals")
    out_p = Path(out_dir) if out_dir else (DATA / "portfolio")
    out_p.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--signals-dir", str(sig_dir),
        "--pairs-manifest", str(pairs_manifest),
        "--min-proba", str(min_proba),
        "--top-k", str(int(top_k)),
        "--out-dir", str(out_p),
    ]
    print("[aggregate] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"aggregate_signals.py failed with exit code {rc}")


def step_report(orders_json: Optional[str] = None,
                backtest_summary: Optional[str] = None,
                out_markdown: Optional[str] = None):
    """Generate a small markdown report."""
    script = Path("portfolio") / "report_latest.py"
    if not script.exists():
        raise SystemExit("portfolio/report_latest.py not found.")
    cmd = [sys.executable, str(script)]
    if orders_json:
        cmd += ["--orders-json", str(orders_json)]
    if backtest_summary:
        cmd += ["--backtest-summary", str(backtest_summary)]
    if out_markdown:
        cmd += ["--out", str(out_markdown)]
    print("[report] running:", " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"report_latest.py failed with exit code {rc}")


# ------------------------- CLI -------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="mntrading CLI")
    ap.add_argument("--mode", required=True,
                    choices=["screen", "ingest", "features", "dataset", "train", "backtest", "select", "promote", "inference", "aggregate", "report"])

    # Common / ingest
    ap.add_argument("--symbols", type=str,
                    help="CSV of symbols OR path/glob to screened_pairs_*.json")
    ap.add_argument("--timeframe", type=str, default="5m")
    ap.add_argument("--since-utc", type=str, default=None)
    ap.add_argument("--limit", type=int, default=1000)

    # features
    ap.add_argument("--beta-window", type=int, default=300)
    ap.add_argument("--z-window", type=int, default=300)

    # dataset
    ap.add_argument("--pairs-manifest", type=str, default=str(FEATURES_DIR / "_manifest.json"))
    ap.add_argument("--label-type", type=str, default="z_threshold",
                    choices=["z_threshold", "revert_direction"])
    ap.add_argument("--zscore-threshold", type=float, default=1.5)
    ap.add_argument("--lag-features", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=None)

    # train/backtest common
    ap.add_argument("--use-dataset", action="store_true")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--gap", type=int, default=24)
    ap.add_argument("--max-train-size", type=int, default=0)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--proba-threshold", type=float, default=0.55)
    ap.add_argument("--signals-from", type=str, default="auto", choices=["auto", "model", "z"])
    ap.add_argument("--fee-rate", type=float, default=0.0005)

    # selection/promote
    ap.add_argument("--summary-path", type=str, default=str(BACKTEST_DIR / "_summary.json"))
    ap.add_argument("--registry-out", type=str, default=str(MODELS_DIR / "registry.json"))
    ap.add_argument("--sharpe-min", type=float, default=0.0)
    ap.add_argument("--maxdd-max", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--production-map-out", type=str, default=str(MODELS_DIR / "production_map.json"))
    ap.add_argument("--registry-in", type=str, default=str(MODELS_DIR / "registry.json"))

    # selection advanced filters (optional)
    ap.add_argument("--require-oof", action="store_true")
    ap.add_argument("--min-auc", type=float, default=None)
    ap.add_argument("--min-rows", type=int, default=None)
    ap.add_argument("--max-per-symbol", type=int, default=None)

    # inference / aggregate / report
    ap.add_argument("--signals-dir", type=str, default=str(DATA / "signals"))
    ap.add_argument("--portfolio-dir", type=str, default=str(DATA / "portfolio"))
    ap.add_argument("--orders-json", type=str, default=str((DATA / "portfolio" / "latest_orders.json")))
    ap.add_argument("--report-out", type=str, default=str((DATA / "portfolio" / "_latest_report.md")))
    ap.add_argument("--n-last", type=int, default=1)
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--skip-flat", action="store_true")

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    mode = args.mode

    if mode == "ingest":
        step_ingest(args.symbols, args.timeframe, args.since_utc, args.limit)

    elif mode == "features":
        step_features(args.symbols, args.beta_window, args.z_window)

    elif mode == "dataset":
        step_dataset(
            args.pairs_manifest, args.label_type, args.zscore_threshold,
            args.lag_features, args.horizon, args.out_dir
        )

    elif mode == "train":
        step_train(args.use_dataset, args.n_splits, args.gap, args.max_train_size, args.early_stopping_rounds, args.proba_threshold)

    elif mode == "backtest":
        step_backtest(args.signals_from, args.proba_threshold, args.fee_rate)

    elif mode == "select":
        step_select(
            args.summary_path, args.registry_out,
            args.sharpe_min, args.maxdd_max, args.top_k,
            require_oof=args.require_oof,
            min_auc=args.min_auc,
            min_rows=args.min_rows,
            max_per_symbol=args.max_per_symbol,
        )

    elif mode == "screen":
        step_screen(args.symbols)

    elif mode == "inference":
        step_inference(
            registry_in=args.registry_in,
            pairs_manifest=str(FEATURES_DIR / "_manifest.json"),
            signals_from=args.signals_from,
            proba_threshold=args.proba_threshold,
            signals_out=args.signals_dir,
            n_last=args.n_last,
            update=bool(args.update),
            skip_flat=bool(args.skip_flat),
        )

    elif mode == "aggregate":
        step_aggregate(
            signals_dir=args.signals_dir,
            pairs_manifest=str(FEATURES_DIR / "_manifest.json"),
            min_proba=args.proba_threshold,
            top_k=args.top_k,
            out_dir=args.portfolio_dir,
        )

    elif mode == "report":
        step_report(
            orders_json=args.orders_json,
            backtest_summary=args.summary_path,
            out_markdown=args.report_out,
        )

    elif mode == "promote":
        step_promote(args.registry_in, args.production_map_out)


if __name__ == "__main__":
    main()
