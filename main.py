#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
"""
Main CLI for the mntrading pipeline:
  screen_pairs.py → (this) ingest → features → dataset → train → backtest → select → promote

Design goals:
- No dependency on any non-existent modules.
- Prefer your custom data loader in data_loader/loader.py when present.
- Save artifacts under data/* exactly as the project expects.
- Produce manifests with explicit file paths (downstream never guesses paths).
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
_loader = None
try:
    # Your custom loader (recommended)
    from data_loader import loader as _loader  # noqa: E402
    HAS_LOADER = True
except Exception:
    HAS_LOADER = False
    _loader = None

HAS_SPREAD = False
_spread = None
try:
    from features import spread as _spread  # noqa: E402
    HAS_SPREAD = True
except Exception:
    HAS_SPREAD = False
    _spread = None

HAS_LABELS = False
_labels = None
try:
    from features.labels import DatasetBuildConfig, build_datasets_for_manifest  # noqa: E402
    HAS_LABELS = True
except Exception:
    HAS_LABELS = False
    _labels = None

HAS_TRAIN = False
_train = None
try:
    from models.train import train_baseline  # noqa: E402
    HAS_TRAIN = True
except Exception:
    HAS_TRAIN = False
    _train = None

HAS_BT = False
_bt = None
try:
    from backtest.runner import run_backtest  # noqa: E402
    HAS_BT = True
except Exception:
    HAS_BT = False
    _bt = None


# ------------------------- Helpers ---------------------------
def _save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True>
    df.to_parquet(path, index=False)


def _load_pairs_from_json(path: str) -> List[Tuple[str, str]]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str]] = []
    # canonical: {"pairs":[{"a":"AAA/USDT","b":"BBB/USDT"}, ...]}
    if isinstance(obj, dict) and "pairs" in obj:
        for it in obj["pairs"]:
            a, b = it.get("a"), it.get("b")
            if a and b:
                pairs.append((a, b))
    elif isinstance(obj, list):
        # allow ["AAA/USDT|BBB/USDT", ...] or [["AAA/USDT","BBB/USDT"], ...]
        for it in obj:
            if isinstance(it, str) and "|" in it:
                a, b = it.split("|", 1)
                pairs.append((a.strip(), b.strip()))
            elif isinstance(it, list) and len(it) == 2:
                pairs.append((str(it[0]), str(it[1])))
    return pairs


def _parse_symbols_arg(arg: str) -> Tuple[List[str], List[Tuple[str, str]], Optional[str]]:
    """
    Returns (unique symbols list, pairs list, pairs_json_path if any).
    --symbols accepts:
      • CSV: "BTC/USDT,ETH/USDT"
      • path/glob to screened_pairs_*.json
    """
    pairs: List[Tuple[str, str]] = []
    symbols: List[str] = []
    pairs_json_path: Optional[str] = None

    paths = sorted(glob.glob(arg))
    if paths:
        # Use the latest JSON (typical: data/pairs/screened_pairs_*.json)
        pairs_json_path = paths[-1]
        for p in paths:
            pairs.extend(_load_pairs_from_json(p))
        symset = set()
        for a, b in pairs:
            symset.add(a)
            symset.add(b)
        symbols = sorted(symset)
        return symbols, pairs, pairs_json_path

    # CSV
    if "," in arg or "/" in arg:
        symbols = sorted({s.strip() for s in arg.split(",") if s.strip()})
    return symbols, pairs, pairs_json_path


def _fallback_ccxt_ingest(symbols: List[str], timeframe: str, since_utc: Optional[str], limit: int):
    """
    Simple CCXT ingest when no custom loader is available.
    Saves to data/raw/ohlcv.parquet with columns [ts, open, high, low, close, volume, symbol]
    """
    import ccxt  # imported on demand
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
    for s in tqdm(symbols, desc="Ingest(ccxt)"):
        if s not in ex.markets:
            print(f"[warn] skip {s}: not in exchange markets")
            continue
        frames.append(_fetch_one(s))
    if not frames:
        raise SystemExit("Nothing ingested (ccxt fallback)")
    ohlcv = pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"])
    _save_parquet(ohlcv, RAW_DIR / "ohlcv.parquet")
    meta = {
        "timeframe": timeframe,
        "since_utc": since_utc,
        "symbols": sorted(ohlcv["symbol"].unique().tolist()),
        "rows": int(len(ohlcv)),
    }
    (RAW_DIR / "ohlcv_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ingest] saved {len(ohlcv)} rows → {RAW_DIR/'ohlcv.parquet'}")


# ------------------------- Steps -----------------------------
def step_ingest(symbols_arg: str, timeframe: str, since_utc: Optional[str], limit: int):
    symbols, _, _ = _parse_symbols_arg(symbols_arg)
    if not symbols:
        raise SystemExit("No symbols provided/resolved for ingest")

    # Prefer your custom loader if available
    if HAS_LOADER:
        # Supported signatures in loader.py (any is OK):
        #  a) get_ohlcv(symbols=[...], timeframe="5m", since_utc="...", limit=1000) -> DataFrame
        #  b) ingest(symbols=[...], timeframe="5m", since_utc="...", limit=1000) -> DataFrame | None (may save itself)
        #  c) fetch_ohlcv(symbol, timeframe, since_utc, limit) -> DataFrame (per-symbol)
        if hasattr(_loader, "get_ohlcv"):
            ohlcv = _loader.get_ohlcv(symbols=symbols, timeframe=timeframe, since_utc=since_utc, limit=limit)
            if not isinstance(ohlcv, pd.DataFrame) or ohlcv.empty:
                raise SystemExit("loader.get_ohlcv returned empty result")
            _save_parquet(ohlcv, RAW_DIR / "ohlcv.parquet")
        elif hasattr(_loader, "ingest"):
            ohlcv = _loader.ingest(symbols=symbols, timeframe=timeframe, since_utc=since_utc, limit=limit)
            if isinstance(ohlcv, pd.DataFrame) and not ohlcv.empty:
                _save_parquet(ohlcv, RAW_DIR / "ohlcv.parquet")
            elif not (RAW_DIR / "ohlcv.parquet").exists():
                raise SystemExit("loader.ingest did not produce data/raw/ohlcv.parquet")
        elif hasattr(_loader, "fetch_ohlcv"):
            frames = []
            for s in tqdm(symbols, desc="Ingest(loader.fetch_ohlcv)"):
                df = _loader.fetch_ohlcv(s, timeframe=timeframe, since_utc=since_utc, limit=limit)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    if "symbol" not in df.columns:
                        df = df.assign(symbol=s)
                    frames.append(df)
            if not frames:
                raise SystemExit("loader.fetch_ohlcv returned no data")
            ohlcv = pd.concat(frames, ignore_index=True).sort_values(["symbol", "ts"])
            _save_parquet(ohlcv, RAW_DIR / "ohlcv.parquet")
        else:
            print("[ingest] No compatible functions in data_loader/loader.py -> using CCXT fallback")
            _fallback_ccxt_ingest(symbols, timeframe, since_utc, limit)
    else:
        _fallback_ccxt_ingest(symbols, timeframe, since_utc, limit)

    # write meta (if not already)
    if (RAW_DIR / "ohlcv.parquet").exists():
        ohlcv = pd.read_parquet(RAW_DIR / "ohlcv.parquet")
        meta = {
            "timeframe": timeframe,
            "since_utc": since_utc,
            "symbols": sorted(ohlcv["symbol"].unique().tolist()) if "symbol" in ohlcv.columns else [],
            "rows": int(len(ohlcv)),
        }
        (RAW_DIR / "ohlcv_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ingest] meta -> {RAW_DIR/'ohlcv_meta.json'}")


def step_features(symbols_arg: str, beta_window: int, z_window: int):
    """
    Build pair features.
    Accepts --symbols as CSV or as path/glob to screened_pairs_*.json.
    Writes individual pair folders:
      data/features/pairs/<A__B>/features.parquet
    and manifest:
      data/features/pairs/_manifest.json with explicit paths.
    """
    raw_path = RAW_DIR / "ohlcv.parquet"
    if not raw_path.exists():
        raise SystemExit(f"{raw_path} not found. Run ingest first.")

    ohlcv = pd.read_parquet(raw_path)
    symbols, pairs, pairs_json_path = _parse_symbols_arg(symbols_arg)
    produced_pairs: List[str] = []

    if HAS_SPREAD and hasattr(_spread, "compute_features_for_pairs"):
        # Prefer file-based mode when a pairs JSON path is available
        if pairs_json_path:
            out = _spread.compute_features_for_pairs(
                pairs_json=pairs_json_path,
                raw_parquet=raw_path,
                out_dir=FEATURES_DIR,
                beta_window=beta_window,
                z_window=z_window,
            )
            if isinstance(out, list):
                produced_pairs = out
        else:
            # CSV symbols → generate all combinations and save
            if not pairs:
                syms = symbols
                pairs = [(syms[i], syms[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))]
            res = _spread.compute_features_for_pairs(
                raw_df=ohlcv, pairs=pairs, beta_window=beta_window, z_window=z_window
            )
            for pk, df in res.items():
                safe = pk.replace("/", "_")
                pair_dir = FEATURES_DIR / safe
                pair_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(pair_dir / "features.parquet", index=False)
                produced_pairs.append(pk)
    else:
        # Minimal internal fallback
        print("[features] WARNING: features.spread not available; using minimal internal implementation.")
        piv = ohlcv.pivot(index="ts", columns="symbol", values="close").sort_index()
        if not pairs:
            syms = symbols
            pairs = [(syms[i], syms[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))]

        def _rolling_beta_alpha(y: pd.Series, x: pd.Series, win: int):
            x_mean = x.rolling(win).mean()
            y_mean = y.rolling(win).mean()
            cov = (x * y).rolling(win).mean() - x_mean * y_mean
            var = x.rolling(win).var()
            beta = cov / (var.replace(0.0, np.nan))
            alpha = y_mean - beta * x_mean
            return beta, alpha

        def _zscore(s: pd.Series, win: int):
            mu = s.rolling(win).mean()
            sd = s.rolling(win).std()
            return (s - mu) / (sd + 1e-12)

        for (a, b) in tqdm(pairs, desc="Features(fallback)"):
            if a not in piv.columns or b not in piv.columns:
                continue
            df = piv[[a, b]].dropna().rename(columns={a: "pa", b: "pb"})
            if len(df) < max(beta_window, z_window) + 5:
                continue
            beta, alpha = _rolling_beta_alpha(df["pa"], df["pb"], beta_window)
            spread = (df["pa"] - (beta * df["pb"] + alpha))
            z = _zscore(spread, z_window)
            out = pd.DataFrame({
                "ts": df.index,
                "a": df["pa"].values,
                "b": df["pb"].values,
                "beta": beta.values,
                "alpha": alpha.values,
                "spread": spread.values,
                "z": z.values,
            }).dropna().reset_index(drop=True)
            pk = f"{a}__{b}"
            safe = pk.replace("/", "_")
            pair_dir = FEATURES_DIR / safe
            pair_dir.mkdir(parents=True, exist_ok=True)
            out.to_parquet(pair_dir / "features.parquet", index=False)
            produced_pairs.append(pk)

    # Build manifest with explicit parquet paths
    items: List[Dict[str, Any]] = []
    for pk in sorted(set(produced_pairs)):
        safe = pk.replace("/", "_")
        path = (FEATURES_DIR / safe / "features.parquet").resolve()
        if path.exists():
            items.append({"pair": pk, "path": str(path), "features": ["a", "b", "beta", "alpha", "spread", "z"]})
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
    """
    Build supervised datasets per pair from features manifest.
    Writes data/datasets/pairs/<A__B>__ds.parquet and data/datasets/_manifest.json
    """
    if not HAS_LABELS:
        raise SystemExit("features.labels not available. Please add features/labels.py with build_datasets_for_manifest().")

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
    """
    Train per-pair models with time-series CV, save OOF predictions and a train report.
    """
    if not HAS_TRAIN:
        raise SystemExit("models.train not available. Please add models/train.py with train_baseline().")
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


def step_backtest(use_dataset: bool, signals_from: str, proba_threshold: float, fee_rate: float):
    """
    Backtest using features + optional OOF probabilities. Writes a summary JSON.
    """
    if not HAS_BT:
        raise SystemExit("backtest.runner not available. Please add backtest/runner.py with run_backtest().")
    summary = run_backtest(
        features_dir=str(FEATURES_DIR),
        datasets_dir=str(DATASETS_DIR),
        models_dir=str(MODELS_DIR),
        out_dir=str(BACKTEST_DIR),
        proba_threshold=proba_threshold,
        fee_rate=fee_rate,
    )
    (BACKTEST_DIR / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # quiet: runner already prints the summary path


def step_select(summary_path: str, registry_out: str, sharpe_min: float, maxdd_max: float, top_k: int,
                require_oof: bool = False, min_auc: Optional[float] = None,
                min_rows: Optional[int] = None, max_per_symbol: Optional[int] = None):
    """
    Champion selection using models/select.py if available; otherwise a simple fallback.
    """
    # Try the richer selector
    try:
        from models.select import select_champions as _select  # local import to avoid hard dep
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
        return
    except Exception as e:
        print(f"[select] models.select not available or failed: {e}")
        # Fallback: simple filter & sort by Sharpe
        p = Path(summary_path)
        if not p.exists():
            raise SystemExit(f"Summary not found: {p}")
        obj = json.loads(p.read_text(encoding="utf-8"))
        pairs = obj.get("pairs", {})
        rows = []
        for pair, data in pairs.items():
            met = (data or {}).get("metrics") or {}
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
    """
    Promote currently selected champions into a simple production map:
      {"pairs":["AAA/USDT__BBB/USDT", ...]}
    """
    p = Path(registry_in)
    if not p.exists():
        raise SystemExit(f"Registry not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    pairs = [it["pair"] for it in obj.get("pairs", [])]
    prod = {"pairs": pairs}
    Path(production_map_out).parent.mkdir(parents=True, exist_ok=True)
    Path(production_map_out).write_text(json.dumps(prod, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[promote] production map -> {production_map_out}")


# ------------------------- CLI -------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="mntrading CLI")
    ap.add_argument("--mode", required=True,
                    choices=["ingest", "features", "dataset", "train", "backtest", "select", "promote"])

    # Common / ingest
    ap.add_argument("--symbols", type=str,
                    help="CSV of symbols OR path/glob to screened_pairs_*.json (for features/ingest)")
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
    ap.add_argument("--horizon", type=int, default=0)  # fixed: was type[int]
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Optional override for datasets output directory (default: data/datasets/pairs)")

    # train
    ap.add_argument("--use-dataset", action="store_true")
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--max-train-size", type=int, default=2000)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--proba-threshold", type=float, default=0.55)

    # backtest
    ap.add_argument("--signals-from", type=str, default="oof")
    ap.add_argument("--fee-rate", type=float, default=0.0005)

    # select/promote
    ap.add_argument("--summary-path", type=str, default=str(BACKTEST_DIR / "_summary.json"))
    ap.add_argument("--registry-out", type=str, default=str(MODELS_DIR / "registry.json"))
    ap.add_argument("--sharpe-min", type=float, default=0.0)
    ap.add_argument("--maxdd-max", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--production-map-out", type=str, default=str(MODELS_DIR / "production_map.json"))
    ap.add_argument("--registry-in", type=str, default=str(MODELS_DIR / "registry.json"))

    # selection advanced filters (optional)
    ap.add_argument("--require-oof", action="store_true",
                    help="Only keep pairs whose backtest used OOF probabilities")
    ap.add_argument("--min-auc", type=float, default=None,
                    help="Filter out pairs whose train AUC is below this value (requires _train_report.json)")
    ap.add_argument("--min-rows", type=int, default=None,
                    help="Filter out pairs with too few dataset rows (requires _train_report.json)")
    ap.add_argument("--max-per-symbol", type=int, default=None,
                    help="Limit how many pairs per base symbol (diversity constraint)")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()
    mode = args.mode.lower()

    if mode == "ingest":
        if not args.symbols:
            raise SystemExit("--symbols is required for ingest (CSV or data/pairs/screened_pairs_*.json)")
        step_ingest(args.symbols, args.timeframe, args.since_utc, args.limit)

    elif mode == "features":
        if not args.symbols:
            raise SystemExit("--symbols is required for features (CSV or data/pairs/screened_pairs_*.json)")
        step_features(args.symbols, args.beta_window, args.z_window)

    elif mode == "dataset":
        step_dataset(
            args.pairs_manifest,
            args.label_type,
            args.zscore_threshold,
            args.lag_features,
            args.horizon,
            out_dir=args.out_dir,
        )

    elif mode == "train":
        step_train(args.use_dataset, args.n_splits, args.gap, args.max_train_size,
                   args.early_stopping_rounds, args.proba_threshold)

    elif mode == "backtest":
        step_backtest(args.use_dataset, args.signals_from, args.proba_threshold, args.fee_rate)

    elif mode == "select":
        step_select(
            args.summary_path, args.registry_out,
            args.sharpe_min, args.maxdd_max, args.top_k,
            require_oof=args.require_oof,
            min_auc=args.min_auc,
            min_rows=args.min_rows,
            max_per_symbol=args.max_per_symbol,
        )

    elif mode == "promote":
        step_promote(args.registry_in, args.production_map_out)


if __name__ == "__main__":
    main()
