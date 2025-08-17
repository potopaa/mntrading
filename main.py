#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified project entrypoint:
  - ingest:   load raw OHLCV (data/raw) using data_loader/loader.py if available, else CCXT fallback
  - features: build features/datasets for pairs (prefers your local builder; has safe fallback)
  - train:    train models per pair (logreg / RF / LightGBM), compare with baseline, log to MLflow
  - backtest: run backtest (delegates to backtest/runner.py if present)
  - select:   select champions across pairs and write data/models/registry.json
  - promote:  build production_map.json from registry

Notes:
  * ASCII-only prints for Windows consoles (avoid unicode arrows).
  * Paths are handled via pathlib for cross-platform compatibility.
  * MLflow endpoint/MinIO are read from env; server side is already set to S3 artifacts.
"""

from __future__ import annotations

import os
import sys
import json
import time
import glob
import math
import argparse
from pathlib import Path
from typing import List, Optional, Dict

# Optional imports guarded at call sites
# import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
FEATURES = DATA / "features" / "pairs"
DATASETS = DATA / "datasets" / "pairs"
MODELS_DIR = DATA / "models"
PAIRS_MODELS = MODELS_DIR / "pairs"
REGISTRY = MODELS_DIR / "registry.json"
PRODUCTION_MAP = MODELS_DIR / "production_map.json"


# ------------- utils ------------- #

def _ensure_dirs():
    for p in [RAW, FEATURES, DATASETS, MODELS_DIR, PAIRS_MODELS]:
        p.mkdir(parents=True, exist_ok=True)


def _ascii_print(msg: str):
    # make sure nothing breaks CP1251 consoles
    try:
        print(msg.encode("cp1251", errors="ignore").decode("cp1251"))
    except Exception:
        # fallback to plain print
        print(msg)


def _load_pairs_from_manifest() -> List[str]:
    mani = FEATURES / "_manifest.json"
    if mani.exists():
        try:
            data = json.loads(mani.read_text(encoding="utf-8"))
            pairs = data.get("pairs") or data.get("symbols") or []
            if isinstance(pairs, list):
                return pairs
        except Exception:
            pass
    # fallback: infer from datasets or features filenames
    pairs = []
    for base in [DATASETS, FEATURES]:
        if base.exists():
            for f in base.rglob("*.parquet"):
                # expect .../pairs/<PAIR>[...].parquet
                name = f.stem
                if "__" in name:  # allow schema like BTC_USDT__ETH_USDT
                    pair = name
                else:
                    pair = name
                if pair not in pairs:
                    pairs.append(pair)
    return sorted(pairs)


def _subproc_python(script: Path, args: List[str]) -> int:
    """Run another project script as a subprocess for best forward-compat."""
    import subprocess
    cmd = [sys.executable, str(script)] + args
    _ascii_print(f"[run] {Path.cwd().name}$ {' '.join(cmd)}")
    return subprocess.call(cmd)


# ------------- steps ------------- #

def step_ingest(symbols: str, timeframe: str, since_utc: Optional[str], limit: Optional[int]) -> None:
    """
    Try to ingest using user's data_loader/loader.py if it exposes a compatible function.
    If not found, use CCXT fallback and write a single parquet at data/raw/ohlcv.parquet
    covering all requested symbols/timeframe (appended with 'symbol' column).
    """
    _ensure_dirs()
    USED = False

    # Try custom loader
    try:
        sys.path.insert(0, str(ROOT))
        import importlib
        dl = importlib.import_module("data_loader.loader")
        # Compatible signatures (we try a few common names)
        for fn_name in ("ingest", "ingest_ohlcv", "run", "main"):
            if hasattr(dl, fn_name):
                _ascii_print(f"[ingest] using data_loader/loader.py::{fn_name}")
                fn = getattr(dl, fn_name)
                # try flexible calling
                try:
                    fn(symbols=symbols, timeframe=timeframe, since_utc=since_utc, limit=limit, out_dir=str(RAW))
                except TypeError:
                    # maybe positional
                    fn(symbols, timeframe, since_utc, limit)
                USED = True
                break
    except ModuleNotFoundError:
        pass
    except Exception as e:
        _ascii_print(f"[ingest] custom loader failed: {e!r}. Will try CCXT fallback.")

    if USED:
        _ascii_print("[ingest] done (custom loader).")
        return

    # CCXT fallback
    _ascii_print("[ingest] No compatible functions in data_loader/loader.py -> using CCXT fallback")
    try:
        import ccxt  # type: ignore
        import pandas as pd
        ex = ccxt.binance()
        all_rows = []
        syms = [s.strip() for s in symbols.split(",") if s.strip()]
        # Parse since (ISO8601) if provided
        since_ms = None
        if since_utc:
            try:
                since_ms = ex.parse8601(since_utc)
            except Exception:
                since_ms = None

        for sym in syms:
            _ascii_print(f"[ingest] fetching {sym} {timeframe} since={since_utc} limit={limit}")
            ohlcv = ex.fetch_ohlcv(sym, timeframe=timeframe, since=since_ms, limit=limit)
            for ts, o, h, l, c, v in ohlcv:
                all_rows.append(
                    {
                        "timestamp": pd.to_datetime(ts, unit="ms"),
                        "open": o, "high": h, "low": l, "close": c, "volume": v,
                        "symbol": sym, "timeframe": timeframe,
                    }
                )
        if not all_rows:
            _ascii_print("[ingest] fetched 0 rows. Check symbols/timeframe/network.")
        else:
            df = pd.DataFrame(all_rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
            out = RAW / "ohlcv.parquet"
            df.to_parquet(out, index=False)
            _ascii_print(f"[ingest] raw parquet -> {out}")
    except Exception as e:
        _ascii_print(f"[ingest] CCXT fallback failed: {e!r}")


def step_features(symbols: Optional[str]) -> None:
    """
    Prefer user's features builder if present:
      - features/build.py: build_pairs(...) or main()
    Fallback: build a tiny set of features from RAW parquet for quick progress.
    """
    _ensure_dirs()
    # Try user's builder
    used = False
    try:
        sys.path.insert(0, str(ROOT))
        import importlib
        fb = importlib.import_module("features.build")
        for fn in ("build_pairs", "main", "run"):
            if hasattr(fb, fn):
                _ascii_print(f"[features] using features/build.py::{fn}")
                func = getattr(fb, fn)
                try:
                    func(symbols=symbols, raw_dir=str(RAW), features_dir=str(FEATURES), datasets_dir=str(DATASETS))
                except TypeError:
                    # attempt positional fallbacks
                    try:
                        func(str(RAW), str(FEATURES), str(DATASETS))
                    except Exception:
                        func()
                used = True
                break
    except ModuleNotFoundError:
        pass
    except Exception as e:
        _ascii_print(f"[features] custom builder failed: {e!r}. Will try fallback.")

    if used:
        _ascii_print("[features] done (custom builder).")
        return

    # Fallback features (very simple)
    _ascii_print("[features] No custom builder detected -> building minimal features from data/raw/ohlcv.parquet")
    import pandas as pd
    raw_path = RAW / "ohlcv.parquet"
    if not raw_path.exists():
        _ascii_print("[features] raw parquet not found. Please run ingest first or provide your own builder.")
        return

    df = pd.read_parquet(raw_path)
    # basic features per symbol
    pairs = sorted(df["symbol"].unique().tolist())
    built = []
    for sym in pairs:
        sub = df[df["symbol"] == sym].sort_values("timestamp").copy()
        if len(sub) < 100:
            _ascii_print(f"[features] skip {sym}: not enough bars ({len(sub)})")
            continue
        sub["ret_1"] = sub["close"].pct_change().fillna(0.0)
        sub["ret_5"] = sub["close"].pct_change(5).fillna(0.0)
        sub["vol_norm"] = (sub["volume"] - sub["volume"].rolling(50).mean()) / (sub["volume"].rolling(50).std() + 1e-9)
        sub["ma_10"] = sub["close"].rolling(10).mean().fillna(method="bfill")
        sub["ma_50"] = sub["close"].rolling(50).mean().fillna(method="bfill")
        # simple label for demo: forward 1-step return > 0
        sub["y"] = (sub["close"].shift(-1) > sub["close"]).astype(int)
        # drop last NaN target row
        sub = sub.iloc[:-1, :]

        # save as dataset (single-asset dataset; pair name == symbol)
        out = DATASETS / f"{sym}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(out, index=False)
        built.append(sym)

    # manifest
    FEATURES.mkdir(parents=True, exist_ok=True)
    manifest = {"pairs": built, "created_ts": int(time.time())}
    (FEATURES / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _ascii_print(f"[features] built {len(built)} datasets -> {DATASETS}")
    _ascii_print(f"[features] manifest -> {FEATURES / '_manifest.json'}")


def step_train(use_dataset: bool, n_splits: int, gap: int, max_train_size: int,
               early_stopping_rounds: int, seed: int) -> None:
    _ensure_dirs()
    _ascii_print("[train] start")
    # Import the project trainer (we ship a robust implementation)
    sys.path.insert(0, str(ROOT))
    from models.train import train_all_pairs  # type: ignore
    report = train_all_pairs(
        use_dataset=use_dataset,
        n_splits=n_splits,
        gap=gap,
        max_train_size=max_train_size,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
    )
    # save compact report
    out = MODELS_DIR / "_train_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _ascii_print(f"[train] report -> {out}")


def step_backtest() -> None:
    """
    Delegate to backtest/runner.py if present. Keep arguments minimal to avoid coupling.
    """
    runner = ROOT / "backtest" / "runner.py"
    if runner.exists():
        rc = _subproc_python(runner, [])
        if rc != 0:
            _ascii_print(f"[backtest] runner exit code {rc}")
    else:
        _ascii_print("[backtest] backtest/runner.py not found. Skipping.")


def _collect_candidates() -> Dict[str, dict]:
    """
    Collect per-pair candidates from models meta files.
    Expect:
      data/models/pairs/<PAIR>/__meta.json  (champion info)
      data/models/pairs/<PAIR>/<model_name>__meta.json (optional)
    """
    result: Dict[str, dict] = {}
    if not PAIRS_MODELS.exists():
        return result
    for d in PAIRS_MODELS.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "__meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                pair = meta.get("pair") or d.name
                result[pair] = meta
            except Exception:
                pass
    return result


def step_select(top_k: int = 20) -> None:
    """
    Build registry of champions. Simple: take each pair's __meta.json
    and create registry.json with top_k by score (AUC first, then accuracy).
    """
    _ensure_dirs()
    cand = _collect_candidates()
    if not cand:
        _ascii_print("[select] no candidates found under data/models/pairs")
        return

    # sort by (auc, acc), desc
    ranked = sorted(
        cand.values(),
        key=lambda m: (float(m.get("metrics", {}).get("auc", 0.0)),
                       float(m.get("metrics", {}).get("accuracy", 0.0))),
        reverse=True
    )
    selected = ranked[:top_k]
    reg = {
        "selected_count": len(selected),
        "total_candidates": len(cand),
        "selected": selected,
        "ts": int(time.time())
    }
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    _ascii_print(f"[select] registry -> {REGISTRY}")


def step_promote() -> None:
    """
    Build production_map.json from registry:
      { "<PAIR>": { "model_path": "...", "model_type": "lgbm/rf/logreg", "ts": ... }, ... }
    """
    _ensure_dirs()
    if not REGISTRY.exists():
        _ascii_print("[promote] registry.json not found. Run select first.")
        return
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    prod_map: Dict[str, dict] = {}
    for item in reg.get("selected", []):
        pair = item.get("pair")
        if not pair:
            continue
        # prefer saved "champion_path" if present
        champ_path = item.get("champion_path")
        if not champ_path:
            # try to compose from pair folder (__champion.pkl)
            pdir = PAIRS_MODELS / pair
            maybe = pdir / "__champion.pkl"
            champ_path = str(maybe) if maybe.exists() else ""
        prod_map[pair] = {
            "pair": pair,
            "model_type": item.get("champion_type", item.get("model_type", "")),
            "model_path": champ_path,
            "ts": int(time.time())
        }
    PRODUCTION_MAP.write_text(json.dumps(prod_map, indent=2), encoding="utf-8")
    _ascii_print(f"[promote] production map -> {PRODUCTION_MAP}")


# ------------- cli ------------- #

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mntrading pipeline")
    sub = p.add_subparsers(dest="mode", required=False)

    # legacy flat flags
    p.add_argument("--mode", type=str, default=None,
                   help="ingest | features | train | backtest | select | promote")

    # common
    p.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT",
                   help="comma-separated symbols or path to JSON with pairs")
    p.add_argument("--timeframe", type=str, default="1h")
    p.add_argument("--since-utc", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)

    # train
    p.add_argument("--use-dataset", action="store_true", help="use data/datasets/pairs for training")
    p.add_argument("--n-splits", type=int, default=3)
    p.add_argument("--gap", type=int, default=5)
    p.add_argument("--max-train-size", type=int, default=2000)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # select
    p.add_argument("--top-k", type=int, default=20)

    return p.parse_args(argv)


def main():
    args = parse_args()

    mode = (args.mode or "").strip().lower()
    if not mode:
        _ascii_print("No --mode given. Nothing to do.")
        return

    if mode == "ingest":
        step_ingest(args.symbols, args.timeframe, args.since_utc, args.limit)
    elif mode == "features":
        # symbols may be a JSON file path containing list of pairs; pass through to custom builder if any.
        syms = args.symbols
        if syms and Path(syms).exists() and syms.lower().endswith(".json"):
            _ascii_print(f"[features] symbols from JSON: {syms}")
        step_features(syms)
    elif mode == "train":
        step_train(
            use_dataset=args.use_dataset,
            n_splits=args.n_splits,
            gap=args.gap,
            max_train_size=args.max_train_size,
            early_stopping_rounds=args.early_stopping_rounds,
            seed=args.seed,
        )
    elif mode == "backtest":
        _ascii_print("Backtesting...")
        step_backtest()
        # many projects write summary here
        _ascii_print(f"[backtest] summary -> {DATA / 'backtest_results' / '_summary.json'}")
    elif mode == "select":
        step_select(top_k=args.top_k)
    elif mode == "promote":
        step_promote()
    else:
        _ascii_print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        _ascii_print(f"failed\n\n    {e.__class__.__name__} -> {e}")
        sys.exit(1)
