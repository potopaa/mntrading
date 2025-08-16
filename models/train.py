#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust baseline training for mntrading.

Interface expected by main.py:
    report = train_baseline(
        datasets_dir: str,
        features_dir: str,
        out_dir: str,
        use_dataset: bool,
        n_splits: int,
        gap: int,
        max_train_size: int,
        early_stopping_rounds: int,
        proba_threshold: float,
    )

Outputs:
- data/models/_train_report.json (returned as dict)
- data/models/pairs/<PAIR>__oof.parquet with columns [ts, y, proba]
- (optional) models can be saved later if нужно
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception as e:
    raise SystemExit("LightGBM is required. Install: pip install lightgbm") from e


# ----------------------- utilities -----------------------
def _load_pairs_from_manifest(features_manifest_path: Path) -> List[str]:
    if not features_manifest_path.exists():
        return []
    obj = json.loads(features_manifest_path.read_text(encoding="utf-8"))
    items = obj.get("items") or []
    pairs = [str(it.get("pair")) for it in items if it.get("pair")]
    return sorted(set(pairs))


def _find_dataset_files(datasets_dir: Path) -> Dict[str, Path]:
    """
    Expect per-pair parquet files like: data/datasets/pairs/<A__B>__ds.parquet
    Returns mapping: "AAA/USDT__BBB/USDT" -> Path
    """
    mp: Dict[str, Path] = {}
    if not datasets_dir.exists():
        return mp
    for p in datasets_dir.glob("*__ds.parquet"):
        key = p.stem.replace("__ds", "")
        # convert "AAA_USDT__BBB_USDT" -> "AAA/USDT__BBB/USDT"
        pair_key = key.replace("_USDT", "/USDT").replace("_USDC", "/USDC").replace("_BTC", "/BTC").replace("_ETH", "/ETH")
        # generic fix (just in case): first split by "__", then reinsert slashes
        if "__" in key and "/" not in pair_key:
            a, b = key.split("__", 1)
            pair_key = a.replace("_", "/") + "__" + b.replace("_", "/")
        mp[pair_key] = p
    return mp


def _time_series_splits(n: int, n_splits: int, gap: int, max_train_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simple expanding/rolling time series split with optional max_train_size and gap.
    Returns list of (train_idx, valid_idx).
    """
    if n_splits <= 0:
        return []
    fold_sizes = []
    # make roughly equal sized valid folds at the tail
    base = n // (n_splits + 1)
    offset = n - base * (n_splits + 1)
    # we use the last n_splits chunks as validation folds
    # start index for first validation
    start_valid = base + offset
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        valid_start = start_valid + i * base
        valid_end = valid_start + base
        if valid_end > n:
            break
        train_end = max(0, valid_start - gap)
        train_start = max(0, train_end - max_train_size) if max_train_size > 0 else 0
        tr_idx = np.arange(train_start, train_end)
        va_idx = np.arange(valid_start, valid_end)
        if len(tr_idx) > 0 and len(va_idx) > 0:
            splits.append((tr_idx, va_idx))
    return splits


def _class_balance_info(y: np.ndarray) -> Tuple[int, int, float]:
    total = len(y)
    pos = int(np.sum(y == 1))
    neg = total - pos
    pos_rate = (pos / total) if total else 0.0
    return pos, neg, pos_rate


def _adaptive_lgb_params(n_train: int) -> Dict:
    """
    Choose LightGBM params based on dataset size to avoid 'no more leaves' warning.
    """
    # make min_data_in_leaf small for small datasets
    min_leaf = max(5, n_train // 50)  # ~2% of train set, but at least 5
    num_leaves = min(63, max(15, n_train // max(1, min_leaf)))  # rough balance
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_leaf),
        "max_depth": -1,
        "min_gain_to_split": 0.0,   # allow splitting
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "max_bin": 255,
        "verbosity": -1,
        "force_col_wise": True,     # stabler on wide data
        "deterministic": True,
        "seed": 42,
    }
    return params


# ----------------------- main trainer -----------------------
def train_baseline(
    datasets_dir: str,
    features_dir: str,
    out_dir: str,
    use_dataset: bool,
    n_splits: int,
    gap: int,
    max_train_size: int,
    early_stopping_rounds: int,
    proba_threshold: float,
) -> Dict:
    """
    Train per-pair classifiers with simple time-series CV and produce OOF probabilities.
    """
    out_root = Path(out_dir)
    out_pairs = out_root / "pairs"
    out_root.mkdir(parents=True, exist_ok=True)
    out_pairs.mkdir(parents=True, exist_ok=True)

    # Figure out which pairs to train
    ds_dir = Path(datasets_dir)
    ds_files = _find_dataset_files(ds_dir)
    if not ds_files:
        # fallback to features manifest (but then we must build labels on the fly)
        feats_manifest = Path(features_dir) / "_manifest.json"
        pairs = _load_pairs_from_manifest(feats_manifest)
        if not pairs:
            raise SystemExit("No datasets or features manifest found for training.")
        # we will load features and create labels from z (|z|>thr)
        from warnings import warn
        warn("Training without datasets: building labels from features |z| > threshold.")
        use_dataset = False
    else:
        pairs = sorted(ds_files.keys())

    summary_pairs: Dict[str, Dict] = {}
    oof_paths: Dict[str, str] = {}

    for pk in pairs:
        try:
            if use_dataset and pk in ds_files:
                df = pd.read_parquet(ds_files[pk])
                # expect columns: ts, y, and feature columns (others)
                if "ts" not in df.columns:
                    # try to infer
                    if "timestamp" in df.columns:
                        df = df.rename(columns={"timestamp": "ts"})
                    else:
                        # create ts from index if needed
                        if isinstance(df.index, pd.DatetimeIndex):
                            df = df.reset_index().rename(columns={"index": "ts"})
                        else:
                            df["ts"] = np.arange(len(df))
                if "y" not in df.columns:
                    # try to build from z if present
                    if "z" in df.columns:
                        df["y"] = (df["z"].abs() > proba_threshold).astype(int)
                    else:
                        raise ValueError("Dataset has no 'y' column and no 'z' to derive labels.")
            else:
                # features mode: load features and build label: y = (|z| > X), default X=1.5
                pair_dir = Path(features_dir) / pk.replace("/", "_") / "features.parquet"
                if not pair_dir.exists():
                    # old layout: features/pairs/<A__B>/features.parquet
                    pair_dir = Path(features_dir) / pk.replace("/", "_") / "features.parquet"
                df = pd.read_parquet(pair_dir)
                if "z" not in df.columns:
                    raise ValueError("Features lack 'z' column; labels cannot be derived.")
                df["y"] = (df["z"].abs() > 1.5).astype(int)  # fixed threshold here
                if "ts" not in df.columns:
                    # make sure we have ts
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index().rename(columns={"index": "ts"})
                    else:
                        df["ts"] = np.arange(len(df))

            # Order by time
            df = df.sort_values("ts").reset_index(drop=True)

            # Features: drop non-numeric and target
            drop_cols = {"ts", "y", "pair"}
            X = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore")
            # keep only numeric
            X = X.select_dtypes(include=[np.number]).copy()
            y = df["y"].astype(int).values

            n = len(X)
            if n < 200:
                # too small to train stably
                summary_pairs[pk] = {"rows": int(n), "auc_mean": None, "note": "too_few_rows"}
                continue

            # Build folds
            splits = _time_series_splits(n, n_splits=n_splits, gap=gap, max_train_size=max_train_size)
            if not splits:
                summary_pairs[pk] = {"rows": int(n), "auc_mean": None, "note": "no_splits"}
                continue

            # Storage for OOF
            oof = np.full(n, np.nan, dtype=float)
            aucs: List[float] = []

            # Iterate folds
            for (tr_idx, va_idx) in splits:
                X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
                X_va, y_va = X.iloc[va_idx], y[va_idx]

                # Skip folds with one class
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                    # mark OOF as 0.5 neutral for this fold
                    oof[va_idx] = 0.5
                    continue

                pos, neg, pos_rate = _class_balance_info(y_tr)
                n_tr = len(y_tr)

                params = _adaptive_lgb_params(n_tr)
                # handle imbalance (approx)
                if 0 < pos < n_tr:
                    params["scale_pos_weight"] = max(1.0, (neg / max(1, pos)))

                dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
                dvalid = lgb.Dataset(X_va, label=y_va, free_raw_data=True)

                # ensure early stopping won't error out on tiny folds
                esr = max(10, min(early_stopping_rounds, len(y_va) // 2))

                model = lgb.train(
                    params=params,
                    train_set=dtrain,
                    valid_sets=[dvalid],
                    num_boost_round=2000,
                    early_stopping_rounds=esr,
                    verbose_eval=False,
                )

                proba = model.predict(X_va, num_iteration=model.best_iteration)
                # clip just in case
                proba = np.clip(proba, 1e-6, 1 - 1e-6)
                oof[va_idx] = proba

                # compute AUC local (safe)
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(y_va, proba)
                    aucs.append(float(auc))
                except Exception:
                    pass

            # finalize OOF and report
            oof = pd.Series(oof).fillna(0.5).values
            df_oof = pd.DataFrame({"ts": df["ts"].values, "y": y, "proba": oof})
            oof_path = (out_pairs / f"{pk.replace('/', '_')}__oof.parquet")
            df_oof.to_parquet(oof_path, index=False)

            auc_mean = float(np.nanmean(aucs)) if aucs else None
            summary_pairs[pk] = {
                "rows": int(n),
                "auc_mean": auc_mean,
                "oof_path": str(oof_path),
                "class_balance": {
                    "pos_rate_overall": float((y == 1).mean()),
                },
            }

        except Exception as e:
            summary_pairs[pk] = {"rows": None, "auc_mean": None, "error": str(e)}
            continue

    report = {
        "pairs": summary_pairs,
        "n_splits": int(n_splits),
        "gap": int(gap),
        "max_train_size": int(max_train_size),
        "proba_threshold": float(proba_threshold),
    }

    (Path(out_dir) / "_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
