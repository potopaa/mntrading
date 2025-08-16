#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.validation import check_is_fitted

# optional libs
_HAS_XGB = False
_HAS_LGB = False
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    pass
try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except Exception:
    pass

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class TrainConfig:
    n_splits: int = 3
    gap: int = 5
    max_train_size: int = 2000
    early_stopping_rounds: int = 50
    proba_threshold: float = 0.55
    label_type: str = "revert_direction"  # recommended for trading


def _read_ds(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # make ts index
    if "ts" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["ts"], utc=True, errors="coerce"))
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index.name = "ts"
    df = df.sort_index()
    # drop duplicates
    df = df[~df.index.duplicated(keep="last")]
    return df


def _get_Xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = [c for c in ["ts", "pair_a", "pair_b"] if c in df.columns]
    X = df.drop(columns=drop_cols + ["y"], errors="ignore")
    if "y" not in df.columns:
        raise ValueError("Dataset must contain 'y' column")
    y = df["y"].astype(int)
    return X, y


def _iter_tscv(n: int, n_splits: int, gap: int, max_train_size: Optional[int]):
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=None, gap=gap)
    # emulate embargo by trimming train tail by 'gap'
    for tr_idx, te_idx in splitter.split(np.arange(n)):
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        tr_end = tr_idx[-1] - gap if gap > 0 else tr_idx[-1]
        tr_idx = tr_idx[: max(0, tr_end + 1)]
        if max_train_size and len(tr_idx) > max_train_size:
            tr_idx = tr_idx[-max_train_size:]
        yield tr_idx, te_idx


def _fit_rf(X_tr, y_tr) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    rf.fit(X_tr, y_tr)
    try:
        # calibrate to improve probabilities
        cal = CalibratedClassifierCV(rf, method="isotonic", cv=3)
        cal.fit(X_tr, y_tr)
        return cal
    except Exception:
        return rf


def _fit_xgb(X_tr, y_tr, X_va, y_va, es_rounds: int):
    if not _HAS_XGB:
        return None
    try:
        clf = XGBClassifier(
            n_estimators=2000,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
            eval_metric="auc",
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=es_rounds,
        )
        return clf
    except Exception:
        return None


def _fit_lgb(X_tr, y_tr, X_va, y_va, es_rounds: int):
    if not _HAS_LGB:
        return None
    try:
        clf = LGBMClassifier(
            n_estimators=5000,
            max_depth=-1,
            learning_rate=0.02,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
            objective="binary",
        )
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="auc",
            callbacks=[],
        )
        return clf
    except Exception:
        return None


def _predict_proba(model, X):
    try:
        p = model.predict_proba(X)[:, 1]
    except Exception:
        p = model.predict(X)
        # squeeze into [0,1]
        p = (p - p.min()) / (p.max() - p.min() + 1e-12)
    return p


def train_baseline(
    datasets_dir: str,
    features_dir: str,
    out_dir: str,
    use_dataset: bool = True,
    n_splits: int = 3,
    gap: int = 5,
    max_train_size: int = 2000,
    early_stopping_rounds: int = 50,
    proba_threshold: float = 0.55,
) -> Dict[str, Any]:
    """
    Train per-pair classifiers using time-series CV and save OOF predictions.
    Returns report JSON with per-pair AUC and basic stats.
    """
    datasets_dir = Path(datasets_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(datasets_dir.glob("*__ds.parquet"))
    report: Dict[str, Any] = {"pairs": {}, "n_splits": n_splits, "gap": gap, "max_train_size": max_train_size}

    for f in files:
        pair_key = f.name.replace("__ds.parquet", "").replace("_", "/")
        try:
            df = _read_ds(f)
            X, y = _get_Xy(df)
            n = len(df)
            if n < 200:
                continue
            oof = pd.Series(index=df.index, dtype=float)
            fold_ids = pd.Series(index=df.index, dtype="Int64")
            aucs: List[float] = []
            for fold, (tr_idx, te_idx) in enumerate(_iter_tscv(n, n_splits, gap, max_train_size), start=1):
                if len(tr_idx) == 0 or len(te_idx) == 0:
                    continue
                X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
                X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
                # validation split for ES
                split = int(0.8 * len(X_tr))
                X_tr2, y_tr2 = X_tr.iloc[:split], y_tr.iloc[:split]
                X_va,  y_va  = X_tr.iloc[split:], y_tr.iloc[split:]

                # fit candidates
                best_model = _fit_xgb(X_tr2, y_tr2, X_va, y_va, early_stopping_rounds) or \
                             _fit_lgb(X_tr2, y_tr2, X_va, y_va, early_stopping_rounds)
                if best_model is None:
                    best_model = _fit_rf(X_tr, y_tr)

                p = _predict_proba(best_model, X_te)
                oof.iloc[te_idx] = p
                fold_ids.iloc[te_idx] = fold
                try:
                    aucs.append(roc_auc_score(y_te, p))
                except Exception:
                    pass

            # write OOF
            pair_safe = f.name.replace("__ds.parquet","")
            pair_dir = pairs_dir / pair_safe
            pair_dir.mkdir(parents=True, exist_ok=True)
            oof_df = pd.DataFrame({"ts": df.index, "y": y.values, "proba": oof.values, "fold": fold_ids.values})
            oof_df.to_parquet(pair_dir / "oof.parquet", index=False)

            report["pairs"][pair_key] = {
                "rows": int(n),
                "auc_mean": float(np.nanmean(aucs)) if aucs else float("nan"),
                "oof_path": str((pair_dir / "oof.parquet").resolve()),
            }
        except Exception as e:
            # skip this pair
            continue

    # write global train report
    (out_dir / "_train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
