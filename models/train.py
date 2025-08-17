# -*- coding: utf-8 -*-
"""
Train per-pair models:
  - Baseline: Logistic Regression (with StandardScaler)
  - Candidates: RandomForest, LightGBM (if available)
  - TimeSeries CV with gaps; OOF predictions and metrics (AUC, logloss, accuracy)
  - Champion selection by AUC (then accuracy tie-break)
  - Artifacts:
      data/models/pairs/<PAIR>/__champion.pkl
      data/models/pairs/<PAIR>/oof.parquet
      data/models/pairs/<PAIR>/__meta.json
    and candidate models:
      data/models/pairs/<PAIR>/lgbm.pkl / rf.pkl / logreg.pkl (+ meta)

MLflow:
  - experiment name comes from env MLFLOW_EXPERIMENT_NAME (default "mntrading_s3")
  - if missing, we CREATE it with artifact_location = s3://<MINIO_BUCKET>/<prefix>/<name>
    where prefix = env MLFLOW_ARTIFACTS_PREFIX (default "experiments")
  - models logged with mlflow.*.log_model(name="...") (new API), fallback to artifact_path on older MLflow.

Safe for Windows consoles and Docker.
"""

from __future__ import annotations

import os
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# Optional LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

ROOT = Path(__file__).resolve().parents[1]  # mntrading/
DATA = ROOT / "data"
FEATURES = DATA / "features" / "pairs"
DATASETS = DATA / "datasets" / "pairs"
MODELS_DIR = DATA / "models"
PAIRS_MODELS = MODELS_DIR / "pairs"

# ----------------- helpers ----------------- #

def _mlflow_log_model_sklearn(model, name: str, input_example=None):
    """Log sklearn model with the 'name=' API, fallback to artifact_path for older MLflow."""
    import mlflow
    try:
        mlflow.sklearn.log_model(sk_model=model, name=name, input_example=input_example)
    except TypeError:
        mlflow.sklearn.log_model(sk_model=model, artifact_path=name, input_example=input_example)

def _mlflow_log_model_lightgbm(model, name: str):
    """Log LightGBM model with the 'name=' API, fallback to artifact_path if needed."""
    import mlflow
    try:
        mlflow.lightgbm.log_model(lgb_model=model, name=name)
    except TypeError:
        mlflow.lightgbm.log_model(lgb_model=model, artifact_path=name)

def _auc_safe(y_true, y_prob) -> float:
    u = np.unique(y_true)
    if len(u) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")

def _logloss_safe(y_true, y_prob) -> float:
    try:
        return float(log_loss(y_true, y_prob, eps=1e-15))
    except Exception:
        return float("nan")

def _acc_from_proba(y_true, y_prob, thr=0.5) -> float:
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    try:
        return float(accuracy_score(y_true, y_pred))
    except Exception:
        return float("nan")

def _find_pair_files(pair: str) -> Optional[Path]:
    # prefer datasets
    cand = DATASETS / f"{pair}.parquet"
    if cand.exists():
        return cand
    for p in (DATASETS / pair).glob("*.parquet"):
        return p
    # fallback to features
    cand2 = FEATURES / f"{pair}.parquet"
    if cand2.exists():
        return cand2
    for p in (FEATURES / pair).glob("*.parquet"):
        return p
    return None

def _list_pairs_from_manifest() -> List[str]:
    mani = FEATURES / "_manifest.json"
    if mani.exists():
        try:
            data = json.loads(mani.read_text(encoding="utf-8"))
            pairs = data.get("pairs") or data.get("symbols") or []
            if isinstance(pairs, list):
                return sorted(pairs)
        except Exception:
            pass
    # fallback: scan datasets and features folders
    pairs = set()
    for base in [DATASETS, FEATURES]:
        if base.exists():
            for f in base.rglob("*.parquet"):
                pairs.add(f.stem)
    return sorted(pairs)

@dataclass
class FoldMetrics:
    auc: float
    logloss: float
    accuracy: float

# ----------------- MLflow experiment helpers ----------------- #

def _ensure_mlflow_experiment() -> str:
    """
    Ensure an MLflow experiment exists and is bound to S3 artifacts.
    Reads:
      - MLFLOW_EXPERIMENT_NAME (default: "mntrading_s3")
      - MINIO_BUCKET (default: "mlflow")
      - MLFLOW_ARTIFACTS_PREFIX (default: "experiments")
    Returns experiment id (string or empty).
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    exp_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "mntrading_s3")
    bucket = os.environ.get("MINIO_BUCKET", "mlflow")
    prefix = os.environ.get("MLFLOW_ARTIFACTS_PREFIX", "experiments")
    artifact_location = f"s3://{bucket}/{prefix}/{exp_name}".rstrip("/")

    client = MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        try:
            exp_id = client.create_experiment(name=exp_name, artifact_location=artifact_location)
        except Exception:
            # fall back to default server-side artifact root if creation with custom location is restricted
            exp_id = client.create_experiment(name=exp_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(exp_name)
    return exp_id

# ----------------- main training ----------------- #

def _prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    # Try to find target column
    y_col = None
    for c in ("y", "label", "target", "signal"):
        if c in df.columns:
            y_col = c
            break
    if y_col is None:
        raise ValueError("No target column found. Expected one of: y, label, target, signal")

    y = df[y_col].astype(int).copy()
    X = df.drop(columns=[y_col]).copy()

    # Drop non-numeric columns except timestamp
    for c in list(X.columns):
        if np.issubdtype(X[c].dtype, np.number):
            continue
        if c != "timestamp":
            X.drop(columns=[c], inplace=True)

    # Replace inf/nan
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y

def _time_series_cv(X: pd.DataFrame, y: pd.Series, n_splits=3, gap=5, max_train_size=2000, seed=42):
    """
    Regular TimeSeriesSplit without shuffling, emulate 'gap' by trimming tail of train.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=None, gap=0)
    N = len(X)
    idx = np.arange(N)
    for train_idx, test_idx in tscv.split(idx):
        if gap > 0:
            mx = test_idx.min()
            train_idx = train_idx[train_idx <= (mx - gap)]
        if max_train_size and len(train_idx) > max_train_size:
            train_idx = train_idx[-max_train_size:]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield train_idx, test_idx

def _fit_eval_models(X: pd.DataFrame, y: pd.Series, n_splits=3, gap=5,
                     max_train_size=2000, early_stopping_rounds=50, seed=42) -> Tuple[Dict, Dict, Dict[str, np.ndarray]]:
    """
    Train baseline and candidates with OOF evaluation.
    Returns:
      metrics_by_model: {name: {"auc":..., "logloss":..., "accuracy":...}}
      models_fitted: {name: estimator}
      oof: {name: np.ndarray}
    """
    rng = np.random.RandomState(seed)

    # Baseline LR (robust)
    pipe_lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None, random_state=seed))
    ])

    # RF
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None,
        min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed
    )

    # LightGBM — стабильные настройки под «плоские»/малые данные
    lgbm = None
    if HAS_LGB:
        lgbm = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            boosting_type="gbdt",
            learning_rate=0.05,
            n_estimators=500,
            num_leaves=31,
            max_depth=-1,
            min_data_in_leaf=5,
            min_sum_hessian_in_leaf=1e-3,
            min_gain_to_split=0.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=-1,
            random_state=seed,
        )
        # иногда помогает на ступенчатых фичах:
        # lgbm.set_params(max_bin=511)

    models = {"logreg": pipe_lr, "rf": rf}
    if lgbm is not None:
        models["lgbm"] = lgbm

    # OOF
    oof = {name: np.full(len(X), np.nan, dtype=float) for name in models}
    for tr, te in _time_series_cv(X, y, n_splits=n_splits, gap=gap, max_train_size=max_train_size, seed=seed):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]

        Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # если целевая константна — пропускаем фолд
        if ytr.nunique() < 2:
            continue

        for name, est in models.items():
            est_ = est
            if HAS_LGB and name == "lgbm":
                try:
                    est_.fit(
                        Xtr, ytr, eval_set=[(Xte, yte)],
                        eval_metric="logloss",
                        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
                    )
                except Exception:
                    est_.fit(Xtr, ytr)
            else:
                est_.fit(Xtr, ytr)

            try:
                p = est_.predict_proba(Xte)[:, 1]
            except Exception:
                s = est_.decision_function(Xte)
                p = 1 / (1 + np.exp(-s))
            oof[name][te] = p

    # Метрики
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    for name, pred in oof.items():
        mask = ~np.isnan(pred)
        if mask.sum() == 0:
            m = {"auc": float("nan"), "logloss": float("nan"), "accuracy": float("nan")}
        else:
            m = {
                "auc": _auc_safe(y[mask].values, pred[mask]),
                "logloss": _logloss_safe(y[mask].values, pred[mask]),
                "accuracy": float(accuracy_score(y[mask].values, (pred[mask] >= 0.5).astype(int))),
            }
        metrics_by_model[name] = m

    # Финальный фит на всём
    models_fitted = {}
    Xfull = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for name, est in models.items():
        try:
            est.fit(Xfull, y)
            models_fitted[name] = est
        except Exception:
            pass

    return metrics_by_model, models_fitted, oof

def _choose_champion(metrics_by_model: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    ranked = sorted(
        metrics_by_model.items(),
        key=lambda kv: (float(kv[1].get("auc") or 0.0), float(kv[1].get("accuracy") or 0.0)),
        reverse=True
    )
    if not ranked:
        return "logreg", {"auc": float("nan"), "logloss": float("nan"), "accuracy": float("nan")}
    return ranked[0][0], ranked[0][1]

def train_one_pair(pair: str, n_splits=3, gap=5, max_train_size=2000,
                   early_stopping_rounds=50, seed=42) -> Dict:
    """Train models for a single pair and save artifacts + mlflow logging."""
    fpath = _find_pair_files(pair)
    if fpath is None or (not fpath.exists()):
        raise FileNotFoundError(f"Dataset/features for pair {pair} not found")

    df = pd.read_parquet(fpath)
    X, y = _prepare_xy(df)

    # safety guards
    if len(X) < max(100, n_splits * 50) or pd.Series(y).nunique() < 2:
        return {"pair": pair, "skipped": True, "reason": "too_small_or_constant_target"}

    metrics_by_model, models_fitted, oof = _fit_eval_models(
        X, y,
        n_splits=n_splits,
        gap=gap,
        max_train_size=max_train_size,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed
    )
    champion_name, champion_metrics = _choose_champion(metrics_by_model)

    out_dir = PAIRS_MODELS / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    # OOF
    oof_df = pd.DataFrame({"idx": np.arange(len(y)), "y": y.values})
    for name, p in oof.items():
        oof_df[f"p_{name}"] = p
    oof_path = out_dir / "oof.parquet"
    oof_df.to_parquet(oof_path, index=False)

    # Models
    saved_paths = {}
    for name, est in models_fitted.items():
        pkl = out_dir / f"{name}.pkl"
        with open(pkl, "wb") as f:
            pickle.dump(est, f)
        saved_paths[name] = str(pkl)

    # Champion copy
    champ_path = ""
    if champion_name in saved_paths:
        cp = out_dir / "__champion.pkl"
        with open(saved_paths[champion_name], "rb") as src, open(cp, "wb") as dst:
            dst.write(src.read())
        champ_path = str(cp)

    meta = {
        "pair": pair,
        "metrics": champion_metrics,
        "metrics_by_model": metrics_by_model,
        "champion": champion_name,
        "champion_type": champion_name,
        "champion_path": champ_path,
        "oof_path": str(oof_path),
    }
    (out_dir / "__meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # MLflow logging
    try:
        import mlflow
        _ensure_mlflow_experiment()
        with mlflow.start_run(run_name=pair):
            mlflow.log_param("n_splits", n_splits)
            mlflow.log_param("gap", gap)
            mlflow.log_param("max_train_size", max_train_size)
            mlflow.log_param("seed", seed)
            mlflow.log_param("features_cols", len(X.columns))

            for model_name, m in metrics_by_model.items():
                for k, v in m.items():
                    if isinstance(v, float) and (math.isfinite(v) or np.isnan(v)):
                        mlflow.log_metric(f"{model_name}_{k}", 0.0 if np.isnan(v) else float(v))
            auc = champion_metrics.get("auc", float("nan"))
            mlflow.log_metric("champion_auc", 0.0 if np.isnan(auc) else float(auc))

            for name, est in models_fitted.items():
                if name == "lgbm" and HAS_LGB:
                    _mlflow_log_model_lightgbm(est, name="lgbm")
                else:
                    _mlflow_log_model_sklearn(est, name=name)

            mlflow.log_artifact(str(oof_path), artifact_path="oof")
            mlflow.log_artifact(str(out_dir / "__meta.json"), artifact_path="meta")
    except Exception as e:
        print(f"[mlflow] warning: {e!r}")

    return {
        "pair": pair,
        "skipped": False,
        "champion": champion_name,
        "metrics": champion_metrics,
        "models": list(models_fitted.keys()),
        "paths": saved_paths,
        "meta_path": str(out_dir / "__meta.json"),
    }

def train_all_pairs(use_dataset: bool = True, n_splits: int = 3, gap: int = 5,
                    max_train_size: int = 2000, early_stopping_rounds: int = 50, seed: int = 42) -> Dict:
    pairs = _list_pairs_from_manifest()
    if not pairs:
        return {"pairs_trained": 0, "items": [], "reason": "no_pairs_found"}

    items = []
    for i, pair in enumerate(pairs, 1):
        try:
            print(f"[train] ({i}/{len(pairs)}) pair={pair}")
            res = train_one_pair(
                pair=pair,
                n_splits=n_splits,
                gap=gap,
                max_train_size=max_train_size,
                early_stopping_rounds=early_stopping_rounds,
                seed=seed
            )
            items.append(res)
        except Exception as e:
            print(f"[train] pair={pair} failed: {e!r}")
            items.append({"pair": pair, "skipped": True, "reason": f"exception: {e!r}"})

    trained = [x for x in items if not x.get("skipped")]
    return {"pairs_trained": len(trained), "pairs_total": len(items), "items": items}
