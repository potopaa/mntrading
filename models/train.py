#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-model training for mntrading with optional MLflow tracking.

- Trains per-pair three models: LogisticRegression (baseline), RandomForest, LightGBM.
- Time-series CV with OOF probabilities per model.
- Selects the best model per pair by mean AUC across folds.
- Writes OOF of the winner to data/models/pairs/<PAIR>__oof.parquet (as before).
- Saves the winner model to data/models/pairs/<PAIR>__model.pkl (+ __meta.json).
- If MLFLOW_TRACKING_URI is set, logs runs/params/metrics/artifacts to MLflow.
  Optionally registers the winner to MLflow Model Registry if env MLFLOW_REGISTER=1.

Expected to be called by main.py: train_baseline(...)
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- third-party models ---
# sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
except Exception as e:
    raise SystemExit("scikit-learn is required. Install: pip install scikit-learn") from e

# lightgbm
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# optional mlflow
try:
    import mlflow
    import mlflow.sklearn
    if HAS_LGBM:
        import mlflow.lightgbm
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False


# ----------------------- utilities -----------------------
def _find_dataset_files(datasets_dir: Path) -> Dict[str, Path]:
    mp: Dict[str, Path] = {}
    if not datasets_dir.exists():
        return mp
    for p in datasets_dir.glob("*__ds.parquet"):
        key = p.stem.replace("__ds", "")
        # normalize "AAA_USDT__BBB_USDT" -> "AAA/USDT__BBB/USDT"
        if "__" in key and "/" not in key:
            a, b = key.split("__", 1)
            pair_key = a.replace("_", "/") + "__" + b.replace("_", "/")
        else:
            pair_key = key
        mp[pair_key] = p
    return mp


def _time_series_splits(n: int, n_splits: int, gap: int, max_train_size: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_splits <= 0:
        return []
    base = n // (n_splits + 1)
    offset = n - base * (n_splits + 1)
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


def _class_balance(y: np.ndarray) -> Tuple[int, int, float]:
    total = len(y)
    pos = int(np.sum(y == 1))
    neg = total - pos
    rate = (pos / total) if total else 0.0
    return pos, neg, rate


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> Optional[float]:
    try:
        return float(roc_auc_score(y_true, proba))
    except Exception:
        return None


def _make_models(seed: int = 42) -> Dict[str, object]:
    """
    Return dict of model_name -> estimator (sklearn-compatible with predict_proba)
    """
    models: Dict[str, object] = {}

    # Baseline: Logistic Regression with standardization
    models["logreg"] = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
        ("clf", LogisticRegression(
            solver="lbfgs", max_iter=2000, n_jobs=None, random_state=seed, class_weight="balanced"
        )),
    ])

    # RandomForest
    models["rf"] = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=5, n_jobs=-1,
        random_state=seed, class_weight="balanced_subsample"
    )

    # LightGBM
    if HAS_LGBM:
        models["lgbm"] = lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=20,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=seed,
            n_jobs=-1,
        )
    return models


def _log_mlflow_start(experiment_name: str, run_name: str, tags: Dict[str, str]):
    if not HAS_MLFLOW:
        return None
    try:
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name, tags=tags)
        return run
    except Exception:
        return None


def _mlflow_log(run, params: Dict = None, metrics: Dict = None, artifacts: List[Tuple[str, str]] = None):
    if not HAS_MLFLOW or run is None:
        return
    try:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if artifacts:
            for local_path, artifact_path in artifacts:
                mlflow.log_artifact(local_path, artifact_path=artifact_path)
    except Exception:
        pass


def _mlflow_log_model(run, model, artifact_subpath: str, model_name: Optional[str] = None, register: bool = False):
    if not HAS_MLFLOW or run is None:
        return
    try:
        uri = None
        if HAS_LGBM and isinstance(model, lgb.LGBMClassifier):
            mlflow.lightgbm.log_model(model, artifact_path=artifact_subpath, registered_model_name=(model_name if register else None))
            uri = mlflow.get_artifact_uri(artifact_subpath)
        else:
            mlflow.sklearn.log_model(model, artifact_path=artifact_subpath, registered_model_name=(model_name if register else None))
            uri = mlflow.get_artifact_uri(artifact_subpath)
        return uri
    except Exception:
        return None


# ----------------------- main trainer -----------------------
def train_baseline(
    datasets_dir: str,
    features_dir: str,
    out_dir: str,
    use_dataset: bool,
    n_splits: int,
    gap: int,
    max_train_size: int,
    early_stopping_rounds: int,  # kept for API compatibility; not used directly here
    proba_threshold: float,      # kept for API compatibility
) -> Dict:
    out_root = Path(out_dir)
    out_pairs = out_root / "pairs"
    out_root.mkdir(parents=True, exist_ok=True)
    out_pairs.mkdir(parents=True, exist_ok=True)

    # Discover pairs/datasets
    ds_dir = Path(datasets_dir)
    ds_files = _find_dataset_files(ds_dir)
    if ds_files:
        pairs = sorted(ds_files.keys())
    else:
        # Fallback: from features manifest (labels will be built from z)
        man = json.loads((Path(features_dir) / "_manifest.json").read_text(encoding="utf-8"))
        pairs = sorted({str(it["pair"]) for it in man.get("items", []) if "pair" in it})

    global_report: Dict[str, Dict] = {}
    seed = 42
    models = _make_models(seed=seed)

    # Optional MLflow global tags
    mlflow_enabled = HAS_MLFLOW and (os.environ.get("MLFLOW_TRACKING_URI") or "").strip() != ""
    register_models = (os.environ.get("MLFLOW_REGISTER") == "1")

    for pk in pairs:
        try:
            # Load data
            if use_dataset and pk in ds_files:
                df = pd.read_parquet(ds_files[pk])
                if "ts" not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index().rename(columns={"index": "ts"})
                    else:
                        df["ts"] = np.arange(len(df))
                if "y" not in df.columns:
                    if "z" in df.columns:
                        df["y"] = (df["z"].abs() > 1.5).astype(int)
                    else:
                        raise ValueError("Dataset has no 'y' and no 'z' to derive labels.")
            else:
                fpath = Path(features_dir) / pk.replace("/", "_") / "features.parquet"
                df = pd.read_parquet(fpath)
                if "z" not in df.columns:
                    raise ValueError("Features lack 'z' column; cannot derive labels.")
                if "ts" not in df.columns:
                    if isinstance(df.index, pd.DatetimeIndex):
                        df = df.reset_index().rename(columns={"index": "ts"})
                    else:
                        df["ts"] = np.arange(len(df))
                df["y"] = (df["z"].abs() > 1.5).astype(int)

            # Sort by time
            df = df.sort_values("ts").reset_index(drop=True)

            # Features: numeric only, drop ['ts','y','pair']
            drop_cols = {"ts", "y", "pair"}
            X = df.drop(columns=[c for c in df.columns if c in drop_cols], errors="ignore")
            X = X.select_dtypes(include=[np.number]).astype("float32").copy()
            y = df["y"].astype(int).values
            n = len(X)
            if n < 400:
                global_report[pk] = {"rows": int(n), "note": "too_few_rows"}
                continue

            # Build splits
            splits = _time_series_splits(n, n_splits=n_splits, gap=gap, max_train_size=max_train_size)
            if not splits:
                global_report[pk] = {"rows": int(n), "note": "no_splits"}
                continue

            # MLflow parent run per pair
            parent_run = None
            if mlflow_enabled:
                parent_run = _log_mlflow_start(
                    experiment_name="mntrading",
                    run_name=f"{pk}",
                    tags={"pair": pk, "stage": "train", "use_dataset": str(bool(use_dataset))}
                )

            # Train/eval each model
            model_reports: Dict[str, Dict] = {}
            oof_by_model: Dict[str, np.ndarray] = {}

            for mname, est in models.items():
                oof = np.full(n, np.nan, dtype=float)
                aucs: List[float] = []
                # nested mlflow run
                child = None
                if mlflow_enabled and parent_run is not None:
                    child = _log_mlflow_start(
                        experiment_name="mntrading",
                        run_name=f"{pk}__{mname}",
                        tags={"pair": pk, "model": mname}
                    )
                    _mlflow_log(child, params={
                        "model": mname, "n_splits": n_splits, "gap": gap, "max_train_size": max_train_size
                    })

                for (tr_idx, va_idx) in splits:
                    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
                    X_va, y_va = X.iloc[va_idx], y[va_idx]

                    # Skip folds with one class
                    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                        oof[va_idx] = 0.5
                        continue

                    # Fit
                    if HAS_LGBM and isinstance(est, lgb.LGBMClassifier):
                        est.set_params(random_state=seed)
                    est.fit(X_tr, y_tr)

                    # Predict
                    if hasattr(est, "predict_proba"):
                        proba = est.predict_proba(X_va)[:, 1]
                    else:
                        # fallback: decision_function -> sigmoid
                        if hasattr(est, "decision_function"):
                            z = est.decision_function(X_va)
                            proba = 1.0 / (1.0 + np.exp(-z))
                        else:
                            proba = np.full(len(y_va), 0.5)
                    proba = np.clip(proba, 1e-6, 1 - 1e-6)
                    oof[va_idx] = proba

                    auc = _safe_auc(y_va, proba)
                    if auc is not None:
                        aucs.append(auc)

                # finalize OOF
                oof = pd.Series(oof).fillna(0.5).values
                auc_mean = float(np.nanmean(aucs)) if len(aucs) else None

                # store report
                model_reports[mname] = {
                    "auc_mean": auc_mean,
                    "aucs": aucs,
                    "rows": int(n),
                    "splits": len(splits),
                }
                oof_by_model[mname] = oof

                # mlflow log
                if mlflow_enabled and child is not None:
                    _mlflow_log(child, metrics={"auc_mean": (auc_mean if auc_mean is not None else -1.0)})
                    # save temporary OOF artifact to pair dir and log
                    tmp_oof = out_pairs / f"{pk.replace('/', '_')}__{mname}__oof.parquet"
                    pd.DataFrame({"ts": df["ts"].values, "y": y, "proba": oof}).to_parquet(tmp_oof, index=False)
                    _mlflow_log(child, artifacts=[(str(tmp_oof), f"oof/{pk.replace('/', '_')}")])
                    try:
                        mlflow.end_run()
                    except Exception:
                        pass

            # pick winner
            best_name = None
            best_auc = -1.0
            for mname, rep in model_reports.items():
                auc = rep.get("auc_mean")
                if auc is not None and auc > best_auc:
                    best_auc = auc
                    best_name = mname
            if best_name is None:
                # nothing worked, default to logreg oof as neutral
                best_name = "logreg"

            # Fit winner on full data and save model
            winner = models[best_name]
            if HAS_LGBM and isinstance(winner, lgb.LGBMClassifier):
                winner.set_params(random_state=seed)
            winner.fit(X, y)

            model_path = out_pairs / f"{pk.replace('/', '_')}__model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(winner, f)

            meta = {
                "pair": pk,
                "winner": best_name,
                "auc_mean": (float(best_auc) if best_auc is not None else None),
                "features": list(X.columns),
                "n_rows": int(n),
                "n_splits": int(n_splits),
            }
            meta_path = out_pairs / f"{pk.replace('/', '_')}__meta.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            # Write OOF of the winner (as expected by backtest)
            oof_winner = oof_by_model.get(best_name)
            if oof_winner is None:
                oof_winner = np.full(n, 0.5)
            df_oof = pd.DataFrame({"ts": df["ts"].values, "y": y, "proba": oof_winner})
            oof_path = out_pairs / f"{pk.replace('/', '_')}__oof.parquet"
            df_oof.to_parquet(oof_path, index=False)

            # Parent MLflow logging + model artifact
            if mlflow_enabled and parent_run is not None:
                _mlflow_log(parent_run, params={"winner": best_name}, metrics={"winner_auc_mean": (best_auc if best_auc is not None else -1.0)})
                # log artifacts: winner model, meta, oof
                _mlflow_log(parent_run, artifacts=[
                    (str(model_path), f"models/{pk.replace('/', '_')}"),
                    (str(meta_path),  f"models/{pk.replace('/', '_')}"),
                    (str(oof_path),   f"oof/{pk.replace('/', '_')}"),
                ])
                # also log as MLflow model artifact and optionally register
                model_uri = _mlflow_log_model(parent_run, winner, artifact_subpath=f"mlflow_models/{pk.replace('/', '_')}",
                                              model_name=(f"mntrading_{pk.replace('/', '_')}" if register_models else None),
                                              register=register_models)
                try:
                    mlflow.end_run()
                except Exception:
                    pass

            # Final report entry
            global_report[pk] = {
                "rows": int(n),
                "winner": best_name,
                "winner_auc_mean": (float(best_auc) if best_auc is not None else None),
                "models": model_reports,
                "oof_path": str(oof_path),
                "model_path": str(model_path),
                "meta_path": str(meta_path),
            }

        except Exception as e:
            global_report[pk] = {"error": str(e)}
            continue

    (Path(out_dir) / "_train_report.json").write_text(json.dumps(global_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return global_report
