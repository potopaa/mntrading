# -*- coding: utf-8 -*-
"""
Training module expected by main.py:
 - exposes train_baseline(...) with signature that main.py calls
 - supports datasets (preferred) or features as source
 - per-pair OOF metrics + champion selection (AUC, tie-break by accuracy)
 - artifacts saved under data/models/pairs/<PAIR>/
 - MLflow logging with experiment created in S3 (MinIO) if needed
"""

from __future__ import annotations

import os
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ---------- utils ----------

def _utf8_stdio():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _safe_auc(y_true, y_proba) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return float("nan")


def _safe_acc(y_true, y_pred) -> float:
    try:
        return float(accuracy_score(y_true, y_pred))
    except Exception:
        return float("nan")


def _safe_logloss(y_true, y_proba) -> float:
    try:
        return float(log_loss(y_true, y_proba, eps=1e-7))
    except Exception:
        return float("nan")


def _read_pair_features(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _resolve_pair_file(source: str, pair: str) -> Optional[Path]:
    # source == "dataset" or "features"
    if source == "dataset":
        p = Path("data/datasets/pairs") / f"{pair}.parquet"
        return p
    else:
        p = Path("data/features/pairs") / f"{pair}.parquet"
        return p if p.exists() else None


def _iter_time_splits(n: int, n_splits: int, gap: int, max_train_size: int):
    """
    Custom CV for time series with optional max_train_size and gap before test.
    Yields (train_idx, test_idx).
    """
    if n_splits <= 1:
        yield np.arange(0, n - gap), np.arange(n - gap, n)
        return
    # crude splitter similar to TimeSeriesSplit but with gap & max_train_size
    fold_size = (n - gap) // n_splits
    for i in range(n_splits):
        tr_end = (i + 1) * fold_size
        te_start = tr_end + gap
        te_end = min(te_start + fold_size, n)
        if te_start >= te_end or tr_end <= 0:
            continue
        tr_start = max(0, tr_end - max_train_size) if max_train_size > 0 else 0
        tr = np.arange(tr_start, tr_end)
        te = np.arange(te_start, te_end)
        if len(tr) == 0 or len(te) == 0:
            continue
        yield tr, te


# ---------- MLflow helpers ----------

def _ensure_mlflow_experiment() -> str:
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
            exp_id = client.create_experiment(name=exp_name)
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(exp_name)
    return exp_id


def _mlflow_log_model_sklearn(model, name: str, input_example=None):
    import mlflow
    try:
        mlflow.sklearn.log_model(sk_model=model, name=name, input_example=input_example)
    except TypeError:
        # MLflow < 2.14
        mlflow.sklearn.log_model(sk_model=model, artifact_path=name, input_example=input_example)


def _mlflow_log_model_lightgbm(model, name: str):
    import mlflow
    try:
        mlflow.lightgbm.log_model(lgb_model=model, name=name)
    except TypeError:
        # MLflow < 2.14
        mlflow.lightgbm.log_model(lgb_model=model, artifact_path=name)


# ---------- core training ----------

def _fit_eval_all(X: pd.DataFrame, y: pd.Series, n_splits: int, gap: int,
                  max_train_size: int, early_stopping_rounds: int):
    """
    Train 2–3 simple models + LightGBM (if available) and compute OOF metrics.
    Returns dict: {'baseline': {...}, 'rf': {...}, 'lgbm': {...?}} with metrics + predictions.
    """
    results = {}

    # Baseline: logistic regression on standardized features
    pipe_lr = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )

    # Optional LGBM
    if HAS_LGB:
        lgbm = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        lgbm = None

    # OOF storage
    oof = {
        "baseline": np.full(len(y), np.nan, dtype=float),
        "rf": np.full(len(y), np.nan, dtype=float),
    }
    if HAS_LGB:
        oof["lgbm"] = np.full(len(y), np.nan, dtype=float)

    # CV
    for tr_idx, te_idx in _iter_time_splits(len(y), n_splits, gap, max_train_size):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        # Baseline
        pipe_lr.fit(X_tr, y_tr)
        oof["baseline"][te_idx] = pipe_lr.predict_proba(X_te)[:, 1]

        # RF
        rf.fit(X_tr, y_tr)
        oof["rf"][te_idx] = rf.predict_proba(X_te)[:, 1]

        # LGB
        if HAS_LGB:
            # keep it robust on small/flat data
            lgbm.fit(
                X_tr, y_tr,
                eval_set=[(X_te, y_te)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=max(10, early_stopping_rounds), verbose=False),
                    lgb.log_evaluation(period=0),
                ]
            )
            oof["lgbm"][te_idx] = lgbm.predict_proba(X_te)[:, 1]

    # Metrics
    out = {}
    for name, pred in oof.items():
        mask = ~np.isnan(pred)
        if mask.sum() == 0:
            out[name] = {"auc": float("nan"), "logloss": float("nan"), "acc": float("nan")}
            continue
        y_true = y.iloc[mask].values
        y_proba = pred[mask]
        y_pred = (y_proba >= 0.5).astype(int)
        out[name] = {
            "auc": _safe_auc(y_true, y_proba),
            "logloss": _safe_logloss(y_true, y_proba),
            "acc": _safe_acc(y_true, y_pred),
        }

    return out, pipe_lr, rf, (lgbm if HAS_LGB else None), oof


def _choose_champion(metrics_by_model: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    """
    Choose best model by AUC; tie-break by accuracy.
    """
    best = None
    best_name = ""
    for name, m in metrics_by_model.items():
        auc = m.get("auc", float("nan"))
        acc = m.get("acc", float("nan"))
        key = (0.0 if math.isnan(auc) else auc, 0.0 if math.isnan(acc) else acc)
        if (best is None) or (key > best):
            best = key
            best_name = name
    return best_name, metrics_by_model.get(best_name, {})


def train_baseline(
    datasets_dir: str,
    features_dir: str,
    out_dir: str,
    use_dataset: bool,
    n_splits: int,
    gap: int,
    max_train_size: int,
    early_stopping_rounds: int,
    proba_threshold: float = 0.5,
) -> Dict:
    """
    Expected by main.py. Trains per-pair models and returns JSON-able report.
    Artifacts saved under <out_dir>/pairs/<PAIR>/.
    """
    _utf8_stdio()

    # Source
    source = "dataset" if use_dataset else "features"

    # Determine pair list from manifest produced by previous steps
    manifest_path = Path("data/features/pairs/_manifest.json")
    if manifest_path.exists():
        try:
            pairs = json.loads(manifest_path.read_text(encoding="utf-8")).get("pairs", [])
        except Exception:
            pairs = []
    else:
        pairs = []

    if not pairs:
        return {"pairs_trained": 0, "pairs_total": 0, "items": [], "reason": "no_pairs_found"}

    # MLflow: optional but preferred
    try:
        import mlflow
        exp_id = _ensure_mlflow_experiment()
        print(f"[mlflow] tracking_uri={mlflow.get_tracking_uri()} exp_id={exp_id}")
        mlflow_ok = True
    except Exception:
        mlflow_ok = False

    results = []

    for i, pair in enumerate(pairs, 1):
        print(f"[train] ({i}/{len(pairs)}) pair={pair}")
        fpath = _resolve_pair_file(source, pair)
        if fpath is None or not fpath.exists():
            results.append({"pair": pair, "skipped": True, "reason": "file_not_found"})
            continue

        df = _read_pair_features(fpath)
        if df is None or len(df) < 50:
            results.append({"pair": pair, "skipped": True, "reason": "too_few_rows"})
            continue

        # Expect columns: features... + 'y'
        if "y" not in df.columns:
            results.append({"pair": pair, "skipped": True, "reason": "no_y"})
            continue

        y = df["y"].astype(int)
        X = df.drop(columns=["y"])
        # sanity: keep only numeric columns
        X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Train + OOF metrics
        metrics_by_model, baseline, rf, lgbm, oof = _fit_eval_all(
            X=X, y=y, n_splits=n_splits, gap=gap, max_train_size=max_train_size,
            early_stopping_rounds=early_stopping_rounds
        )

        champion_name, champ_metrics = _choose_champion(metrics_by_model)

        # Save artifacts under out_dir/pairs/<PAIR>/
        base_dir = Path(out_dir)
        pair_dir = base_dir / "pairs" / pair
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Save champion model
        champ_path = pair_dir / f"model_{champion_name}.pkl"
        model_obj = {"baseline": baseline, "rf": rf, "lgbm": (lgbm if HAS_LGB else None)}.get(champion_name)
        with open(champ_path, "wb") as f:
            pickle.dump(model_obj, f)

        # Save OOF predictions
        oof_path = pair_dir / "oof.parquet"
        pd.DataFrame(oof).to_parquet(oof_path, index=False)

        meta = {
            "pair": pair,
            "source": source,
            "champion": champion_name,
            "metrics": metrics_by_model,
            "champion_metrics": champ_metrics,
            "oof_path": str(oof_path),
            "champion_path": str(champ_path),
        }
        (pair_dir / "__meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # MLflow logging (explicit experiment_id + run_id in logs)
        if mlflow_ok:
            try:
                import mlflow
                # end any previous active run (safety)
                if mlflow.active_run():
                    mlflow.end_run()
                # start run bound to our experiment id
                with mlflow.start_run(run_name=pair, experiment_id=exp_id) as r:
                    print(f"[mlflow] run started pair={pair} run_id={r.info.run_id}")
                    mlflow.log_param("n_splits", n_splits)
                    mlflow.log_param("gap", gap)
                    mlflow.log_param("max_train_size", max_train_size)
                    mlflow.log_param("features_cols", len(X.columns))

                    for mn, mm in metrics_by_model.items():
                        for k, v in mm.items():
                            vv = 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)
                            mlflow.log_metric(f"{mn}_{k}", vv)
                    mlflow.log_metric(
                        "champion_auc",
                        0.0 if math.isnan(champ_metrics.get("auc", float("nan"))) else float(champ_metrics.get("auc", 0.0))
                    )

                    # log models (best effort)
                    models_full = {"baseline": baseline, "rf": rf}
                    if HAS_LGB and (lgbm is not None):
                        models_full["lgbm"] = lgbm
                    for name, est in models_full.items():
                        try:
                            if name == "lgbm" and HAS_LGB:
                                _mlflow_log_model_lightgbm(est, name="lgbm")
                            else:
                                _mlflow_log_model_sklearn(est, name=name)
                        except Exception as e:
                            print(f"[mlflow] model_log warn ({name}): {e!r}")

                    mlflow.log_artifact(str(oof_path), artifact_path="oof")
                    mlflow.log_artifact(str(pair_dir / "__meta.json"), artifact_path="meta")
            except Exception as e:
                print(f"[mlflow] warning: {e!r}")

        # Append report item
        results.append({
            "pair": pair,
            "champion": champion_name,
            "metrics": metrics_by_model,
        })

    report = {
        "pairs_trained": sum(1 for r in results if not r.get("skipped")),
        "pairs_total": len(results),
        "items": results,
    }
    # write global train report
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[train] report → data/models/_train_report.json")
    return report
