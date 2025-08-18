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

def _safe_auc(y_true, y_prob) -> float:
    u = np.unique(y_true)
    if len(u) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _safe_logloss(y_true, y_prob) -> float:
    try:
        return float(log_loss(y_true, y_prob, eps=1e-15))
    except Exception:
        return float("nan")


def _choose_champion(metrics: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    ranked = sorted(
        metrics.items(),
        key=lambda kv: (float(kv[1].get("auc") or 0.0), float(kv[1].get("accuracy") or 0.0)),
        reverse=True,
    )
    if not ranked:
        return "logreg", {"auc": float("nan"), "logloss": float("nan"), "accuracy": float("nan")}
    return ranked[0][0], ranked[0][1]


def _detect_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    # целевая колонка — любая из типичных
    y_col = next((c for c in ["y", "label", "target", "signal"] if c in df.columns), None)
    if y_col is None:
        raise ValueError("No target column found. Expected one of: y, label, target, signal")

    y = df[y_col].astype(int).copy()
    X = df.drop(columns=[y_col]).copy()

    # выкинем нечисловые, кроме timestamp/ts (пусть пройдёт вниз по конвейеру)
    for c in list(X.columns):
        if np.issubdtype(X[c].dtype, np.number):
            continue
        if c not in ("timestamp", "ts"):
            X.drop(columns=[c], inplace=True)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, y


def _list_pairs_from_manifest(base: Path) -> List[str]:
    # пробуем <base>/pairs/_manifest.json, затем <base>/_manifest.json
    for cand in [base / "pairs" / "_manifest.json", base / "_manifest.json"]:
        if cand.exists():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                pairs = data.get("pairs") or data.get("symbols") or []
                if isinstance(pairs, list):
                    # элементы могут быть строками "A/B|C/D"
                    norm: List[str] = []
                    for x in pairs:
                        if isinstance(x, str):
                            norm.append(x)
                        elif isinstance(x, dict) and "pair" in x:
                            norm.append(x["pair"])
                    return sorted(set(norm or pairs))
            except Exception:
                pass

    # fallback: просканировать parquet-файлы
    pairs = set()
    for root in [base / "pairs", base]:
        if root.exists():
            for f in root.rglob("*.parquet"):
                pairs.add(f.stem)
    return sorted(pairs)


def _resolve_pair_file(base: Path, pair: str) -> Optional[Path]:
    # ищем в <base>/pairs/<PAIR>.parquet, затем в <base>/<PAIR>.parquet
    cand = base / "pairs" / f"{pair}.parquet"
    if cand.exists():
        return cand
    cand2 = base / f"{pair}.parquet"
    if cand2.exists():
        return cand2
    # ещё вариант: папка <base>/pairs/<PAIR>/*.parquet
    folder = base / "pairs" / pair
    if folder.exists():
        ps = list(folder.glob("*.parquet"))
        if ps:
            return ps[0]
    return None


def _time_series_folds(n: int, n_splits=3, gap=5, max_train_size=2000):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=0)
    idx = np.arange(n)
    for tr, te in tscv.split(idx):
        if gap > 0:
            mx = te.min()
            tr = tr[tr <= (mx - gap)]
        if max_train_size and len(tr) > max_train_size:
            tr = tr[-max_train_size:]
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
        mlflow.sklearn.log_model(sk_model=model, artifact_path=name, input_example=input_example)


def _mlflow_log_model_lightgbm(model, name: str):
    import mlflow
    try:
        mlflow.lightgbm.log_model(lgb_model=model, name=name)
    except TypeError:
        mlflow.lightgbm.log_model(lgb_model=model, artifact_path=name)


# ---------- core training ----------

def _fit_eval_all(X: pd.DataFrame, y: pd.Series, n_splits: int, gap: int,
                  max_train_size: int, early_stopping_rounds: int, seed: int = 42):
    # Baseline
    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None,
        min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed,
    )

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
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            n_jobs=-1,
            random_state=seed,
        )

    models = {"logreg": lr, "rf": rf}
    if lgbm is not None:
        models["lgbm"] = lgbm

    oof = {k: np.full(len(X), np.nan) for k in models}
    for tr, te in _time_series_folds(len(X), n_splits=n_splits, gap=gap, max_train_size=max_train_size):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        Xte, yte = X.iloc[te], y.iloc[te]
        if ytr.nunique() < 2:
            continue

        for name, est in models.items():
            est_ = est
            if HAS_LGB and name == "lgbm":
                try:
                    est_.fit(
                        Xtr, ytr,
                        eval_set=[(Xte, yte)],
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
                p = 1.0 / (1.0 + np.exp(-s))
            oof[name][te] = p

    metrics_by_model: Dict[str, Dict[str, float]] = {}
    for name, p in oof.items():
        mask = ~np.isnan(p)
        if mask.sum() == 0:
            m = {"auc": float("nan"), "logloss": float("nan"), "accuracy": float("nan")}
        else:
            m = {
                "auc": _safe_auc(y[mask].values, p[mask]),
                "logloss": _safe_logloss(y[mask].values, p[mask]),
                "accuracy": float(accuracy_score(y[mask].values, (p[mask] >= 0.5).astype(int))),
            }
        metrics_by_model[name] = m

    # финальная дообучка на всём
    models_full: Dict[str, object] = {}
    Xfull = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for name, est in models.items():
        try:
            est.fit(Xfull, y)
            models_full[name] = est
        except Exception:
            pass

    return metrics_by_model, models_full, oof


# ---------- public API for main.py ----------

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
    ds_base = Path(datasets_dir)
    fe_base = Path(features_dir)
    out_base = Path(out_dir)
    out_pairs = out_base / "pairs"
    out_pairs.mkdir(parents=True, exist_ok=True)

    source = ds_base if use_dataset and ds_base.exists() else fe_base
    pairs = _list_pairs_from_manifest(source)
    if not pairs:
        return {"pairs_trained": 0, "pairs_total": 0, "items": [], "reason": "no_pairs_found"}

    # MLflow: не обязателен, но попробуем
    try:
        import mlflow
        _ensure_mlflow_experiment()
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

        df = pd.read_parquet(fpath)
        try:
            X, y = _detect_xy(df)
        except Exception as e:
            results.append({"pair": pair, "skipped": True, "reason": f"bad_dataset: {e}"})
            continue

        # sanity-guard
        if len(X) < max(100, n_splits * 50) or pd.Series(y).nunique() < 2:
            results.append({"pair": pair, "skipped": True, "reason": "too_small_or_constant"})
            continue

        metrics_by_model, models_full, oof = _fit_eval_all(
            X, y, n_splits=n_splits, gap=gap, max_train_size=max_train_size,
            early_stopping_rounds=early_stopping_rounds
        )
        champ_name, champ_metrics = _choose_champion(metrics_by_model)

        # save artifacts
        pair_dir = out_pairs / pair
        pair_dir.mkdir(parents=True, exist_ok=True)

        # OOF parquet
        oof_df = pd.DataFrame({"idx": np.arange(len(y)), "y": y.values})
        for name, p in oof.items():
            oof_df[f"p_{name}"] = p
        oof_path = pair_dir / "oof.parquet"
        oof_df.to_parquet(oof_path, index=False)

        # models .pkl
        saved = {}
        for name, est in models_full.items():
            pkl = pair_dir / f"{name}.pkl"
            with open(pkl, "wb") as f:
                pickle.dump(est, f)
            saved[name] = str(pkl)

        # champion copy
        champ_path = ""
        if champ_name in saved:
            cp = pair_dir / "__champion.pkl"
            with open(saved[champ_name], "rb") as src, open(cp, "wb") as dst:
                dst.write(src.read())
            champ_path = str(cp)

        meta = {
            "pair": pair,
            "champion": champ_name,
            "metrics": champ_metrics,
            "metrics_by_model": metrics_by_model,
            "oof_path": str(oof_path),
            "champion_path": champ_path,
        }
        (pair_dir / "__meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # MLflow logging
        if mlflow_ok:
            try:
                import mlflow
                with mlflow.start_run(run_name=pair):
                    mlflow.log_param("n_splits", n_splits)
                    mlflow.log_param("gap", gap)
                    mlflow.log_param("max_train_size", max_train_size)
                    mlflow.log_param("features_cols", len(X.columns))

                    for mn, mm in metrics_by_model.items():
                        for k, v in mm.items():
                            vv = 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)
                            mlflow.log_metric(f"{mn}_{k}", vv)
                    mlflow.log_metric("champion_auc", 0.0 if math.isnan(champ_metrics.get("auc", float("nan"))) else float(champ_metrics.get("auc", 0.0)))

                    # log models
                    for name, est in models_full.items():
                        if name == "lgbm" and HAS_LGB:
                            _mlflow_log_model_lightgbm(est, name="lgbm")
                        else:
                            _mlflow_log_model_sklearn(est, name=name)

                    mlflow.log_artifact(str(oof_path), artifact_path="oof")
                    mlflow.log_artifact(str(pair_dir / "__meta.json"), artifact_path="meta")
            except Exception as e:
                print(f"[mlflow] warning: {e!r}")

        results.append({
            "pair": pair,
            "skipped": False,
            "champion": champ_name,
            "metrics": champ_metrics,
            "models": list(models_full.keys()),
            "meta_path": str(pair_dir / "__meta.json"),
        })

    trained = [r for r in results if not r.get("skipped")]
    report = {
        "pairs_trained": len(trained),
        "pairs_total": len(results),
        "items": results,
    }
    return report
