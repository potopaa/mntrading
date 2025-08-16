#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# optional boosters
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature


# ==== utils: trading metrics on validation ====

def _to_signals_from_proba_and_z(proba: pd.Series, z: pd.Series, thr: float) -> pd.Series:
    sig = pd.Series(0, index=proba.index, dtype=int)
    m = proba >= thr
    if "z" in z.index.names or z.index.equals(proba.index):
        sig.loc[m] = np.where(z.loc[m] < 0, 1, -1)
    else:
        sig.loc[m] = 0
    return sig


def _val_trading_metrics(sig: pd.Series, price: pd.Series) -> Dict[str, float]:
    ret = sig.shift(1).fillna(0) * np.log(price).diff().fillna(0)
    eq = (1 + ret).cumprod()
    if ret.std(ddof=0) > 0:
        sharpe = float(np.sqrt(252) * ret.mean() / ret.std(ddof=0))
    else:
        sharpe = float("nan")
    dd = np.maximum.accumulate(eq) - eq
    maxdd = float((dd / np.maximum.accumulate(eq)).max()) if len(dd) else float("nan")
    return {"val_sharpe": sharpe, "val_maxdd": maxdd}


# ==== ES helpers ====

def _fit_with_es(clf, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame, y_te: np.ndarray, es_rounds: int):
    """
    Универсальная ранняя остановка:
    - пробуем старый kwarg `early_stopping_rounds`;
    - если не сработало — callbacks (новые XGBoost/LightGBM);
    - если ES недоступен — обучаем без него.
    """
    if es_rounds is None or es_rounds <= 0:
        clf.fit(X_tr, y_tr)
        return clf

    try:
        clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], early_stopping_rounds=es_rounds)
        return clf
    except TypeError:
        pass

    try:
        if xgb is not None and clf.__class__.__module__.startswith("xgboost"):
            cb = [xgb.callback.EarlyStopping(rounds=es_rounds, save_best=True)]
            clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=cb)
            return clf
    except Exception as e:
        warnings.warn(f"XGBoost ES via callbacks failed: {e}")

    try:
        if lgb is not None and clf.__class__.__module__.startswith("lightgbm"):
            cb = [lgb.early_stopping(es_rounds, verbose=False)]
            clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=cb)
            return clf
    except Exception as e:
        warnings.warn(f"LightGBM ES via callbacks failed: {e}")

    warnings.warn("Early stopping not available for this classifier. Fitting without ES.")
    clf.fit(X_tr, y_tr)
    return clf


# ==== MLflow helpers (signature + alias, без deprecated API) ====

def _latest_version_via_search(client: MlflowClient, model_name: str) -> Optional[str]:
    """Берём последнюю версию через search_model_versions (без Stage API)."""
    try:
        vers = client.search_model_versions(f"name = '{model_name}'")
        if not vers:
            return None
        # максимальная по номеру
        return str(max(int(v.version) for v in vers))
    except Exception:
        return None


def _mlflow_log_and_register_model(clf, model_name: str, X_example: Optional[pd.DataFrame] = None):
    """
    Логируем и регистрируем модель с подписью:
      - используем новый аргумент `name="model"` (если недоступен — fallback на `artifact_path`);
      - передаём signature и input_example;
      - alias 'staging' ставим через set_model_version_alias, версию берём через search_model_versions.
    Возвращает dict {version, run_id}.
    """
    signature = None
    input_example = None
    try:
        if X_example is not None and len(X_example) > 0:
            X_ex = X_example.head(5)
            # попытаемся вывести целевую форму выхода для сигнатуры
            try:
                y_ex = clf.predict_proba(X_ex)[:, 1]
            except Exception:
                y_ex = clf.predict(X_ex)
            signature = infer_signature(X_ex, y_ex)
            input_example = X_ex
    except Exception:
        pass

    # лог модели
    try:
        mlflow.sklearn.log_model(
            sk_model=clf,
            name="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example
        )
    except TypeError:
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=model_name,
            signature=signature,
            input_example=input_example
        )

    client = MlflowClient()
    ver = _latest_version_via_search(client, model_name)

    if ver is not None:
        try:
            client.set_model_version_alias(name=model_name, version=ver, alias="staging")
        except Exception:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                try:
                    client.transition_model_version_stage(name=model_name, version=ver, stage="Staging")
                except Exception:
                    pass

    rid = None
    try:
        rid = mlflow.active_run().info.run_id
    except Exception:
        pass

    return {"version": ver, "run_id": rid}


# ==== config dataclass ====

@dataclass
class TrainCfg:
    n_splits: int
    gap: int
    max_train_size: Optional[int]
    early_stopping_rounds: int
    calibrate_rf: bool
    proba_threshold: float
    random_state: int = 42
    experiment_name: str = "baseline_pair_trading"


def _make_models(random_state: int) -> Dict[str, object]:
    models = {}

    rf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state
        ))
    ])
    models["rf"] = rf

    if lgb is not None:
        lgbm = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", lgb.LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.02,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=random_state
            ))
        ])
        models["lgbm"] = lgbm

    if xgb is not None:
        xgbm = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", xgb.XGBClassifier(
                n_estimators=3000,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=random_state,
                tree_method="hist",
                eval_metric="logloss",
            ))
        ])
        models["xgb"] = xgbm

    return models


def _time_series_splits(n: int, n_splits: int, gap: int, max_train_size: Optional[int]):
    base = TimeSeriesSplit(n_splits=n_splits, test_size=None, gap=gap)
    indices = np.arange(n)
    for tr, te in base.split(indices):
        if max_train_size and len(tr) > max_train_size:
            tr = tr[-max_train_size:]
        yield tr, te


# ==== calibrator helper (совместимость с версиями sklearn) ====

def _make_calibrator(base_clf, method: str = "isotonic", cv: int = 3):
    """
    Возвращает CalibratedClassifierCV поверх base_clf.
    - sklearn>=1.4: аргумент называется `estimator`
    - старые версии: `base_estimator`
    """
    try:
        return CalibratedClassifierCV(estimator=base_clf, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_clf, method=method, cv=cv)


def train_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    pair_key: str,
    aux_df: pd.DataFrame,
    fee_rate: float,
    proba_threshold: float,
    n_splits: int,
    gap: int,
    max_train_size: Optional[int],
    early_stopping_rounds: int,
    calibrate_rf: bool,
    refit_on_full: bool,
    random_state: int,
    experiment_name: str = "baseline_pair_trading",
) -> Dict:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = pd.Series(y).astype(int)
    assert len(X) == len(y), "X and y must have same length"

    cfg = TrainCfg(
        n_splits=n_splits,
        gap=gap,
        max_train_size=max_train_size,
        early_stopping_rounds=early_stopping_rounds,
        calibrate_rf=calibrate_rf,
        proba_threshold=proba_threshold,
        random_state=random_state,
        experiment_name=experiment_name
    )

    os.makedirs("data/models", exist_ok=True)
    out_dir = os.path.join("data/models", pair_key.replace("/", "_"))
    os.makedirs(out_dir, exist_ok=True)

    mlflow.set_experiment(cfg.experiment_name)

    models = _make_models(cfg.random_state)
    oof_all: Dict[str, pd.Series] = {}
    fold_reports: Dict[str, List[Dict]] = {name: [] for name in models.keys()}

    feature_cols = list(X.columns)

    for mdl_name, pipe in models.items():
        oof = pd.Series(index=X.index, dtype=float)

        for fold_id, (tr_idx, te_idx) in enumerate(_time_series_splits(len(X), cfg.n_splits, cfg.gap, cfg.max_train_size)):
            X_tr = X.iloc[tr_idx].copy()
            y_tr = y.iloc[tr_idx].values
            X_te = X.iloc[te_idx].copy()
            y_te = y.iloc[te_idx].values

            if mdl_name == "rf" and cfg.calibrate_rf:
                base_clf = pipe.named_steps["clf"]
                pipe.named_steps["clf"] = _make_calibrator(base_clf, method="isotonic", cv=3)

            pipe.named_steps["clf"] = _fit_with_es(
                pipe.named_steps["clf"], X_tr, y_tr, X_te, y_te, cfg.early_stopping_rounds
            )

            try:
                p = pipe.named_steps["clf"].predict_proba(X_te)[:, 1]
            except Exception:
                pred = pipe.named_steps["clf"].predict(X_te)
                p = pred if isinstance(pred, np.ndarray) else np.asarray(pred)
            oof.iloc[te_idx] = p

            # валидационные метрики по фолду
            try:
                roc = roc_auc_score(y_te, p)
            except Exception:
                roc = float("nan")

            if aux_df is not None and "pa" in aux_df.columns:
                z = aux_df["z"].reindex(X_te.index) if "z" in aux_df.columns else pd.Series(np.nan, index=X_te.index)
                sig = _to_signals_from_proba_and_z(pd.Series(p, index=X_te.index), z, cfg.proba_threshold)
                m = _val_trading_metrics(sig, aux_df["pa"].reindex(X_te.index).dropna())
            else:
                m = {"val_sharpe": float("nan"), "val_maxdd": float("nan")}

            fold_reports[mdl_name].append({
                "fold": fold_id,
                "roc_auc": float(roc),
                **m
            })

        oof_all[mdl_name] = oof

    # агрегируем по фолдам и выбираем лучшую модель
    cv_summary = {}
    for mdl_name, reps in fold_reports.items():
        if not reps:
            cv_summary[mdl_name] = {"val_sharpe_mean": float("nan"), "roc_auc_mean": float("nan")}
            continue
        sharpe_mean = float(np.nanmean([r["val_sharpe"] for r in reps]))
        roc_mean = float(np.nanmean([r["roc_auc"] for r in reps]))
        cv_summary[mdl_name] = {"val_sharpe_mean": sharpe_mean, "roc_auc_mean": roc_mean}

    best_name = max(cv_summary.keys(), key=lambda k: (cv_summary[k]["val_sharpe_mean"], cv_summary[k]["roc_auc_mean"]))
    best_oof = oof_all[best_name]

    # метрики OOF: считаем только по валидным точкам (без NaN)
    valid_mask = np.isfinite(best_oof.values)
    oof_auc_val = float(roc_auc_score(y.loc[best_oof.index][valid_mask], best_oof.values[valid_mask])) if valid_mask.any() and len(np.unique(y)) > 1 else float("nan")

    with mlflow.start_run(run_name=f"{pair_key}__{best_name}"):
        mlflow.log_params({
            "pair": pair_key,
            "n_splits": cfg.n_splits,
            "gap": cfg.gap,
            "max_train_size": cfg.max_train_size or 0,
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "calibrate_rf": cfg.calibrate_rf,
            "proba_threshold": cfg.proba_threshold,
            "features_count": len(feature_cols),
            "model": best_name
        })

        for mdl_name, stats in cv_summary.items():
            mlflow.log_metric(f"{mdl_name}_val_sharpe_mean", stats["val_sharpe_mean"])
            mlflow.log_metric(f"{mdl_name}_roc_auc_mean", stats["roc_auc_mean"])

        if np.isfinite(oof_auc_val):
            mlflow.log_metric("oof_roc_auc", oof_auc_val)

        # refit на всём датасете лучшей моделью (с малым вал-срезом для ES)
        final_model = _make_models(cfg.random_state)[best_name]
        if best_name == "rf" and cfg.calibrate_rf:
            base_clf = final_model.named_steps["clf"]
            final_model.named_steps["clf"] = _make_calibrator(base_clf, method="isotonic", cv=3)

        final_model.named_steps["clf"] = _fit_with_es(
            final_model.named_steps["clf"],
            X, y.values,
            X.iloc[-min(len(X), 1000):], y.iloc[-min(len(y), 1000):].values,
            cfg.early_stopping_rounds
        )

        # регистрируем в MLflow с подписью
        model_name = f"mntrading__{pair_key.replace('/','_')}"
        reg_info = _mlflow_log_and_register_model(final_model.named_steps["clf"], model_name, X_example=X.head(20))
        run_id = mlflow.active_run().info.run_id

    # сохраняем OOF и мета локально
    local_dir = os.path.join(out_dir, best_name)
    os.makedirs(local_dir, exist_ok=True)

    pd.DataFrame({
        "y_true": y.loc[best_oof.index].values,
        "y_proba": best_oof.values
    }, index=best_oof.index).to_parquet(os.path.join(local_dir, "oof_predictions.parquet"))

    meta = {
        "pair": pair_key,
        "model_name": best_name,
        "features": feature_cols,
        "oof_metrics": {"roc_auc": oof_auc_val},
        "cv_val_means": cv_summary[best_name],
        "run_id": run_id,
        "model_version": reg_info.get("version") if isinstance(reg_info, dict) else None
    }
    with open(os.path.join(local_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "best_model": best_name,
        "cv_val_means": cv_summary[best_name],
        "oof_metrics": meta["oof_metrics"],
        "run_id": run_id,
        "model_version": meta["model_version"],
        "features": feature_cols
    }
