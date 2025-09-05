from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException


def _read_json(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _canon_key(pair: str) -> str:
    return pair.replace("/", "_").strip()


def _ensure_experiment(tracking_uri: str, name: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(name)
    except MlflowException:
        # Try restore if it was deleted
        client = MlflowClient(tracking_uri=tracking_uri)
        for exp in client.search_experiments(view_type=ViewType.ALL):
            if exp.name == name and exp.lifecycle_stage == "deleted":
                client.restore_experiment(exp.experiment_id)
                break
        mlflow.set_experiment(name)


def _pairs_from_registry_json(registry_path: Path, top_k: int) -> List[str]:
    obj = _read_json(registry_path)
    arr: Optional[List[dict]] = None
    for key in ("pairs", "items", "selected", "champions", "models", "registry"):
        if isinstance(obj.get(key), list):
            arr = obj[key]
            break
    if arr is None and isinstance(obj, dict):
        tmp = []
        for k, v in obj.items():
            if isinstance(v, dict):
                tmp.append({"pair": k, **v})
        arr = tmp

    if not arr:
        return []

    def _score(d: dict) -> float:
        return float((d.get("metrics") or {}).get("sharpe", 0.0))

    arr = sorted(arr, key=_score, reverse=True)
    if top_k > 0:
        arr = arr[:top_k]
    out = []
    for it in arr:
        pair = it.get("pair") or it.get("name") or it.get("key")
        if isinstance(pair, str):
            out.append(pair)
    return out


def _model_name_for_pair(prefix: str, pair: str) -> str:
    return f"{prefix}_{_canon_key(pair)}"


def _build_mapping_from_mlflow(
    client: MlflowClient,
    pairs: List[str],
    prefix: str,
    stage: str,
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for pair in pairs:
        name = _model_name_for_pair(prefix, pair)
        try:
            versions = client.search_model_versions(f"name='{name}'")
            if not versions:
                continue
            if stage and stage.lower() not in ("none",):
                ok = any(v.current_stage.lower() == stage.lower() for v in versions)
                if not ok:
                    continue
            uri = f"models:/{name}/{stage}" if stage and stage != "None" else f"models:/{name}"
            mapping[_canon_key(pair)] = uri
        except Exception:
            continue
    return mapping


class RouterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pandas as pd
        import numpy as np
        import mlflow

        self.pd = pd
        self.np = np

        cfg = json.loads(Path(context.artifacts["router_config"]).read_text(encoding="utf-8"))
        self.proba_threshold = float(cfg.get("proba_threshold", 0.55))
        self.mapping: Dict[str, str] = cfg.get("uri_mapping", {})

        self.models: Dict[str, mlflow.pyfunc.PyFuncModel] = {}
        for key, uri in self.mapping.items():
            try:
                self.models[key] = mlflow.pyfunc.load_model(uri)
            except Exception:
                self.models[key] = None

    def _predict_proba1(self, model, X):
        import numpy as np
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X)
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                return p[:, 1]
            if p.ndim == 1:
                return p
        if hasattr(model, "decision_function"):
            d = np.asarray(model.decision_function(X), dtype="float64")
            return 1.0 / (1.0 + np.exp(-d))
        if hasattr(model, "predict"):
            y = np.asarray(model.predict(X), dtype="float64")
            return np.clip(y, 0.0, 1.0)
        return np.full((len(X),), 0.5, dtype="float64")

    def predict(self, context, model_input):
        import numpy as np
        import pandas as pd

        df = model_input.copy()
        if "pair" not in df.columns:
            raise ValueError("Input must include 'pair' column")

        z_col = "z" if "z" in df.columns else None
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        feat_cols = [c for c in num_cols if c not in ("side",)]
        if z_col and z_col in feat_cols:
            feat_cols.remove(z_col)

        out = []
        for pair, grp in df.groupby("pair"):
            key = _canon_key(pair)
            sub = self.models.get(key)
            if sub is None:
                for _ in range(len(grp)):
                    out.append({"pair": pair, "proba": 0.0, "side": 0})
                continue

            X = grp[feat_cols].to_numpy(dtype="float64", copy=False)
            try:
                pred = sub.predict(grp[feat_cols])
                if hasattr(pred, "values"):
                    pred = pred.values
                proba = np.asarray(pred, dtype="float64").reshape(-1)
                if proba.min() < 0.0 or proba.max() > 1.0:
                    proba = 1.0 / (1.0 + np.exp(-proba))
            except Exception:
                proba = self._predict_proba1(sub, X)

            if z_col and z_col in grp.columns:
                sign = np.sign(grp[z_col].to_numpy(dtype="float64", copy=False))
                side = np.where(proba >= self.proba_threshold, np.where(sign > 0, 1, -1), 0)
            else:
                side = np.where(proba >= self.proba_threshold, 1, 0)

            for i in range(len(grp)):
                out.append({"pair": pair, "proba": float(proba[i]), "side": int(side[i])})

        return pd.DataFrame(out, columns=["pair", "proba", "side"])



def main():
    ap = argparse.ArgumentParser(description="Build and register MLflow Serving router model (loads sub-models from MLflow Registry)")
    ap.add_argument("--registry", type=Path, required=True, help="Path to data/models/registry.json (used to pick top-K pairs)")
    ap.add_argument("--registered-name", default="mntrading_router", help="Registered router model name")
    ap.add_argument("--experiment", default="mntrading", help="MLflow experiment name")
    ap.add_argument("--prefix", default="mntrading", help="Prefix of per-pair models in MLflow (e.g., mntrading_BTC_USDT__ETH_USDT)")
    ap.add_argument("--stage", default="Production", choices=["None", "Staging", "Production", "Archived"], help="Stage of sub-models to use")
    ap.add_argument("--top-k", type=int, default=20, help="Take top-K pairs from registry.json")
    ap.add_argument("--proba-threshold", type=float, default=0.55, help="Threshold used inside the router for side decision")
    args = ap.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    _ensure_experiment(tracking_uri, args.experiment)

    pairs = _pairs_from_registry_json(args.registry, args.top_k)
    if not pairs:
        print("[router] nothing to pack (empty pairs in registry)")
        return

    client = MlflowClient(tracking_uri=tracking_uri)
    uri_mapping = _build_mapping_from_mlflow(client, pairs=pairs, prefix=args.prefix, stage=args.stage)
    if not uri_mapping:
        print("[router] nothing to pack (no MLflow models found for requested pairs/stage)")
        return

    cfg = {
        "proba_threshold": float(args.proba_threshold),
        "uri_mapping": uri_mapping,  # pair_key -> models:/name/stage
    }
    tmp_cfg = Path("router_config.json")
    tmp_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=f"build_router_{args.registered_name}") as run:
        mlflow.pyfunc.log_model(
            artifact_path="model",  # using artifact_path for compatibility
            python_model=RouterModel(),
            artifacts={"router_config": str(tmp_cfg)},
            pip_requirements=[
                "mlflow",
                "pandas",
                "numpy",
                "scikit-learn",
                "joblib",
                "cloudpickle",
                "requests",
            ],
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    try:
        client.get_registered_model(args.registered_name)
    except MlflowException:
        client.create_registered_model(args.registered_name)

    mv = client.create_model_version(name=args.registered_name, source=model_uri, run_id=run.info.run_id)
    print(f"[router] registered {args.registered_name} v{mv.version}")

    if args.stage != "None":
        client.transition_model_version_stage(
            name=args.registered_name,
            version=mv.version,
            stage=args.stage,
            archive_existing_versions=False,
        )
        print(f"[router] transitioned to stage {args.stage}")

    try:
        tmp_cfg.unlink(missing_ok=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()