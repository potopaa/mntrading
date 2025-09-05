from __future__ import annotations
import argparse
import os
from typing import List, Dict
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd


class RouterModel(mlflow.pyfunc.PythonModel):

    def __init__(self, model_uris: List[str]):
        self.model_uris = list(model_uris)
        self._submodels: Dict[int, mlflow.pyfunc.PyFuncModel] = {}

    def load_context(self, context):
        pass

    def _load_if_needed(self, idx: int):
        if idx in self._submodels:
            return self._submodels[idx]
        import mlflow.pyfunc as pyfunc
        m = pyfunc.load_model(self.model_uris[idx])
        self._submodels[idx] = m
        return m

    def predict(self, context, model_input):
        max_len = len(model_input) if hasattr(model_input, "__len__") else 0
        cols = {}
        for i, _ in enumerate(self.model_uris):
            col = f"m{i}"
            try:
                m = self._load_if_needed(i)
                y = m.predict(model_input)
                if isinstance(y, pd.Series):
                    cols[col] = y.reset_index(drop=True)
                elif isinstance(y, pd.DataFrame):
                    cols[col] = y.iloc[:, 0].reset_index(drop=True)
                else:
                    s = pd.Series(y)
                    cols[col] = s.reset_index(drop=True) if len(s) == max_len else pd.Series([None] * max_len)
            except Exception:
                cols[col] = pd.Series([None] * max_len)
        # Align lengths
        for k, v in cols.items():
            if len(v) != max_len:
                cols[k] = pd.Series([None] * max_len)
        return pd.DataFrame(cols)


def list_models_with_stage(prefix: str, stage: str, top_k: int) -> List[str]:
    c = MlflowClient()
    names = [m.name for m in c.search_registered_models()]
    selected = []
    for name in sorted(names):
        if not name.startswith(prefix):
            continue
        try:
            vers = c.get_latest_versions(name, [stage])
            if vers:
                selected.append(name)
        except Exception:
            continue
        if len(selected) >= top_k:
            break
    return selected


def build_and_register_router(
    *,
    prefix: str = "mntrading_",
    pair_stage: str = "Staging",
    top_k: int = 20,
    registered_name: str = "mntrading_router",
    router_stage: str = "Production",
    experiment: str = "mntrading",
):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    if experiment:
        mlflow.set_experiment(experiment)

    model_names = list_models_with_stage(prefix=prefix, stage=pair_stage, top_k=top_k)
    if not model_names:
        raise RuntimeError(f"No per-pair models found for stage={pair_stage} with prefix={prefix}")

    model_uris = [f"models:/{name}/{pair_stage}" for name in model_names]

    with mlflow.start_run(run_name="build_router_from_mlflow"):
        python_model = RouterModel(model_uris=model_uris)
        result = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=python_model,
            registered_model_name=registered_name,
        )
        run_id = mlflow.active_run().info.run_id
        print(f"[router] logged run_id={run_id}, artifact_uri={result.artifact_path}, submodels={len(model_uris)}")

    c = MlflowClient()
    vers = c.get_latest_versions(registered_name)
    if not vers:
        raise RuntimeError(f"Router '{registered_name}' not found after logging.")
    latest_v = sorted(vers, key=lambda v: int(v.version))[-1]
    c.transition_model_version_stage(registered_name, latest_v.version, router_stage)
    print(f"[router] {registered_name} v{latest_v.version} -> {router_stage}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="mntrading_", help="Prefix of per-pair registered model names.")
    p.add_argument("--pair-stage", default="Staging", help="Stage to pick per-pair models from.")
    p.add_argument("--top-k", type=int, default=20, help="How many per-pair models to include.")
    p.add_argument("--registered-name", default="mntrading_router", help="Router registered model name.")
    p.add_argument("--router-stage", default="Production", help="Target stage for the router.")
    p.add_argument("--experiment", default="mntrading", help="MLflow experiment name.")
    args = p.parse_args()

    build_and_register_router(
        prefix=args.prefix,
        pair_stage=args.pair_stage,
        top_k=args.top_k,
        registered_name=args.registered_name,
        router_stage=args.router_stage,
        experiment=args.experiment,
    )


if __name__ == "__main__":
    main()