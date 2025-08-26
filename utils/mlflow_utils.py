import os
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", TRACKING_URI)

def ensure_experiment(name: str) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
    else:
        exp_id = exp.experiment_id
    return exp_id

def register_sklearn_model(model, model_name: str, run_id: str, artifact_subpath: str = "model"):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(REGISTRY_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=REGISTRY_URI)
    model_uri = f"runs:/{run_id}/{artifact_subpath}"
    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    return mv

def get_production_version(model_name: str):
    client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=REGISTRY_URI)
    for v in client.search_model_versions(f"name='{model_name}'"):
        if v.current_stage == "Production":
            return v
    return None

def get_run_metric(run_id: str, key: str):
    client = MlflowClient(tracking_uri=TRACKING_URI)
    r = client.get_run(run_id)
    return r.data.metrics.get(key)

def promote_if_better(model_name: str, new_mv, metric_key: str, direction: str = "higher", min_improve: float = 0.0):
    client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=REGISTRY_URI)
    prod = get_production_version(model_name)
    new_score = get_run_metric(new_mv.run_id, metric_key)
    if prod is None:
        client.transition_model_version_stage(model_name, new_mv.version, stage="Production")
        return {"action": "promoted_first", "new_metric": new_score}
    old_score = get_run_metric(prod.run_id, metric_key)
    better = (new_score - old_score) >= min_improve if direction == "higher" else (old_score - new_score) >= min_improve
    if better:
        client.transition_model_version_stage(model_name, prod.version, stage="Archived")
        client.transition_model_version_stage(model_name, new_mv.version, stage="Production")
        return {"action": "promoted", "old_metric": old_score, "new_metric": new_score}
    else:
        client.transition_model_version_stage(model_name, new_mv.version, stage="Staging")
        return {"action": "staged_only", "old_metric": old_score, "new_metric": new_score}
