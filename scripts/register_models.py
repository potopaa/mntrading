import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import numpy as np
import pandas as pd
try:
    import joblib
except Exception:
    joblib = None
import pickle


# --------------------------- helpers ---------------------------

def sanitize_pair(pair: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", pair).strip("_")


def ensure_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    return prefix if prefix.endswith("_") else prefix + "_"


def load_local_model(model_dir: Path):
    pkl = model_dir / "model.pkl"
    jbl = model_dir / "model.joblib"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    if jbl.exists() and joblib is not None:
        return joblib.load(jbl)
    for cand in list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib")):
        try:
            with open(cand, "rb") as f:
                return pickle.load(f) if cand.suffix == ".pkl" else joblib.load(cand)
        except Exception:
            continue
    return None


def discover_features_df(features_root: Path, pair_id: str) -> Optional[pd.DataFrame]:
    cand = features_root / pair_id / "features.parquet"
    if cand.exists():
        try:
            df = pd.read_parquet(cand, engine="pyarrow")
            if df.empty:
                return None
            return df.tail(5)
        except Exception:
            return None
    return None


def pick_input_columns(df: pd.DataFrame) -> List[str]:
    drop_like = {"y", "label", "target", "ts", "time", "timestamp", "pair",
                 "symbol", "symbol_a", "symbol_b", "run_id", "fold", "split_id"}
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feat_cols = [c for c in num_cols if c not in drop_like]
    return feat_cols or num_cols


class SklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import pickle as _pkl
        model_path = context.artifacts["model"]
        with open(model_path, "rb") as f:
            self.model = _pkl.load(f)

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(model_input)
            if isinstance(proba, (list, tuple)):
                proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                out = proba[:, 1]
            else:
                out = np.max(proba, axis=1)
            return pd.DataFrame({"proba": out})
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(model_input)
            return pd.DataFrame({"score": np.asarray(scores).ravel()})
        preds = self.model.predict(model_input)
        return pd.DataFrame({"pred": np.asarray(preds).ravel()})


def log_and_register_single(
    pair_id: str,
    model_obj,
    input_example: Optional[pd.DataFrame],
    experiment: str,
    registered_name: str,
    stage: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"register_{registered_name}") as run:
        tmp_model_path = Path(mlflow.get_artifact_uri()).joinpath("sk_model.pkl")
        tmp_model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_model_path, "wb") as f:
            pickle.dump(model_obj, f)

        signature = None
        output_example = None
        if input_example is not None and not input_example.empty:
            try:
                test_df = input_example.head(1).copy()
                if hasattr(model_obj, "predict_proba"):
                    proba = model_obj.predict_proba(test_df)
                    if isinstance(proba, (list, tuple)):
                        proba = np.asarray(proba)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        output_example = pd.DataFrame({"proba": proba[:1, 1]})
                    else:
                        output_example = pd.DataFrame({"proba": np.max(proba[:1], axis=1)})
                elif hasattr(model_obj, "decision_function"):
                    output_example = pd.DataFrame({"score": np.asarray(model_obj.decision_function(test_df)).ravel()[:1]})
                else:
                    output_example = pd.DataFrame({"pred": np.asarray(model_obj.predict(test_df)).ravel()[:1]})
                signature = infer_signature(model_input=test_df, model_output=output_example)
            except Exception:
                try:
                    signature = infer_signature(model_input=input_example.head(1))
                except Exception:
                    signature = None

        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SklearnWrapper(),
            artifacts={"model": str(tmp_model_path)},
            input_example=input_example.head(1) if input_example is not None and not input_example.empty else None,
            signature=signature,
        )

        mv = mlflow.register_model(model_uri=model_info.model_uri, name=registered_name)
        registered_version = str(mv.version)

    if stage and stage not in {"None", "none"}:
        try:
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(name=registered_name, version=registered_version, stage=stage)
        except Exception:
            pass

    return run.info.run_id if 'run' in locals() else None, registered_version


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Register pair models to MLflow with signature and input_example.")
    ap.add_argument("--registry", required=True, help="Path to data/models/registry.json")
    ap.add_argument("--models-dir", default="/app/data/models/pairs", help="Root dir with per-pair local models")
    ap.add_argument("--features-dir", default="/app/data/features/pairs", help="Root dir with per-pair features parquet")
    ap.add_argument("--experiment", default="mntrading", help="MLflow experiment name")
    ap.add_argument("--prefix", default="mntrading_", help="Registered model name prefix, e.g. 'mntrading_'")
    ap.add_argument("--stage", default="Staging", choices=["None", "none", "Staging", "Production", "Archived"], help="Target stage")
    ap.add_argument("--top-k", type=int, default=50, help="Limit number of pairs to register")
    args = ap.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.exists():
        print(f"[register] registry not found: {registry_path}", file=sys.stderr)
        sys.exit(1)

    prefix = ensure_prefix(args.prefix)
    models_root = Path(args.models_dir)
    features_root = Path(args.features_dir)

    obj = json.load(open(registry_path, "r", encoding="utf-8"))
    pairs = obj.get("pairs") or obj.get("models") or []
    if isinstance(pairs, dict):
        pairs = list(pairs.values())
    pair_ids = []
    for e in pairs:
        if isinstance(e, str):
            pair_ids.append(sanitize_pair(e))
        elif isinstance(e, dict) and "pair" in e:
            pair_ids.append(sanitize_pair(str(e["pair"])))
    pair_ids = pair_ids[: int(args.top_k)]

    if not pair_ids:
        print("[register] nothing to do (0 pairs in registry)", flush=True)
        return

    print(f"[register] pairs to process: {len(pair_ids)}")
    ok = 0
    for i, pair in enumerate(pair_ids, start=1):
        pair_dir = models_root / pair
        if not pair_dir.exists():
            print(f"[register][{i}/{len(pair_ids)}] skip {pair}: models dir not found -> {pair_dir}")
            continue

        model_obj = load_local_model(pair_dir)
        if model_obj is None:
            print(f"[register][{i}/{len(pair_ids)}] skip {pair}: no model.pkl/joblib in {pair_dir}")
            continue

        feat_df = discover_features_df(features_root, pair)
        input_df = None
        if feat_df is not None:
            cols = pick_input_columns(feat_df)
            if cols:
                input_df = feat_df[cols].copy()

        registered_name = f"{prefix}{pair}"
        try:
            run_id, version = log_and_register_single(
                pair_id=pair,
                model_obj=model_obj,
                input_example=input_df,
                experiment=args.experiment,
                registered_name=registered_name,
                stage=args.stage,
            )
            ok += 1
            print(f"[register][{i}/{len(pair_ids)}] OK {registered_name} -> v{version}")
        except Exception as e:  # noqa: BLE001
            print(f"[register][{i}/{len(pair_ids)}] FAIL {registered_name}: {e}", file=sys.stderr)

    print(f"[register] done, registered={ok}, pairs_in_input={len(pair_ids)}")


if __name__ == "__main__":
    main()