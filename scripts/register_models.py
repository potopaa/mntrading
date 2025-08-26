#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/register_models.py
# All comments are in English by request.

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow.pyfunc


def _ensure_experiment(tracking_uri: str, name: str) -> str:
    """
    Ensure MLflow experiment exists and is active.
    If deleted, try to restore; otherwise create '<name>_v2'.
    """
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(name)
        return name
    except MlflowException as e:
        msg = str(e).lower()
        if "deleted" in msg:
            client = MlflowClient(tracking_uri=tracking_uri)
            for exp in client.search_experiments(view_type=ViewType.ALL):
                if exp.name == name and exp.lifecycle_stage == "deleted":
                    client.restore_experiment(exp.experiment_id)
                    mlflow.set_experiment(name)
                    return name
        alt = f"{name}_v2"
        mlflow.set_experiment(alt)
        return alt


def _read_json(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _canon_key(pair: str) -> str:
    """Turn 'BNB/USDT__XRP/USDT' into 'BNB_USDT__XRP_USDT'."""
    return pair.replace("/", "_")


def _find_model_fallback(models_dir: Path, pair: str) -> Optional[Path]:
    """
    If registry has no model_path, try to locate any artifact under models_dir[/pairs]/<pair_key>.
    Accept extensions: .pkl, .joblib, .json
    """
    key = _canon_key(pair)
    roots = [models_dir / key, models_dir / "pairs" / key]
    root = next((r for r in roots if r.exists()), None)
    if not root:
        return None
    cands: List[Path] = []
    for ext in (".pkl", ".joblib", ".json"):
        cands.extend(sorted(root.rglob(f"*{ext}")))
    return cands[-1] if cands else None


class FileArtifactModel(mlflow.pyfunc.PythonModel):
    """
    Minimal pyfunc wrapper that stores the original model file as artifact.
    Predict is not implemented because offline inference in this project
    uses local files. This wrapper exists to create a valid MLflow Model
    directory for Model Registry.
    """

    def load_context(self, context):
        # Keep path to the original file inside artifacts
        self.model_file = context.artifacts.get("model_file")

    def predict(self, context, model_input):
        # This model is not intended for online predict in this project.
        # Return None or raise to make the behavior explicit.
        return None


def _collect_pairs(registry_path: Path, models_dir: Path, top_k: Optional[int]) -> List[Tuple[str, Optional[str], dict]]:
    """
    Returns a list of tuples: (pair, model_path, metrics_dict).
    """
    obj = _read_json(registry_path)
    arr = None
    for key in ("pairs", "items", "selected", "champions", "models", "registry"):
        if isinstance(obj.get(key), list):
            arr = obj[key]
            break
    if arr is None and isinstance(obj, dict):
        # mapping form: {'PAIR': {...}}
        tmp = []
        for k, v in obj.items():
            if k in ("pairs", "items", "selected", "champions", "models", "registry"):
                continue
            if isinstance(v, dict):
                tmp.append({"pair": k, **v})
        arr = tmp

    if not isinstance(arr, list):
        raise RuntimeError(f"Unsupported registry format: {registry_path}")

    out: List[Tuple[str, Optional[str], dict]] = []
    for it in arr:
        if not isinstance(it, dict):
            continue
        pair = it.get("pair") or it.get("name") or it.get("key")
        if not pair:
            continue
        mpath = it.get("model_path")
        if not mpath:
            fp = _find_model_fallback(models_dir, pair)
            mpath = str(fp) if fp else None
        met = it.get("metrics") or {}
        out.append((str(pair), mpath, met))

    # optional: sort by sharpe desc and cut top_k
    out.sort(key=lambda x: float(x[2].get("sharpe", 0.0)), reverse=True)
    if top_k is not None and top_k > 0:
        out = out[:top_k]
    return out


def main():
    ap = argparse.ArgumentParser(description="Register models to MLflow Model Registry from registry.json")
    ap.add_argument("--registry", type=Path, required=True, help="Path to data/models/registry.json")
    ap.add_argument("--models-dir", type=Path, default=Path("data/models/pairs"), help="Where to look for model files if registry has none")
    ap.add_argument("--experiment", default="mntrading", help="MLflow experiment name")
    ap.add_argument("--prefix", default="mntrading", help="Registered model name prefix")
    ap.add_argument("--stage", default="Staging", choices=["None", "Staging", "Production", "Archived"], help="Stage to transition after registration")
    ap.add_argument("--top-k", type=int, default=0, help="Limit pairs to first K by Sharpe (0 = all)")
    args = ap.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    _ensure_experiment(tracking_uri, args.experiment)

    pairs = _collect_pairs(args.registry, args.models_dir, args.top_k if args.top_k > 0 else None)
    if not pairs:
        print("[register] no pairs found to register")
        return

    client = MlflowClient(tracking_uri=tracking_uri)
    registered_count = 0

    for pair, model_path, metrics in pairs:
        key = _canon_key(pair)
        reg_name = f"{args.prefix}_{key}"

        with mlflow.start_run(run_name=f"register_{key}") as run:
            # Log basic params/metrics for traceability
            mlflow.log_param("pair", pair)
            for mk, mv in (metrics or {}).items():
                try:
                    mlflow.log_metric(mk, float(mv))
                except Exception:
                    pass

            if model_path is None:
                # Still create a tiny placeholder model artifact to register
                placeholder = FileArtifactModel()
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=placeholder,
                    artifacts={},  # no original file available
                    pip_requirements=["mlflow"],
                )
            else:
                placeholder = FileArtifactModel()
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=placeholder,
                    artifacts={"model_file": str(model_path)},
                    pip_requirements=["mlflow"],
                )

            model_uri = f"runs:/{run.info.run_id}/model"

        # Ensure registered model exists and create new version
        try:
            try:
                client.get_registered_model(reg_name)
            except MlflowException:
                client.create_registered_model(reg_name)

            mv = client.create_model_version(name=reg_name, source=model_uri, run_id=run.info.run_id)
            print(f"[register] created {reg_name} v{mv.version}")

            # Transition stage if requested
            if args.stage != "None":
                client.transition_model_version_stage(
                    name=reg_name,
                    version=mv.version,
                    stage=args.stage,
                    archive_existing_versions=False,
                )
                print(f"[register] {reg_name} v{mv.version} -> {args.stage}")

            registered_count += 1
        except Exception as e:
            print(f"[register] failed for {pair}: {e}")

    print(f"[register] done, registered={registered_count}, pairs_in_input={len(pairs)}")


if __name__ == "__main__":
    main()
