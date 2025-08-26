#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/log_to_mlflow.py
# All comments are in English by request.

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType


def _read_json(p: Path) -> Optional[dict]:
    if not p or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _glob_models(models_dir: Path, max_files: int = 10) -> List[Path]:
    out: List[Path] = []
    if not models_dir.exists():
        return out
    for ext in (".pkl", ".joblib", ".json"):
        out.extend(sorted(models_dir.rglob(f"*{ext}")))
    out.sort(key=lambda p: p.stat().st_mtime)
    return out[-max_files:]


def _ensure_experiment(tracking_uri: str, name: str) -> str:
    """
    Ensure experiment exists and is active.
    If deleted, try to restore; if restoration fails, create '<name>_v2'.
    Returns the final experiment name to use.
    """
    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(name)
        return name
    except MlflowException as e:
        msg = str(e).lower()
        if "deleted experiment" in msg or "deleted" in msg:
            client = MlflowClient(tracking_uri=tracking_uri)
            for exp in client.search_experiments(view_type=ViewType.ALL):
                if exp.name == name and exp.lifecycle_stage == "deleted":
                    client.restore_experiment(exp.experiment_id)
                    # try again
                    mlflow.set_experiment(name)
                    return name
        # fallback to a new name
        alt = f"{name}_v2"
        mlflow.set_experiment(alt)
        return alt


def main():
    ap = argparse.ArgumentParser(description="Log training/backtest artifacts to MLflow.")
    ap.add_argument("--experiment", required=False, default="mntrading", help="MLflow experiment name")
    ap.add_argument("--train-report", type=Path, required=False, help="data/models/_train_report.json")
    ap.add_argument("--backtest-summary", type=Path, required=False, help="data/backtest_results/_summary.json")
    ap.add_argument("--models-dir", type=Path, required=False, default=Path("data/models/pairs"))
    ap.add_argument("--artifacts", type=Path, required=False, action="append", help="Extra artifact files to log")
    ap.add_argument("--registry", type=Path, required=False, help="data/models/registry.json")
    ap.add_argument("--prod-map", type=Path, required=False, help="data/models/production_map.json")
    args = ap.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    exp_name = _ensure_experiment(tracking_uri, args.experiment)

    with mlflow.start_run(run_name=f"mntrading_{os.environ.get('HOSTNAME','local')}"):
        # train report: params, metrics, artifact
        tr = _read_json(args.train_report) if args.train_report else None
        if tr:
            params = tr.get("params") or tr.get("config") or {}
            for k, v in params.items():
                try: mlflow.log_param(k, v)
                except Exception: pass
            metrics = tr.get("metrics") or {}
            for k, v in metrics.items():
                try: mlflow.log_metric(k, float(v))
                except Exception: pass
            mlflow.log_artifact(str(args.train_report))

        # backtest summary: artifact + aggregate metrics
        bs = _read_json(args.backtest_summary) if args.backtest_summary else None
        if bs:
            mlflow.log_artifact(str(args.backtest_summary))
            pairs = bs.get("pairs")
            if isinstance(pairs, dict):
                sh = [float((d.get("metrics") or {}).get("sharpe", 0.0)) for d in pairs.values() if isinstance(d, dict)]
                dd = [float((d.get("metrics") or {}).get("maxdd", 0.0)) for d in pairs.values() if isinstance(d, dict)]
                if sh: mlflow.log_metric("backtest_sharpe_mean", sum(sh)/len(sh))
                if dd: mlflow.log_metric("backtest_maxdd_mean", sum(dd)/len(dd))

        # registry / prod-map
        if args.registry and args.registry.exists(): mlflow.log_artifact(str(args.registry))
        if args.prod_map and args.prod_map.exists(): mlflow.log_artifact(str(args.prod_map))

        # model artifacts (limited)
        for p in _glob_models(args.models_dir, max_files=20):
            mlflow.log_artifact(str(p))

        # extra artifacts
        if args.artifacts:
            for art in args.artifacts:
                if art.exists():
                    mlflow.log_artifact(str(art))

        print(f"[mlflow] run logged to {tracking_uri} (experiment={exp_name})")


if __name__ == "__main__":
    main()
