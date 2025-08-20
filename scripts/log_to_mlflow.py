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
    # keep last N by mtime
    out.sort(key=lambda p: p.stat().st_mtime)
    return out[-max_files:]


def main():
    ap = argparse.ArgumentParser(description="Log training artifacts to MLflow.")
    ap.add_argument("--experiment", required=False, default="mntrading", help="MLflow experiment name")
    ap.add_argument("--train-report", type=Path, required=False, help="data/models/_train_report.json")
    ap.add_argument("--backtest-summary", type=Path, required=False, help="data/backtest_results/_summary.json")
    ap.add_argument("--models-dir", type=Path, required=False, default=Path("data/models/pairs"))
    ap.add_argument("--artifacts", type=Path, required=False, action="append", help="Extra artifact files to log")
    ap.add_argument("--registry", type=Path, required=False, help="data/models/registry.json")
    ap.add_argument("--prod-map", type=Path, required=False, help="data/models/production_map.json")
    args = ap.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name=f"mntrading_{os.environ.get('HOSTNAME','local')}"):
        # Log params/metrics from train report if present
        tr = _read_json(args.train_report) if args.train_report else None
        if tr:
            # params
            params = tr.get("params") or tr.get("config") or {}
            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass
            # metrics
            metrics = tr.get("metrics") or {}
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass
            mlflow.log_artifact(str(args.train_report))

        # Backtest summary as artifact + basic aggregate metrics
        bs = _read_json(args.backtest_summary) if args.backtest_summary else None
        if bs:
            mlflow.log_artifact(str(args.backtest_summary))
            pairs = bs.get("pairs")
            if isinstance(pairs, dict):
                # average sharpe/maxdd
                sh = [float((d.get("metrics") or {}).get("sharpe", 0.0)) for d in pairs.values() if isinstance(d, dict)]
                dd = [float((d.get("metrics") or {}).get("maxdd", 0.0)) for d in pairs.values() if isinstance(d, dict)]
                if sh:
                    mlflow.log_metric("backtest_sharpe_mean", sum(sh)/len(sh))
                if dd:
                    mlflow.log_metric("backtest_maxdd_mean", sum(dd)/len(dd))

        # Registry / production map
        if args.registry and args.registry.exists():
            mlflow.log_artifact(str(args.registry))
        if args.prod_map and args.prod_map.exists():
            mlflow.log_artifact(str(args.prod_map))

        # Model files (limited)
        for p in _glob_models(args.models_dir, max_files=20):
            mlflow.log_artifact(str(p))

        # Extra artifacts
        if args.artifacts:
            for art in args.artifacts:
                if art.exists():
                    mlflow.log_artifact(str(art))

        print(f"[mlflow] run logged to {tracking_uri}")


if __name__ == "__main__":
    main()
