#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/resolve_router_uri.py
# Prints the best model URI for serving: prefer models:/mntrading_router/Production,
# otherwise fallback to runs:/<last build_router run>/model.

import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

ROUTER_NAME = os.environ.get("ROUTER_NAME", "mntrading_router")
EXPERIMENT   = os.environ.get("ROUTER_EXPERIMENT", "mntrading")
STAGE        = os.environ.get("ROUTER_STAGE", "Production")

def main():
    client = MlflowClient()  # uses MLFLOW_TRACKING_URI
    # 1) Try registered model@stage
    try:
        vers = client.get_latest_versions(ROUTER_NAME, [STAGE])
        if vers:
            print(f"models:/{ROUTER_NAME}/{STAGE}")
            return 0
    except Exception:
        pass

    # 2) Fallback: last build_router_* run in the experiment
    exps = [e for e in client.search_experiments() if e.name == EXPERIMENT]
    if not exps:
        print(f"[resolve] experiment not found: {EXPERIMENT}", file=sys.stderr)
        return 2
    exp_id = exps[0].experiment_id
    runs = client.search_runs(
        [exp_id],
        "tags.mlflow.runName LIKE 'build_router_%'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs:
        run = runs[0]
        print(f"runs:/{run.info.run_id}/model")
        return 0

    print("[resolve] no router registered and no build_router runs found", file=sys.stderr)
    return 3

if __name__ == "__main__":
    sys.exit(main())
