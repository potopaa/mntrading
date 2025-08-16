#!/usr/bin/env python3
"""
Copy the latest MLflow model artifacts into ml_model/ directory.
"""
import os
import shutil

last_run_file = os.path.join("data", "raw", "last_run_id.txt")
if not os.path.exists(last_run_file):
    raise FileNotFoundError(f"{last_run_file} not found. Run training first.")

run_id = open(last_run_file).read().strip()
src = os.path.join("mlruns", "0", run_id, "artifacts", "model")
dst = "ml_model"

if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"[update_model] Copied model run {run_id} → {dst}")
