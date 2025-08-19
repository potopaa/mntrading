# -*- coding: utf-8 -*-
"""
Minimal FastAPI wrapper for MNTrading pipeline.

Endpoints:
- GET /health
- POST /run/short_cycle
- GET /run/short_cycle_now    (alias for short pipeline)
- GET /artifacts              (list some produced artifacts)

This service executes main.py steps in sequence via subprocess.
"""

from __future__ import annotations
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException

APP_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = APP_ROOT / "main.py"
DATA_DIR = APP_ROOT / "data"

app = FastAPI(title="MNTrading API", version="0.1.0")

def run_step(mode: str, extra: Optional[List[str]] = None) -> int:
    """Run a pipeline step via main.py and return exit code."""
    cmd = [sys.executable, str(MAIN_PY), "--mode", mode]
    if extra:
        cmd.extend(extra)
    env = os.environ.copy()
    proc = subprocess.run(cmd, cwd=str(APP_ROOT), env=env)
    return proc.returncode

def run_short_pipeline() -> Dict[str, int]:
    """Run the short pipeline steps and return per-step exit codes."""
    plan = [
        ("screen", []),
        ("ingest", []),
        ("features", []),
        ("dataset", ["--label-type", "z_threshold", "--z-th", "1.5", "--lag-features", "10", "--horizon", "3"]),
        ("train", []),
        ("backtest", []),
        ("select", []),
    ]
    result: Dict[str, int] = {}
    for mode, extra in plan:
        rc = run_step(mode, extra)
        result[mode] = rc
        if rc != 0:
            break
    return result

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run/short_cycle")
def post_short_cycle():
    if not MAIN_PY.exists():
        raise HTTPException(status_code=500, detail="main.py not found")
    result = run_short_pipeline()
    if any(rc != 0 for rc in result.values()):
        return {"status": "failed", "steps": result}
    return {"status": "ok", "steps": result}

@app.get("/run/short_cycle_now")
def get_short_cycle_now():
    """Alias GET endpoint matching README."""
    return post_short_cycle()

@app.get("/artifacts")
def list_artifacts():
    """List some produced artifacts under data/."""
    data = {}
    for sub in ["features", "datasets", "backtest_results", "models", "portfolio", "report"]:
        p = DATA_DIR / sub
        if p.exists():
            files = []
            for child in p.rglob("*"):
                if child.is_file():
                    try:
                        size = child.stat().st_size
                    except Exception:
                        size = None
                    files.append({"path": str(child.relative_to(DATA_DIR)), "size": size})
            data[sub] = files[:500]  # cap output
        else:
            data[sub] = []
    return {"data": data}
