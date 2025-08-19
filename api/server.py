# api/server.py
# -*- coding: utf-8 -*-
"""
FastAPI service to expose health and simple pipeline triggers.
All comments in English.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
from typing import List

app = FastAPI(title="MNTrading API")

def run_cmd(cmd: List[str]) -> int:
    return subprocess.run(cmd).returncode

class RunResponse(BaseModel):
    status: str
    step: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run/short_cycle", response_model=RunResponse)
def run_short_cycle():
    steps = [
        ["python","/app/main.py","--mode","screen"],
        ["python","/app/main.py","--mode","features","--symbols","/app/data/pairs/screened_pairs_*.json"],
        ["python","/app/main.py","--mode","dataset","--label-type","z_threshold","--zscore-threshold","1.5","--lag-features","10","--horizon","3"],
        ["python","/app/main.py","--mode","train","--use-dataset","--n-splits","5","--gap","24","--proba-threshold","0.55"],
        ["python","/app/main.py","--mode","backtest","--signals-from","auto","--proba-threshold","0.55","--fee-rate","0.0005"],
        ["python","/app/main.py","--mode","select"],
    ]
    last = "none"
    for step in steps:
        last = " ".join(step)
        rc = run_cmd(step)
        if rc != 0:
            return RunResponse(status="failed", step=last)
    return RunResponse(status="ok", step=last)

@app.post("/run/full_cycle", response_model=RunResponse)
def run_full_cycle():
    steps = [
        ["python","/app/main.py","--mode","screen"],
        ["python","/app/main.py","--mode","features","--symbols","/app/data/pairs/screened_pairs_*.json"],
        ["python","/app/main.py","--mode","dataset","--label-type","z_threshold","--zscore-threshold","1.5","--lag-features","10","--horizon","3"],
        ["python","/app/main.py","--mode","train","--use-dataset","--n-splits","5","--gap","24","--proba-threshold","0.55"],
        ["python","/app/main.py","--mode","backtest","--signals-from","auto","--proba-threshold","0.55","--fee-rate","0.0005"],
        ["python","/app/main.py","--mode","select"],
        ["python","/app/main.py","--mode","promote"],
        ["python","/app/main.py","--mode","inference","--registry-in","/app/data/models/production_map.json","--update"],
        ["python","/app/main.py","--mode","aggregate","--top-k","10","--proba-threshold","0.55"],
        ["python","/app/main.py","--mode","report"],
    ]
    last = "none"
    for step in steps:
        last = " ".join(step)
        rc = run_cmd(step)
        if rc != 0:
            return RunResponse(status="failed", step=last)
    return RunResponse(status="ok", step=last)
