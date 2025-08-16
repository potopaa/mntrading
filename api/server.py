# -*- coding: utf-8 -*-
"""
Лёгкий API-сервер без тяжёлых импортов на старте.
- /health
- /pairs/manifest
- /artifacts/{production_map|registry|backtest_summary}
- /logs/last_api_run
- /tasks/last
- /run/short_cycle   (POST)
- /run/short_cycle_get (GET)
- /run/bootstrap_quick (POST)

Пайплайны исполняются отдельным subprocess, логи — в data/portfolio/_last_api_run.log,
статус последней задачи — в памяти и в data/portfolio/_last_task.json.
"""

import os
import json
import time
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Пути
APP_DIR = Path("/app")
DATA_DIR = APP_DIR / "data"
PORTFOLIO_DIR = DATA_DIR / "portfolio"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
BACKTEST_DIR = DATA_DIR / "backtest_results"
LOG_FILE = PORTFOLIO_DIR / "_last_api_run.log"
TASK_FILE = PORTFOLIO_DIR / "_last_task.json"

PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

# --- Глобальное состояние последней задачи
_last_task: Dict[str, Any] = {
    "id": "",
    "name": "",
    "status": "idle",  # idle|running|finished|error
    "started": None,
    "finished": None,
    "logs_path": str(LOG_FILE),
}

# --- FastAPI app
app = FastAPI(title="mntrading API", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _write_log(line: str) -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8", errors="ignore") as f:
        f.write(line.rstrip() + "\n")


def _set_task(name: str, status: str, task_id: Optional[str] = None) -> None:
    global _last_task
    now = time.time()
    if status == "running":
        _last_task = {
            "id": task_id or f"{int(now*1e6):x}",
            "name": name,
            "status": "running",
            "started": now,
            "finished": None,
            "logs_path": str(LOG_FILE),
        }
        LOG_FILE.write_text("", encoding="utf-8")  # reset
        _write_log(f"[task] {name} started")
    elif status in ("finished", "error"):
        _last_task["status"] = status
        _last_task["finished"] = now
        _write_log("[task] finished")
    TASK_FILE.write_text(json.dumps(_last_task), encoding="utf-8")


def _run_cmd(cmd: List[str]) -> int:
    """Запускает команду и пишет stdout/stderr в лог."""
    _write_log(f"[cmd] {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(APP_DIR),
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _write_log(line.rstrip())
    proc.wait()
    return int(proc.returncode or 0)


def _run_short_cycle(params: Dict[str, Any]) -> int:
    """Короткий цикл: inference -> aggregate."""
    tf = str(params.get("timeframe", "5m"))
    limit = int(params.get("limit", 1000))
    proba_th = float(params.get("proba_threshold", 0.55))
    top_k = int(params.get("top_k", 10))
    equity = float(params.get("equity", 10000.0))
    leverage = float(params.get("leverage", 1.0))
    lookback_bars = int(params.get("lookback_bars", 2000))

    # 1) inference
    cmd_inf = [
        "python",
        "inference.py",
        "--registry",
        str(MODELS_DIR / "registry.json"),
        "--pairs-manifest",
        str(FEATURES_DIR / "pairs" / "_manifest.json"),
        "--timeframe",
        tf,
        "--limit",
        str(limit),
        "--proba-threshold",
        str(proba_th),
        "--update",
        "--n-last",
        "1",
        "--out",
        str(DATA_DIR / "signals"),
    ]
    rc = _run_cmd(cmd_inf)
    if rc != 0:
        return rc

    # 2) aggregate
    cmd_agg = [
        "python",
        "portfolio/aggregate_signals.py",
        "--signals-dir",
        str(DATA_DIR / "signals"),
        "--pairs-manifest",
        str(FEATURES_DIR / "pairs" / "_manifest.json"),
        "--min-proba",
        str(proba_th),
        "--top-k",
        str(top_k),
        "--scheme",
        "equal_weight",
        "--equity",
        str(equity),
        "--leverage",
        str(leverage),
        "--lookback-bars",
        str(lookback_bars),
    ]
    rc = _run_cmd(cmd_agg)
    return rc


def _run_bootstrap_quick() -> int:
    """Быстрый bootstrap: screen(1h)->ingest(5m)->features->dataset->train->backtest->select."""
    # screen (1h)
    cmd_screen = [
        "python",
        "screen_pairs.py",
        "--universe",
        "BTC,ETH,SOL,BNB,XRP,ADA,MATIC,TRX,LTC,DOT",
        "--quote",
        "USDT",
        "--source",
        "ccxt",
        "--exchange",
        "binance",
        "--min-samples",
        "200",
        "--corr-threshold",
        "0.3",
        "--alpha",
        "0.25",
        "--page-size",
        "1000",
        "--top-k",
        "50",
    ]
    rc = _run_cmd(cmd_screen)
    if rc != 0:
        return rc

    # последний manifest
    pairs_dir = DATA_DIR / "pairs"
    last_pairs = ""
    if pairs_dir.exists():
        files = sorted(pairs_dir.glob("screened_pairs_*.json"))
        if files:
            last_pairs = str(files[-1])

    # ingest 5m
    cmd_ingest = [
        "python",
        "main.py",
        "--mode",
        "ingest",
        "--symbols",
        last_pairs or "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,MATIC/USDT,TRX/USDT,LTC/USDT,DOT/USDT",
        "--timeframe",
        "5m",
        "--limit",
        "1000",
    ]
    rc = _run_cmd(cmd_ingest)
    if rc != 0:
        return rc

    # features (pairs)
    cmd_feat = [
        "python",
        "main.py",
        "--mode",
        "features",
        "--symbols",
        last_pairs or "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,ADA/USDT,MATIC/USDT,TRX/USDT,LTC/USDT,DOT/USDT",
        "--beta-window",
        "300",
        "--z-window",
        "300",
    ]
    rc = _run_cmd(cmd_feat)
    if rc != 0:
        return rc

    # dataset
    cmd_ds = [
        "python",
        "main.py",
        "--mode",
        "dataset",
        "--pairs-manifest",
        str(FEATURES_DIR / "pairs" / "_manifest.json"),
        "--label-type",
        "z_threshold",
        "--zscore-threshold",
        "1.5",
        "--lag-features",
        "1",
    ]
    rc = _run_cmd(cmd_ds)
    if rc != 0:
        return rc

    # train
    cmd_train = [
        "python",
        "main.py",
        "--mode",
        "train",
        "--use-dataset",
        "--n-splits",
        "3",
        "--gap",
        "5",
        "--max-train-size",
        "2000",
        "--early-stopping-rounds",
        "50",
        "--proba-threshold",
        "0.55",
    ]
    rc = _run_cmd(cmd_train)
    if rc != 0:
        return rc

    # backtest
    cmd_bt = [
        "python",
        "main.py",
        "--mode",
        "backtest",
        "--signals-from",
        "oof",
    ]
    rc = _run_cmd(cmd_bt)
    if rc != 0:
        return rc

    # select
    cmd_sel = [
        "python",
        "main.py",
        "--mode",
        "select",
    ]
    rc = _run_cmd(cmd_sel)
    return rc


def _start_thread(name: str, target, kwargs: Optional[Dict[str, Any]] = None) -> None:
    _set_task(name, "running")
    def _runner():
        try:
            rc = target(**(kwargs or {}))
            if rc == 0:
                _set_task(name, "finished")
            else:
                _write_log(f"[error] return code {rc}")
                _set_task(name, "error")
        except Exception as e:
            _write_log(f"[exception] {type(e).__name__}: {e}")
            _set_task(name, "error")
    t = threading.Thread(target=_runner, daemon=True)
    t.start()


# --------- API ---------

@app.get("/")
def root():
    return {"ok": True, "service": "mntrading-api"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/pairs/manifest")
def pairs_manifest():
    p = FEATURES_DIR / "pairs" / "_manifest.json"
    if not p.exists():
        return {"manifest": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e), "manifest": []}


@app.get("/artifacts/production_map")
def get_production_map():
    p = MODELS_DIR / "production_map.json"
    if not p.exists():
        return {"pairs": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e), "pairs": {}}


@app.get("/artifacts/registry")
def get_registry():
    p = MODELS_DIR / "registry.json"
    if not p.exists():
        return {"pairs": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e), "pairs": {}}


@app.get("/artifacts/backtest_summary")
def get_backtest_summary():
    p = BACKTEST_DIR / "_summary.json"
    if not p.exists():
        return {"pairs": {}}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e), "pairs": {}}


@app.get("/logs/last_api_run")
def last_api_run():
    if not LOG_FILE.exists():
        return {"logs": ""}
    try:
        return {"logs": LOG_FILE.read_text(encoding="utf-8", errors="ignore")}
    except Exception as e:
        return {"logs": f"(failed to read logs: {e})"}


@app.get("/tasks/last")
def tasks_last():
    return _last_task


class IngestInferParams(BaseModel):
    timeframe: str = "5m"
    limit: int = 1000
    proba_threshold: float = 0.55
    top_k: int = 10
    equity: float = 10000.0
    leverage: float = 1.0
    lookback_bars: int = 2000


@app.post("/run/short_cycle")
def run_short_cycle(params: Optional[IngestInferParams] = None):
    _start_thread("short_cycle", lambda **_: _run_short_cycle((params or IngestInferParams()).model_dump()))
    return {"status": "started", "task": _last_task}


@app.get("/run/short_cycle_get")
def run_short_cycle_get(
    timeframe: str = Query("5m"),
    limit: int = Query(1000),
    proba_threshold: float = Query(0.55),
    top_k: int = Query(10),
    equity: float = Query(10000.0),
    leverage: float = Query(1.0),
    lookback_bars: int = Query(2000),
):
    payload = dict(
        timeframe=timeframe,
        limit=limit,
        proba_threshold=proba_threshold,
        top_k=top_k,
        equity=equity,
        leverage=leverage,
        lookback_bars=lookback_bars,
    )
    _start_thread("short_cycle", lambda **_: _run_short_cycle(payload))
    return {"status": "started", "task": _last_task}


@app.post("/run/bootstrap_quick")
def run_bootstrap_quick():
    _start_thread("bootstrap_quick", lambda **_: _run_bootstrap_quick())
    return {"status": "started", "task": _last_task}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
