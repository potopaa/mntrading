# -*- coding: utf-8 -*-
"""
Airflow DAGs for mntrading:
- mntrading_bootstrap_daily: once a day, run full bootstrap_quick via API and wait until finished
- mntrading_short_cycle_15m: every 15m, run short_cycle via API (ingest -> features -> inference -> aggregate -> report)
"""
from __future__ import annotations

import os
import time
import requests
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

API_BASE = os.environ.get("API_BASE_URL", "http://api:8000")


def _post_json(url: str, payload=None):
    r = requests.post(url, json=payload or {}, timeout=60)
    r.raise_for_status()
    return r.json()


def _get_json(url: str):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def trigger_bootstrap_quick(**context):
    return _post_json(f"{API_BASE}/run/bootstrap_quick")


def trigger_short_cycle(**context):
    # можно параметризовать по контексту, если нужно
    payload = {
        "timeframe": "5m",
        "limit": 1000,
        "proba_threshold": 0.55,
        "top_k": 10,
        "equity": 10000.0,
        "leverage": 1.0,
        "lookback_bars": 2000,
    }
    return _post_json(f"{API_BASE}/run/short_cycle", payload)


def wait_until_finished(max_minutes: int = 120, **context):
    """
    Poll /tasks/last until status in {finished, error} or timeout.
    """
    deadline = time.time() + max_minutes * 60
    last_status = ""
    while time.time() < deadline:
        try:
            js = _get_json(f"{API_BASE}/tasks/last")
            status = js.get("status", "")
            if status != last_status:
                print("status:", status)
                last_status = status
            if status in ("finished", "error"):
                # лог в stdout
                print(js.get("logs", "")[-6000:])
                if status == "error":
                    raise RuntimeError("Pipeline finished with ERROR")
                return
        except Exception as e:
            print("Polling error:", e)
        time.sleep(10)
    raise TimeoutError("Pipeline not finished in time")


# ===================== mntrading_bootstrap_daily =====================
with DAG(
    dag_id="mntrading_bootstrap_daily",
    description="Daily bootstrap: screen→ingest→features→dataset→train→backtest→select",
    start_date=datetime(2025, 8, 1),
    schedule_interval="30 0 * * *",  # 00:30 UTC daily
    catchup=False,
    default_args={"retries": 0, "owner": "airflow"},
) as dag_bootstrap:
    t_trigger = PythonOperator(
        task_id="trigger_bootstrap_quick",
        python_callable=trigger_bootstrap_quick,
    )
    t_wait = PythonOperator(
        task_id="wait_until_finished",
        python_callable=wait_until_finished,
        op_kwargs={"max_minutes": 240},
    )
    t_trigger >> t_wait


# ===================== mntrading_short_cycle_15m =====================
with DAG(
    dag_id="mntrading_short_cycle_15m",
    description="Intraday short cycle: ingest→features→inference→aggregate→report",
    start_date=datetime(2025, 8, 1),
    schedule_interval="*/15 * * * *",  # every 15 minutes
    catchup=False,
    default_args={"retries": 0, "owner": "airflow"},
) as dag_short:
    t_trigger = PythonOperator(
        task_id="trigger_short_cycle",
        python_callable=trigger_short_cycle,
    )
    t_wait = PythonOperator(
        task_id="wait_until_finished",
        python_callable=wait_until_finished,
        op_kwargs={"max_minutes": 30},
    )
    t_trigger >> t_wait
