# -*- coding: utf-8 -*-
import os
import json
import time
from typing import Any, Dict, Optional

import requests
import pandas as pd
import streamlit as st

# --- optional autorefresh (fallback if package is missing)
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:  # pragma: no cover
    def st_autorefresh(*args, **kwargs):
        return None

# -------- Config --------
# локально по умолчанию 127.0.0.1:8000; в Docker UI выставляем API_BASE_URL=http://api:8000
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="mntrading dashboard",
    page_icon="📈",
    layout="wide",
)

# -------- Helpers --------
def api_get(path: str, timeout: int = 25) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {"raw": r.text}
    except Exception as e:
        return {"error": f"GET {path}: {e}"}


def api_post(path: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 25) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    try:
        r = requests.post(url, json=payload or {}, timeout=timeout)
        r.raise_for_status()
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
    except Exception as e:
        return {"error": f"POST {path}: {e}"}


def flatten_production_map(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    pairs = (data or {}).get("pairs", {})
    for pair, info in pairs.items():
        rows.append({
            "pair": pair,
            "model_name": info.get("model_name"),
            "run_id": info.get("run_id"),
            "model_version": info.get("model_version"),
            "roc_auc(oof)": (info.get("oof_metrics") or {}).get("roc_auc"),
            "roc_auc_mean(cv)": (info.get("cv_val_means") or {}).get("roc_auc_mean"),
            "val_sharpe_mean": (info.get("cv_val_means") or {}).get("val_sharpe_mean"),
            "features": ",".join(info.get("features", [])),
        })
    return pd.DataFrame(rows)


def code_logs_block(logs_text: str, max_chars: int = 8000) -> None:
    if not logs_text:
        st.info("No logs yet.")
        return
    txt = logs_text[-max_chars:] if len(logs_text) > max_chars else logs_text
    # avoid non-ASCII arrows in some terminals
    txt = txt.replace("\u2192", "->")
    st.code(txt, language="log")


# -------- Session state --------
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = False
if "last_run_payload" not in st.session_state:
    st.session_state.last_run_payload = None
if "last_bootstrap_ts" not in st.session_state:
    st.session_state.last_bootstrap_ts = None

# ------------- Header -------------
st.title("mntrading — control panel")

with st.container():
    cols = st.columns([1, 1, 2])
    with cols[0]:
        health = api_get("/health")
        ok = "status" in health and health["status"] == "ok"
        st.metric("API health", "ok" if ok else "error")
        st.caption(f"API: {API_BASE_URL}")

    with cols[1]:
        # persistent checkbox
        auto = st.checkbox("Auto refresh (10s)", value=st.session_state.auto_refresh, key="auto_refresh")
        if auto:
            st_autorefresh(interval=10_000, key="auto_refresh_tick")

    with cols[2]:
        task = api_get("/tasks/last")
        task_status = task.get("status", "idle")
        started = task.get("started", None)
        finished = task.get("finished", None)
        st.write(f"**Last task:** {task.get('name','–')} | **status:** `{task_status}`")
        if started:
            st.caption(f"started: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(started)))}")
        if finished:
            st.caption(f"finished: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(float(finished)))}")

st.divider()

# ------------- Tabs -------------
tab_run, tab_artifacts, tab_logs = st.tabs(["▶ Run", "📦 Artifacts", "📜 Logs"])

# ========== RUN TAB ==========
with tab_run:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Short cycle (inference → aggregate)")
        with st.form("short_cycle_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                timeframe = st.selectbox("Timeframe", ["5m", "15m"], index=0)
                limit = st.number_input("Limit bars per symbol", min_value=100, max_value=5000, value=1000, step=50)
            with c2:
                proba_threshold = st.slider("Proba threshold", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
                top_k = st.number_input("Top-K pairs to trade", min_value=1, max_value=50, value=10, step=1)
            with c3:
                equity = st.number_input("Equity, $", min_value=1000.0, max_value=1_000_000.0, value=10_000.0, step=100.0)
                leverage = st.number_input("Leverage", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
            lookback_bars = st.number_input("Report lookback bars", min_value=100, max_value=10000, value=2000, step=50)

            submitted = st.form_submit_button("Run short cycle", use_container_width=True)
            if submitted:
                payload = dict(
                    timeframe=timeframe,
                    limit=int(limit),
                    proba_threshold=float(proba_threshold),
                    top_k=int(top_k),
                    equity=float(equity),
                    leverage=float(leverage),
                    lookback_bars=int(lookback_bars),
                )
                st.session_state.last_run_payload = payload
                resp = api_post("/run/short_cycle", payload)
                if "error" in resp:
                    st.error(resp["error"])
                else:
                    st.success("Started short cycle.")
                    st.json(resp)

        if st.session_state.last_run_payload:
            with st.expander("Last short cycle payload"):
                st.json(st.session_state.last_run_payload)

    with right:
        st.subheader("Bootstrap (quick)")
        st.caption("End-to-end: screen 1h → ingest 5m → features → dataset → train → backtest → select")

        if st.button("Start bootstrap quick", type="primary", use_container_width=True):
            st.session_state.last_bootstrap_ts = time.time()
            resp = api_post("/run/bootstrap_quick")
            if "error" in resp:
                st.error(resp["error"])
            else:
                st.success("Bootstrap started.")
                st.json(resp)

        if st.session_state.last_bootstrap_ts:
            st.caption(
                "Last bootstrap trigger at: "
                + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.session_state.last_bootstrap_ts))
            )

# ========== ARTIFACTS TAB ==========
with tab_artifacts:
    st.subheader("Artifacts overview")

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Production map**")
        pm = api_get("/artifacts/production_map")
        if "error" in pm:
            st.error(pm["error"])
        else:
            df_pm = flatten_production_map(pm)
            if not df_pm.empty:
                st.dataframe(df_pm, use_container_width=True, height=300)
            else:
                st.info("Empty production map.")

    with a2:
        st.markdown("**Registry**")
        reg = api_get("/artifacts/registry")
        if "error" in reg:
            st.error(reg["error"])
        else:
            st.json(reg)

    st.markdown("**Backtest summary**")
    bs = api_get("/artifacts/backtest_summary")
    if "error" in bs:
        st.error(bs["error"])
    else:
        st.json(bs)

# ========== LOGS TAB ==========
with tab_logs:
    st.subheader("Last task logs")
    logs = api_get("/logs/last_api_run")
    if "error" in logs:
        st.error(logs["error"])
    else:
        code_logs_block(logs.get("logs", ""))

    st.caption("Logs path (in API container): `/app/data/portfolio/_last_api_run.log`")
