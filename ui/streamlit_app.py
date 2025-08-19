# ui/streamlit_app.py
# -*- coding: utf-8 -*-
"""
Streamlit UI to run the pipeline with sidebar parameters:
- Run full or short pipeline with one click
- Configure step parameters in sidebar
- Show backtest summary and latest report
All comments are in English.
"""
import os
import json
import subprocess
from pathlib import Path

import streamlit as st
import pandas as pd

# Optional: lazy fetch from MinIO when local artifacts are missing
S3_ENABLED = any(os.getenv(k) for k in ("MLFLOW_S3_ENDPOINT_URL", "MINIO_BUCKET"))
SINK = None
if S3_ENABLED:
    try:
        from utils.minio_io import MinioSink  # requires boto3 in the image
        SINK = MinioSink.from_env(enabled=True)
    except Exception:
        SINK = None

def fetch_if_missing(local_path: Path, s3_key: str | None = None, s3_prefix: str | None = None, is_dir: bool = False):
    """Try to fetch artifact from MinIO if not present locally."""
    if local_path.exists():
        return
    if not SINK:
        return
    try:
        if is_dir and s3_prefix:
            SINK.download_dir(s3_prefix, local_path)
        elif s3_key:
            if SINK.exists(s3_key):
                local_path.parent.mkdir(parents=True, exist_ok=True)
                SINK.download_file(s3_key, local_path)
    except Exception:
        pass

APP_ROOT = Path("/app").resolve()
DATA_DIR = APP_ROOT / "data"
BT_SUMMARY = DATA_DIR / "backtest_results" / "_summary.json"
REPORT_MD = DATA_DIR / "portfolio" / "_latest_report.md"

def run_cmd(cmd: list[str]) -> str:
    """Run a command and capture combined output."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return out.strip()

st.set_page_config(page_title="MNTrading", layout="wide")
st.title("MNTrading — pipeline control")

with st.sidebar:
    st.header("Parameters")
    # Ingest 1h universe
    uni_symbols = st.text_area("Universe symbols (CSV)", "BTC/USDT,ETH/USDT,SOL/USDT")
    since_1h = st.text_input("since-utc for 1h", "2025-01-01T00:00:00Z")
    limit_1h = st.number_input("limit per page (1h)", min_value=100, max_value=20000, value=5000, step=100)

    st.markdown("---")
    # Screen -> auto ingest 5m
    since_5m = st.text_input("since-utc for 5m (after screen)", "2025-01-01T00:00:00Z")
    limit_5m = st.number_input("limit per page (5m)", min_value=100, max_value=20000, value=1000, step=100)

    st.markdown("---")
    # Features
    beta_window = st.number_input("beta window", min_value=50, max_value=5000, value=1000, step=50)
    z_window = st.number_input("z window", min_value=50, max_value=5000, value=300, step=10)

    st.markdown("---")
    # Dataset
    label_type = st.selectbox("label type", ["z_threshold", "revert_direction"])
    z_th = st.number_input("zscore threshold", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    lag_features = st.number_input("lag features", min_value=0, max_value=100, value=10, step=1)
    horizon = st.number_input("horizon", min_value=1, max_value=100, value=3, step=1)

    st.markdown("---")
    # Train & Backtest
    n_splits = st.number_input("n_splits", min_value=2, max_value=20, value=5, step=1)
    gap = st.number_input("gap", min_value=0, max_value=1000, value=24, step=1)
    proba_th = st.number_input("proba threshold", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
    signals_from = st.selectbox("signals from", ["auto","model","z"])
    fee_rate = st.number_input("fee rate", min_value=0.0, max_value=0.01, value=0.0005, step=0.0001)

    st.markdown("---")
    top_k = st.number_input("top-K portfolio", min_value=1, max_value=100, value=10, step=1)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Universe (1h)")
    if st.button("Ingest 1h universe"):
        cmd = ["python","/app/main.py","--mode","ingest","--symbols",uni_symbols,"--timeframe","1h","--since-utc",since_1h,"--limit",str(int(limit_1h))]
        st.code(" ".join(cmd), language="bash")
        st.text(run_cmd(cmd))

with col2:
    st.subheader("Screen (1h) → auto 5m")
    if st.button("Screen cointegration and ingest 5m"):
        cmd = ["python","/app/main.py","--mode","screen","--since-utc-5m",since_5m,"--limit-5m",str(int(limit_5m))]
        st.code(" ".join(cmd), language="bash")
        st.text(run_cmd(cmd))

with col3:
    st.subheader("Features on 5m")
    if st.button("Build features"):
        cmd = ["python","/app/main.py","--mode","features","--symbols","/app/data/pairs/screened_pairs_*.json","--beta-window",str(int(beta_window)),"--z-window",str(int(z_window))]
        st.code(" ".join(cmd), language="bash")
        st.text(run_cmd(cmd))

st.markdown("---")
col4, col5, col6 = st.columns(3)
with col4:
    st.subheader("Dataset")
    if st.button("Build dataset"):
        cmd = ["python","/app/main.py","--mode","dataset","--label-type",label_type,"--zscore-threshold",str(float(z_th)),"--lag-features",str(int(lag_features)),"--horizon",str(int(horizon))]
        st.code(" ".join(cmd), language="bash")
        st.text(run_cmd(cmd))

with col5:
    st.subheader("Train & Backtest")
    if st.button("Train models"):
        cmd = ["python","/app/main.py","--mode","train","--use-dataset","--n-splits",str(int(n_splits)),"--gap",str(int(gap)),"--proba-threshold",str(float(proba_th))]
        st.code(" ".join(cmd), language="bash")
        st.text(run_cmd(cmd))
    if st.button("Backtest"):
        cmd = ["python","/app/main.py","--mode","backtest","--signals-from",signals_from,"--proba-threshold",str(float(proba_th)),"--fee-rate",str(float(fee_rate))]
        st.code(" ".join(cmd), language="bash")
        st.text(run_cmd(cmd))

with col6:
    st.subheader("Select & Promote")
    if st.button("Select champions"):
        cmd = ["python","/app/main.py","--mode","select"]
        st.code(" ".join(cmd), language="bash")
        st.text(run_cmd(cmd))
    if st.button("Promote"):
        cmd = ["python","/app/main.py","--mode","promote"]
        st.code(" ".join(cmd), language="bash")
        st.text(run_cmd(cmd))

st.markdown("---")
st.subheader("Run pipeline shortcuts")
c1, c2 = st.columns(2)
with c1:
    if st.button("Run SHORT pipeline"):
        steps = [
            ["python","/app/main.py","--mode","screen"],
            ["python","/app/main.py","--mode","features","--symbols","/app/data/pairs/screened_pairs_*.json","--beta-window",str(int(beta_window)),"--z-window",str(int(z_window))],
            ["python","/app/main.py","--mode","dataset","--label-type",label_type,"--zscore-threshold",str(float(z_th)),"--lag-features",str(int(lag_features)),"--horizon",str(int(horizon))],
            ["python","/app/main.py","--mode","train","--use-dataset","--n-splits",str(int(n_splits)),"--gap",str(int(gap)),"--proba-threshold",str(float(proba_th))],
            ["python","/app/main.py","--mode","backtest","--signals-from",signals_from,"--proba-threshold",str(float(proba_th)),"--fee-rate",str(float(fee_rate))],
            ["python","/app/main.py","--mode","select"],
        ]
        for cmd in steps:
            st.code(" ".join(cmd), language="bash")
            st.text(run_cmd(cmd))

with c2:
    if st.button("Run FULL pipeline (except ingest 1h)"):
        steps = [
            ["python","/app/main.py","--mode","screen","--since-utc-5m",since_5m,"--limit-5m",str(int(limit_5m))],
            ["python","/app/main.py","--mode","features","--symbols","/app/data/pairs/screened_pairs_*.json","--beta-window",str(int(beta_window)),"--z-window",str(int(z_window))],
            ["python","/app/main.py","--mode","dataset","--label-type",label_type,"--zscore-threshold",str(float(z_th)),"--lag-features",str(int(lag_features)),"--horizon",str(int(horizon))],
            ["python","/app/main.py","--mode","train","--use-dataset","--n-splits",str(int(n_splits)),"--gap",str(int(gap)),"--proba-threshold",str(float(proba_th))],
            ["python","/app/main.py","--mode","backtest","--signals-from",signals_from,"--proba-threshold",str(float(proba_th)),"--fee-rate",str(float(fee_rate))],
            ["python","/app/main.py","--mode","select"],
            ["python","/app/main.py","--mode","promote"],
            ["python","/app/main.py","--mode","inference","--registry-in","/app/data/models/production_map.json","--update"],
            ["python","/app/main.py","--mode","aggregate","--top-k",str(int(top_k)),"--proba-threshold",str(float(proba_th))],
            ["python","/app/main.py","--mode","report"],
        ]
        for cmd in steps:
            st.code(" ".join(cmd), language="bash")
            st.text(run_cmd(cmd))

st.markdown("---")
st.subheader("Artifacts")

# Try to fetch missing artifacts before displaying
fetch_if_missing(BT_SUMMARY, s3_key="backtest/_summary.json")
fetch_if_missing(REPORT_MD, s3_key="portfolio/_latest_report.md")

colA, colB = st.columns(2)
with colA:
    st.write("Backtest summary")
    if BT_SUMMARY.exists():
        obj = json.loads(BT_SUMMARY.read_text(encoding="utf-8"))
        st.json(obj)
    else:
        st.info("No backtest summary yet.")

with colB:
    st.write("Latest report")
    if REPORT_MD.exists():
        st.markdown(REPORT_MD.read_text(encoding="utf-8"))
    else:
        st.info("No report yet.")

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
st.caption(f"MLflow: {mlflow_uri}")
