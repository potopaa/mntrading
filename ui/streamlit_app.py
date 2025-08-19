# -*- coding: utf-8 -*-
"""
Streamlit UI for MNTrading pipeline:
- Short and Full pipeline buttons
- Step-by-step execution controls
- Artifacts preview (datasets/backtest/portfolio)
- Simple charts (backtest curves, equity)
- Links to MLflow UI

This app assumes it runs inside the Docker container with /app as project root.
"""

import os
import sys
import json
import glob
import time
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import streamlit as st

APP_ROOT = Path("/app").resolve()
DATA_DIR = APP_ROOT / "data"
MLFLOW_UI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").replace("mlflow", "localhost")

st.set_page_config(page_title="MNTrading Pipeline", layout="wide")

# -------------------------------
# Helpers
# -------------------------------
def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    """Run a shell command and stream output into Streamlit."""
    st.write(f"`$ {' '.join(cmd)}`")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(APP_ROOT),
        env=env or os.environ.copy(),
    )
    lines = []
    status = st.status("Running...", expanded=True)
    with status:
        for line in iter(proc.stdout.readline, ""):
            lines.append(line)
            st.write(line.rstrip("\n"))
    proc.wait()
    rc = proc.returncode
    if rc == 0:
        status.update(label="Done", state="complete")
    else:
        status.update(label=f"Failed (exit={rc})", state="error")
    return rc

def run_step(mode: str, extra: Optional[List[str]] = None) -> int:
    """Run one pipeline step via main.py --mode <mode>."""
    cmd = [sys.executable, str(APP_ROOT / "main.py"), "--mode", mode]
    if extra:
        cmd.extend(extra)
    return run_cmd(cmd)

def run_pipeline(steps: List[List[str]]) -> bool:
    """Run multiple steps: each element is ['mode', *extra]. Stop on first failure."""
    for spec in steps:
        mode, *extra = spec
        st.subheader(f"Step: {mode}")
        rc = run_step(mode, extra)
        if rc != 0:
            st.error(f"Step `{mode}` failed with exit code {rc}")
            return False
    return True

def read_any_parquet_or_csv(paths: List[Path]) -> Optional[pd.DataFrame]:
    """Read the first existing parquet/csv from given list of paths."""
    for p in paths:
        if p.suffix.lower() == ".parquet" and p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        if p.suffix.lower() == ".csv" and p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                pass
    return None

def list_files(pattern: str, limit: int = 20) -> List[Path]:
    """List files by glob pattern."""
    return [Path(p) for p in sorted(glob.glob(pattern))][-limit:]

def mlflow_link() -> str:
    """Return MLflow UI link."""
    if MLFLOW_UI.startswith("http"):
        return MLFLOW_UI
    return "http://localhost:5000"

# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.title("Controls")

with st.sidebar.expander("Pipeline steps", expanded=True):
    step_names = [
        "screen", " ingest", " features", " dataset", " train",
        " backtest", " select", " promote", " inference", " aggregate", " report"
    ]
    st.write(", ".join(step_names))

short_btn = st.sidebar.button("Run SHORT pipeline", use_container_width=True)
full_btn = st.sidebar.button("Run FULL pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f"[Open MLflow]({mlflow_link()})")

# -------------------------------
# Main layout with tabs
# -------------------------------
tab_overview, tab_run, tab_artifacts, tab_reports, tab_mlflow = st.tabs(
    ["Overview", "Run pipeline", "Artifacts", "Reports/Charts", "MLflow"]
)

with tab_overview:
    st.markdown("### MNTrading — Demo UI")
    st.write(
        "Use the sidebar buttons to run SHORT or FULL pipelines. "
        "You can also execute steps one-by-one from the **Run pipeline** tab."
    )
    st.info("Airflow is recommended for scheduled runs; this UI is for interactive demo.")

with tab_run:
    st.markdown("### Run steps individually")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("screen"):
            run_step("screen")
        if st.button("features"):
            run_step("features")
        if st.button("train"):
            run_step("train")
        if st.button("select"):
            run_step("select")
        if st.button("aggregate"):
            run_step("aggregate")

    with col2:
        if st.button("ingest"):
            run_step("ingest")
        if st.button("dataset"):
            # Default label params; adjust if needed
            extra = ["--label-type", "z_threshold", "--z-th", "1.5", "--lag-features", "10", "--horizon", "3"]
            run_step("dataset", extra)
        if st.button("backtest"):
            run_step("backtest")
        if st.button("promote"):
            run_step("promote")
        if st.button("report"):
            run_step("report")

    st.markdown("---")
    st.markdown("#### One-click")
    if short_btn:
        st.success("Starting SHORT pipeline…")
        short_flow = [
            ["screen"],
            ["ingest"],
            ["features"],
            ["dataset", "--label-type", "z_threshold", "--z-th", "1.5", "--lag-features", "10", "--horizon", "3"],
            ["train"],
            ["backtest"],
            ["select"],
        ]
        ok = run_pipeline(short_flow)
        if ok:
            st.success("SHORT pipeline completed.")

    if full_btn:
        st.success("Starting FULL pipeline…")
        full_flow = [
            ["screen"],
            ["ingest"],
            ["features"],
            ["dataset", "--label-type", "z_threshold", "--z-th", "1.5", "--lag-features", "10", "--horizon", "3"],
            ["train"],
            ["backtest"],
            ["select"],
            ["promote"],
            ["inference"],
            ["aggregate"],
            ["report"],
        ]
        ok = run_pipeline(full_flow)
        if ok:
            st.success("FULL pipeline completed.")

with tab_artifacts:
    st.markdown("### Artifacts preview")

    # Datasets manifest
    ds_manifest = DATA_DIR / "datasets" / "_manifest.json"
    if ds_manifest.exists():
        st.subheader("datasets/_manifest.json")
        st.code(ds_manifest.read_text()[:10000])
    else:
        st.warning("datasets/_manifest.json not found")

    # Backtest summary files (try parquet/csv)
    st.subheader("Backtest results")
    candidates = list_files(str(DATA_DIR / "backtest_results" / "**" / "summary.parquet")) \
              +  list_files(str(DATA_DIR / "backtest_results" / "**" / "summary.csv"))
    if candidates:
        df_bt = read_any_parquet_or_csv(candidates)
        if df_bt is not None and not df_bt.empty:
            st.dataframe(df_bt.tail(200), use_container_width=True)
        else:
            st.info("No readable summary found.")
    else:
        st.info("No backtest summaries found.")

    # Portfolio/equity
    st.subheader("Portfolio / equity curve")
    eq_candidates = [
        DATA_DIR / "portfolio" / "equity_curve.parquet",
        DATA_DIR / "portfolio" / "equity_curve.csv",
    ]
    df_eq = read_any_parquet_or_csv(eq_candidates)
    if df_eq is not None and not df_eq.empty:
        numeric_cols = df_eq.select_dtypes("number").columns.tolist()
        if numeric_cols:
            st.line_chart(df_eq[numeric_cols])
        st.dataframe(df_eq.tail(200), use_container_width=True)
    else:
        st.info("No equity curve found.")

with tab_reports:
    st.markdown("### Reports")
    rpt_files = list_files(str(DATA_DIR / "report" / "**" / "*.html")) \
             +  list_files(str(DATA_DIR / "report" / "**" / "*.md")) \
             +  list_files(str(DATA_DIR / "report" / "**" / "*.txt"))
    if rpt_files:
        for p in rpt_files:
            st.write(f"**{p.relative_to(DATA_DIR)}**")
            try:
                st.code(p.read_text()[:10000])
            except Exception:
                st.write("Binary or non-UTF8 file — preview skipped.")
    else:
        st.info("No report files found.")

with tab_mlflow:
    st.markdown("### MLflow")
    st.write("Open MLflow UI to inspect experiments and artifacts:")
    st.markdown(f"- {mlflow_link()}")
