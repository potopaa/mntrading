
# ui/streamlit_app.py (EN-only)
import os
import json
import glob
import subprocess
from pathlib import Path
from typing import Tuple

import streamlit as st
from mlflow.tracking import MlflowClient
import mlflow

APP_ROOT = Path("/app")
DATA_DIR = APP_ROOT / "data"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "mntrading_model")
ROUTER_NAME = os.getenv("MODEL_NAME", "mntrading_router")
ROUTER_STAGE = os.getenv("MODEL_STAGE", "Production")

def run_cmd(cmd: str) -> Tuple[int, str, str]:
    """Run a shell command and return (code, stdout, stderr)."""
    st.code(cmd, language="bash")
    proc = subprocess.run(["bash","-lc", cmd], text=True, capture_output=True)
    if proc.stdout:
        st.text(proc.stdout)
    if proc.returncode != 0:
        st.error(proc.stderr or f"Exit code {proc.returncode}")
    return proc.returncode, proc.stdout, proc.stderr

def mlflow_ready() -> Tuple[bool, str, MlflowClient]:
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_REGISTRY_URI)
        _ = client.list_experiments()
        return True, "OK", client
    except Exception as e:
        return False, str(e), None  # type: ignore

def header():
    st.set_page_config(page_title="mntrading dashboard", layout="wide")
    st.title("mntrading • MLOps dashboard")
    ok, msg, _client = mlflow_ready()
    if ok:
        st.success(f"MLflow reachable at {MLFLOW_TRACKING_URI}")
    else:
        st.error(f"MLflow not reachable: {msg}")

def sidebar_params():
    st.sidebar.header("Quick parameters")
    exchange = st.sidebar.text_input("Exchange", value="binance")
    quote = st.sidebar.text_input("Quote", value="USDT")
    top = st.sidebar.number_input("Top N symbols", value=100, min_value=10, max_value=2000, step=10)
    since_1h = st.sidebar.text_input("Since (1h, ISO-8601)", value="2024-01-01T00:00:00Z")
    limit_1h = st.sidebar.number_input("Limit per request (1h)", value=1000, min_value=100, max_value=1000, step=100)
    max_candles = st.sidebar.number_input("Max candles per symbol", value=5000, min_value=1000, max_value=30000, step=1000)

    n_splits = st.sidebar.number_input("CV folds", value=5, min_value=2, max_value=10)
    gap = st.sidebar.number_input("CV gap (bars)", value=24, min_value=0, max_value=240)
    proba_thr = st.sidebar.slider("Probability threshold", min_value=0.5, max_value=0.9, value=0.55, step=0.01)

    return dict(exchange=exchange, quote=quote, top=top, since_1h=since_1h,
                limit_1h=limit_1h, max_candles=max_candles,
                n_splits=n_splits, gap=gap, proba_thr=proba_thr)

def section_screening(p):
    st.header("1) Screening (1h) and fetch 5m for screened pairs")
    if st.button("Run screening now"):
        code, *_ = run_cmd("python /app/main.py --mode screen")
        if code == 0:
            st.success("Screening completed.")

    pairs = sorted(DATA_DIR.glob("pairs/screened_pairs_*.json"))
    if pairs:
        latest = pairs[-1]
        st.success(f"Found {len(pairs)} screened list(s). Latest: {latest.name}")
        try:
            st.json(json.loads(latest.read_text(encoding='utf-8')))
        except Exception:
            st.info("Latest pairs JSON is large; showing file path only.")
            st.text(str(latest))
    else:
        st.info("No screened pairs yet.")

def section_features():
    st.header("2) Features and datasets")
    pairs = sorted(DATA_DIR.glob("pairs/screened_pairs_*.json"))
    sym_arg = f"--symbols /app/{pairs[-1].as_posix()}" if pairs else ""
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Build features"):
            run_cmd(f"python /app/main.py --mode features {sym_arg} --beta-window 1000 --z-window 300")
    with c2:
        if st.button("Build datasets"):
            run_cmd("python /app/main.py --mode dataset --use-manifest")

    mani = DATA_DIR / "features/pairs/_manifest.json"
    if mani.exists():
        st.success("Feature manifest:")
        try:
            st.json(json.loads(mani.read_text(encoding='utf-8')))
        except Exception:
            st.text(str(mani))
    else:
        st.info("Feature manifest not found yet.")

def section_training_registry(p):
    st.header("3) Training and Model Registry (MLflow)")
    if st.button("Train (5 folds) + log to MLflow + register + promote"):
        run_cmd(f"python /app/main.py --mode train --use-dataset --n-splits {int(p['n_splits'])} --gap {int(p['gap'])} --proba-threshold {float(p['proba_thr'])}")

    ok, msg, client = mlflow_ready()
    if not ok:
        st.warning("MLflow not available; skip registry view.")
        return

    st.subheader("Experiments")
    try:
        for e in client.list_experiments():
            st.write(f"• **{e.name}** (`{e.experiment_id}`)")
    except Exception as e:
        st.warning(f"Failed to read experiments: {e}")

    st.subheader("Registered models")
    try:
        for rm in client.search_registered_models():
            st.write(f"• **{rm.name}**")
            for v in rm.latest_versions:
                st.write(f"  └ v{v.version} • stage: {v.current_stage} • run_id={v.run_id}")
    except Exception as e:
        st.warning(f"Failed to read registry: {e}")

def section_backtest_report(p):
    st.header("4) Backtest and report")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Run backtest (OOF signals)"):
            run_cmd(f"python /app/main.py --mode backtest --signals-from oof --proba-threshold {float(p['proba_thr'])}")
    with c2:
        if st.button("Aggregate signals"):
            run_cmd("python /app/main.py --mode aggregate")
    with c3:
        if st.button("Generate report"):
            run_cmd("python /app/main.py --mode report")

    summary = DATA_DIR / "backtest_results/_summary.json"
    if summary.exists():
        st.success("Backtest summary (json):")
        try:
            st.json(json.loads(summary.read_text(encoding='utf-8')))
        except Exception:
            st.text(str(summary))
    else:
        st.info("No backtest summary yet.")

    report_md = DATA_DIR / "portfolio/latest_report.md"
    if report_md.exists():
        st.subheader("Latest report (markdown)")
        st.markdown(report_md.read_text(encoding='utf-8'))
    else:
        st.info("No report yet. Click **Generate report** after aggregate.")

def section_serving():
    st.header("5) Serving (router)")
    st.write(f"Serving expects a Production version of `{ROUTER_NAME}` in MLflow Model Registry.")
    if st.button("Restart serving container"):
        run_cmd("supervisorctl restart serving || true")
        st.caption("Check logs: `docker compose -f docker-compose.serving.yml logs -f serving`")

def main():
    header()
    p = sidebar_params()
    tabs = st.tabs(["Screening", "Features/Datasets", "Training/Registry", "Backtest/Report", "Serving"])
    with tabs[0]:
        section_screening(p)
    with tabs[1]:
        section_features()
    with tabs[2]:
        section_training_registry(p)
    with tabs[3]:
        section_backtest_report(p)
    with tabs[4]:
        section_serving()

if __name__ == "__main__":
    main()
