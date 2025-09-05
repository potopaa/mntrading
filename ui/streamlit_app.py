<<<<<<< HEAD
import subprocess
from pathlib import Path
import streamlit as st
import requests

=======

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
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700

APP_ROOT = Path("/app")
DATA_DIR = APP_ROOT / "data"

<<<<<<< HEAD

def run(cmd: str) -> str:
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.stdout
=======
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "mntrading_model")
ROUTER_NAME = os.getenv("MODEL_NAME", "mntrading_router")
ROUTER_STAGE = os.getenv("MODEL_STAGE", "Production")
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700

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

<<<<<<< HEAD
    st.sidebar.subheader("Serving")
    serving_url = st.sidebar.text_input("Serving URL", "http://serving:5001")

    return dict(
        exchange=exchange, quote=quote, top=int(top),
        since_1h=since_1h, limit_1h=int(limit_1h), max_candles=int(max_candles),
        since_5m=since_5m, limit_5m=int(limit_5m),
        beta_window=int(beta_window), z_window=int(z_window),
        label_type=label_type, z_thr=float(z_thr), lag_features=int(lag_features), horizon=int(horizon),
        n_splits=int(n_splits), gap=int(gap), early_stop=int(early_stop),
        proba_thr=float(proba_thr), top_k=int(top_k),
        serving_url=serving_url,
    )
=======
def section_training_registry(p):
    st.header("3) Training and Model Registry (MLflow)")
    if st.button("Train (5 folds) + log to MLflow + register + promote"):
        run_cmd(f"python /app/main.py --mode train --use-dataset --n-splits {int(p['n_splits'])} --gap {int(p['gap'])} --proba-threshold {float(p['proba_thr'])}")
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700

    ok, msg, client = mlflow_ready()
    if not ok:
        st.warning("MLflow not available; skip registry view.")
        return

<<<<<<< HEAD
def section_actions(p: dict):
    st.header("Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Data")
        if st.button("Ingest(1h)"):
            cmd = (
                "python /app/main.py --mode ingest "
                f"--symbols-auto --exchange {p['exchange']} --quote {p['quote']} --top {p['top']} "
                f"--timeframe 1h --since-utc {p['since_1h']} --limit {p['limit_1h']} --max-candles {p['max_candles']}"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Screen pairs (and ingest 5m)"):
            cmd = (
                "python /app/main.py --mode screen"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Features"):
            cmd = (
                "python /app/main.py --mode features "
                "--symbols $(ls -1t /app/data/pairs/screened_pairs_*.json | head -n1) "
                f"--beta-window {p['beta_window']} --z-window {p['z_window']}"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Dataset"):
            cmd = (
                "python /app/main.py --mode dataset "
                f"--label-type {p['label_type']} --zscore-threshold {p['z_thr']} "
                f"--lag-features {p['lag_features']} --horizon {p['horizon']}"
            )
            st.code(cmd)
            st.text(run(cmd))

    with col2:
        st.subheader("Modeling")
        if st.button("Train"):
            cmd = (
                "python /app/main.py --mode train --use-dataset "
                f"--n-splits {p['n_splits']} --gap {p['gap']} "
                f"--early-stopping-rounds {p['early_stop']} --proba-threshold {p['proba_thr']}"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Backtest"):
            cmd = (
                "python /app/main.py --mode backtest "
                f"--signals-from auto --proba-threshold {p['proba_thr']} --fee-rate 0.0005"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Select champions"):
            cmd = (
                "python /app/main.py --mode select "
                "--summary-path /app/data/backtest_results/_summary.json "
                "--registry-out /app/data/models/registry.json "
                "--sharpe-min 0.0 --maxdd-max 1.0 "
                f"--top-k {p['top_k']}"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Promote to production map"):
            cmd = (
                "python /app/main.py --mode promote "
                "--registry-in /app/data/models/registry.json "
                "--production-map-out /app/data/models/production_map.json"
            )
            st.code(cmd)
            st.text(run(cmd))

    with col3:
        st.subheader("MLflow & Serving")

        hc_col1, hc_col2 = st.columns(2)
        with hc_col1:
            if st.button("Serving /ping"):
                try:
                    r = requests.get(f"{p['serving_url'].rstrip('/')}/ping", timeout=3)
                    st.write(f"Status: {r.status_code}, Body: {r.text[:200]}")
                except Exception as e:
                    st.error(f"Serving ping failed: {e}")
        with hc_col2:
            if st.button("Serving /version"):
                try:
                    r = requests.get(f"{p['serving_url'].rstrip('/')}/version", timeout=3)
                    st.write(f"Status: {r.status_code}, Body: {r.text[:200]}")
                except Exception as e:
                    st.error(f"Serving version failed: {e}")

        st.divider()

        if st.button("Register pairs to MLflow (→ Staging)"):
            cmd = (
                "python /app/scripts/register_models.py "
                "--registry /app/data/models/registry.json "
                "--models-dir /app/data/models/pairs "
                "--experiment mntrading "
                "--prefix mntrading "
                "--stage Staging "
                f"--top-k {p['top_k']}"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Build & register Router from MLflow"):
            cmd = (
                "python /app/scripts/build_router_from_mlflow.py "
                "--prefix mntrading_ "
                "--pair-stage Staging "
                f"--top-k {p['top_k']} "
                "--registered-name mntrading_router "
                "--router-stage Production "
                "--experiment mntrading"
            )
            st.code(cmd)
            st.text(run(cmd))
            st.info("Now (on host) run:  docker compose -f docker-compose.yml -f docker-compose.streamlit.yml -f docker-compose.serving.yml up -d --force-recreate serving")

        if st.button("Inference via MLflow Serving"):
            cmd = (
                "python /app/scripts/serving_inference.py "
                "--pairs-manifest /app/data/features/pairs/_manifest.json "
                f"--n-last 1 "
                f"--serving-url {p['serving_url']} "
                "--out /app/data/signals"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Aggregate"):
            cmd = (
                "python /app/main.py --mode aggregate "
                "--signals-dir /app/data/signals "
                "--portfolio-dir /app/data/portfolio "
                f"--proba-threshold {p['proba_thr']} --top-k {p['top_k']}"
            )
            st.code(cmd)
            st.text(run(cmd))
=======
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
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700

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
<<<<<<< HEAD
            cmd = (
                "python /app/portfolio/report_latest.py "
                "--orders-json /app/data/portfolio/latest_orders.json "
                "--backtest-summary /app/data/backtest_results/_summary.json "
                "--registry /app/data/models/registry.json "
                "--out /app/data/portfolio/_latest_report.md"
            )
            st.code(cmd)
            st.text(run(cmd))
=======
            run_cmd("python /app/main.py --mode report")
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700

    summary = DATA_DIR / "backtest_results/_summary.json"
    if summary.exists():
        st.success("Backtest summary (json):")
        try:
            st.json(json.loads(summary.read_text(encoding='utf-8')))
        except Exception:
            st.text(str(summary))
    else:
        st.info("No backtest summary yet.")

<<<<<<< HEAD
        if st.button("Log to MLflow"):
            cmd = (
                "python /app/scripts/log_to_mlflow.py "
                "--experiment mntrading "
                "--train-report /app/data/models/_train_report.json "
                "--backtest-summary /app/data/backtest_results/_summary.json "
                "--models-dir /app/data/models/pairs "
                "--artifacts /app/data/portfolio/_latest_report.md "
                "--artifacts /app/data/models/registry.json"
            )
            st.code(cmd)
            st.text(run(cmd))

    st.header("Latest report preview")
    report = Path("/app/data/portfolio/_latest_report.md")
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
=======
    report_md = DATA_DIR / "portfolio/latest_report.md"
    if report_md.exists():
        st.subheader("Latest report (markdown)")
        st.markdown(report_md.read_text(encoding='utf-8'))
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700
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