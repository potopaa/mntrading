# ui/streamlit_app.py
# All comments are in English by request.

import os
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st


APP_ROOT = Path("/app")
DATA = APP_ROOT / "data"

# Helper to run shell commands inside the container.
def run(cmd: str) -> str:
    """Run a shell command and capture combined output."""
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(APP_ROOT),
    )
    return proc.stdout


def header():
    st.set_page_config(page_title="MNTrading", layout="wide")
    st.title("MNTrading — Pipeline UI")
    st.caption("Buttons call the same CLI steps you ran from PowerShell (one-liners).")


def sidebar_params():
    st.sidebar.header("Pipeline parameters")

    st.sidebar.subheader("Universe / Ingest(1h)")
    exchange = st.sidebar.text_input("Exchange", "binance")
    quote = st.sidebar.text_input("Quote", "USDT")
    top = st.sidebar.number_input("Top symbols", min_value=10, max_value=500, value=200, step=10)
    since_1h = st.sidebar.text_input("since-utc (1h)", "2025-01-01T00:00:00Z")
    limit_1h = st.sidebar.number_input("limit per call (1h)", min_value=100, max_value=5000, value=1000, step=100)
    max_candles = st.sidebar.number_input("max-candles (0=all)", min_value=0, max_value=10_000_000, value=0, step=1000)

    st.sidebar.subheader("Screen → Ingest(5m)")
    since_5m = st.sidebar.text_input("since-utc-5m", "2025-01-01T00:00:00Z")
    limit_5m = st.sidebar.number_input("limit-5m", min_value=100, max_value=50000, value=1000, step=100)

    st.sidebar.subheader("Features")
    beta_window = st.sidebar.number_input("beta-window", min_value=50, max_value=5000, value=1000, step=50)
    z_window = st.sidebar.number_input("z-window", min_value=50, max_value=2000, value=300, step=50)

    st.sidebar.subheader("Dataset")
    label_type = st.sidebar.selectbox("label-type", options=["z_threshold", "revert_direction"], index=0)
    z_thr = st.sidebar.number_input("zscore-threshold", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    lag_features = st.sidebar.number_input("lag-features", min_value=0, max_value=100, value=10, step=1)
    horizon = st.sidebar.number_input("horizon", min_value=1, max_value=100, value=3, step=1)

    st.sidebar.subheader("Training/CV")
    n_splits = st.sidebar.number_input("n-splits", min_value=2, max_value=20, value=5, step=1)
    gap = st.sidebar.number_input("gap", min_value=0, max_value=1000, value=24, step=1)
    early_stop = st.sidebar.number_input("early-stopping-rounds", min_value=0, max_value=1000, value=50, step=5)

    st.sidebar.subheader("Signals / Portfolio")
    proba_thr = st.sidebar.number_input("proba-threshold", min_value=0.5, max_value=1.0, value=0.55, step=0.01)
    top_k = st.sidebar.number_input("top-k", min_value=1, max_value=200, value=20, step=1)

    return dict(
        exchange=exchange, quote=quote, top=int(top),
        since_1h=since_1h, limit_1h=int(limit_1h), max_candles=int(max_candles),
        since_5m=since_5m, limit_5m=int(limit_5m),
        beta_window=int(beta_window), z_window=int(z_window),
        label_type=label_type, z_thr=float(z_thr), lag_features=int(lag_features), horizon=int(horizon),
        n_splits=int(n_splits), gap=int(gap), early_stop=int(early_stop),
        proba_thr=float(proba_thr), top_k=int(top_k),
    )


def section_actions(p):
    st.header("Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Ingest 1h (auto universe)"):
            cmd = (
                "python /app/main.py --mode ingest --symbols-auto "
                f"--exchange {p['exchange']} --quote {p['quote']} --top {p['top']} "
                f"--timeframe 1h --since-utc {p['since_1h']} --limit {p['limit_1h']} --max-candles {p['max_candles']}"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Screen + Ingest 5m for screened"):
            cmd = (
                "python /app/main.py --mode screen "
                f"--since-utc-5m {p['since_5m']} --limit-5m {p['limit_5m']}"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Features"):
            cmd = (
                "python /app/main.py --mode features "
                "--symbols '/app/data/pairs/screened_pairs_*.json' "
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
        if st.button("Inference"):
            cmd = (
                "python /app/main.py --mode inference "
                "--registry-in /app/data/models/registry.json "
                f"--signals-from model --proba-threshold {p['proba_thr']} "
                "--n-last 1 --update"
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

        if st.button("Generate report"):
            cmd = (
                "python /app/portfolio/report_latest.py "
                "--orders-json /app/data/portfolio/latest_orders.json "
                "--backtest-summary /app/data/backtest_results/_summary.json "
                "--out /app/data/portfolio/_latest_report.md"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Upload to MinIO"):
            cmd = (
                "python /app/scripts/upload_to_minio.py "
                "/app/data/features /app/data/datasets /app/data/models "
                "/app/data/backtest_results /app/data/portfolio"
            )
            st.code(cmd)
            st.text(run(cmd))

        if st.button("Log to MLflow"):
            cmd = (
                "python /app/scripts/log_to_mlflow.py "
                "--experiment mntrading "
                "--train-report /app/data/models/_train_report.json "
                "--backtest-summary /app/data/backtest_results/_summary.json "
                "--models-dir /app/data/models/pairs "
                "--artifacts /app/data/portfolio/_latest_report.md "
                "--artifacts /app/data/portfolio/latest_orders.json "
                "--registry /app/data/models/registry.json "
                "--prod-map /app/data/models/production_map.json"
            )
            st.code(cmd)
            st.text(run(cmd))

    st.subheader("Full pipeline (1-click)")
    if st.button("Run FULL pipeline"):
        with st.status("Running full pipeline...", expanded=True) as status:
            steps = [
                ("Ingest 1h", f"python /app/main.py --mode ingest --symbols-auto --exchange {p['exchange']} --quote {p['quote']} --top {p['top']} --timeframe 1h --since-utc {p['since_1h']} --limit {p['limit_1h']} --max-candles {p['max_candles']}"),
                ("Screen+5m", f"python /app/main.py --mode screen --since-utc-5m {p['since_5m']} --limit-5m {p['limit_5m']}"),
                ("Features", f"python /app/main.py --mode features --symbols '/app/data/pairs/screened_pairs_*.json' --beta-window {p['beta_window']} --z-window {p['z_window']}"),
                ("Dataset", f"python /app/main.py --mode dataset --label-type {p['label_type']} --zscore-threshold {p['z_thr']} --lag-features {p['lag_features']} --horizon {p['horizon']}"),
                ("Train", f"python /app/main.py --mode train --use-dataset --n-splits {p['n_splits']} --gap {p['gap']} --early-stopping-rounds {p['early_stop']} --proba-threshold {p['proba_thr']}"),
                ("Backtest", f"python /app/main.py --mode backtest --signals-from auto --proba-threshold {p['proba_thr']} --fee-rate 0.0005"),
                ("Select", f"python /app/main.py --mode select --summary-path /app/data/backtest_results/_summary.json --registry-out /app/data/models/registry.json --sharpe-min 0.0 --maxdd-max 1.0 --top-k {p['top_k']}"),
                ("Promote", "python /app/main.py --mode promote --registry-in /app/data/models/registry.json --production-map-out /app/data/models/production_map.json"),
                ("Inference", f"python /app/main.py --mode inference --registry-in /app/data/models/registry.json --signals-from model --proba-threshold {p['proba_thr']} --n-last 1 --update"),
                ("Aggregate", f"python /app/main.py --mode aggregate --signals-dir /app/data/signals --portfolio-dir /app/data/portfolio --proba-threshold {p['proba_thr']} --top-k {p['top_k']}"),
                ("Report", "python /app/portfolio/report_latest.py --orders-json /app/data/portfolio/latest_orders.json --backtest-summary /app/data/backtest_results/_summary.json --out /app/data/portfolio/_latest_report.md"),
            ]
            for name, cmd in steps:
                st.write(f"**{name}** → `{cmd}`")
                st.text(run(cmd))
            status.update(label="Full pipeline finished", state="complete")

    st.subheader("Artifacts")
    report = DATA / "portfolio" / "_latest_report.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    else:
        st.info("No report yet. Click **Generate report** after aggregate.")


def main():
    header()
    p = sidebar_params()
    section_actions(p)


if __name__ == "__main__":
    main()
