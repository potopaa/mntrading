#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit UI for the mntrading pipeline with a clean, aligned layout.

Key points:
- Sidebar holds all parameters → main area stays tidy.
- Buttons are laid out in evenly sized columns.
- Works with or without the API; falls back to local subprocesses.
- Logs are shown in an expandable panel to avoid layout jumps.
"""

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests
import streamlit as st

# ---------------------- Basic config ----------------------
API_BASE = (os.getenv("MNTRADING_API") or "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="mntrading — UI", page_icon="📈", layout="wide")

# Subtle CSS polish: aligned buttons, narrower content width, wrapped code
st.markdown("""
<style>
/* tighten main container width for better readability */
.block-container {max-width: 1180px; padding-top: 0.75rem;}
/* make all Streamlit buttons full-width within their column */
.button-row .stButton>button {width: 100%; padding: 0.6rem 0.8rem;}
/* keep code blocks from overflowing */
pre, code {white-space: pre-wrap; word-break: break-word;}
/* little status pill style */
.status-pill {display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; font-weight:600;}
.status-ok {background:#DCFCE7; color:#065F46;}
.status-warn {background:#FEF9C3; color:#713F12;}
</style>
""", unsafe_allow_html=True)


# ---------------------- Helpers ----------------------
def find_repo_root() -> Path:
    """Detect project root so subprocesses run in the right place."""
    here = Path(__file__).resolve().parent
    for c in [here, here.parent, here.parent.parent, Path.cwd()]:
        if (c / "main.py").exists() and (c / "screen_pairs.py").exists():
            return c
    return Path.cwd()

ROOT = find_repo_root()

def api_alive() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.ok
    except Exception:
        return False

def api_get(path: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=20); r.raise_for_status(); return r.json()
    except Exception as e:
        return {"error": str(e)}

def api_post(path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=300); r.raise_for_status(); return r.json()
    except Exception as e:
        return {"error": str(e)}

def py() -> str:
    return sys.executable

def run(cmd: List[str]) -> int:
    """Run a subprocess and stream logs into the UI."""
    import subprocess
    ph = st.empty()
    ph.code(" ".join(cmd), language="bash")
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    logs = []
    for line in p.stdout:
        logs.append(line.rstrip("\n"))
        ph.code("\n".join(logs[-800:]), language="bash")
    p.wait()
    return p.returncode


# ---------------------- Sidebar (parameters) ----------------------
st.sidebar.title("Parameters")
with st.sidebar:
    timeframe = st.selectbox("Timeframe", ["5m", "1m", "15m"], 0)
    limit = st.number_input("Limit (bars)", 100, 5000, 1000, 100)
    since_utc = st.text_input("Since UTC (optional)", "")
    universe = st.text_input("Universe (CSV of bases)", "BTC,ETH,SOL,BNB,XRP,ADA,MATIC,TRX,LTC,DOT")
    quote = st.text_input("Quote", "USDT")
    corr_thr = st.slider("|corr| ≥", 0.0, 1.0, 0.3, 0.05)
    alpha = st.slider("max p-value (Engle–Granger)", 0.0, 1.0, 0.25, 0.01)
    top_k = st.number_input("Top-K pairs", 1, 200, 50, 1)

    st.markdown("---")
    beta_win = st.number_input("β/α window", 50, 2000, 300, 50)
    z_win = st.number_input("z window", 50, 2000, 300, 50)
    z_th = st.slider("|z| threshold (dataset label)", 0.5, 5.0, 1.5, 0.1)
    lags = st.number_input("lag features", 0, 10, 1, 1)
    horizon = st.number_input("horizon (bars)", 0, 10, 0, 1)

    st.markdown("---")
    proba_th = st.slider("proba threshold", 0.5, 0.9, 0.55, 0.01)
    fee_rate = st.number_input("fee rate", 0.0, 0.01, 0.0005, 0.0001)
    top_signals = st.number_input("Top-K signals", 1, 100, 10, 1)
    equity = st.number_input("Equity", 1000, 1_000_000, 10000, 1000)
    leverage = st.number_input("Leverage", 1.0, 10.0, 1.0, 0.5)

    st.markdown("---")
    st.subheader("Champion selection")
    sharpe_min_ui = st.number_input("Sharpe min", -2.0, 5.0, 0.0, 0.1)
    maxdd_max_ui = st.number_input("Max drawdown (≤)", 0.0, 1.0, 1.0, 0.01)
    top_k_ui = st.number_input("Top-K (final)", 1, 200, 20, 1)
    require_oof_ui = st.checkbox("Require OOF probabilities", value=False)
    with st.expander("Advanced filters", expanded=False):
        min_auc_ui = st.text_input("Min AUC (optional)", value="")
        min_rows_ui = st.text_input("Min rows (optional)", value="")
        max_per_symbol_ui = st.text_input("Max per base symbol (optional)", value="")

    st.markdown("---")
    st.caption("API base (read-only):")
    st.code(API_BASE, language="bash")


# ---------------------- Header ----------------------
st.title("mntrading — Operations")
alive = api_alive()
status_class = "status-ok" if alive else "status-warn"
status_text = "🟢 online" if alive else "🟡 offline (local mode)"
st.markdown(f'API status: <span class="status-pill {status_class}">{status_text}</span>', unsafe_allow_html=True)

# Determine symbols source for subsequent steps
pairs_dir = ROOT / "data" / "pairs"
pairs_dir.mkdir(parents=True, exist_ok=True)
pairs_jsons = sorted(pairs_dir.glob("screened_pairs_*.json"))
symbols_arg = str(pairs_jsons[-1]) if pairs_jsons else ",".join(
    [f"{s.strip().upper()}/{quote}" for s in universe.split(",") if s.strip()]
)

# ---------------------- Controls ----------------------
tabs = st.tabs(["Pipeline", "Artifacts", "API (optional)"])

# === PIPELINE TAB ===
with tabs[0]:
    st.markdown("#### Run steps")
    # Row 1
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("1) Screen (1h) → pairs JSON")
        if st.button("Screen", key="btn_screen"):
            cmd = [py(), str(ROOT / "screen_pairs.py"),
                   "--universe", universe, "--quote", quote, "--source", "ccxt", "--exchange", "binance",
                   "--since-utc", since_utc or "2025-01-01",
                   "--min-samples", "200", "--corr-threshold", str(corr_thr),
                   "--alpha", str(alpha), "--top-k", str(top_k)]
            st.info("Screening pairs...")
            st.success("done" if run(cmd) == 0 else "failed")

    with c2:
        st.caption("2) Ingest (5m) → data/raw/ohlcv.parquet")
        if st.button("Ingest", key="btn_ingest"):
            cmd = [py(), str(ROOT / "main.py"), "--mode", "ingest", "--symbols", symbols_arg,
                   "--timeframe", timeframe, "--limit", str(limit)]
            if since_utc:
                cmd += ["--since-utc", since_utc]
            st.info("Ingesting OHLCV...")
            st.success("done" if run(cmd) == 0 else "failed")

    with c3:
        st.caption("3) Features → per-pair parquet + manifest")
        if st.button("Features", key="btn_features"):
            cmd = [py(), str(ROOT / "main.py"), "--mode", "features", "--symbols", symbols_arg,
                   "--beta-window", str(beta_win), "--z-window", str(z_win)]
            st.info("Building features...")
            st.success("done" if run(cmd) == 0 else "failed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.caption("4) Dataset → data/datasets/* + manifest")
        if st.button("Dataset", key="btn_dataset"):
            cmd = [py(), str(ROOT / "main.py"), "--mode", "dataset",
                   "--pairs-manifest", str(ROOT / "data" / "features" / "pairs" / "_manifest.json"),
                   "--label-type", "z_threshold", "--zscore-threshold", str(z_th),
                   "--lag-features", str(lags), "--horizon", str(horizon)]
            st.info("Building dataset...")
            st.success("done" if run(cmd) == 0 else "failed")

    with d2:
        st.caption("5) Train → OOF per pair + _train_report.json")
        if st.button("Train", key="btn_train"):
            cmd = [py(), str(ROOT / "main.py"), "--mode", "train", "--use-dataset",
                   "--n-splits", "3", "--gap", "5", "--max-train-size", "2000",
                   "--early-stopping-rounds", "50", "--proba-threshold", "0.55"]
            st.info("Training models...")
            st.success("done" if run(cmd) == 0 else "failed")

    with d3:
        st.caption("6) Backtest → _summary.json")
        if st.button("Backtest", key="btn_backtest"):
            cmd = [py(), str(ROOT / "main.py"), "--mode", "backtest", "--use-dataset",
                   "--signals-from", "oof", "--proba-threshold", "0.55", "--fee-rate", str(fee_rate)]
            st.info("Backtesting...")
            st.success("done" if run(cmd) == 0 else "failed")
    st.markdown('</div>', unsafe_allow_html=True)

    # Row 3
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.caption("7) Select champions → registry.json")
        if st.button("Select", key="btn_select"):
            cmd = [py(), str(ROOT / "main.py"), "--mode", "select",
                   "--summary-path", str(ROOT / "data" / "backtest_results" / "_summary.json"),
                   "--registry-out", str(ROOT / "data" / "models" / "registry.json"),
                   "--sharpe-min", str(sharpe_min_ui),
                   "--maxdd-max", str(maxdd_max_ui),
                   "--top-k", str(top_k_ui)]
            if require_oof_ui:
                cmd += ["--require-oof"]
            # optional advanced filters (only add if user provided a value)
            if min_auc_ui.strip():
                cmd += ["--min-auc", min_auc_ui.strip()]
            if min_rows_ui.strip():
                cmd += ["--min-rows", min_rows_ui.strip()]
            if max_per_symbol_ui.strip():
                cmd += ["--max-per-symbol", max_per_symbol_ui.strip()]
            st.info("Selecting champions...")
            st.success("done" if run(cmd) == 0 else "failed")

    with r2:
        st.caption("8) Promote → production_map.json")
        if st.button("Promote", key="btn_promote"):
            cmd = [py(), str(ROOT / "main.py"), "--mode", "promote",
                   "--production-map-out", str(ROOT / "data" / "models" / "production_map.json")]
            st.info("Promoting models...")
            st.success("done" if run(cmd) == 0 else "failed")

    with r3:
        st.caption("9) Inference → data/signals/*.jsonl")
        if st.button("Inference", key="btn_infer"):
            cmd = [py(), str(ROOT / "inference.py"),
                   "--registry", str(ROOT / "data" / "models" / "production_map.json"),
                   "--pairs-manifest", str(ROOT / "data" / "features" / "pairs" / "_manifest.json"),
                   "--timeframe", timeframe, "--limit", str(limit),
                   "--proba-threshold", "0.55", "--update", "--n-last", "1",
                   "--out", str(ROOT / "data" / "signals")]
            st.info("Running inference...")
            st.success("done" if run(cmd) == 0 else "failed")

    with r4:
        st.caption("10) Aggregate + mini-report")
        if st.button("Aggregate + Report", key="btn_agg"):
            cmd1 = [py(), str(ROOT / "portfolio" / "aggregate_signals.py"),
                    "--signals-dir", str(ROOT / "data" / "signals"),
                    "--pairs-manifest", str(ROOT / "data" / "features" / "pairs" / "_manifest.json"),
                    "--min-proba", "0.55", "--top-k", str(top_signals),
                    "--scheme", "equal_weight", "--equity", str(equity), "--leverage", str(leverage)]
            st.info("Aggregating signals...")
            rc1 = run(cmd1)
            cmd2 = [py(), str(ROOT / "portfolio" / "report_latest.py"),
                    "--orders-dir", str(ROOT / "data" / "portfolio"),
                    "--backtests-dir", str(ROOT / "data" / "backtest_results"),
                    "--lookback-bars", "2000"]
            st.info("Creating report...")
            rc2 = run(cmd2)
            st.success("done" if rc1 == 0 and rc2 == 0 else f"failed rc={rc1}/{rc2}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Logs/Artifacts
    with st.expander("Artifacts (local view)", expanded=False):
        colA, colB = st.columns(2)
        try:
            mf = json.loads((ROOT / "data" / "features" / "pairs" / "_manifest.json").read_text(encoding="utf-8"))
            colA.markdown("**Features manifest**"); colA.json(mf)
        except Exception:
            colA.info("No features manifest yet")
        try:
            summ = json.loads((ROOT / "data" / "backtest_results" / "_summary.json").read_text(encoding="utf-8"))
            colB.markdown("**Backtest summary**"); colB.json(summ)
        except Exception:
            colB.info("No backtest summary yet")

# === ARTIFACTS TAB ===
with tabs[1]:
    st.markdown("#### Quick peek")
    col1, col2 = st.columns(2)
    try:
        pm = json.loads((ROOT / "data" / "models" / "production_map.json").read_text(encoding="utf-8"))
        col1.markdown("**Production map**")
        col1.json(pm)
    except Exception:
        col1.info("No production map yet")

    try:
        tr = json.loads((ROOT / "data" / "models" / "_train_report.json").read_text(encoding="utf-8"))
        col2.markdown("**Train report**")
        col2.json(tr)
    except Exception:
        col2.info("No train report yet")

# === API TAB (optional) ===
with tabs[2]:
    st.markdown("Use server-side flows if the API is running.")
    colx, coly = st.columns(2)
    if colx.button("API: bootstrap_quick"):
        st.json(api_post("/run/bootstrap_quick"))
    if coly.button("API: short_cycle"):
        st.json(api_post("/run/short_cycle", {"timeframe": timeframe, "limit": int(limit)}))
    st.json(api_get("/artifacts/production_map"))
    st.json(api_get("/artifacts/backtest_summary"))
