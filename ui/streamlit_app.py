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
import glob
import json
import shlex
import subprocess
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).name == "streamlit_app.py" and (Path(__file__).parent.name == "ui")) else Path(__file__).resolve().parent
# If file is at repo root, adjust ROOT to repo root
if (ROOT / "main.py").exists():
    pass
elif (ROOT.parent / "main.py").exists():
    ROOT = ROOT.parent

# ---------- helpers ----------

def py() -> str:
    return sys.executable or "python"

def run(cmd, cwd=None, env=None) -> int:
    """Run and stream output to UI."""
    if isinstance(cmd, (list, tuple)):
        printable = " ".join(shlex.quote(str(x)) for x in cmd)
    else:
        printable = str(cmd)
    st.code(printable, language="bash")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env if env else os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        log = st.empty()
        lines = []
        for line in proc.stdout:
            lines.append(line.rstrip("\n"))
            if len(lines) > 400:
                lines = lines[-400:]
            log.text("\n".join(lines))
        proc.wait()
        rc = int(proc.returncode or 0)
        if rc == 0:
            st.success("OK")
        else:
            st.error(f"Exit code {rc}")
        return rc
    except Exception as e:
        st.error(f"Failed to run: {e!r}")
        return 1

def latest_screened_json() -> str | None:
    paths = sorted(glob.glob(str(ROOT / "data" / "pairs" / "screened_pairs_*.json")))
    return paths[-1] if paths else None

def build_symbols_csv(universe_csv: str, quote: str, use_all: bool) -> str:
    """Build CSV like 'BTC/USDT,ETH/USDT,...'. If use_all=True, pull full spot universe from Binance via ccxt."""
    quote = quote.strip().upper()
    if use_all:
        try:
            import ccxt  # type: ignore
            ex = ccxt.binance()
            ex.load_markets()
            syms = []
            for m in ex.markets.values():
                if m.get("spot") and m.get("quote") == quote:
                    base = m.get("base")
                    sym = m.get("symbol")
                    # filter out leveraged tokens and weird synthetics
                    if base and sym and all(x not in base for x in ("UP", "DOWN", "BULL", "BEAR")) and ":" not in sym:
                        syms.append(sym)
            syms = sorted(set(syms))
            return ",".join(syms)
        except Exception as e:
            st.warning(f"Failed to load full universe via ccxt: {e}. Falling back to manual list.")
    # fallback to manual universe list
    bases = [s.strip().upper() for s in universe_csv.split(",") if s.strip()]
    return ",".join([f"{b}/{quote}" for b in bases])

# ---------- UI ----------

st.set_page_config(page_title="mntrading — Operations", layout="wide")
st.title("mntrading — Operations")

# Status block (simple)
with st.sidebar:
    st.header("Parameters")
    timeframe = st.selectbox("Timeframe (ingest for screening)", ["1h", "5m", "15m"], 0)
    limit = st.number_input("Limit (bars)", 100, 5000, 1000, 100)
    since_utc = st.text_input("Since UTC (optional)", "")
    universe = st.text_input("Universe (CSV of bases)", "BTC,ETH,SOL,BNB,XRP,ADA,TRX,LTC,DOT")
    quote = st.text_input("Quote", "USDT")
    use_all = st.checkbox("Use ALL Binance SPOT symbols for this quote", value=False, help="Ignores 'Universe' above and fetches full list via ccxt")

    st.markdown("---")
    st.subheader("Screening")
    corr_thr = st.slider("|corr| ≥", 0.0, 1.0, 0.30, 0.05)
    alpha = st.slider("max p-value (Engle–Granger)", 0.0, 1.0, 0.25, 0.01)
    top_k = st.number_input("Top-K pairs", 1, 300, 50, 1)

    st.markdown("---")
    st.subheader("Features")
    beta_win = st.number_input("β/α window", 50, 5000, 1000, 50)
    z_win = st.number_input("z-score window", 50, 2000, 300, 10)

    st.markdown("---")
    st.subheader("Dataset")
    label_type = st.selectbox("Label type", ["z_threshold", "revert_direction"], index=0)
    z_thr = st.number_input("zscore threshold", 0.5, 5.0, 1.5, 0.1)
    lag_feats = st.number_input("Lag features", 0, 20, 1, 1)
    horizon = st.number_input("Horizon", 0, 50, 0, 1)

    st.markdown("---")
    st.subheader("Training / Selection")
    n_splits = st.number_input("CV splits", 2, 10, 3, 1)
    gap = st.number_input("Gap", 0, 50, 5, 1)
    max_train = st.number_input("Max train size", 200, 200_000, 2000, 100)
    es_rounds = st.number_input("Early stopping rounds", 5, 500, 50, 5)
    top_k_sel = st.number_input("Top-K to promote", 1, 100, 20, 1)

# CSS for tidy layout
st.markdown("""
<style>
.button-row { display: flex; gap: 12px; }
.button-row > div { flex: 1; }
</style>
""", unsafe_allow_html=True)

# Row 1
st.markdown('<div class="button-row">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.caption("1) Screen (1h) → pairs JSON")
    if st.button("Screen", key="btn_screen"):
        symbols_csv = build_symbols_csv(universe, quote, use_all)

        # 1) Ensure raw 1h parquet exists (ingest via main)
        cmd_ing = [py(), str(ROOT / "main.py"), "--mode", "ingest",
                   "--symbols", symbols_csv, "--timeframe", "1h", "--limit", str(limit)]
        if since_utc:
            cmd_ing += ["--since-utc", since_utc]
        st.info("Ingesting 1h OHLCV for screening...")
        rc_ing = run(cmd_ing)

        # 2) Run the screener with supported flags
        if rc_ing == 0:
            cmd = [py(), str(ROOT / "screen_pairs.py"),
                   "--raw-parquet", str(ROOT / "data" / "raw" / "ohlcv.parquet"),
                   "--symbols", symbols_csv,
                   "--min-bars", "200",
                   "--min-corr", str(corr_thr),
                   "--max-pvalue", str(alpha),
                   "--top-k", str(top_k)]
            st.info("Screening pairs...")
            st.success("done" if run(cmd) == 0 else "failed")
        else:
            st.error("Ingest failed — screening skipped")

with c2:
    st.caption("2) Features → per-pair parquet")
    if st.button("Features", key="btn_features"):
        # Prefer the latest screened pairs JSON if any:
        latest_json = latest_screened_json()
        symbols_arg = latest_json if latest_json else build_symbols_csv(universe, quote, use_all)
        if latest_json:
            st.info(f"Using screened pairs: {Path(latest_json).name}")
        cmd = [py(), str(ROOT / "main.py"), "--mode", "features",
               "--symbols", symbols_arg,
               "--beta-window", str(beta_win), "--z-window", str(z_win)]
        st.info("Building features...")
        st.success("done" if run(cmd) == 0 else "failed")

with c3:
    st.caption("3) Dataset → data/datasets/* + manifest")
    if st.button("Dataset", key="btn_dataset"):
        cmd = [py(), str(ROOT / "main.py"), "--mode", "dataset",
               "--pairs-manifest", str(ROOT / "data" / "features" / "pairs" / "_manifest.json"),
               "--label-type", label_type,
               "--zscore-threshold", str(z_thr),
               "--lag-features", str(lag_feats),
               "--horizon", str(horizon),
               "--out-dir", str(ROOT / "data" / "datasets" / "pairs")]
        st.info("Building dataset...")
        st.success("done" if run(cmd) == 0 else "failed")
st.markdown('</div>', unsafe_allow_html=True)

# Row 2
st.markdown('<div class="button-row">', unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)
with d1:
    st.caption("4) Train (CV) → models + MLflow")
    if st.button("Train", key="btn_train"):
        cmd = [py(), str(ROOT / "main.py"), "--mode", "train", "--use-dataset",
               "--n-splits", str(n_splits), "--gap", str(gap),
               "--max-train-size", str(max_train),
               "--early-stopping-rounds", str(es_rounds)]
        st.info("Training models...")
        st.success("done" if run(cmd) == 0 else "failed")

with d2:
    st.caption("5) Backtest → report")
    if st.button("Backtest", key="btn_backtest"):
        cmd = [py(), str(ROOT / "main.py"), "--mode", "backtest"]
        st.info("Backtesting...")
        st.success("done" if run(cmd) == 0 else "failed")

with d3:
    st.caption("6) Select & Promote → production map")
    if st.button("Select & Promote", key="btn_select"):
        cmd1 = [py(), str(ROOT / "main.py"), "--mode", "select", "--top-k", str(top_k_sel)]
        cmd2 = [py(), str(ROOT / "main.py"), "--mode", "promote"]
        st.info("Selecting champions...")
        ok1 = (run(cmd1) == 0)
        st.info("Promoting to production...")
        st.success("done" if ok1 and run(cmd2) == 0 else "failed")
st.markdown('</div>', unsafe_allow_html=True)
