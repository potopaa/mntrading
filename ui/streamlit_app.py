#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import shlex
import subprocess
from pathlib import Path

import streamlit as st

# --------- paths ----------
ROOT = Path(__file__).resolve().parent.parent if (Path(__file__).parent.name == "ui") else Path(__file__).resolve().parent

def py() -> str:
    return sys.executable or "python"

def run(cmd, cwd=None, env=None) -> int:
    """Run and live-stream logs into UI."""
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
        holder = st.empty()
        lines = []
        for line in proc.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            if len(lines) > 400:
                lines = lines[-400:]
            holder.text("\n".join(lines))
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

# ---------- helpers ----------
def latest_screened_json() -> str | None:
    lst = sorted(glob.glob(str(ROOT / "data" / "pairs" / "screened_pairs_*.json")))
    return lst[-1] if lst else None

def extract_symbols_from_pairs_json(pairs_json: str) -> str:
    if not pairs_json or not Path(pairs_json).exists():
        return ""
    obj = json.loads(Path(pairs_json).read_text(encoding="utf-8"))
    pairs = obj.get("pairs") if isinstance(obj, dict) else obj
    syms = set()
    for it in pairs or []:
        if isinstance(it, dict):
            a, b = it.get("a"), it.get("b")
        elif isinstance(it, list) and len(it) == 2:
            a, b = it
        elif isinstance(it, str) and "|" in it:
            a, b = it.split("|", 1)
        else:
            a = b = None
        if a and b:
            syms.add(str(a).strip())
            syms.add(str(b).strip())
    return ",".join(sorted(syms))

def build_symbols_csv(universe_csv: str, quote: str, use_all: bool) -> str:
    quote = quote.strip().upper()
    if use_all:
        try:
            import ccxt
            ex = ccxt.binance()
            ex.load_markets()
            syms = []
            for m in ex.markets.values():
                if m.get("spot") and m.get("quote") == quote:
                    base = m.get("base")
                    sym = m.get("symbol")
                    if base and sym and all(x not in base for x in ("UP", "DOWN", "BULL", "BEAR")) and ":" not in sym:
                        syms.append(sym)
            syms = sorted(set(syms))
            return ",".join(syms)
        except Exception as e:
            st.warning(f"Failed to load full universe via ccxt: {e}. Falling back to manual list.")
    bases = [s.strip().upper() for s in universe_csv.split(",") if s.strip()]
    return ",".join([f"{b}/{quote}" for b in bases])

# ---------- page ----------
st.set_page_config(page_title="mntrading — Operations", layout="wide")
st.title("mntrading — Operations")

with st.sidebar:
    st.header("Parameters")
    # Universe
    timeframe_screen = st.selectbox("Screen timeframe", ["1h"], index=0, help="Скринер всегда на 1h")
    limit = st.number_input("Limit (bars)", 100, 5000, 1000, 100)
    since_utc = st.text_input("Since UTC (optional)", "")
    universe = st.text_input("Universe (CSV of bases)", "BTC,ETH,SOL,BNB,XRP,ADA,TRX,LTC,DOT")
    quote = st.text_input("Quote", "USDT")
    use_all = st.checkbox("Use ALL Binance SPOT symbols for this quote", value=False)

    st.markdown("---")
    st.subheader("Screening")
    min_bars = st.number_input("min bars", 50, 5000, 200, 50)
    corr_thr = st.slider("|corr| ≥", 0.0, 1.0, 0.30, 0.05)
    alpha = st.slider("max p-value (Engle–Granger)", 0.0, 1.0, 0.25, 0.01)
    top_k = st.number_input("Top-K pairs", 1, 300, 50, 1)

    st.markdown("---")
    st.subheader("Features (5m)")
    beta_win = st.number_input("β/α window", 50, 5000, 1000, 50)
    z_win = st.number_input("z-score window", 50, 2000, 300, 10)

    st.markdown("---")
    st.subheader("Dataset / Train / Select")
    label_type = st.selectbox("Label type", ["z_threshold", "revert_direction"], index=0)
    z_thr = st.number_input("zscore threshold", 0.5, 5.0, 1.5, 0.1)
    lag_feats = st.number_input("Lag features", 0, 20, 1, 1)
    horizon = st.number_input("Horizon", 0, 50, 0, 1)
    n_splits = st.number_input("CV splits", 2, 10, 3, 1)
    gap = st.number_input("Gap", 0, 50, 5, 1)
    max_train = st.number_input("Max train size", 200, 200_000, 2000, 100)
    es_rounds = st.number_input("Early stopping rounds", 5, 500, 50, 5)
    top_k_sel = st.number_input("Top-K to promote", 1, 100, 20, 1)

# ---- row styles ----
st.markdown("""
<style>
.row { display:flex; gap:12px; }
.row > div { flex:1; }
</style>
""", unsafe_allow_html=True)

# ---------- Row A: 1h ingest → screen ----------
st.markdown('<div class="row">', unsafe_allow_html=True)
a1, a2, a3 = st.columns(3)
with a1:
    st.caption("1) Ingest UNIVERSE (1h)")
    if st.button("Ingest 1h Universe"):
        symbols_csv = build_symbols_csv(universe, quote, use_all)
        cmd = [py(), str(ROOT / "main.py"), "--mode", "ingest",
               "--symbols", symbols_csv,
               "--timeframe", "1h",
               "--limit", str(limit)]
        if since_utc:
            cmd += ["--since-utc", since_utc]
        st.info("Downloading 1h OHLCV…")
        st.success("done" if run(cmd) == 0 else "failed")

with a2:
    st.caption("2) Screen cointegrated pairs (from 1h)")
    if st.button("Screen pairs"):
        raw_1h = ROOT / "data" / "raw" / "ohlcv_1h.parquet"
        symbols_csv = build_symbols_csv(universe, quote, use_all)
        cmd = [py(), str(ROOT / "screen_pairs.py"),
               "--raw-parquet", str(raw_1h),
               "--symbols", symbols_csv,
               "--min-bars", str(min_bars),
               "--min-corr", str(corr_thr),
               "--max-pvalue", str(alpha),
               "--top-k", str(top_k)]
        st.info("Screening on 1h…")
        st.success("done" if run(cmd) == 0 else "failed")


# ---------- Row B: 5m ingest (only selected pairs) → features ----------
st.markdown('<div class="row">', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
with b1:
    st.caption("3) Ingest SELECTED PAIRS (5m)")
    if st.button("Ingest 5m for pairs"):
        pairs_json = latest_screened_json()
        if not pairs_json:
            st.error("No screened_pairs_*.json found. Run screening first.")
        else:
            symbols_csv = extract_symbols_from_pairs_json(pairs_json)
            if not symbols_csv:
                st.error("Pairs JSON is empty.")
            else:
                cmd = [py(), str(ROOT / "main.py"), "--mode", "ingest",
                       "--symbols", symbols_csv,
                       "--timeframe", "5m",
                       "--limit", str(max(limit, 1000))]
                if since_utc:
                    cmd += ["--since-utc", since_utc]
                st.info("Downloading 5m OHLCV for selected pairs…")
                st.success("done" if run(cmd) == 0 else "failed")

with b2:
    st.caption("4) Build FEATURES (5m) for selected pairs")
    if st.button("Features"):
        pairs_json = latest_screened_json()
        if not pairs_json:
            st.error("No screened_pairs_*.json found. Run screening first.")
        else:
            cmd = [py(), str(ROOT / "main.py"), "--mode", "features",
                   "--symbols", pairs_json,
                   "--beta-window", str(beta_win), "--z-window", str(z_win)]
            st.info("Building features…")
            st.success("done" if run(cmd) == 0 else "failed")

with b3:
    st.caption("5) Build DATASET")
    if st.button("Dataset"):
        cmd = [py(), str(ROOT / "main.py"), "--mode", "dataset",
               "--pairs-manifest", str(ROOT / "data" / "features" / "pairs" / "_manifest.json"),
               "--label-type", label_type,
               "--zscore-threshold", str(z_thr),
               "--lag-features", str(lag_feats),
               "--horizon", str(horizon),
               "--out-dir", str(ROOT / "data" / "datasets" / "pairs")]
        st.info("Building dataset…")
        st.success("done" if run(cmd) == 0 else "failed")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Row C: Train / Backtest / Select & Promote ----------
st.markdown('<div class="row">', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.caption("6) Train (CV) + log to MLflow")
    if st.button("Train"):
        cmd = [py(), str(ROOT / "main.py"), "--mode", "train", "--use-dataset",
               "--n-splits", str(n_splits), "--gap", str(gap),
               "--max-train-size", str(max_train),
               "--early-stopping-rounds", str(es_rounds)]
        st.info("Training…")
        st.success("done" if run(cmd) == 0 else "failed")

with c2:
    st.caption("7) Backtest")
    if st.button("Backtest"):
        cmd = [py(), str(ROOT / "main.py"), "--mode", "backtest"]
        st.info("Backtesting…")
        st.success("done" if run(cmd) == 0 else "failed")

with c3:
    st.caption("8) Select & Promote")
    if st.button("Select & Promote"):
        cmd1 = [py(), str(ROOT / "main.py"), "--mode", "select", "--top-k", str(top_k_sel)]
        cmd2 = [py(), str(ROOT / "main.py"), "--mode", "promote"]
        st.info("Selecting champions…")
        ok1 = (run(cmd1) == 0)
        st.info("Promoting…")
        st.success("done" if ok1 and run(cmd2) == 0 else "failed")
st.markdown('</div>', unsafe_allow_html=True)

# ---------- Row D: Inference / Portfolio / Report ----------
st.markdown('<div class="row">', unsafe_allow_html=True)
d1, d2, d3 = st.columns(3)
with d1:
    st.caption("9) Inference (produce signals)")
    if st.button("Inference"):
        cmd = [py(), str(ROOT / "inference.py")]
        st.info("Running inference…")
        st.success("done" if run(cmd) == 0 else "failed")

with d2:
    st.caption("10) Portfolio aggregation")
    if st.button("Portfolio"):
        cmd = [py(), str(ROOT / "portfolio" / "aggregate_signals.py"),
               "--min-proba", "0.55", "--top-k", "20"]
        st.info("Aggregating signals…")
        st.success("done" if run(cmd) == 0 else "failed")

with d3:
    st.caption("11) Report")
    if st.button("Report"):
        cmd = [py(), str(ROOT / "portfolio" / "report_latest.py")]
        st.info("Generating latest report…")
        st.success("done" if run(cmd) == 0 else "failed")
st.markdown('</div>', unsafe_allow_html=True)
