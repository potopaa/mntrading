#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a small human-readable report for the latest portfolio orders.

Defaults:
  - orders json: data/portfolio/latest_orders.json
  - backtest summary: data/backtest_results/_summary.json
  - markdown out: data/portfolio/_latest_report.md

Works with no args (so a naked subprocess call from Streamlit is fine).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import pandas as pd


# ------------------------- helpers -------------------------
def _utf8_stdio():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _load_latest_orders(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        arr = obj.get("orders") if isinstance(obj, dict) else obj
        if isinstance(arr, list):
            # basic validation
            out = []
            for it in arr:
                if isinstance(it, dict) and "pair" in it:
                    out.append(it)
            return out
        return []
    except Exception:
        return []


def _load_backtest_summary(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        pairs = obj.get("pairs") or {}
        mp: Dict[str, dict] = {}
        for k, v in pairs.items():
            met = (v or {}).get("metrics") or {}
            mp[str(k)] = {
                "sharpe": met.get("sharpe"),
                "maxdd": met.get("maxdd"),
                "equity_last": met.get("equity_last"),
                "n_bars": met.get("n_bars"),
                "n_trades": met.get("n_trades"),
            }
        return mp
    except Exception:
        return {}


def _fmt(x, digits=4):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "-"
        if isinstance(x, float):
            return f"{x:.{digits}f}"
        return str(x)
    except Exception:
        return "-"


# ------------------------- core -------------------------
def build_markdown(orders: List[dict], bt: Dict[str, dict]) -> str:
    ts = now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    if not orders:
        return (
            f"# Latest Portfolio Report\n\n"
            f"_Generated: {ts}_\n\n"
            f"**No orders found.** Make sure you ran the aggregator.\n"
        )

    # summary
    total = len(orders)
    gross_total = float(sum(max(0.0, float(it.get("gross_notional", 0.0))) for it in orders))
    long_n = sum(1 for it in orders if (it.get("side_spread") == "long_spread"))
    short_n = sum(1 for it in orders if (it.get("side_spread") == "short_spread"))

    md = []
    md.append("# Latest Portfolio Report")
    md.append("")
    md.append(f"_Generated: {ts}_")
    md.append("")
    md.append(f"- Orders: **{total}**  |  Gross notional: **{_fmt(gross_total, 2)}**")
    md.append(f"- Sides: long={long_n}, short={short_n}")
    md.append("")

    # table header
    md.append("| Pair | Side | Proba | Gross | A_px | B_px | Beta | Sharpe | MaxDD |")
    md.append("|:-----|:-----|------:|------:|-----:|-----:|-----:|------:|------:|")

    for it in orders:
        pair = str(it.get("pair", "-"))
        side = str(it.get("side_spread", "-"))
        proba = _fmt(it.get("proba"), 3)
        gross = _fmt(it.get("gross_notional"), 2)
        px = it.get("px") or {}
        a_px = _fmt(px.get("a"), 6)
        b_px = _fmt(px.get("b"), 6)
        beta = _fmt(px.get("beta"), 4)

        met = bt.get(pair) or {}
        sharpe = _fmt(met.get("sharpe"), 3)
        maxdd = _fmt(met.get("maxdd"), 3)

        md.append(f"| {pair} | {side} | {proba} | {gross} | {a_px} | {b_px} | {beta} | {sharpe} | {maxdd} |")

    md.append("")
    md.append("> Sources: data/portfolio/latest_orders.json, data/backtest_results/_summary.json")
    md.append("")
    return "\n".join(md)


# --------------------------- CLI ---------------------------
@click.command()
@click.option("--orders-json", default="data/portfolio/latest_orders.json", show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--backtest-summary", default="data/backtest_results/_summary.json", show_default=True,
              type=click.Path(exists=False, dir_okay=False))
@click.option("--out-markdown", default="data/portfolio/_latest_report.md", show_default=True,
              type=click.Path(dir_okay=False))
def main(orders_json: str, backtest_summary: str, out_markdown: str):
    _utf8_stdio()

    orders = _load_latest_orders(Path(orders_json))
    bt = _load_backtest_summary(Path(backtest_summary))

    md = build_markdown(orders, bt)
    out_p = Path(out_markdown)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(md, encoding="utf-8")

    # Friendly console output for Streamlit to show
    print("[report] wrote markdown:", out_p)
    if orders:
        print("[report] orders:", len(orders), "gross_total:", _fmt(sum(float(it.get('gross_notional', 0.0)) for it in orders), 2))
    else:
        print("[report] no orders to report")

if __name__ == "__main__":
    main()
