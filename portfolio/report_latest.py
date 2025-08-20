#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# portfolio/report_latest.py
# All comments are in English by request.

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import click


def _read_json(p: Path) -> Any:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    """Read JSONL file into a list of dicts."""
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _try_load_orders(orders_json: Path) -> List[Dict[str, Any]]:
    """
    Try to load orders from latest_orders.json.
    Supported shapes:
      A) {"orders": [ {...}, ... ]}
      B) {"orders_path": "...jsonl"} -> load from JSONL
      C) {"selected": [ {"pair":..., "side":..., "proba":...}, ... ]}  (fallback)
    """
    obj = _read_json(orders_json)
    # A)
    if isinstance(obj, dict) and isinstance(obj.get("orders"), list):
        return list(obj["orders"])
    # B)
    orders_path = obj.get("orders_path") if isinstance(obj, dict) else None
    if isinstance(orders_path, str):
        jp = Path(orders_path)
        if not jp.is_absolute():
            jp = orders_json.parent / jp
        return _read_jsonl(jp)
    # C) fallback: "selected"
    sel = obj.get("selected") if isinstance(obj, dict) else None
    if isinstance(sel, list):
        return list(sel)
    # As a last resort, accept list top-level
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported latest_orders.json format: {orders_json}")


def _load_backtest_metrics(summary_path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    """
    Return mapping pair -> {"sharpe":..., "maxdd":...}, robust to formats:
      {"pairs": { "PAIR": {"metrics": {...}} }}  or
      {"pairs": [ {"pair":"...", "metrics": {...}}, ... ]}
    """
    if not summary_path:
        return {}
    obj = _read_json(summary_path)
    pairs = obj.get("pairs")
    out: Dict[str, Dict[str, float]] = {}
    if isinstance(pairs, dict):
        for pair, d in pairs.items():
            met = d.get("metrics", {}) if isinstance(d, dict) else {}
            if isinstance(met, dict):
                out[str(pair)] = {
                    "sharpe": float(met.get("sharpe", 0.0)),
                    "maxdd": float(met.get("maxdd", 0.0)),
                }
    elif isinstance(pairs, list):
        for it in pairs:
            if not isinstance(it, dict):
                continue
            pair = it.get("pair") or it.get("name") or it.get("key")
            met = it.get("metrics", {})
            if pair and isinstance(met, dict):
                out[str(pair)] = {
                    "sharpe": float(met.get("sharpe", 0.0)),
                    "maxdd": float(met.get("maxdd", 0.0)),
                }
    return out


def _fmt_side(v: Any) -> str:
    """
    Normalize side to pretty text.
    Accept +/-1, "long"/"short", "buy"/"sell".
    """
    if isinstance(v, (int, float)):
        if v > 0:
            return "LONG"
        if v < 0:
            return "SHORT"
        return "FLAT"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("long", "buy", "1", "+1"):
            return "LONG"
        if s in ("short", "sell", "-1"):
            return "SHORT"
        return s.upper()
    return str(v)


def _render_markdown(orders: List[Dict[str, Any]], metrics: Dict[str, Dict[str, float]]) -> str:
    dt = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines.append(f"# MNTrading — Latest Orders Report")
    lines.append("")
    lines.append(f"_Generated at: **{dt}**_")
    lines.append("")
    lines.append("## Selected Orders")
    lines.append("")
    lines.append("| Pair | Side | Proba | Notes |")
    lines.append("|---|---:|---:|---|")
    for o in orders:
        pair = str(o.get("pair") or o.get("symbol") or o.get("pair_key") or "?")
        side = _fmt_side(o.get("side"))
        proba = o.get("proba")
        try:
            proba_s = f"{float(proba):.3f}" if proba is not None else ""
        except Exception:
            proba_s = str(proba) if proba is not None else ""
        note = o.get("note") or ""
        lines.append(f"| {pair} | {side} | {proba_s} | {note} |")

    # Metrics section (optional)
    if metrics:
        lines.append("")
        lines.append("## Backtest Metrics (per pair)")
        lines.append("")
        lines.append("| Pair | Sharpe | MaxDD |")
        lines.append("|---|---:|---:|")
        for pair, met in metrics.items():
            lines.append(f"| {pair} | {met.get('sharpe', 0.0):.3f} | {met.get('maxdd', 0.0):.4f} |")

    lines.append("")
    return "\n".join(lines)


@click.command()
@click.option("--orders-json", type=click.Path(path_type=Path), required=True, help="Path to latest_orders.json produced by aggregator.")
@click.option("--backtest-summary", type=click.Path(path_type=Path), required=False, help="Path to backtest _summary.json for metrics (optional).")
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True, help="Output Markdown file path.")
def main(orders_json: Path, backtest_summary: Optional[Path], out_path: Path):
    """Build human-readable Markdown report from latest_orders.json (and optionally backtest summary)."""
    orders = _try_load_orders(orders_json)
    metrics = _load_backtest_metrics(backtest_summary) if backtest_summary else {}

    md = _render_markdown(orders, metrics)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")

    # Also maintain a convenient symlink-like copy name if user prefers
    # (on Windows/FAT we just duplicate the file).
    latest_alias = out_path.parent / "_latest_report.md"
    try:
        latest_alias.write_text(md, encoding="utf-8")
    except Exception:
        pass

    print(f"[report] wrote {out_path}")


if __name__ == "__main__":
    main()
