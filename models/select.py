#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Champion selection for pairs based on backtest metrics (and optionally train report).

Inputs
------
- backtest summary JSON produced by backtest/runner.py:
    {
      "pairs": {
        "AAA/USDT__BBB/USDT": {"metrics": {"sharpe": ..., "maxdd": ..., "equity_last": ...}, "oof_used": true},
        ...
      },
      "proba_threshold": ...,
      "fee_rate": ...
    }

- OPTIONAL: train report JSON produced by models/train.py:
    {
      "pairs": {
        "AAA/USDT__BBB/USDT": {"rows": N, "auc_mean": 0.61, "oof_path": "..."},
        ...
      },
      "n_splits": 3,
      "gap": 5
    }

Selection logic
---------------
1) Filter by thresholds: sharpe >= sharpe_min, maxdd <= maxdd_max.
2) (optional) Require OOF probs to be used in backtest (require_oof=True).
3) (optional) Filter by min_auc and min_rows if train report provided.
4) Sort by Sharpe desc, then equity_last desc.
5) (optional) Diversity constraint by base symbol (max_per_symbol).
6) Keep top_k and write registry JSON:

    {
      "criteria": {...},
      "pairs": [
        {"pair": "AAA/USDT__BBB/USDT", "rank": 1, "metrics": {...}, "train": {"auc_mean": 0.61, "rows": 1234}},
        ...
      ]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _base(sym: str) -> str:
    """Extract base from 'BASE/QUOTE'."""
    return sym.split("/", 1)[0] if "/" in sym else sym


def _split_pair_key(pk: str) -> Tuple[str, str]:
    """'AAA/USDT__BBB/USDT' → ('AAA/USDT', 'BBB/USDT')"""
    if "__" in pk:
        a, b = pk.split("__", 1)
        return a, b
    # fallback: try single delimiter
    if "|" in pk:
        a, b = pk.split("|", 1)
        return a, b
    return pk, pk  # degenerate


def select_champions(
    summary_path: str,
    registry_out: str,
    *,
    sharpe_min: float = 0.0,
    maxdd_max: float = 1.0,
    top_k: int = 20,
    require_oof: bool = False,
    # optional stability filters (require train report)
    train_report_path: Optional[str] = None,
    min_auc: Optional[float] = None,
    min_rows: Optional[int] = None,
    # optional diversity constraint by base currency (BTC, ETH, ...)
    max_per_symbol: Optional[int] = None,
) -> Dict[str, Any]:
    summary_p = Path(summary_path)
    if not summary_p.exists():
        raise FileNotFoundError(f"Backtest summary not found: {summary_p}")

    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    pairs_info: Dict[str, Any] = summary.get("pairs", {}) or {}

    train = {}
    if train_report_path:
        p = Path(train_report_path)
        if p.exists():
            train = json.loads(p.read_text(encoding="utf-8")).get("pairs", {}) or {}

    # 1) collect candidates
    cands: List[Tuple[str, float, float, float, bool, float, Optional[float], Optional[int]]] = []
    # (pair, sharpe, maxdd, equity_last, oof_used, sort_key, auc, rows)

    for pk, data in pairs_info.items():
        met = (data or {}).get("metrics") or {}
        sharpe = float(met.get("sharpe") or np.nan)
        maxdd = float(met.get("maxdd") or np.nan)
        eq = float(met.get("equity_last") or 1.0)
        oof_used = bool((data or {}).get("oof_used"))

        if np.isnan(sharpe) or np.isnan(maxdd):
            continue
        if sharpe < float(sharpe_min) or maxdd > float(maxdd_max):
            continue
        if require_oof and not oof_used:
            continue

        auc = rows = None
        if pk in train:
            auc = train[pk].get("auc_mean")
            rows = train[pk].get("rows")
            if min_auc is not None and (auc is None or np.isnan(float(auc)) or float(auc) < float(min_auc)):
                continue
            if min_rows is not None and (rows is None or int(rows) < int(min_rows)):
                continue

        # sort_key: prioritize Sharpe, then equity_last
        sort_key = (sharpe * 10.0) + (eq - 1.0)  # simple mix
        cands.append((pk, sharpe, maxdd, eq, oof_used, sort_key, auc, rows))

    # 2) sort by score desc
    cands.sort(key=lambda t: t[5], reverse=True)

    # 3) diversity constraint (greedy)
    selected: List[Tuple[str, float, float, float, bool, Optional[float], Optional[int]]] = []
    counts: Dict[str, int] = {}
    for pk, sharpe, maxdd, eq, oof_used, _, auc, rows in cands:
        if max_per_symbol is not None and max_per_symbol > 0:
            a, b = _split_pair_key(pk)
            ba, bb = _base(a), _base(b)
            if counts.get(ba, 0) >= max_per_symbol or counts.get(bb, 0) >= max_per_symbol:
                continue
            counts[ba] = counts.get(ba, 0) + 1
            counts[bb] = counts.get(bb, 0) + 1
        selected.append((pk, sharpe, maxdd, eq, oof_used, auc, rows))
        if len(selected) >= int(top_k):
            break

    # 4) build registry
    registry = {
        "criteria": {
            "sharpe_min": float(sharpe_min),
            "maxdd_max": float(maxdd_max),
            "top_k": int(top_k),
            "require_oof": bool(require_oof),
            "min_auc": float(min_auc) if min_auc is not None else None,
            "min_rows": int(min_rows) if min_rows is not None else None,
            "max_per_symbol": int(max_per_symbol) if max_per_symbol is not None else None,
        },
        "pairs": []
    }
    for rank, (pk, sharpe, maxdd, eq, oof_used, auc, rows) in enumerate(selected, start=1):
        itm = {
            "pair": pk,
            "rank": rank,
            "metrics": {"sharpe": float(sharpe), "maxdd": float(maxdd), "equity_last": float(eq), "oof_used": bool(oof_used)},
        }
        if auc is not None or rows is not None:
            itm["train"] = {}
            if auc is not None:  itm["train"]["auc_mean"] = float(auc)
            if rows is not None: itm["train"]["rows"] = int(rows)
        registry["pairs"].append(itm)

    out_p = Path(registry_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry
