#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Screen asset pairs for pairs trading.

Reads raw OHLCV parquet (long format: ts/symbol/close) and outputs screened pairs JSON:
{
  "pairs": [
    {"a": "BTC/USDT", "b": "ETH/USDT", "metrics": {"corr": 0.82, "pvalue": 0.01, "bars": 25000}},
    ...
  ],
  ...
}

Use with the pipeline:
  1) python screen_pairs.py --raw-parquet data/raw/ohlcv.parquet --top-k 200
  2) python main.py --mode features --symbols data/pairs/screened_pairs_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

# Optional cointegration test
try:
    from statsmodels.tsa.stattools import coint  # type: ignore
    HAS_SM = True
except Exception:
    HAS_SM = False


# ------------------------- Helpers -------------------------
def _utf8_stdio():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def build_close_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw OHLCV to a wide close-price matrix with DatetimeIndex (UTC).
    Supports long format with columns ['symbol','close'] and a time column among:
    ['ts','timestamp','time','date','datetime'].
    """
    lower = {str(c).lower(): c for c in raw.columns}
    if "symbol" in lower and "close" in lower:
        time_col = None
        for cand in ("ts", "timestamp", "time", "date", "datetime"):
            if cand in lower:
                time_col = lower[cand]
                break
        if time_col is None:
            raise ValueError("Time column not found in raw parquet (expected ts/timestamp/time/date/datetime)")
        idx = pd.to_datetime(raw[time_col], utc=True, errors="coerce")
        wide = raw.assign(_ts=idx).pivot(index="_ts", columns=lower["symbol"], values=lower["close"])
        wide.index.name = "ts"
        return wide.sort_index()

    # assume wide already
    df = raw.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index.name = "ts"
    return df.sort_index()


def screen_pairs(
    px: pd.DataFrame,
    symbols: Optional[List[str]],
    min_bars: int,
    min_corr: float,
    max_pvalue: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Simple screener:
      - compute log-returns corr for overlapping window;
      - if statsmodels available, compute Engle-Granger cointegration p-value on log-prices;
      - filter by min_bars, min_corr, (pvalue <= max_pvalue);
      - sort by (pvalue asc, corr desc, bars desc) if cointegration is available, else by corr, bars.
    """
    cols = symbols if symbols else [c for c in px.columns if pd.api.types.is_numeric_dtype(px[c])]
    cols = [c for c in cols if c in px.columns]
    if len(cols) < 2:
        return []

    lpx = np.log(px[cols].astype("float64").clip(1e-12))
    rets = lpx.diff().dropna()

    results: List[Tuple[str, str, float, float, int]] = []  # a,b,corr,pval,n
    ncols = len(cols)
    for i in range(ncols):
        a = cols[i]
        for j in range(i + 1, ncols):
            b = cols[j]
            rpair = rets[[a, b]].dropna()
            n = len(rpair)
            if n < min_bars:
                continue
            corr = float(rpair[a].corr(rpair[b]))
            if not np.isfinite(corr) or corr < min_corr:
                continue

            pval = 1.0
            if HAS_SM:
                try:
                    ab = lpx[[a, b]].dropna()
                    if len(ab) >= min_bars:
                        _, pval, _ = coint(ab[a].values, ab[b].values, trend="c", autolag="AIC")
                        if not np.isfinite(pval):
                            pval = 1.0
                except Exception:
                    pval = 1.0

            results.append((a, b, corr, pval, n))

    if not results:
        return []

    if HAS_SM:
        results.sort(key=lambda t: (t[3], -t[2], -t[4]))
        filtered = [r for r in results if r[3] <= max_pvalue]
    else:
        results.sort(key=lambda t: (-t[2], -t[4]))
        filtered = results

    if top_k > 0:
        filtered = filtered[:top_k]

    out = [
        {"a": a, "b": b, "metrics": {"corr": round(c, 6), "pvalue": (round(p, 6) if HAS_SM else None), "bars": int(n)}}
        for (a, b, c, p, n) in filtered
    ]
    return out


# --------------------------- CLI ---------------------------
@click.command()
@click.option("--raw-parquet", default="data/raw/ohlcv.parquet", show_default=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option("--symbols", default="", help="CSV list of symbols to consider. If empty, use all from raw.")
@click.option("--min-bars", default=2000, show_default=True, type=int, help="Minimum overlapping bars per pair")
@click.option("--min-corr", default=0.6, show_default=True, type=float, help="Min correlation of log-returns")
@click.option("--max-pvalue", default=0.05, show_default=True, type=float,
              help="Max cointegration p-value (ignored if statsmodels is not installed)")
@click.option("--top-k", default=200, show_default=True, type=int, help="Keep top-K pairs after filters")
@click.option("--out", "out_path", default="", help="Output JSON path. Default: data/pairs/screened_pairs_<UTC>.json")
def main(raw_parquet: str, symbols: str, min_bars: int, min_corr: float, max_pvalue: float, top_k: int, out_path: str):
    _utf8_stdio()

    raw = pd.read_parquet(raw_parquet)
    px = build_close_matrix(raw)
    syms = [s.strip() for s in symbols.split(",") if s.strip()] if symbols else None

    pairs = screen_pairs(
        px=px,
        symbols=syms,
        min_bars=int(min_bars),
        min_corr=float(min_corr),
        max_pvalue=float(max_pvalue),
        top_k=int(top_k),
    )

    out_dir = Path("data/pairs")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_path:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"screened_pairs_{ts}.json"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pairs": pairs,
        "raw_parquet": str(Path(raw_parquet).resolve()),
        "min_bars": int(min_bars),
        "min_corr": float(min_corr),
        "max_pvalue": (float(max_pvalue) if HAS_SM else None),
        "top_k": int(top_k),
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[screen] wrote {len(pairs)} pairs -> {out_path}")
    if not HAS_SM:
        print("[warn] statsmodels not installed; cointegration filter skipped (using correlation only)")


if __name__ == "__main__":
    main()
