# -*- coding: utf-8 -*-
"""
screen_pairs.py — screening cointegration/correlation pairs.

Usage examples:
  python screen_pairs.py --raw-parquet data/raw/ohlcv.parquet --symbols ALL --quote USDT --min-bars 150 --min-corr 0.10 --max-pvalue 1.0 --top-k 200
  python screen_pairs.py --raw-parquet data/raw/ohlcv.parquet --symbols "BTC/USDT,ETH/USDT,SOL/USDT"
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Sequence, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.tsa.stattools as smt
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


def _load_all_symbols_from_parquet(parquet_path: Path, quote: Optional[str], max_symbols: Optional[int]) -> List[str]:
    df = pd.read_parquet(parquet_path)
    if "symbol" not in df.columns:
        raise ValueError(f"'symbol' column not found in {parquet_path}")
    syms = sorted(df["symbol"].dropna().astype(str).unique().tolist())
    if quote:
        syms = [s for s in syms if s.upper().endswith(f"/{quote.upper()}")]
    if max_symbols and max_symbols > 0:
        syms = syms[:max_symbols]
    if not syms:
        raise ValueError("No symbols discovered from parquet with given filters")
    return syms


def _parse_symbols_arg(symbols_arg: Optional[str]) -> Optional[List[str]]:
    if not symbols_arg:
        return None
    s = symbols_arg.strip()
    if s.upper() in {"ALL", "__ALL__"}:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _engle_granger_pvalue(y: pd.Series, x: pd.Series) -> float:
    # Simple Engle–Granger cointegration test p-value
    # (y and x should be aligned, same index)
    if not HAS_STATSMODELS:
        return 1.0  # effectively "pass everything"
    try:
        res = smt.coint(y, x)
        return float(res[1])
    except Exception:
        return 1.0


def screen_pairs(
    parquet_path: Path,
    symbols: Sequence[str],
    min_bars: int = 150,
    min_corr: float = 0.10,
    max_pvalue: float = 1.0,
    top_k: int = 200,
    out_dir: Path = Path("data/pairs"),
) -> Path:
    df = pd.read_parquet(parquet_path)
    tcol = next(c for c in ["ts", "timestamp", "time", "date", "datetime"] if c in df.columns)
    px = df[df["symbol"].isin(symbols)].pivot(index=tcol, columns="symbol", values="close")
    px.index = pd.to_datetime(px.index, utc=True, errors="coerce")
    rets = np.log(px).diff()

    corr = rets.corr().abs()
    cols = corr.columns.tolist()

    candidates: List[Tuple[str, str, float, int, float]] = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            sub = rets[[a, b]].dropna()
            n = int(sub.shape[0])
            if n < min_bars:
                continue
            c = float(corr.loc[a, b])
            if not np.isfinite(c) or c < min_corr:
                continue
            # (optional) cointegration
            pv = _engle_granger_pvalue(px[a].dropna(), px[b].dropna()) if HAS_STATSMODELS else 0.999
            if pv <= max_pvalue:
                candidates.append((a, b, c, n, pv))

    candidates.sort(key=lambda x: (x[2], x[3]), reverse=True)
    if top_k and top_k > 0:
        candidates = candidates[:top_k]

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"screened_pairs_{ts}.json"

    payload = [
        {"a": a, "b": b, "corr": round(c, 6), "bars": n, "pvalue": round(pv, 6)}
        for a, b, c, n, pv in candidates
    ]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[screen] wrote {len(payload)} pairs -> {out_path}")
    if not HAS_STATSMODELS:
        print("[warn] statsmodels not installed — skipping cointegration test (using pvalue ~ 1.0 pass-through).")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-parquet", type=Path, default=Path("data/raw/ohlcv.parquet"))
    ap.add_argument("--symbols", type=str, default="ALL",
                    help='Comma-separated list or "ALL" to use every symbol found in parquet.')
    ap.add_argument("--quote", type=str, default="USDT", help="Filter symbols by quote currency (e.g., USDT).")
    ap.add_argument("--max-symbols", type=int, default=0, help="Optional cap for discovered symbols (0 = no cap).")
    ap.add_argument("--min-bars", type=int, default=150)
    ap.add_argument("--min-corr", type=float, default=0.10)
    ap.add_argument("--max-pvalue", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--universe-json", type=Path, default=None,
                    help="Optional JSON file with explicit universe ['BTC/USDT', ...]. Overrides --symbols/--quote discovery.")
    args = ap.parse_args()

    if args.universe_json and args.universe_json.exists():
        syms = json.loads(args.universe_json.read_text(encoding="utf-8"))
        if not isinstance(syms, list) or not syms:
            raise ValueError("--universe-json should be a JSON array of symbols")
    else:
        parsed = _parse_symbols_arg(args.symbols)
        if parsed is None:
            syms = _load_all_symbols_from_parquet(args.raw_parquet, args.quote, args.max_symbols)
        else:
            syms = parsed

    screen_pairs(
        parquet_path=args.raw_parquet,
        symbols=syms,
        min_bars=args.min_bars,
        min_corr=args.min_corr,
        max_pvalue=args.max_pvalue,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
