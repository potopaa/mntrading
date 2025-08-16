#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Screen pairs on 1h data and save a manifest with selected pairs.

Pipeline usage (examples):
  python screen_pairs.py --universe "BTC,ETH,SOL,BNB,XRP,ADA,MATIC,TRX,LTC,DOT" --quote USDT \
    --source ccxt --exchange binance --since-utc 2025-01-01 \
    --min-samples 200 --corr-threshold 0.3 --alpha 0.25 --page-size 1000 --top-k 50

Output:
  data/pairs/screened_pairs_YYYYMMDD_HHMMSSZ.json
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

# tqdm is optional; keep ascii logs friendly on Windows terminals
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# ======================
# Time helpers (UTC)
# ======================

def _parse_since_utc(since_utc: Optional[str]) -> int:
    """
    Accepts:
      - "2025"                         -> 2025-01-01T00:00:00Z
      - "2025-01-01"                   -> 2025-01-01T00:00:00Z
      - "2025-01-01T00:00:00Z"         -> as-is (UTC)
      - None or ""                     -> 0 (fetch all)
    Returns: UNIX epoch milliseconds (int, UTC).
    """
    if not since_utc:
        return 0

    s = str(since_utc).strip()
    if re.fullmatch(r"\d{4}", s):
        ts = pd.Timestamp(f"{s}-01-01T00:00:00Z")
    else:
        ts = pd.Timestamp(s)

    # If tz-naive -> localize, if tz-aware -> convert to UTC
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    # pandas .value is ns since epoch; convert to ms
    return int(ts.value // 10**6)


def _utcnow_iso() -> str:
    now_utc = datetime.now(timezone.utc).replace(microsecond=0)
    return now_utc.isoformat()  # e.g. 2025-08-15T09:25:00+00:00


def _utcnow_stamp() -> str:
    now_utc = datetime.now(timezone.utc)
    return now_utc.strftime("%Y%m%d_%H%M%SZ")


# ======================
# CCXT fetch (1h)
# ======================

@dataclass
class FetchParams:
    exchange: str = "binance"
    timeframe: str = "1h"
    page_size: int = 1000
    max_bars: Optional[int] = None


def _load_ccxt_exchange(exchange: str):
    import ccxt  # lazy import
    ex_name = exchange.lower().strip()
    if not hasattr(ccxt, ex_name):
        raise ValueError(f"Unknown exchange for ccxt: {exchange}")
    ex = getattr(ccxt, ex_name)({"enableRateLimit": True})
    ex.load_markets()
    return ex


def _fetch_symbol_ohlcv_1h_ccxt(ex, symbol: str, since_ms: int, limit: int, max_bars: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch OHLCV for a symbol via ccxt, timeframe=1h, looping by 'since' until no rows returned.
    Returns DataFrame with columns: [timestamp, open, high, low, close, volume]
    """
    rows: List[List[float]] = []
    next_since = since_ms if since_ms else None
    fetched = 0

    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe="1h", since=next_since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        fetched += len(batch)
        if max_bars and fetched >= max_bars:
            break
        # move cursor; add +1ms to avoid duplicate last bar
        next_since = batch[-1][0] + 1

        # ccxt safety: if we received less than limit -> likely reached the end
        if len(batch) < limit:
            break

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    # drop duplicates by timestamp, keep last
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _fetch_ohlcv_1h(symbols: List[str], exchange: str, since_utc: Optional[str], page_size: int, max_bars: Optional[int] = None) -> pd.DataFrame:
    """
    Fetch 1h close prices for all symbols and return a wide dataframe:
    index = timestamp (ns), columns = symbol (e.g., BTC/USDT), values = close
    """
    since_ms = _parse_since_utc(since_utc)
    ex = _load_ccxt_exchange(exchange)

    series_map: Dict[str, pd.Series] = {}
    for sym in tqdm(symbols, desc="Symbols", ascii=True):
        try:
            df = _fetch_symbol_ohlcv_1h_ccxt(ex, sym, since_ms, page_size, max_bars=max_bars)
            if df.empty:
                continue
            s = pd.Series(df["close"].values, index=pd.to_datetime(df["timestamp"], unit="ms", utc=True), name=sym)
            # Drop duplicates on the index (hourly bars must be unique per timestamp)
            s = s[~s.index.duplicated(keep="last")].sort_index()
            series_map[sym] = s
        except Exception as e:
            print(f"[warn] fetch failed for {sym}: {e}")

    if not series_map:
        return pd.DataFrame()

    dfw = pd.DataFrame(series_map)
    # make sure index sorted and duplicated timestamps removed globally as well
    dfw = dfw[~dfw.index.duplicated(keep="last")].sort_index()
    return dfw


# ======================
# Screening logic
# ======================

@dataclass
class PairInfo:
    a: str
    b: str
    corr: float
    samples: int
    alpha: float
    beta: float


def _ols_alpha_beta(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Simple OLS: y = alpha + beta * x
    Returns (alpha, beta)
    """
    # add constant term
    X = np.column_stack([np.ones_like(x), x])
    # solve (X'X)^{-1} X'y
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        alpha, beta = float(coef[0]), float(coef[1])
    except Exception:
        alpha, beta = float("nan"), float("nan")
    return alpha, beta


def _screen_pairs(dfw: pd.DataFrame, corr_threshold: float, min_samples: int, top_k: Optional[int]) -> List[PairInfo]:
    """
    dfw: wide df with 1h close prices per symbol (columns are symbols).
    1) Compute log returns per symbol
    2) Compute Pearson corr per pair (on overlapping returns)
    3) Keep pairs with corr >= threshold and >= min_samples
    4) For each kept pair, estimate alpha/beta between log prices (y on x)
    """
    out: List[PairInfo] = []
    if dfw.empty or dfw.shape[1] < 2:
        return out

    # log returns (hourly)
    lr = np.log(dfw).diff().dropna(how="all")

    cols = list(dfw.columns)
    n = len(cols)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = cols[i], cols[j]
            # overlapping returns
            sub = lr[[a, b]].dropna()
            n_samp = int(len(sub))
            if n_samp < min_samples:
                continue
            r = float(sub[a].corr(sub[b]))
            if math.isnan(r) or r < corr_threshold:
                continue

            # alpha/beta on log prices on overlapping timestamps
            sub_px = np.log(dfw[[a, b]].dropna())
            sub_px = sub_px.dropna()
            # align columns in case of slight differences
            sub_px = sub_px[[a, b]].dropna()
            if len(sub_px) >= 2:
                alpha, beta = _ols_alpha_beta(sub_px[a].values, sub_px[b].values)
            else:
                alpha, beta = float("nan"), float("nan")

            out.append(PairInfo(a=a, b=b, corr=r, samples=n_samp, alpha=alpha, beta=beta))

    # sort by correlation desc, then samples desc
    out.sort(key=lambda p: (p.corr, p.samples), reverse=True)

    if top_k and top_k > 0 and len(out) > top_k:
        out = out[:top_k]

    return out


# ======================
# CLI
# ======================

@click.command()
@click.option("--universe", type=str, required=True, help="Comma-separated base symbols, e.g. 'BTC,ETH,SOL,...'")
@click.option("--quote", type=str, default="USDT", show_default=True, help="Quote asset, e.g. USDT")
@click.option("--source", type=click.Choice(["ccxt"]), default="ccxt", show_default=True, help="Data source")
@click.option("--exchange", type=str, default="binance", show_default=True, help="ccxt exchange id")
@click.option("--since-utc", type=str, default=None, help="Start point (UTC). Examples: '2025', '2025-01-01', '2025-01-01T00:00:00Z'")
@click.option("--page-size", type=int, default=1000, show_default=True, help="Fetch page size (ccxt limit)")
@click.option("--max-bars", type=int, default=None, help="Optional hard cap on bars per symbol")
@click.option("--min-samples", type=int, default=200, show_default=True, help="Min overlapping hourly returns per pair")
@click.option("--corr-threshold", type=float, default=0.3, show_default=True, help="Min Pearson corr between hourly returns")
@click.option("--alpha", "alpha_param", type=float, default=0.25, show_default=True, help="Reserved param (kept for compatibility)")
@click.option("--top-k", type=int, default=50, show_default=True, help="Keep top-K pairs by corr")
@click.option("--out-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=Path("data/pairs"), show_default=True)
def main(
    universe: str,
    quote: str,
    source: str,
    exchange: str,
    since_utc: Optional[str],
    page_size: int,
    max_bars: Optional[int],
    min_samples: int,
    corr_threshold: float,
    alpha_param: float,
    top_k: int,
    out_dir: Path,
):
    """
    Screen pairs on 1h closes for the given universe/quote.
    Saves JSON manifest consumable by downstream steps (features/dataset/...).
    """
    bases = [s.strip().upper() for s in universe.split(",") if s.strip()]
    symbols = [f"{b}/{quote.upper()}" for b in bases]
    print(f"[step] screen 1h (universe={','.join(bases)}, quote={quote.upper()}, exchange={exchange}, since_utc={since_utc})")

    if source != "ccxt":
        raise click.UsageError(f"Unsupported source: {source}")

    dfw = _fetch_ohlcv_1h(symbols, exchange=exchange, since_utc=since_utc, page_size=page_size, max_bars=max_bars)
    if dfw.empty:
        print("[done] saved 0 pairs -> (no data)")
        return

    pairs = _screen_pairs(dfw, corr_threshold=corr_threshold, min_samples=min_samples, top_k=top_k)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utcnow_stamp()
    out_path = out_dir / f"screened_pairs_{stamp}.json"

    manifest = {
        "generated_at_utc": _utcnow_iso(),
        "params": {
            "universe": bases,
            "quote": quote.upper(),
            "source": source,
            "exchange": exchange,
            "since_utc": since_utc,
            "page_size": page_size,
            "max_bars": max_bars,
            "min_samples": min_samples,
            "corr_threshold": corr_threshold,
            "alpha": alpha_param,
            "top_k": top_k,
        },
        "symbols": symbols,
        "pairs": [asdict(p) for p in pairs],
    }

    # Save ASCII json (ensure_ascii=True) to avoid Windows cp1251 issues in terminals
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=True, indent=2)

    print(f"[done] saved {len(pairs)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
