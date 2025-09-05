from __future__ import annotations
from typing import Iterable, List, Optional
import time

import ccxt
import pandas as pd


STABLES = {"USDT", "BUSD", "USDC", "TUSD", "FDUSD", "DAI"}


def _ex_from_name(name: str):
    name = (name or "binance").lower()
    if not hasattr(ccxt, name):
        raise ValueError(f"Unknown exchange: {name}")
    ex = getattr(ccxt, name)()
    ex.load_markets()
    return ex


def list_spot_symbols(
    exchange: str = "binance",
    quote: str = "USDT",
    top: Optional[int] = None,
    exclude_stables: bool = True,
    use_tickers_sort: bool = True,
) -> List[str]:
    ex = _ex_from_name(exchange)
    quote = quote.upper().strip()

    # Collect all active spot symbols with the given quote
    syms: List[str] = []
    for sym, m in ex.markets.items():
        if not m.get("active"):
            continue
        if not m.get("spot"):
            continue
        if m.get("quote") != quote:
            continue
        base = str(m.get("base", "")).upper().strip()
        if exclude_stables and base in STABLES:
            continue
        syms.append(sym)

    if not syms:
        return syms

    if use_tickers_sort:
        try:
            # fetch_tickers can be heavy; fetch for filtered list only
            tickers = ex.fetch_tickers(syms)
            def _quote_vol(t: dict) -> float:
                # try common fields in 'info' or top-level
                info = t.get("info") or {}
                for k in ("quoteVolume", "volumeQuote", "qv"):
                    v = info.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
                v = t.get("quoteVolume")
                try:
                    return float(v) if v is not None else 0.0
                except Exception:
                    return 0.0
            syms.sort(key=lambda s: _quote_vol(tickers.get(s, {})), reverse=True)
        except Exception:
            syms.sort()

    if top:
        syms = syms[: int(top)]
    return syms


def _fetch_ohlcv_all(
    ex,
    symbol: str,
    timeframe: str,
    since_ms: Optional[int],
    per_page: int,
    rate_limit_ms: int,
    max_candles: Optional[int] = None,
) -> pd.DataFrame:
    """
    Paginated OHLCV fetch:
    - 1000 per page (Binance cap) or per_page provided
    - continue while we make forward progress
    - stop if max_candles reached (if provided)
    """
    rows_all = []
    last_ts = None
    for _ in range(100000):  # defensive upper bound
        rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=per_page)
        if not rows:
            break
        if last_ts is not None and rows[-1][0] <= last_ts:
            # no forward progress -> stop
            break
        rows_all.extend(rows)
        last_ts = rows[-1][0]
        since_ms = last_ts + 1
        time.sleep((rate_limit_ms or 500) / 1000.0)
        if max_candles and len(rows_all) >= max_candles:
            break

    if not rows_all:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "symbol"])

    df = pd.DataFrame(rows_all, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["symbol"] = symbol
    return df


def load_ohlcv_for_symbols(
    symbols: Iterable[str],
    timeframe: str = "1h",
    since_utc: Optional[str] = None,
    limit: int = 1000,
    max_candles: Optional[int] = None,
    exchange: str = "binance",
) -> pd.DataFrame:
    """
    Robust multi-symbol OHLCV loader with pagination.

    Args:
      symbols: list/iterable of symbols, e.g. ['BTC/USDT', 'ETH/USDT', ...]
      timeframe: '1h', '5m', etc.
      since_utc: ISO8601 start time, exchange local time is not used
      limit: per-page limit (capped at 1000 for Binance)
      max_candles: optional total cap per symbol (e.g. 20000)
      exchange: exchange name for ccxt (default 'binance')

    Returns:
      pandas DataFrame with columns ['ts','open','high','low','close','volume','symbol'].
    """
    ex = _ex_from_name(exchange)
    per_page = min(int(limit) if limit else 1000, 1000)
    since_ms = ex.parse8601(since_utc) if since_utc else None

    frames: List[pd.DataFrame] = []
    for sym in symbols:
        if sym not in ex.markets:
            # skip unknown symbol gracefully
            continue
        df = _fetch_ohlcv_all(
            ex=ex,
            symbol=sym,
            timeframe=timeframe,
            since_ms=since_ms,
            per_page=per_page,
            rate_limit_ms=getattr(ex, "rateLimit", 500),
            max_candles=max_candles,
        )
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "symbol"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["symbol", "ts"]).reset_index(drop=True)
    return out
