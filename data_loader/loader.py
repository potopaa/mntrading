import ccxt
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize exchange (public access)
exchange = ccxt.binance({"enableRateLimit": True})

def get_all_symbols(market: str = "USDT") -> List[str]:
    """
    Return all trading symbols on Binance that end with the given market.
    """
    markets = exchange.load_markets()
    return [symbol for symbol in markets if symbol.endswith(f"/{market}")]

def get_ohlcv_incremental(
    symbol: str,
    timeframe: str,
    since_ts: int,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch only new candles since since_ts (ms) via paginated requests.
    """
    all_bars = []
    pbar = tqdm(desc=f"{symbol} bars", unit="bars")

    while True:
        bars = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit)
        if not bars:
            break
        all_bars.extend(bars)
        pbar.update(len(bars))
        since_ts = bars[-1][0] + 1
        if len(bars) < limit:
            break

    pbar.close()

    df = pd.DataFrame(all_bars, columns=[
        "timestamp", "open", "high", "low", "close", "volume"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp")

def get_multiple_ohlcv_incremental(
    symbols: List[str],
    timeframe: str,
    since_ts: int,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch new OHLCV for multiple symbols incrementally and combine.
    """
    frames = []
    for sym in tqdm(symbols, desc="Symbols"):
        df_sym = get_ohlcv_incremental(sym, timeframe, since_ts, limit)
        df_sym.columns = pd.MultiIndex.from_product([[sym], df_sym.columns])
        frames.append(df_sym)
    return pd.concat(frames, axis=1).sort_index()
