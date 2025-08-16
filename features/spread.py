# -*- coding: utf-8 -*-
"""
Feature engineering for pairs trading:
- Rolling OLS beta/alpha between two instruments (a,b)
- Spread = a - (beta * b + alpha)
- Z-score over the spread
- A dual-interface helper `compute_features_for_pairs`:
    Mode A (in-memory):  pass raw_df + pairs -> Dict[pair_key, DataFrame]
    Mode B (file IO):    pass pairs_json + raw_parquet + out_dir -> List[pair_key], and it saves each pair's features to:
                         <out_dir>/<A__B>/features.parquet
This design keeps compatibility with both older and newer callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd


# -------------------------- Low-level utils --------------------------
def compute_spread(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """
    Legacy helper: simple log-spread between a and b (close prices).
    This is kept for compatibility with any older code that might import it.
    """
    if a not in df.columns or b not in df.columns:
        return pd.Series(dtype=float)
    return (np.log(df[a]) - np.log(df[b])).rename("spread")


def rolling_ols(y: pd.Series, x: pd.Series, window: int, min_periods: Optional[int] = None) -> Tuple[pd.Series, pd.Series]:
    """
    Cheap rolling OLS approximation (beta, alpha) without statsmodels.
    """
    minp = min_periods or window
    x_mean = x.rolling(window, min_periods=minp).mean()
    y_mean = y.rolling(window, min_periods=minp).mean()
    cov = (x * y).rolling(window, min_periods=minp).mean() - x_mean * y_mean
    var = x.rolling(window, min_periods=minp).var()
    beta = cov / (var + 1e-12)
    alpha = y_mean - beta * x_mean
    return beta.rename("beta"), alpha.rename("alpha")


def rolling_zscore(s: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """
    Rolling z-score with ddof=0 to match trading usage.
    """
    minp = min_periods or window
    mu = s.rolling(window, min_periods=minp).mean()
    sd = s.rolling(window, min_periods=minp).std(ddof=0)
    return ((s - mu) / (sd + 1e-12)).rename("z")


def _guess_time_index(series: pd.Series) -> pd.DatetimeIndex:
    """
    Convert a time-like series to UTC DatetimeIndex.
    Tries ns/ms/s based on magnitude if the dtype is numeric.
    """
    vals = pd.to_numeric(series, errors="coerce")
    if np.isfinite(vals).any():
        sample = float(np.nanmedian(vals.iloc[: min(10, len(vals))]))
        unit: Optional[str] = None
        if sample > 1e14:      # ~ ns
            unit = "ns"
        elif sample > 1e11:    # ~ ms
            unit = "ms"
        elif sample > 1e9:     # ~ s
            unit = "s"
        # else let pandas infer
        idx = pd.to_datetime(series, unit=unit, utc=True, errors="coerce") if unit else pd.to_datetime(series, utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.DatetimeIndex(idx)


def build_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw OHLCV into a wide 'close' matrix: index=datetime, columns=symbol.
    Supports:
      - MultiIndex columns with 'close' on the last level,
      - Long format with columns ['symbol','close'] and any time column among: ['ts','timestamp','time','date','datetime'],
      - Already-wide frames.
    """
    # Case 1: MultiIndex columns (try to pick 'close' from the last level)
    if isinstance(df.columns, pd.MultiIndex):
        names = df.columns.names or []
        last_level = names[-1] if names else None
        if last_level:
            try:
                out = df.xs("close", axis=1, level=last_level)
                return out.sort_index()
            except (KeyError, ValueError):
                pass
        try:
            out = df.xs("close", axis=1, level=-1)
            return out.sort_index()
        except (KeyError, ValueError):
            pass

    # Case 2: Long format
    lower = {str(c).lower(): c for c in df.columns}
    if "symbol" in lower and "close" in lower:
        # detect a time column; prefer common names
        time_col = None
        for cand in ("ts", "timestamp", "time", "date", "datetime"):
            if cand in lower:
                time_col = lower[cand]
                break

        if time_col is not None:
            idx = _guess_time_index(df[time_col])
        else:
            # fallback: try to use current index as time
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index
            else:
                idx = pd.to_datetime(df.index, utc=True, errors="coerce")

        wide = df.set_index(idx).pivot(columns=lower["symbol"], values=lower["close"])
        wide.index.name = "ts"
        return wide.sort_index()

    # Case 3: already wide
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to coerce the index if it looks like time-like integers
        try:
            df = df.copy()
            df.index = _guess_time_index(df.index.to_series())
        except Exception:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index.name = "ts"
    return df.sort_index()


def compute_pair_features_from_prices(
    px: pd.DataFrame,
    a: str,
    b: str,
    beta_window: int = 300,
    z_window: int = 300,
    min_periods: Optional[int] = None
) -> pd.DataFrame:
    """
    Build pair features (a,b): beta, alpha, spread, z.
    Returns a DataFrame with columns:
      ['ts','a','b','beta','alpha','spread','z','pair']
    """
    if a not in px.columns or b not in px.columns:
        return pd.DataFrame()

    df = px[[a, b]].dropna().copy()
    df.columns = ["pa", "pb"]

    beta, alpha = rolling_ols(df["pa"], df["pb"], beta_window, min_periods)
    spread = (df["pa"] - (beta * df["pb"] + alpha)).rename("spread")
    z = rolling_zscore(spread, z_window, min_periods)

    out = pd.concat([df, beta, alpha, spread, z], axis=1).dropna()
    out.index.name = "ts"
    out = out.reset_index()
    out["a"] = a
    out["b"] = b
    out["pair"] = f"{a}__{b}"
    # rename for consistency: original price columns are not needed downstream
    out = out.rename(columns={"pa": "a_close", "pb": "b_close"})
    return out


# ------------------- Dual-interface orchestrator -------------------
def compute_features_for_pairs(
    raw_df: Optional[pd.DataFrame] = None,
    pairs: Optional[List[Tuple[str, str]]] = None,
    beta_window: int = 300,
    z_window: int = 300,
    min_periods: Optional[int] = None,
    *,
    pairs_json: Optional[Union[str, Path]] = None,
    raw_parquet: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
) -> Union[Dict[str, pd.DataFrame], List[str]]:
    """
    Two modes:

    Mode A (in-memory):
        compute_features_for_pairs(raw_df=df, pairs=[("AAA/USDT","BBB/USDT"), ...], ...)
      → returns dict {'AAA/USDT__BBB/USDT': DataFrame, ...}

    Mode B (file IO):
        compute_features_for_pairs(pairs_json=<path>, raw_parquet=<path>, out_dir=<dir>, ...)
      → saves each pair to <out_dir>/<A__B>/features.parquet and returns list of pair keys
    """
    # File-based mode
    if pairs_json is not None or raw_parquet is not None or out_dir is not None:
        if not (pairs_json and raw_parquet and out_dir):
            raise ValueError("File-based mode requires pairs_json, raw_parquet and out_dir")

        p_json = Path(pairs_json)
        p_raw = Path(raw_parquet)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # load pairs from JSON
        with p_json.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        pair_list: List[Tuple[str, str]] = []
        entries = obj.get("pairs") or []
        for e in entries:
            a = e.get("a") or e.get("A") or e.get("base") or e.get("x") or e.get("left")
            b = e.get("b") or e.get("B") or e.get("quote") or e.get("y") or e.get("right")
            if a and b:
                pair_list.append((str(a), str(b)))
        if not pair_list:
            # allow strings like "AAA/USDT__BBB/USDT"
            arr = obj.get("items") or obj.get("list") or []
            for s in arr:
                if isinstance(s, str) and "__" in s:
                    a, b = s.split("__", 1)
                    pair_list.append((a.replace("_", "/"), b.replace("_", "/")))
        if not pair_list:
            raise ValueError(f"No pairs found in {p_json}")

        # read raw parquet and build a close matrix
        raw = pd.read_parquet(p_raw)
        px = build_close_matrix(raw)

        produced: List[str] = []
        for a, b in pair_list:
            feat = compute_pair_features_from_prices(px, a, b, beta_window, z_window, min_periods)
            if feat.empty:
                continue
            key_safe = f"{a.replace('/', '_')}__{b.replace('/', '_')}"
            pair_dir = out / key_safe
            pair_dir.mkdir(parents=True, exist_ok=True)
            feat.to_parquet(pair_dir / "features.parquet", index=False)
            produced.append(f"{a}__{b}")
        return produced

    # In-memory mode
    if raw_df is None or pairs is None:
        raise ValueError("Pass either (raw_df, pairs) or (pairs_json, raw_parquet, out_dir)")
    px = build_close_matrix(raw_df)
    results: Dict[str, pd.DataFrame] = {}
    for (a, b) in pairs:
        feat = compute_pair_features_from_prices(px, a, b, beta_window, z_window, min_periods)
        key = f"{a}__{b}"
        if not feat.empty:
            results[key] = feat
    return results
