# -*- coding: utf-8 -*-
"""
Feature builder for pair spreads.

Public entry:
    compute_features_for_pairs(...)

Supports two calling styles:

1) File-based (used by main.py when --symbols points to screened_pairs_*.json):
    compute_features_for_pairs(
        pairs_json: str | Path,
        raw_parquet: str | Path,
        out_dir: str | Path,
        beta_window: int,
        z_window: int,
    ) -> List[str]   # writes per-pair features.parquet and returns list of pair keys

2) In-memory (used by main.py when --symbols is CSV; main.py writes parquet itself):
    compute_features_for_pairs(
        raw_df: pd.DataFrame,
        pairs: List[Tuple[str, str]],
        beta_window: int,
        z_window: int,
    ) -> Dict[str, pd.DataFrame]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union
import json
import math

import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except Exception:  # tqdm is optional
    def tqdm(x, **kwargs):
        return x


# ----------------------------- Pair parsing helpers ----------------------------- #

def _decode_pair_key(s: str) -> Tuple[str, str]:
    """
    Accepts:
      - 'BTC_USDT__ETH_USDT'
      - 'BTC/USDT,ETH/USDT'
      - 'BTC/USDT__ETH/USDT'
    Returns: ('BTC/USDT', 'ETH/USDT')
    """
    s = (s or "").strip()
    if "__" in s:
        a, b = s.split("__", 1)
        return a.replace("_", "/"), b.replace("_", "/")
    if "," in s:
        a, b = [x.strip() for x in s.split(",", 1)]
        return a, b
    raise ValueError(f"Cannot decode pair string: {s!r}")


def _pairs_from_any(obj: Any) -> List[Tuple[str, str]]:
    """
    Normalize *anything* (dict/list/str/path) into list[(a,b)] with symbols 'XXX/USDT'.
    Supported shapes:
      - {"pairs": [...]} or {"data": [...]} or {"results": [...]} — elements may be dict/list/str
      - [ {"a": "...", "b": "..."}, {"pair": "BTC_USDT__ETH_USDT"}, ["BTC/USDT","ETH/USDT"], "BTC_USDT__ETH_USDT", ... ]
      - path to .json containing any of the above
      - single string with comma-separated pair keys
    """
    # If it's path to JSON — load and recurse
    if isinstance(obj, (str, Path)):
        p = Path(str(obj))
        # path to JSON file
        if p.suffix.lower() == ".json" and p.exists():
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
                return _pairs_from_any(j)
            except Exception:
                # not json or unreadable -> fall back to string parsing
                pass

    # unwrap container
    if isinstance(obj, dict):
        items = obj.get("pairs") or obj.get("data") or obj.get("results") or []
    elif isinstance(obj, list):
        items = obj
    else:
        s = str(obj).strip()
        items = [x.strip() for x in s.split(",") if x.strip()] if s else []

    out: List[Tuple[str, str]] = []
    for it in items:
        a = b = None
        if isinstance(it, dict):
            a = it.get("a") or it.get("sym_a") or it.get("left") or it.get("x")
            b = it.get("b") or it.get("sym_b") or it.get("right") or it.get("y")
            if not (a and b):
                pair_key = it.get("pair") or it.get("pair_key") or it.get("id") or ""
                if pair_key:
                    try:
                        a, b = _decode_pair_key(pair_key)
                    except Exception:
                        a = b = None
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            a, b = str(it[0]), str(it[1])
        elif isinstance(it, str):
            try:
                a, b = _decode_pair_key(it)
            except Exception:
                a = b = None

        if a and b:
            out.append((a, b))
    return out


# ----------------------------- Core computations ----------------------------- #

def _rolling_beta_alpha(y: pd.Series, x: pd.Series, win: int) -> Tuple[pd.Series, pd.Series]:
    """
    Fast rolling OLS via moments:
        beta = Cov(x,y)/Var(x)
        alpha = E[y] - beta * E[x]
    """
    x_mean = x.rolling(win).mean()
    y_mean = y.rolling(win).mean()
    cov = (x * y).rolling(win).mean() - x_mean * y_mean
    var = x.rolling(win).var()
    beta = cov / (var.replace(0.0, np.nan))
    alpha = y_mean - beta * x_mean
    return beta, alpha


def _zscore(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - mu) / (sd + 1e-12)


def _ensure_ts_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize time column to UTC DatetimeIndex named 'ts'.
    """
    df = df.copy()
    tcol = None
    for c in ("ts", "timestamp", "time", "date", "datetime"):
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        raise ValueError("Raw OHLCV must contain a time column (ts/timestamp/time/date/datetime).")
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df.assign(ts=ts).dropna(subset=["ts"]).set_index("ts").sort_index()
    return df


def _pivot_close(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Raw dataframe -> pivot prices by symbol, index ts, value close.
    """
    for col in ("symbol", "close"):
        if col not in df_raw.columns:
            raise ValueError(f"Raw OHLCV must contain '{col}' column.")
    df = _ensure_ts_index(df_raw)
    piv = df.pivot(columns="symbol", values="close")
    return piv


def _build_features_for_pair(piv: pd.DataFrame, a: str, b: str,
                             beta_window: int, z_window: int) -> Optional[pd.DataFrame]:
    if a not in piv.columns or b not in piv.columns:
        return None
    df = piv[[a, b]].dropna().rename(columns={a: "pa", b: "pb"})
    if len(df) < max(beta_window, z_window) + 5:
        return None

    beta, alpha = _rolling_beta_alpha(df["pa"], df["pb"], beta_window)
    spread = df["pa"] - (beta * df["pb"] + alpha)
    z = _zscore(spread, z_window)
    out = pd.DataFrame(
        {
            "ts": df.index,
            "a": df["pa"].values,
            "b": df["pb"].values,
            "beta": beta.values,
            "alpha": alpha.values,
            "spread": spread.values,
            "z": z.values,
        }
    ).dropna().reset_index(drop=True)
    return out


# ----------------------------- Public entry ----------------------------- #

def c(
    *,
    # file-based variant
    pairs_json: Optional[Union[str, Path]] = None,
    raw_parquet: Optional[Union[str, Path]] = None,
    out_dir: Optional[Union[str, Path]] = None,
    # in-memory variant
    raw_df: Optional[pd.DataFrame] = None,
    pairs: Optional[Iterable[Tuple[str, str]]] = None,
    # common params
    beta_window: int = 1000,
    z_window: int = 300,
):
    """
    Either:
      - file-based: pass pairs_json + raw_parquet + out_dir -> returns List[str] of pair keys (and writes parquet)
      - in-memory: pass raw_df + pairs -> returns Dict[str, DataFrame] (caller writes parquet)
    """
    is_file_mode = pairs_json is not None or raw_parquet is not None or out_dir is not None
    is_mem_mode = raw_df is not None or pairs is not None

    if is_file_mode and is_mem_mode:
        raise ValueError("Pass either file-based args (pairs_json, raw_parquet, out_dir) OR in-memory args (raw_df, pairs).")

    if is_file_mode:
        if pairs_json is None or raw_parquet is None or out_dir is None:
            raise ValueError("File-based mode requires pairs_json, raw_parquet, out_dir.")
        pairs_list = _pairs_from_any(pairs_json)
        if not pairs_list:
            raise ValueError(f"No pairs parsed from {pairs_json}")
        raw_path = Path(raw_parquet)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw parquet not found: {raw_path}")
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        raw = pd.read_parquet(raw_path)
        piv = _pivot_close(raw)

        produced: List[str] = []
        for (a, b) in tqdm(pairs_list, desc="Features"):
            df = _build_features_for_pair(piv, a, b, beta_window, z_window)
            if df is None or df.empty:
                continue
            pair_key = f"{a}__{b}"
            safe = pair_key.replace("/", "_")
            pair_dir = out_path / safe
            pair_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pair_dir / "features.parquet", index=False)
            produced.append(pair_key)

        return produced

    # in-memory
    if raw_df is None or pairs is None:
        raise ValueError("In-memory mode requires raw_df and pairs.")
    piv = _pivot_close(raw_df)
    result: Dict[str, pd.DataFrame] = {}
    for (a, b) in tqdm(list(pairs), desc="Features(mem)"):
        df = _build_features_for_pair(piv, a, b, beta_window, z_window)
        if df is None or df.empty:
            continue
        pair_key = f"{a}__{b}"
        result[pair_key] = df
    return result
