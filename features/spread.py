from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional
import os

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from tqdm import tqdm

DEBUG = os.getenv("FEATURES_DEBUG", "0").lower() in ("1", "true", "yes", "on")

@dataclass
class SpreadConfig:
    beta_window: int = 300
    z_window: int = 300

def _log(msg: str) -> None:
    if DEBUG:
        print(f"[features/spread] {msg}")

def _normalize_symbol(s: str) -> str:
    s = str(s).strip()
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    return s


def _ensure_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    for cand in ["ts", "timestamp", "time", "date", "datetime"]:
        if cand in df.columns:
            if cand != "ts":
                df = df.rename(columns={cand: "ts"})
            break
    else:
        raise ValueError("Raw dataframe must contain a time column (ts/timestamp/time/date/datetime).")

    for cand in ["close", "Close", "CLOSE", "c"]:
        if cand in df.columns:
            if cand != "close":
                df = df.rename(columns={cand: "close"})
            break
    else:
        raise ValueError("Raw dataframe must contain a close column (close/Close/c).")

    for cand in ["symbol", "ticker", "pair", "asset"]:
        if cand in df.columns:
            if cand != "symbol":
                df = df.rename(columns={cand: "symbol"})
            break
    else:
        raise ValueError("Raw dataframe must contain a symbol column (symbol/ticker/pair/asset).")

    if not is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        if is_datetime64tz_dtype(df["ts"]):
            try:
                df["ts"] = df["ts"].dt.tz_convert("UTC")
            except Exception:
                df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        else:
            df["ts"] = df["ts"].dt.tz_localize("UTC")

    df = df.dropna(subset=["ts", "symbol", "close"])
    df["symbol"] = df["symbol"].astype(str).map(_normalize_symbol)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float64")
    df = df.dropna(subset=["close"])
    df = df.sort_values(["symbol", "ts"]).reset_index(drop=True)

    _log(f"rows={len(df)}, symbols={sorted(df['symbol'].unique().tolist())[:10]}")
    return df

def _rolling_beta_alpha(y: pd.Series, x: pd.Series, win: int) -> Tuple[pd.Series, pd.Series]:
    x_mean = x.rolling(win).mean()
    y_mean = y.rolling(win).mean()
    cov = (x * y).rolling(win).mean() - x_mean * y_mean
    var = x.rolling(win).var()
    beta = cov / var.replace(0.0, np.nan)
    alpha = y_mean - beta * x_mean
    return beta, alpha


def _pair_key(a: str, b: str) -> str:
    return f"{a.replace('/', '_')}__{b.replace('/', '_')}"


def compute_features_for_pairs(
    raw_df: pd.DataFrame,
    pairs: Iterable[Tuple[str, str]],
    beta_window: int = 300,
    z_window: int = 300,
) -> Dict[str, pd.DataFrame]:

    if beta_window <= 1 or z_window <= 1:
        raise ValueError("beta_window and z_window must be > 1")

    df = _ensure_schema(raw_df)

    norm_pairs: List[Tuple[str, str]] = [(_normalize_symbol(a), _normalize_symbol(b)) for a, b in pairs]

    piv = df.pivot(index="ts", columns="symbol", values="close").sort_index()
    existing_cols = set(piv.columns)

    out: Dict[str, pd.DataFrame] = {}
    todo = list(norm_pairs)

    for a, b in tqdm(todo, desc="Features(mem)"):
        if a not in existing_cols or b not in existing_cols:
            _log(f"skip {a}__{b}: missing ({a in existing_cols=}, {b in existing_cols=})")
            continue

        px = piv[[a, b]].dropna().sort_index()
        nmin = max(beta_window, z_window) + 10
        if len(px) < nmin:
            _log(f"skip {a}__{b}: too few rows {len(px)} < {nmin}")
            continue

        a_close = px[a].astype("float64")
        b_close = px[b].astype("float64")

        beta, alpha = _rolling_beta_alpha(a_close, b_close, int(beta_window))
        spread = a_close - (beta * b_close + alpha)

        mean_sp = spread.rolling(int(z_window)).mean()
        std_sp = spread.rolling(int(z_window)).std()
        z = (spread - mean_sp) / std_sp.replace(0.0, np.nan)

        res = pd.DataFrame(
            {
                "ts": px.index,
                "a_close": a_close,
                "b_close": b_close,
                "beta": beta,
                "alpha": alpha,
                "spread": spread,
                "z": z,
            }
        ).dropna().reset_index(drop=True)

        if len(res) == 0:
            _log(f"skip {a}__{b}: all rows dropped after rolling windows")
            continue

        out[_pair_key(a, b)] = res
        _log(f"ok {a}__{b}: rows={len(res)}")

    if not out:
        _log("no pairs produced — check symbol names and data length")

    return out