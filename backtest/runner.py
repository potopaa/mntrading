#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numerically stable backtester for mntrading.

Inputs
------
- features_dir: data/features/pairs/<A__B>/features.parquet
  Required columns: ['ts','a_close','b_close','beta','alpha','spread','z'] (others are ignored)
- models_dir (optional, for OOF): data/models/pairs/<A__B>__oof.parquet
  Required columns: ['ts','y','proba']
- datasets_dir: not used directly here, but kept for signature parity

Config
------
- proba_threshold: threshold for OOF probabilities (signals_from='oof')
- fee_rate: proportional fee per unit notional when position changes
- bars_per_year: annualization factor for Sharpe; default assumes 5m bars

Outputs
-------
- summary dict and JSON at out_dir/_summary.json:
    {
      "proba_threshold": 0.55,
      "fee_rate": 0.0005,
      "bars_per_year": 105120,
      "pairs": {
        "AAA/USDT__BBB/USDT": {
          "oof_used": true,
          "metrics": {
            "sharpe": 0.83,
            "maxdd": 0.12,
            "equity_last": 1.17,
            "n_bars": 1234,
            "n_trades": 56
          }
        },
        ...
      }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _pair_key_to_path(pair_key: str) -> str:
    # "AAA/USDT__BBB/USDT" -> "AAA_USDT__BBB_USDT"
    return pair_key.replace("/", "_")


def _load_features(features_dir: Path, pair_key: str) -> Optional[pd.DataFrame]:
    p = features_dir / _pair_key_to_path(pair_key) / "features.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    # Normalize required columns
    cols = {c.lower(): c for c in df.columns}
    # accept legacy names a,b or a_close,b_close
    if "a_close" not in cols and "a" in cols:
        df = df.rename(columns={cols["a"]: "a_close"})
    if "b_close" not in cols and "b" in cols:
        df = df.rename(columns={cols["b"]: "b_close"})
    # ensure ts
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("features parquet must have 'ts' column or DatetimeIndex")
    # keep only needed
    need = ["ts", "a_close", "b_close", "beta", "alpha", "spread", "z"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"features for {pair_key} missing columns: {missing}")
    # sort by ts and cast to float64
    df = df.sort_values("ts").reset_index(drop=True)
    for c in ["a_close", "b_close", "beta", "alpha", "spread", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def _load_oof(models_dir: Path, pair_key: str) -> Optional[pd.DataFrame]:
    p = models_dir / "pairs" / f"{_pair_key_to_path(pair_key)}__oof.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if "ts" not in df.columns or "proba" not in df.columns:
        return None
    df = df[["ts", "proba"]].copy().sort_values("ts").reset_index(drop=True)
    df["proba"] = pd.to_numeric(df["proba"], errors="coerce").astype("float64").clip(1e-6, 1 - 1e-6)
    return df


def _sharpe_from_logrets(log_r: np.ndarray, bars_per_year: int) -> Optional[float]:
    # convert log returns to simple returns for sharpe approximation
    r = np.expm1(log_r)  # simple return per bar
    r = r[np.isfinite(r)]
    if r.size < 2:
        return None
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return None
    return float(np.sqrt(bars_per_year) * (mu / sd))


def _max_drawdown_from_equity(equity: np.ndarray) -> Optional[float]:
    if equity.size < 2:
        return None
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.where(peak == 0, 1.0, peak)
    dd = dd[np.isfinite(dd)]
    if dd.size == 0:
        return 0.0
    return float(np.max(dd))


def _compute_pair_backtest(
    feat: pd.DataFrame,
    oof: Optional[pd.DataFrame],
    proba_threshold: float,
    fee_rate: float,
    bars_per_year: int = 105120,  # 5m bars
) -> Dict:
    """
    Core backtest:
    - log-returns of (a, b)
    - hedge by beta (use beta_{t-1} to avoid look-ahead)
    - signal direction = -sign(z_{t-1}) when |z_{t-1}| >= z_entry (z_entry = 1.5)
    - if OOF provided: position magnitude = normalized (proba - threshold)
    - fees on position change (per-leg), proportional to fee_rate
    """
    df = feat.copy()
    df = df.sort_values("ts").reset_index(drop=True)

    # log prices
    df["la"] = np.log(df["a_close"].clip(1e-12))
    df["lb"] = np.log(df["b_close"].clip(1e-12))
    # log returns
    df["ra"] = df["la"].diff().fillna(0.0).astype("float64")
    df["rb"] = df["lb"].diff().fillna(0.0).astype("float64")

    # use beta shifted by 1 bar (no look-ahead)
    df["beta_lag"] = df["beta"].shift(1).replace([np.inf, -np.inf], np.nan).fillna(method="ffill")
    df["beta_lag"] = df["beta_lag"].fillna(1.0).astype("float64")

    # signal direction from z (mean-reversion): if z>entry -> short spread => sign=-1; if z<-entry -> long spread => sign=+1
    z_entry = 1.5
    z_lag = df["z"].shift(1)
    sign = (-np.sign(z_lag)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    active = (z_lag.abs() >= z_entry).astype("float64")

    # oof-based magnitude (0..1)
    if oof is not None and len(oof) > 0:
        df = df.merge(oof, on="ts", how="left")
        proba = df["proba"].astype("float64")
        mag = ((proba - float(proba_threshold)) / max(1e-6, 1.0 - float(proba_threshold))).clip(0.0, 1.0).fillna(0.0)
        oof_used = True
    else:
        mag = active.copy()
        oof_used = False

    # final position per leg (units of "notional", hedge with beta)
    pos_a = (sign * mag * active).astype("float64")
    pos_b = (-sign * mag * active * df["beta_lag"]).astype("float64")

    # turnover -> fees
    pos_a_lag = pos_a.shift(1).fillna(0.0)
    pos_b_lag = pos_b.shift(1).fillna(0.0)
    dpos = (pos_a - pos_a_lag).abs() + (pos_b - pos_b_lag).abs()
    fees = (fee_rate * dpos).astype("float64")

    # portfolio log-return per bar (approx): w_a * r_a + w_b * r_b - fees
    log_r = (pos_a * df["ra"] + pos_b * df["rb"] - fees).fillna(0.0).astype("float64").values

    # cumulate in log-space to avoid overflow
    cum_log = np.cumsum(log_r, dtype="float64")
    equity = np.exp(cum_log)  # starts at 1

    # metrics
    sharpe = _sharpe_from_logrets(log_r, bars_per_year)
    maxdd = _max_drawdown_from_equity(equity)
    equity_last = float(equity[-1]) if equity.size else 1.0

    # count trades ~ times we switch from 0 to non-zero pos_a
    trades = int(np.sum((pos_a_lag == 0.0) & (pos_a != 0.0)))

    metrics = {
        "sharpe": float(sharpe) if sharpe is not None else None,
        "maxdd": float(maxdd) if maxdd is not None else None,
        "equity_last": float(equity_last),
        "n_bars": int(len(df)),
        "n_trades": trades,
    }
    return {"metrics": metrics, "oof_used": bool(oof_used)}


def run_backtest(
    features_dir: str,
    datasets_dir: str,  # kept for signature compatibility
    models_dir: str,
    out_dir: str,
    proba_threshold: float = 0.55,
    fee_rate: float = 0.0005,
    bars_per_year: int = 105120,  # 5m bars ~= 365 * 24 * 12
) -> Dict:
    features_dir_p = Path(features_dir)
    models_dir_p = Path(models_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # Discover pairs from features manifest
    manifest_p = features_dir_p / "_manifest.json"
    if not manifest_p.exists():
        raise SystemExit(f"features manifest not found: {manifest_p}")
    man = json.loads(manifest_p.read_text(encoding="utf-8"))
    items = man.get("items") or []
    pairs = [str(it.get("pair")) for it in items if it.get("pair")]
    pairs = sorted(set(pairs))

    summary = {
        "proba_threshold": float(proba_threshold),
        "fee_rate": float(fee_rate),
        "bars_per_year": int(bars_per_year),
        "pairs": {}
    }

    for pk in pairs:
        try:
            feat = _load_features(features_dir_p, pk)
            if feat is None or len(feat) < 5:
                continue
            oof = _load_oof(models_dir_p, pk)
            res = _compute_pair_backtest(
                feat=feat,
                oof=oof,
                proba_threshold=float(proba_threshold),
                fee_rate=float(fee_rate),
                bars_per_year=int(bars_per_year),
            )
            summary["pairs"][pk] = res
        except Exception as e:
            summary["pairs"][pk] = {"error": str(e)}

    out_p = out_dir_p / "_summary.json"
    out_p.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[backtest] summary -> {out_p}")
    return summary
