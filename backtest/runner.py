from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_index(df: pd.DataFrame) -> pd.DataFrame:
    if "ts" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["ts"], utc=True, errors="coerce"))
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "ts"
    return df


def _load_features(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return _safe_index(df)


def _load_oof(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return _safe_index(df)


def _equity_metrics(equity: pd.Series) -> Dict[str, float]:
    rets = equity.pct_change().dropna()
    sharpe = float(np.sqrt(252*24*12) * rets.mean() / (rets.std() + 1e-12)) if len(rets) else float("nan")
    dd = (equity / equity.cummax() - 1.0).min() if len(equity) else float("nan")
    return {"sharpe": sharpe, "maxdd": float(abs(dd)) if pd.notna(dd) else float("nan")}


def _run_pair_backtest(
    feat: pd.DataFrame,
    oof: Optional[pd.DataFrame],
    proba_threshold: float,
    fee_rate: float
) -> Dict[str, Any]:
    """
    Backtest on spread using either OOF probabilities (preferred) or z-rule fallback.
    Strategy:
      - position is in "spread" (long=+1, short=-1).
      - If OOF present and y is 'revert_direction' (encoded {-1,+1}->{0,1} during training),
        we interpret proba as P(long spread). pos = +1 if proba>thr else -1 if proba<(1-thr), else 0.
      - Else fallback: z-rule mean reversion: pos = -1 if z>z_th; +1 if z<-z_th; else 0.
    PnL is pos_{t-1} * Δspread_t - fee * |pos_t - pos_{t-1}|.
    """
    idx = feat.index
    spread = feat["spread"].astype(float)
    z = feat["z"].astype(float)
    # derive positions
    pos = pd.Series(0.0, index=idx)
    if oof is not None and "proba" in oof.columns:
        oof = oof.reindex(idx).ffill().bfill()
        p = oof["proba"].clip(0, 1)
        up = p > proba_threshold
        dn = p < (1.0 - proba_threshold)
        pos[up] = +1.0
        pos[dn] = -1.0
        # neutral otherwise
    else:
        # z-rule fallback with default threshold 1.5
        z_th = 1.5
        pos[z > z_th] = -1.0
        pos[z < -z_th] = +1.0

    # PnL on spread changes
    dS = spread.diff().fillna(0.0)
    pos_prev = pos.shift(1).fillna(0.0)
    pnl = pos_prev * (-dS)  # revert: profit when spread moves toward zero
    turnover = (pos - pos_prev).abs()
    costs = turnover * fee_rate
    equity = (1.0 + pnl - costs).cumprod()
    metrics = _equity_metrics(equity)
    res = {
        "equity_last": float(equity.iloc[-1]) if len(equity) else 1.0,
        "sharpe": metrics["sharpe"],
        "maxdd": metrics["maxdd"],
    }
    return res


def run_backtest(
    features_dir: str,
    datasets_dir: str,
    models_dir: str,
    out_dir: str,
    proba_threshold: float = 0.55,
    fee_rate: float = 0.0005
) -> Dict[str, Any]:
    """
    Iterate over features and available OOF predictions, evaluate per-pair,
    write a summary JSON, and per-pair metrics.
    Expected files:
      - features: data/features/pairs/<A__B>/features.parquet
      - OOF (optional): data/models/pairs/<A__B>/oof.parquet with columns [proba]
    """
    features_dir = Path(features_dir)
    models_dir = Path(models_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"pairs": {}, "proba_threshold": proba_threshold, "fee_rate": fee_rate}
    for p in sorted(features_dir.glob("*/*")):
        if p.name != "features.parquet":
            continue
        pair_key = p.parent.name.replace("_", "/").replace("//", "/").replace("/USDT", "/USDT")  # best-effort
        feat = _load_features(p)
        oof_path = models_dir / "pairs" / p.parent.name / "oof.parquet"
        oof = _load_oof(oof_path)
        res = _run_pair_backtest(feat, oof, proba_threshold, fee_rate)
        summary["pairs"][pair_key] = {"metrics": res, "oof_used": bool(oof is not None)}
    return summary
