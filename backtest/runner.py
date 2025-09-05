
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import json
import math

import numpy as np
import pandas as pd


def _read_features_manifest(features_dir: Path) -> List[Dict]:
    candidates = [
        features_dir.parent / "_manifest.json",      # data/features/_manifest.json
        features_dir / "_manifest.json",             # data/features/pairs/_manifest.json
    ]
    for p in candidates:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            items = obj.get("items") or []
            out = []
            for it in items:
                pair = it.get("pair") or it.get("name") or it.get("key")
                path = it.get("path") or it.get("parquet") or it.get("file")
                if pair and path:
                    out.append({"pair": str(pair), "path": str(path)})
            return out
    return []


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    # ts
    ts_col = None
    for c in ["ts", "timestamp", "time", "date", "datetime"]:
        if c in df.columns:
            ts_col = c; break
    if ts_col is None:
        raise ValueError("features parquet must contain a time column (ts/timestamp/...).")
    if ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    # z
    z_col = None
    for c in ["z", "zscore", "z_score", "spread_z"]:
        if c in df.columns:
            z_col = c; break
    if z_col is None:
        raise ValueError("features parquet must contain z-score column (z/zscore/z_score/spread_z).")
    if z_col != "z":
        df = df.rename(columns={z_col: "z"})
    df["z"] = pd.to_numeric(df["z"], errors="coerce").astype("float64")

    # spread (optional)
    if "spread" not in df.columns:
        for c in ["spr", "residual"]:
            if c in df.columns:
                df = df.rename(columns={c: "spread"}); break
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").astype("float64")

    df = df.dropna(subset=["ts", "z"]).sort_values("ts").reset_index(drop=True)
    return df


def _compute_pair_pnl(df: pd.DataFrame, fee_rate: float, z_entry: float = 1.5, z_exit: float = 0.5) -> Dict[str, float]:
    df = df.copy()
    if "spread" in df.columns:
        serie = df["spread"].astype("float64")
    else:
        serie = df["z"].astype("float64")

    # positions
    pos = np.zeros(len(df), dtype="int8")
    for i in range(len(df)):
        z = df.at[i, "z"]
        p = pos[i-1] if i > 0 else 0
        if p == 0:
            if z >= z_entry:
                p = -1
            elif z <= -z_entry:
                p = +1
        else:
            if abs(z) <= z_exit:
                p = 0
        pos[i] = p

    # returns
    dS = serie.diff().fillna(0.0).astype("float64")
    ret = (- (pd.Series(pos).shift(1).fillna(0.0)) * dS).astype("float64")

    # costs on position changes
    dpos = pd.Series(pos).diff().fillna(0.0).abs()
    cost = (fee_rate * 2.0) * dpos
    ret = ret - cost

    # annualization factor for 5m bars (approx): 12 bars/hour * 24 * 365
    ann = math.sqrt(12 * 24 * 365)

    mu = float(ret.mean())
    sd = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0
    sharpe = (mu / sd) * ann if sd > 0 else 0.0

    equity = ret.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    maxdd = float(dd.min()) if len(dd) else 0.0

    return {"sharpe": sharpe, "maxdd": abs(maxdd), "trades": int(dpos.sum() / 2)}


def run_backtest(
    features_dir: str,
    datasets_dir: str,
    models_dir: str,
    out_dir: str,
    *,
    signals_from: str = "auto",
    proba_threshold: float = 0.55,
    fee_rate: float = 0.0005,
) -> Dict:
    fdir = Path(features_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    items = _read_features_manifest(fdir if fdir.name == "pairs" else fdir / "pairs")
    if not items:
        items = _read_features_manifest(fdir)

    if not items:
        raise RuntimeError(f"No features manifest found under {features_dir}")

    pairs_metrics: Dict[str, Dict[str, float]] = {}
    processed = 0
    for it in items:
        pair = str(it["pair"])
        p = Path(it["path"])
        if not p.exists():
            guess = Path("data/features/pairs") / pair.replace("/", "_") / "features.parquet"
            if guess.exists():
                p = guess
            else:
                continue
        try:
            df = pd.read_parquet(p)
            df = _ensure_schema(df)
            met = _compute_pair_pnl(df, fee_rate=fee_rate, z_entry=1.5, z_exit=0.5)
            pairs_metrics[pair] = {"metrics": met}
            processed += 1
        except Exception as e:
            continue

    summary = {"pairs": pairs_metrics, "processed": processed}
    (out / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
