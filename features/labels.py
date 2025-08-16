# -*- coding: utf-8 -*-
"""
Build supervised datasets (X, y) for pairs from features parquet files.

- Robust manifest reader: accepts path (str/Path), dict, or list of entries.
- Safe datetime index handling: sort, drop duplicate timestamps.
- Lagged features (to avoid lookahead).
- Forecast horizon: shift labels forward.
- Label types:
    * "z_threshold": y = 1 if abs(z.shift(-H)) > z_th else 0
    * "revert_direction":
        y_raw ∈ {-1, 0, +1}:
            if z > z_th  -> y_raw = -1  (short spread; expect mean reversion down)
            if z < -z_th -> y_raw = +1  (long spread;  expect mean reversion up)
            else         -> y_raw = 0   (no signal)
        We drop y_raw==0 and convert {-1,+1}→{0,1} for binary classifiers.
- Outputs:
    * Per-pair dataset parquet files in data/datasets/pairs
    * Manifest at data/datasets/_manifest.json (list of items with path)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

import json
import pandas as pd
import numpy as np


@dataclass
class DatasetBuildConfig:
    label_type: str = "z_threshold"
    zscore_threshold: float = 1.5
    lag_features: int = 1
    horizon: int = 0  # shift label into the future by H bars


def _to_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure we have a datetime index named 'ts'
    if "ts" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["ts"], utc=True, errors="coerce"))
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "ts"
    return df


def _read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return _to_dtindex(df)


def _read_manifest(man: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(man, (str, Path)):
        obj = json.loads(Path(man).read_text(encoding="utf-8"))
    else:
        obj = man
    if isinstance(obj, dict):
        if "items" in obj:
            items = obj["items"]
        elif "pairs" in obj:
            # legacy format: list of pair keys without paths
            # try to infer paths under the same directory
            base = Path(man).parent if isinstance(man, (str, Path)) else Path("data/features/pairs")
            items = []
            for pk in obj["pairs"]:
                safe = pk.replace("/", "_")
                items.append({"pair": pk, "path": str((base / safe / "features.parquet").resolve())})
        else:
            items = []
    elif isinstance(obj, list):
        items = obj
    else:
        items = []
    # normalize fields
    norm = []
    for it in items:
        pair = it.get("pair") or f"{it.get('a','')}__{it.get('b','')}"
        path = it.get("path")
        if pair and path:
            norm.append({"pair": pair, "path": path})
    return norm


def build_dataset_from_features(
    df: pd.DataFrame,
    cfg: DatasetBuildConfig
) -> pd.DataFrame:
    """
    Given a features DataFrame with columns ['a','b','beta','alpha','spread','z'],
    produce a supervised dataset with lagged features and labels.
    """
    df = _to_dtindex(df)
    cols = ["spread", "z", "beta", "alpha", "a", "b"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not in features")
    X = df[["z", "spread", "beta", "alpha"]].copy()
    # add lags
    for k in range(1, int(cfg.lag_features) + 1):
        X[f"z_lag{k}"] = X["z"].shift(k)
        X[f"spread_lag{k}"] = X["spread"].shift(k)
    # labels
    if cfg.label_type == "z_threshold":
        y = (df["z"].shift(-cfg.horizon).abs() > float(cfg.zscore_threshold)).astype("int8")
        y.name = "y"
        out = pd.concat([X, y], axis=1).dropna()
    elif cfg.label_type == "revert_direction":
        z = df["z"]
        y_raw = pd.Series(0, index=z.index, dtype="int8")
        y_raw[z > float(cfg.zscore_threshold)] = -1
        y_raw[z < -float(cfg.zscore_threshold)] = +1
        # shift into the future if horizon>0 (we bet now on direction over next H bars)
        if cfg.horizon:
            y_raw = y_raw.shift(-cfg.horizon)
        # keep only signals (drop 0)
        mask = y_raw != 0
        X = X.loc[mask]
        y = y_raw.loc[mask].map({-1:0, +1:1}).astype("int8")
        y.name = "y"
        out = pd.concat([X, y], axis=1).dropna()
    else:
        raise ValueError(f"Unknown label_type: {cfg.label_type}")
    out.index.name = "ts"
    return out.reset_index()


def build_datasets_for_manifest(
    features_manifest: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
    out_dir: Union[str, Path],
    cfg: Optional[DatasetBuildConfig] = None
) -> Dict[str, Any]:
    """
    For each features parquet in manifest, build a dataset parquet and manifest.
    """
    cfg = cfg or DatasetBuildConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = _read_manifest(features_manifest)
    built_items: List[Dict[str, Any]] = []
    for it in items:
        pair = it["pair"]
        path = it["path"]
        try:
            fdf = _read_parquet(path)
            ds = build_dataset_from_features(fdf, cfg)
            safe = pair.replace("/", "_")
            p_out = out_dir / f"{safe}__ds.parquet"
            ds.to_parquet(p_out, index=False)
            built_items.append({"pair": pair, "path": str(p_out.resolve()), "rows": int(len(ds))})
        except Exception as e:
            # skip pair on error
            continue
    man = {"items": built_items, "label_type": cfg.label_type, "zscore_threshold": cfg.zscore_threshold,
           "lag_features": cfg.lag_features, "horizon": cfg.horizon}
    # write datasets manifest one level up (data/datasets/_manifest.json)
    manifest_path = out_dir.parent / "_manifest.json"
    manifest_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return man
