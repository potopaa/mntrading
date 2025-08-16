# -*- coding: utf-8 -*-
"""
Build supervised datasets (X, y) for pairs from features parquet files.

Key features:
- Robust manifest reader: accepts path (str/Path), dict, or list of entries.
- Safe datetime index handling: sort, drop duplicate timestamps.
- Lagged features (to avoid lookahead).
- Forecast horizon: shift labels forward.
- Label type: "z_threshold" (binary), configurable threshold.

Outputs:
- Per-pair dataset parquet files in data/datasets/pairs
- Manifest at data/datasets/_manifest.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ==========================
# Utils: IO & Index handling
# ==========================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a tz-aware datetime index (UTC). Handle common cases:
    - 'ts' column with ISO or epoch
    - already indexed by datetime (naive or tz-aware)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "ts" in df.columns:
            df = df.set_index("ts")

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _read_parquet_features(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _to_datetime_index(df)
    return df


# ==========================
# Manifest handling (flex)
# ==========================

def _pair_from_path(p: Union[str, Path]) -> str:
    """
    Infer pair like 'BNB/USDT__ETH/USDT' from filename 'BNB_USDT__ETH_USDT.parquet'
    """
    stem = Path(p).stem
    if "__" in stem:
        a, b = stem.split("__", 1)

        def norm(x: str) -> str:
            parts = x.split("_", 1)
            return "/".join(parts) if len(parts) == 2 else x

        return f"{norm(a)}__{norm(b)}"
    return stem


def _normalize_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize one manifest entry to { "pair": str, "path": str, "features": Optional[List[str]] }
    Accepts various keys: path/data_path/file/parquet, pair/name/key.
    """
    p = e.get("path") or e.get("data_path") or e.get("file") or e.get("parquet")
    if not p:
        raise ValueError(f"Manifest entry has no path-like key: {e}")
    pair = e.get("pair") or e.get("name") or e.get("key") or _pair_from_path(p)
    features = e.get("features")
    return {"pair": pair, "path": str(p), "features": features}


def _extract_entries_from_obj(obj: Any) -> List[Dict[str, Any]]:
    """
    Find a list of entries within a loaded manifest object.
    We accept:
      - {"items": [ ... ]}  (preferred)
      - {"pairs": [ ... ]}  (if entries have path)
      - a bare list [ ... ]
    """
    if isinstance(obj, list):
        entries = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("items"), list):
            entries = obj["items"]
        elif isinstance(obj.get("pairs"), list):
            entries = obj["pairs"]
        else:
            candidates = [v for v in obj.values() if isinstance(v, list)]
            if not candidates:
                raise ValueError("Unsupported manifest dict structure; no list field found.")
            entries = candidates[0]
    else:
        raise TypeError(f"Unsupported manifest type: {type(obj)}")

    normed = []
    for e in entries:
        if not isinstance(e, dict):
            e = {"path": e}
        normed.append(_normalize_entry(e))
    return normed


def _read_pairs_from_manifest(manifest: Union[str, Path, Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
    """
    Accept path|dict|list and return normalized entries with fields: pair, path, (optional) features.
    """
    if isinstance(manifest, (str, Path)):
        with open(manifest, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return _extract_entries_from_obj(obj)
    elif isinstance(manifest, (dict, list)):
        return _extract_entries_from_obj(manifest)
    else:
        raise TypeError(f"Unsupported manifest type: {type(manifest)}")


# ==========================
# Dataset building
# ==========================

@dataclass
class DatasetBuildConfig:
    label_type: str = "z_threshold"
    zscore_threshold: float = 1.5
    lag_features: int = 1
    horizon: int = 0
    out_dir: Path = Path("data/datasets/pairs")


def _build_single_dataset(
    features_path: Union[str, Path],
    cfg: DatasetBuildConfig,
    features_whitelist: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Build a single (X,y) dataset from features parquet.
    - features_whitelist: if provided, restrict feature columns (e.g., ["pa","pb","beta","alpha","spread","z"])
    """
    df = _read_parquet_features(features_path)

    # Feature columns
    cols = list(df.columns)
    if features_whitelist:
        feat_cols = [c for c in cols if c in set(features_whitelist)]
    else:
        ignore = {"y"}
        feat_cols = [c for c in cols if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]

    # Lag features to prevent leakage
    X = df[feat_cols].copy()
    if cfg.lag_features and cfg.lag_features > 0:
        X = X.shift(cfg.lag_features)

    # Labels
    if cfg.label_type == "z_threshold":
        if "z" not in df.columns:
            raise ValueError(f"'z' column not found in {features_path}, cannot build z_threshold label.")
        y = (df["z"].abs() >= float(cfg.zscore_threshold)).astype(int)
    else:
        raise ValueError(f"Unsupported label_type: {cfg.label_type}")

    # Forecast horizon: move target to the future
    if cfg.horizon and cfg.horizon > 0:
        y = y.shift(-cfg.horizon)

    dataset = pd.concat({"y": y}, axis=1).join(X, how="outer")
    dataset = dataset.dropna(axis=0, how="any")
    dataset = dataset.sort_index()
    dataset = dataset[~dataset.index.duplicated(keep="last")]
    return dataset


def build_datasets_for_manifest(
    manifest_path_or_obj: Union[str, Path, Dict[str, Any], List[Any]],
    label_type: str = "z_threshold",
    zscore_threshold: float = 1.5,
    lag_features: int = 1,
    horizon: int = 0,
    out_dir: Union[str, Path] = Path("data/datasets/pairs"),
    features: Optional[List[str]] = None,   # <<< NEW: global whitelist to stay compatible with main.py
    **_,
) -> List[Dict[str, Any]]:
    """
    Build datasets for each pair listed in the features manifest (or list/dict of entries).
    Returns a list of manifest entries for produced datasets:
      [{"pair": "...", "path": "data/datasets/pairs/XXX.parquet", "features": [...]}]
    Also writes a consolidated manifest: data/datasets/_manifest.json
    """
    entries = _read_pairs_from_manifest(manifest_path_or_obj)

    cfg = DatasetBuildConfig(
        label_type=label_type,
        zscore_threshold=zscore_threshold,
        lag_features=lag_features,
        horizon=horizon,
        out_dir=Path(out_dir),
    )

    _ensure_dir(cfg.out_dir)
    produced: List[Dict[str, Any]] = []

    for e in entries:
        pair = e["pair"]
        fpath = e["path"]

        # Per-entry features (from features manifest) OR global whitelist from function arg
        per_entry_feats = e.get("features")
        effective_feats = features if features else per_entry_feats

        ds = _build_single_dataset(fpath, cfg, features_whitelist=effective_feats)

        stem = Path(fpath).stem
        out_path = cfg.out_dir / f"{stem}.parquet"
        ds.to_parquet(out_path, index=True)

        produced.append({"pair": pair, "path": str(out_path), "features": [c for c in ds.columns if c != "y"]})

    # Write datasets manifest
    datasets_root = cfg.out_dir.parent
    manifest_out = datasets_root / "_manifest.json"
    manifest_obj = {"items": produced}
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, ensure_ascii=True, indent=2)

    return produced
