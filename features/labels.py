from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os
from pathlib import Path

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

# ----- config -----

@dataclass
class DatasetBuildConfig:
    label_type: str = "z_threshold"   # "z_threshold" / "revert_direction"
    zscore_threshold: float = 1.5
    lag_features: int = 10
    horizon: int = 3

DATASET_MIN_ROWS = int(os.getenv("DATASET_MIN_ROWS", "200"))
DATASET_DEBUG = os.getenv("DATASET_DEBUG", "0").lower() in ("1", "true", "yes", "on")

def _d(msg: str) -> None:
    if DATASET_DEBUG:
        print(f"[labels] {msg}")

# ----- helpers -----

def _norm_pair_key(pair: str) -> str:
    return pair.replace("/", "_")

def _read_manifest(features_manifest: str | Path) -> List[Dict]:
    p = Path(features_manifest)
    if not p.exists():
        raise FileNotFoundError(f"Features manifest not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    items = obj.get("items") or []
    out = []
    for it in items:
        pair = it.get("pair") or it.get("name") or it.get("key")
        path = it.get("path") or it.get("parquet") or it.get("file")
        if pair and path:
            out.append({"pair": str(pair), "path": str(path)})
    return out

def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    # ts
    ts_col = None
    for c in ["ts","timestamp","time","date","datetime"]:
        if c in df.columns:
            ts_col = c; break
    if ts_col is None:
        raise ValueError("Features frame must contain time column (ts/timestamp/...).")
    if ts_col != "ts":
        df = df.rename(columns={ts_col: "ts"})

    # z-score
    z_col = None
    for c in ["z","zscore","z_score","spread_z"]:
        if c in df.columns:
            z_col = c; break
    if z_col is None:
        raise ValueError("Features frame must contain z-score column (z/zscore/z_score/spread_z).")
    if z_col != "z":
        df = df.rename(columns={z_col: "z"})

    # optional: spread, beta, alpha
    if "spread" not in df.columns:
        for c in ["spr","residual"]:
            if c in df.columns:
                df = df.rename(columns={c: "spread"}); break
    if "beta" not in df.columns:
        for c in ["coef","slope"]:
            if c in df.columns:
                df = df.rename(columns={c: "beta"}); break
    if "alpha" not in df.columns:
        for c in ["intercept","const"]:
            if c in df.columns:
                df = df.rename(columns={c: "alpha"}); break

    # dtypes
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

    df = df.dropna(subset=["ts", "z"]).sort_values("ts").reset_index(drop=True)
    df["z"] = pd.to_numeric(df["z"], errors="coerce").astype("float64")
    if "spread" in df.columns:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").astype("float64")
    if "beta" in df.columns:
        df["beta"] = pd.to_numeric(df["beta"], errors="coerce").astype("float64")
    if "alpha" in df.columns:
        df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce").astype("float64")
    return df

def _make_lags(s: pd.Series, n: int, prefix: str) -> pd.DataFrame:
    out = {}
    for i in range(1, int(n)+1):
        out[f"{prefix}_lag{i}"] = s.shift(i)
    return pd.DataFrame(out)

# ----- targets -----

def _target_z_threshold(df: pd.DataFrame, horizon: int, thr: float) -> pd.Series:
    z = df["z"].astype("float64")
    z_fut = z.shift(-int(horizon))
    event = (z.abs() >= float(thr))
    revert = (z * z_fut <= 0) | (z_fut.abs() < z.abs())
    y = (event & revert).astype("int8")
    return y

def _target_revert_direction(df: pd.DataFrame, horizon: int) -> pd.Series:
    z = df["z"].astype("float64")
    z_fut = z.shift(-int(horizon))
    y = (z_fut.abs() < z.abs()).astype("int8")
    return y

# ----- build single dataset -----

def _build_single_pair_dataset(pair: str, feat_path: Path, cfg: DatasetBuildConfig) -> Optional[pd.DataFrame]:
    if not feat_path.exists():
        _d(f"skip {pair}: features parquet missing: {feat_path}")
        return None

    df = pd.read_parquet(feat_path)
    df = _ensure_schema(df)

    feats = {"z": df["z"]}
    if "spread" in df.columns: feats["spread"] = df["spread"]
    if "beta" in df.columns:   feats["beta"] = df["beta"]
    if "alpha" in df.columns:  feats["alpha"] = df["alpha"]
    X = pd.DataFrame(feats, index=df.index)

    if cfg.lag_features and cfg.lag_features > 0:
        X = pd.concat(
            [
                X,
                _make_lags(df["z"], cfg.lag_features, "z"),
                _make_lags(df["spread"], cfg.lag_features, "spread") if "spread" in df.columns else pd.DataFrame(index=df.index),
            ],
            axis=1,
        )

    if cfg.label_type == "z_threshold":
        y = _target_z_threshold(df, cfg.horizon, cfg.zscore_threshold)
        rows = (df["z"].abs() >= float(cfg.zscore_threshold))
    elif cfg.label_type == "revert_direction":
        y = _target_revert_direction(df, cfg.horizon)
        rows = pd.Series(True, index=df.index)
    else:
        raise ValueError(f"Unknown label_type: {cfg.label_type}")

    out = pd.DataFrame({"ts": df["ts"]})
    out = pd.concat([out, X, y.rename("y")], axis=1)
    out = out.loc[rows].dropna().reset_index(drop=True)

    if len(out) < DATASET_MIN_ROWS:
        _d(f"skip {pair}: rows {len(out)} < DATASET_MIN_ROWS({DATASET_MIN_ROWS})")
        return None

    out["pair"] = pair
    _d(f"ok {pair}: rows={len(out)}")
    return out

# ----- public API -----

def build_datasets_for_manifest(
    features_manifest: str | Path,
    out_dir: str | Path,
    cfg: DatasetBuildConfig,
) -> Dict:
    items = _read_manifest(features_manifest)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    built = []
    total_pairs = 0

    for it in items:
        pair = str(it["pair"]); total_pairs += 1
        feat_path = Path(it["path"])
        if not feat_path.exists():
            guess = Path("data/features/pairs") / _norm_pair_key(pair) / "features.parquet"
            if guess.exists(): feat_path = guess

        ds = _build_single_pair_dataset(pair, feat_path, cfg)
        if ds is None:
            continue

        pair_key = _norm_pair_key(pair)
        pdir = out_dir / pair_key; pdir.mkdir(parents=True, exist_ok=True)
        out_path = pdir / "dataset.parquet"
        ds.to_parquet(out_path, index=False)
        built.append({"pair": pair, "path": str(out_path), "rows": int(len(ds)),
                      "features": [c for c in ds.columns if c not in ("ts","y","pair")]})

    manifest = {
        "items": built,
        "total_pairs": total_pairs,
        "built_pairs": len(built),
        "min_rows": DATASET_MIN_ROWS,
        "label_type": cfg.label_type,
        "zscore_threshold": cfg.zscore_threshold,
        "lag_features": cfg.lag_features,
        "horizon": cfg.horizon,
        "out_dir": str(Path(out_dir).resolve()),
    }
    (out_dir.parent / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
