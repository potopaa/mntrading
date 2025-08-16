#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference for mntrading (auto/model/z modes).

- Reads a registry of champion pairs (production_map.json / registry.json / list / features manifest).
- Finds features parquet paths via features manifest (pairs -> features.parquet).
- For each pair emits last N bars as JSONL records: data/signals/<PAIR>.jsonl

Signal modes:
  * auto   (default): try model; if unavailable -> fallback to z-based
  * model  : use saved winner model (<PAIR>__model.pkl + __meta.json)
  * z      : use z-score thresholds only (logistic proba from |z|)

Direction:
  mean-reversion: side = "long_spread" if z < 0 else "short_spread".
  Entry/exit: enter when |z| >= z_entry AND proba >= proba_threshold; exit (flat) when |z| <= z_exit.

Examples:
  python inference.py ^
    --registry data/models/production_map.json ^
    --pairs-manifest data/features/pairs/_manifest.json ^
    --signals-from auto --proba-threshold 0.55 --z-entry 1.5 --z-exit 0.5 ^
    --update --n-last 1 ^
    --out data/signals
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd


# -------------------- Utils --------------------
def _ensure_utf8_stdout():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _normalize_pair_key(s: str) -> str:
    """Accept both AAA/USDT__BBB/USDT and AAA_USDT__BBB_USDT."""
    if "__" in s and "/" not in s:
        a, b = s.split("__", 1)
        a = a.replace("_", "/")
        b = b.replace("_", "/")
        return f"{a}__{b}"
    return s


def _split_pair(pair_key: str) -> Tuple[str, str]:
    if "__" not in pair_key:
        return pair_key, pair_key
    a, b = pair_key.split("__", 1)
    return a, b


def _parse_registry(path: str) -> List[str]:
    """
    Supported formats:
      - {"pairs": ["AAA/USDT__BBB/USDT", ...]}
      - {"pairs": [{"pair":"AAA/USDT__BBB/USDT"}, ...]}
      - ["AAA/USDT__BBB/USDT", ...]
      - {"items":[{"pair":"AAA/USDT__BBB/USDT"}]}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Registry not found: {p}")
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Cannot read registry JSON {p}: {e}")

    pairs: List[str] = []
    if isinstance(obj, dict) and "pairs" in obj:
        for it in obj["pairs"] or []:
            if isinstance(it, str):
                pairs.append(_normalize_pair_key(it))
            elif isinstance(it, dict) and isinstance(it.get("pair"), str):
                pairs.append(_normalize_pair_key(it["pair"]))
            elif isinstance(it, dict) and "a" in it and "b" in it:
                pairs.append(f"{str(it['a'])}__{str(it['b'])}")
    if not pairs and isinstance(obj, dict) and "items" in obj:
        for it in obj["items"] or []:
            if isinstance(it, dict) and "pair" in it:
                pairs.append(_normalize_pair_key(str(it["pair"])))
    if not pairs and isinstance(obj, list):
        for s in obj:
            if isinstance(s, str):
                pairs.append(_normalize_pair_key(s))

    pairs = sorted({p for p in pairs if isinstance(p, str) and "__" in p})
    if not pairs:
        raise ValueError(f"Unsupported or empty registry format: {path}")
    return pairs


def _load_manifest_pairs_paths(manifest_path: str) -> Dict[str, str]:
    """
    Returns mapping: pair_key -> absolute path to features.parquet
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Pairs manifest not found: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    mp: Dict[str, str] = {}
    for it in obj.get("items", []):
        pk = _normalize_pair_key(str(it.get("pair")))
        path = it.get("path")
        if pk and path:
            mp[pk] = str(Path(path).resolve())
    return mp


def _read_last_jsonl_ts(path: Path) -> Optional[pd.Timestamp]:
    """Read last non-empty line's ts (UTC)."""
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            buf = bytearray()
            i = size - 1
            while i >= 0:
                f.seek(i)
                ch = f.read(1)
                if ch == b"\n" and buf:
                    break
                buf.extend(ch)
                i -= 1
        line = bytes(reversed(buf)).decode("utf-8").strip()
        if not line:
            return None
        obj = json.loads(line)
        ts = obj.get("ts")
        if ts is None:
            return None
        return pd.to_datetime(ts, utc=True, errors="coerce")
    except Exception:
        return None


def _append_jsonl(out_path: Path, rows: List[dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------- Signal helpers --------------------
def _logistic_proba(z_abs: float, z_entry: float, k: float = 1.5) -> float:
    """Map |z| relative to z_entry into probability in (0.5..0.99)."""
    x = z_abs - float(z_entry)
    p = 1.0 / (1.0 + math.exp(-k * x))
    return float(min(0.99, max(0.5, p)))


def _load_features_df(features_parquet: str, need_cols: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_parquet(features_parquet)
    # ensure ts
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError(f"{features_parquet} must have 'ts' column or DatetimeIndex")
    if need_cols:
        # make sure all required columns exist (fill absent numeric with 0.0)
        for c in need_cols:
            if c not in df.columns:
                df[c] = 0.0
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.sort_values("ts").reset_index(drop=True)


# -------------------- Per-pair inference (Z mode) --------------------
def _infer_z_for_pair(
    features_parquet: str,
    pair_key: str,
    n_last: int,
    z_entry: float,
    z_exit: float,
    proba_threshold: float,
    min_proba_to_write: float,
    update: bool,
    existing_last_ts: Optional[pd.Timestamp],
    skip_flat: bool,
) -> List[dict]:
    df = _load_features_df(features_parquet, need_cols=["z"])
    if update and existing_last_ts is not None:
        df = df[df["ts"] > existing_last_ts]
    if n_last > 0:
        df = df.tail(int(n_last))
    if df.empty:
        return []

    a, b = _split_pair(pair_key)
    rows: List[dict] = []
    for _, r in df.iterrows():
        z = float(r["z"])
        z_abs = abs(z)
        ts_iso = pd.Timestamp(r["ts"]).isoformat()

        proba = _logistic_proba(z_abs, z_entry)
        if z_abs >= z_entry and proba >= proba_threshold and proba >= min_proba_to_write:
            side = "long_spread" if z < 0 else "short_spread"
            rows.append({"ts": ts_iso, "pair": pair_key, "a": a, "b": b, "side": side, "proba": proba, "z": z})
        elif z_abs <= z_exit and not skip_flat:
            rows.append({"ts": ts_iso, "pair": pair_key, "a": a, "b": b, "side": "flat", "proba": 0.5, "z": z})
        else:
            # between thresholds or below min_proba_to_write -> no record
            pass
    return rows


# -------------------- Per-pair inference (Model mode) --------------------
def _load_pair_model(model_dir: Path, pair_key: str) -> Tuple[Optional[object], Optional[dict]]:
    base = pair_key.replace("/", "_")
    pkl = model_dir / f"{base}__model.pkl"
    meta = model_dir / f"{base}__meta.json"
    if not pkl.exists() or not meta.exists():
        return None, None
    try:
        with open(pkl, "rb") as f:
            model = pickle.load(f)
        meta_obj = json.loads(meta.read_text(encoding="utf-8"))
        return model, meta_obj
    except Exception:
        return None, None


def _infer_model_for_pair(
    features_parquet: str,
    pair_key: str,
    model_dir: Path,
    n_last: int,
    z_entry: float,
    z_exit: float,
    proba_threshold: float,
    min_proba_to_write: float,
    update: bool,
    existing_last_ts: Optional[pd.Timestamp],
    skip_flat: bool,
) -> Tuple[List[dict], bool]:
    """
    Returns (rows, used_model)
    used_model=False means we fell back to z mode.
    """
    model, meta = _load_pair_model(model_dir, pair_key)
    if model is None or not isinstance(meta, dict) or not meta.get("features"):
        # no model -> fallback to z mode (handled by caller)
        return [], False

    feat_cols: List[str] = list(meta["features"])
    need_cols = sorted(set(feat_cols + ["z"]))  # also need z for direction/gating
    df = _load_features_df(features_parquet, need_cols=need_cols)

    if update and existing_last_ts is not None:
        df = df[df["ts"] > existing_last_ts]
    if n_last > 0:
        df = df.tail(int(n_last))
    if df.empty:
        return [], True  # used model but nothing to write

    # Build X in the same order as during training
    X = df[feat_cols].select_dtypes(include=[np.number]).astype("float32").fillna(0.0)
    # Predict proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        if hasattr(model, "decision_function"):
            zraw = model.decision_function(X)
            proba = 1.0 / (1.0 + np.exp(-zraw))
        else:
            proba = np.full(len(X), 0.5)
    proba = np.clip(proba.astype("float64"), 1e-6, 1 - 1e-6)

    a, b = _split_pair(pair_key)
    rows: List[dict] = []
    for i, r in df.iterrows():
        z = float(r["z"])
        z_abs = abs(z)
        p = float(proba[df.index.get_loc(i)])  # align by position
        ts_iso = pd.Timestamp(r["ts"]).isoformat()

        # Gate by z thresholds and proba
        if z_abs >= z_entry and p >= proba_threshold and p >= min_proba_to_write:
            side = "long_spread" if z < 0 else "short_spread"
            rows.append({"ts": ts_iso, "pair": pair_key, "a": a, "b": b, "side": side, "proba": p, "z": z})
        elif z_abs <= z_exit and not skip_flat:
            rows.append({"ts": ts_iso, "pair": pair_key, "a": a, "b": b, "side": "flat", "proba": 0.5, "z": z})
        else:
            pass

    return rows, True


# -------------------- CLI --------------------
@click.command()
@click.option("--registry", "registry_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to production_map.json or registry.json")
@click.option("--pairs-manifest", "pairs_manifest", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to features manifest (data/features/pairs/_manifest.json)")
@click.option("--signals-from", type=click.Choice(["auto", "model", "z"]), default="auto", show_default=True,
              help="Where to take probabilities from")
@click.option("--model-dir", default="data/models/pairs", show_default=True, type=click.Path(file_okay=False),
              help="Directory with <PAIR>__model.pkl and __meta.json saved by train")
@click.option("--strict-model", is_flag=True, help="If set with signals-from=model, do not fallback to z when model is missing")
@click.option("--timeframe", default="5m", show_default=True, help="Informational")
@click.option("--limit", default=1000, show_default=True, type=int, help="Informational")
@click.option("--proba-threshold", default=0.55, show_default=True, type=float,
              help="Minimal proba to consider a trade (combined with z thresholds)")
@click.option("--min-proba-to-write", default=None, type=float,
              help="Optional extra filter for writing; default = --proba-threshold")
@click.option("--z-entry", default=1.5, show_default=True, type=float, help="Entry threshold on |z|")
@click.option("--z-exit", default=0.5, show_default=True, type=float, help="Exit threshold on |z| (flat)")
@click.option("--n-last", default=1, show_default=True, type=int, help="How many latest bars to emit per pair")
@click.option("--update", is_flag=True, help="Append only new records after last ts in the JSONL")
@click.option("--skip-flat", is_flag=True, help="Do not write explicit flat records")
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False),
              help="Directory to store signals JSONL files")
def main(
    registry_path: str,
    pairs_manifest: str,
    signals_from: str,
    model_dir: str,
    strict_model: bool,
    timeframe: str,
    limit: int,
    proba_threshold: float,
    min_proba_to_write: Optional[float],
    z_entry: float,
    z_exit: float,
    n_last: int,
    update: bool,
    skip_flat: bool,
    out_dir: str,
):
    _ensure_utf8_stdout()

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = _parse_registry(registry_path)
    mp = _load_manifest_pairs_paths(pairs_manifest)
    model_root = Path(model_dir)

    if min_proba_to_write is None:
        min_proba_to_write = proba_threshold

    total = 0
    wrote = 0
    used_models = 0

    for pk in pairs:
        total += 1
        fpath = mp.get(pk)
        if not fpath or not Path(fpath).exists():
            print(f"[infer] skip {pk}: features parquet not found in manifest")
            continue

        out_path = out_root / (pk.replace("/", "_") + ".jsonl")
        last_ts = _read_last_jsonl_ts(out_path) if update else None

        try:
            rows: List[dict] = []
            used_model = False

            if signals_from in ("auto", "model"):
                rows, used_model = _infer_model_for_pair(
                    features_parquet=fpath,
                    pair_key=pk,
                    model_dir=model_root,
                    n_last=n_last,
                    z_entry=z_entry,
                    z_exit=z_exit,
                    proba_threshold=proba_threshold,
                    min_proba_to_write=float(min_proba_to_write),
                    update=update,
                    existing_last_ts=last_ts,
                    skip_flat=skip_flat,
                )

            if not rows and (signals_from == "z" or (signals_from == "auto" and not used_model) or (signals_from == "model" and not strict_model)):
                # fallback to z-based
                rows = _infer_z_for_pair(
                    features_parquet=fpath,
                    pair_key=pk,
                    n_last=n_last,
                    z_entry=z_entry,
                    z_exit=z_exit,
                    proba_threshold=proba_threshold,
                    min_proba_to_write=float(min_proba_to_write),
                    update=update,
                    existing_last_ts=last_ts,
                    skip_flat=skip_flat,
                )

            if rows:
                _append_jsonl(out_path, rows)
                wrote += len(rows)
                used_models += int(used_model)
                print(f"[infer] {pk}: +{len(rows)} rows -> {out_path} (mode={'model' if used_model else 'z'})")
            else:
                print(f"[infer] {pk}: nothing to write")
        except Exception as e:
            print(f"[infer] {pk}: error: {e}")

    print(f"[infer] done: pairs={total}, wrote_rows={wrote}, used_model_pairs={used_models}, out_dir={out_root}")


if __name__ == "__main__":
    main()
