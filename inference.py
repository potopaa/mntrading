#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Inference for mntrading:
- Reads a registry of champion pairs (production_map or registry)
- Reads features manifest to locate per-pair features.parquet
- Computes latest signals per pair using z-score mean-reversion rule
- Appends JSONL signals to data/signals/<PAIR>.jsonl

Usage (example):
    python inference.py \
      --registry data/models/production_map.json \
      --pairs-manifest data/features/pairs/_manifest.json \
      --timeframe 5m --limit 1000 \
      --proba-threshold 0.55 --z-entry 1.5 --z-exit 0.5 \
      --update --n-last 1 \
      --out data/signals
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd


# -------------------- IO helpers --------------------
def _ensure_utf8_stdout():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _normalize_pair_key(s: str) -> str:
    """
    Accepts both "AAA/USDT__BBB/USDT" and "AAA_USDT__BBB_USDT".
    Returns normalized "AAA/USDT__BBB/USDT".
    """
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
    Supported registry formats:
    - production_map.json: {"pairs": ["AAA/USDT__BBB/USDT", ...]}
    - registry.json:       {"pairs": [{"pair":"AAA/USDT__BBB/USDT", ...}, ...]}
    - plain list:          ["AAA/USDT__BBB/USDT", ...]
    - features manifest:   {"items":[{"pair":"AAA/USDT__BBB/USDT", "path":"..."}]}
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
        arr = obj["pairs"] or []
        for it in arr:
            if isinstance(it, str):
                pairs.append(_normalize_pair_key(it))
            elif isinstance(it, dict):
                if "pair" in it and isinstance(it["pair"], str):
                    pairs.append(_normalize_pair_key(it["pair"]))
                elif "a" in it and "b" in it:
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
    """
    Read last non-empty line and return its ts as pandas Timestamp (UTC), if any.
    """
    if not path.exists():
        return None
    try:
        # read from the end
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


# -------------------- Signal logic --------------------
def _logistic_proba(z_abs: float, z_entry: float, k: float = 1.5) -> float:
    """Map |z| relative to z_entry into probability in (0.5..0.99)."""
    x = z_abs - float(z_entry)
    # logistic centered at 0 => 0.5 at threshold
    p = 1.0 / (1.0 + math.exp(-k * x))
    # clip to avoid extremes
    return float(min(0.99, max(0.5, p)))


def _infer_signals_for_pair(
    features_parquet: str,
    pair_key: str,
    n_last: int,
    z_entry: float,
    z_exit: float,
    update: bool,
    existing_last_ts: Optional[pd.Timestamp],
) -> List[dict]:
    """
    Build signals for the last n_last bars from features parquet.
    If update=True and existing_last_ts provided, only emit bars strictly after that ts.
    """
    df = pd.read_parquet(features_parquet)
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError(f"{features_parquet} must have 'ts' column or DatetimeIndex")
    if "z" not in cols:
        raise ValueError(f"{features_parquet} lacks 'z' column")

    df = df[["ts", cols["z"]]].rename(columns={cols["z"]: "z"}).copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    if update and existing_last_ts is not None:
        df = df[df["ts"] > existing_last_ts]

    if n_last > 0:
        df = df.tail(int(n_last))

    if df.empty:
        return []

    # make hysteresis with previous state if we had one (from existing file)
    # but since мы не читаем весь файл (дорого), гистерезис реализуем без истории:
    # вход при |z| >= z_entry, выход при |z| <= z_exit; если между — не сигналим.
    a, b = _split_pair(pair_key)
    rows: List[dict] = []
    for _, r in df.iterrows():
        z = float(r["z"])
        ts_iso = pd.Timestamp(r["ts"]).isoformat()
        z_abs = abs(z)

        if z_abs >= z_entry:
            side = "long_spread" if z < 0 else "short_spread"  # mean-reversion: negative z -> long spread
            proba = _logistic_proba(z_abs, z_entry)
            rows.append({
                "ts": ts_iso,
                "pair": pair_key,
                "a": a,
                "b": b,
                "side": side,
                "proba": proba,
                "z": z,
            })
        elif z_abs <= z_exit:
            # emit explicit flat signal to allow downstream to close
            rows.append({
                "ts": ts_iso,
                "pair": pair_key,
                "a": a,
                "b": b,
                "side": "flat",
                "proba": 0.5,
                "z": z,
            })
        else:
            # between entry and exit — оставляем без записи
            pass

    return rows


# -------------------- CLI --------------------
@click.command()
@click.option("--registry", "registry_path", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to production_map.json or registry.json")
@click.option("--pairs-manifest", "pairs_manifest", required=True, type=click.Path(exists=True, dir_okay=False),
              help="Path to features manifest (data/features/pairs/_manifest.json)")
@click.option("--timeframe", default="5m", show_default=True, help="Timeframe (informational)")
@click.option("--limit", default=1000, show_default=True, type=int, help="Bars to consider (informational)")
@click.option("--proba-threshold", default=0.55, show_default=True, type=float,
              help="Threshold used downstream (informational here, affects logistic scale)")
@click.option("--z-entry", default=1.5, show_default=True, type=float, help="Entry threshold on |z|")
@click.option("--z-exit", default=0.5, show_default=True, type=float, help="Exit threshold on |z|")
@click.option("--n-last", default=1, show_default=True, type=int, help="How many latest bars to emit per pair")
@click.option("--update", is_flag=True, help="Append only new records after last ts in the JSONL")
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False),
              help="Directory to store signals JSONL files")
def main(
    registry_path: str,
    pairs_manifest: str,
    timeframe: str,
    limit: int,
    proba_threshold: float,
    z_entry: float,
    z_exit: float,
    n_last: int,
    update: bool,
    out_dir: str,
):
    _ensure_utf8_stdout()

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = _parse_registry(registry_path)
    mp = _load_manifest_pairs_paths(pairs_manifest)

    # Emit signals per pair
    total = 0
    wrote = 0
    for pk in pairs:
        total += 1
        fpath = mp.get(pk)
        if not fpath or not Path(fpath).exists():
            print(f"[infer] skip {pk}: features parquet not found in manifest")
            continue

        out_path = out_root / (pk.replace("/", "_") + ".jsonl")
        last_ts = _read_last_jsonl_ts(out_path) if update else None

        try:
            rows = _infer_signals_for_pair(
                features_parquet=fpath,
                pair_key=pk,
                n_last=n_last,
                z_entry=z_entry,
                z_exit=z_exit,
                update=update,
                existing_last_ts=last_ts,
            )
            if rows:
                _append_jsonl(out_path, rows)
                wrote += len(rows)
                print(f"[infer] {pk}: +{len(rows)} rows -> {out_path}")
            else:
                print(f"[infer] {pk}: nothing to write")
        except Exception as e:
            print(f"[infer] {pk}: error: {e}")

    print(f"[infer] done: pairs={total}, wrote_rows={wrote}, out_dir={out_root}")


if __name__ == "__main__":
    main()
