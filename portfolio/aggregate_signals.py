#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# portfolio/aggregate_signals.py
# All comments are in English by request.

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import pandas as pd


def _canon_pair_key(s: str) -> str:
    """Normalize pair key so that 'BNB/USDT__XRP/USDT' == 'BNB_USDT__XRP_USDT'."""
    return s.replace("/", "_").strip()


def _read_manifest_pairs(manifest_path: Path) -> Dict[str, Path]:
    """
    Read features pairs manifest and return mapping:
        canonical_pair_key -> features.parquet path
    Supports 'items': [{"pair": "...", "path": "..."}] and friends.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pairs manifest not found: {manifest_path}")
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = obj.get("items") or []
    out: Dict[str, Path] = {}
    for it in items:
        pair = it.get("pair") or it.get("name") or it.get("key")
        path = it.get("path") or it.get("parquet") or it.get("file")
        if pair and path:
            out[_canon_pair_key(str(pair))] = Path(str(path))
    return out


@dataclass
class SignalRow:
    ts: pd.Timestamp
    pair: str
    side: int      # +1 long, -1 short
    proba: float


def _read_signals(signals_dir: Path) -> List[SignalRow]:
    """
    Scan signals_dir/*/signals.parquet and build a list of SignalRow.
    If 'pair' column is missing in a file, infer from folder name.
    """
    rows: List[SignalRow] = []
    for p in signals_dir.rglob("signals.parquet"):
        try:
            df = pd.read_parquet(p)
            # infer pair
            if "pair" in df.columns:
                pair = str(df["pair"].iloc[0])
            else:
                pair = p.parent.name.replace("_", "/")  # reverse-guess
            # ensure columns
            tcol = "ts" if "ts" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
            if not tcol:
                continue
            scol = "side"
            pcol = "proba"
            if scol not in df.columns or pcol not in df.columns:
                continue
            # cast
            d = df[[tcol, scol, pcol]].copy()
            d.rename(columns={tcol: "ts"}, inplace=True)
            d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
            d["side"] = pd.to_numeric(d["side"], errors="coerce").fillna(0).astype("int8")
            d["proba"] = pd.to_numeric(d["proba"], errors="coerce").fillna(0.0).astype("float64")
            d = d.dropna(subset=["ts"]).sort_values("ts")

            # keep last row per file (usually one row if --n-last=1)
            if len(d):
                r = d.iloc[-1]
                rows.append(SignalRow(ts=r["ts"], pair=pair, side=int(r["side"]), proba=float(r["proba"])))
        except Exception:
            # skip unreadable file
            pass
    return rows


def _select_top(rows: List[SignalRow], min_proba: float, top_k: int) -> List[SignalRow]:
    """
    Deduplicate by pair keeping the most recent ts; filter by proba; take top_k by proba.
    """
    latest: Dict[str, SignalRow] = {}
    for r in rows:
        key = _canon_pair_key(r.pair)
        if (key not in latest) or (r.ts > latest[key].ts):
            latest[key] = r
    # filter and sort
    filt = [r for r in latest.values() if r.proba >= float(min_proba)]
    filt.sort(key=lambda x: x.proba, reverse=True)
    return filt[: max(top_k, 0)]


def _write_orders(out_dir: Path, orders: List[SignalRow]) -> Tuple[Path, Dict]:
    """
    Write JSONL with orders and a summary latest_orders.json next to it.
    Returns (path_to_jsonl, summary_obj).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_tag = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    jsonl = out_dir / f"orders_{ts_tag}.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in orders:
            f.write(json.dumps({
                "ts": r.ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "pair": r.pair,
                "side": int(r.side),
                "proba": float(r.proba),
            }) + "\n")
    summary = {
        "orders_path": str(jsonl.relative_to(out_dir)),
        "orders": [{
            "ts": r.ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pair": r.pair,
            "side": int(r.side),
            "proba": float(r.proba),
        } for r in orders],
    }
    (out_dir / "latest_orders.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return jsonl, summary


@click.command()
@click.option("--signals-dir", type=click.Path(path_type=Path), required=True, help="Directory containing */signals.parquet")
@click.option("--pairs-manifest", type=click.Path(path_type=Path), required=True, help="Features pairs manifest (data/features/pairs/_manifest.json)")
@click.option("--min-proba", type=float, default=0.55, show_default=True)
@click.option("--top-k", type=int, default=20, show_default=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True, help="Where to write orders and summary")
def main(signals_dir: Path, pairs_manifest: Path, min_proba: float, top_k: int, out_dir: Path):
    """Aggregate signals into tradable orders with manifest-aware filtering."""
    feats = _read_manifest_pairs(pairs_manifest)
    rows = _read_signals(signals_dir)

    print(f"[dbg] signals_dir={signals_dir}, files_found={len(list(signals_dir.rglob('signals.parquet')))}")
    print(f"[dbg] latest-signal rows loaded: {len(rows)}")

    selected: List[SignalRow] = []
    skipped = 0
    for r in _select_top(rows, min_proba=min_proba, top_k=top_k*3):  # pick more, will filter by manifest below
        k = _canon_pair_key(r.pair)
        if k not in feats:
            print(f"[skip] {r.pair}: features parquet not found in manifest")
            skipped += 1
            continue
        selected.append(r)
        if len(selected) >= top_k:
            break

    print(f"[ok] selected: {len(selected)} pairs (min_proba={min_proba}, top_k={top_k})")
    jsonl_path, summary = _write_orders(out_dir, selected)
    print(f"[ok] wrote {len(selected)} orders -> {jsonl_path}")
    print(f"[ok] summary -> {out_dir / 'latest_orders.json'}")


if __name__ == "__main__":
    main()
