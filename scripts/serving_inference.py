#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# scripts/serving_inference.py
# All comments are in English by request.

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import requests


def _canon_pair_key(s: str) -> str:
    return s.replace("/", "_").strip()


def _read_manifest_pairs(manifest_path: Path) -> Dict[str, Path]:
    obj = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    items = obj.get("items") or []
    out: Dict[str, Path] = {}
    for it in items:
        pair = it.get("pair") or it.get("name") or it.get("key")
        path = it.get("path") or it.get("parquet") or it.get("file")
        if pair and path:
            out[_canon_pair_key(str(pair))] = Path(str(path))
    return out


def _select_last_rows(per_pair_path: Dict[str, Path], n_last: int) -> pd.DataFrame:
    rows = []
    for key, p in per_pair_path.items():
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        df = df.sort_values("ts")
        take = df.tail(n_last).copy()
        # add original 'pair' with slashes
        pair = key.replace("_", "/")
        take["pair"] = pair
        rows.append(take)
    if not rows:
        return pd.DataFrame()
    cat = pd.concat(rows, ignore_index=True)
    # keep numeric + pair + z if present
    keep = cat.select_dtypes(include=["number"]).columns.tolist()
    if "z" in cat.columns and "z" not in keep:
        keep.append("z")
    keep.append("pair")
    return cat[keep]


def _invoke(serving_url: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calls MLflow pyfunc REST /invocations.
    """
    payload = {
        "dataframe_split": {
            "columns": list(df.columns),
            "data": df.to_numpy().tolist(),
        }
    }
    headers = {"Content-Type": "application/json"}
    r = requests.post(serving_url.rstrip("/") + "/invocations", data=json.dumps(payload), headers=headers, timeout=120)
    r.raise_for_status()
    obj = r.json()
    # pyfunc returns array-like or dataframe JSON; our router returns list of dicts or a table
    if isinstance(obj, dict) and "columns" in obj and "data" in obj:
        out = pd.DataFrame(obj["data"], columns=obj["columns"])
    else:
        out = pd.DataFrame(obj)
    return out


def _write_signals(out_dir: Path, infer_df: pd.DataFrame) -> int:
    """
    Save signals like existing pipeline expects:
      data/signals/<PAIR_KEY>/signals.parquet with columns ['ts','pair','side','proba'].
    We do not have per-row ts from server; reuse current UTC time for all rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    count = 0
    for pair, grp in infer_df.groupby("pair"):
        key = _canon_pair_key(pair)
        d = out_dir / key
        d.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "ts": [now] * len(grp),
            "pair": [pair] * len(grp),
            "side": grp["side"].astype("int32").tolist(),
            "proba": grp["proba"].astype("float64").tolist(),
        })
        (d / "signals.parquet").unlink(missing_ok=True)
        df.to_parquet(d / "signals.parquet", index=False)
        count += len(df)
    return count


def main():
    ap = argparse.ArgumentParser(description="Call MLflow Serving router and write signals")
    ap.add_argument("--serving-url", default="http://serving:5001", help="Base URL of mlflow models serve")
    ap.add_argument("--pairs-manifest", type=Path, default=Path("data/features/pairs/_manifest.json"))
    ap.add_argument("--n-last", type=int, default=1, help="Rows per pair to send")
    ap.add_argument("--out", type=Path, default=Path("data/signals"))
    args = ap.parse_args()

    per_pair = _read_manifest_pairs(args.pairs_manifest)
    feats = _select_last_rows(per_pair, args.n_last)
    if feats.empty:
        print("[serving] no features to send")
        return

    pred = _invoke(args.serving_url, feats)
    if not {"pair", "proba", "side"}.issubset(set(pred.columns)):
        raise RuntimeError(f"Serving response must contain columns ['pair','proba','side'], got: {list(pred.columns)}")

    n = _write_signals(args.out, pred)
    print(f"[serving] wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
