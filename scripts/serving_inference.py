<<<<<<< HEAD
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import requests


def _sanitize_pair(p: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(p)).strip("_")


def _read_registry_pairs(registry_path: Path, top_k: Optional[int]) -> List[str]:
    obj = json.loads(registry_path.read_text(encoding="utf-8"))
    pairs = obj.get("pairs") or obj.get("models") or []
    out: List[str] = []
    if isinstance(pairs, dict):
        out = [_sanitize_pair(k) for k in pairs.keys()]
    else:
        for it in pairs:
            if isinstance(it, str):
                out.append(_sanitize_pair(it))
            elif isinstance(it, dict) and "pair" in it:
                out.append(_sanitize_pair(it["pair"]))
    if top_k and top_k > 0:
        out = out[: top_k]
    return out


def _collect_features_from_pairs(features_root: Path, pairs: List[str], n_last: int) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in pairs:
        f = features_root / p / "features.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f, engine="pyarrow")
            if df.empty:
                continue
            drop_like = {
                "y", "label", "target", "ts", "time", "timestamp", "pair",
                "symbol", "symbol_a", "symbol_b", "run_id", "fold", "split_id",
            }
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            feat_cols = [c for c in num_cols if c not in drop_like] or num_cols
            take = df[feat_cols].tail(max(1, int(n_last))).copy()
            take.insert(0, "pair", p)
            frames.append(take)
        except Exception:
            continue
    if not frames:
        raise RuntimeError("No features found to build inference payload.")
    return pd.concat(frames, axis=0, ignore_index=True)


def _to_mlflow_dataframe_split(df: pd.DataFrame) -> dict:
    return {"dataframe_split": {"columns": list(df.columns), "data": df.values.tolist()}}


def _post(url: str, payload: dict, timeout_connect: float, timeout_read: float) -> Tuple[int, str]:
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=(timeout_connect, timeout_read))
    return resp.status_code, resp.text


def main():
    ap = argparse.ArgumentParser(description="Send inference payload to MLflow Serving router.")
    ap.add_argument("--serving-url", required=True, help="MLflow serving base URL, e.g. http://serving:5001")
    ap.add_argument("--registry", default="/app/data/models/registry.json", help="Path to registry.json")
    ap.add_argument("--features-dir", default="/app/data/features/pairs", help="Root with per-pair features")
    ap.add_argument("--n-last", type=int, default=1, help="Take last N rows per pair")
    ap.add_argument("--top-k", type=int, default=20, help="Limit number of pairs from registry")
    ap.add_argument("--out", default="/app/data/signals", help="Directory to store predictions")
    ap.add_argument("--timeout-connect", type=float, default=3.0, help="Connect timeout seconds")
    ap.add_argument("--timeout-read", type=float, default=120.0, help="Read timeout seconds (increase for cold start)")
    ap.add_argument("--warmup", action="store_true", help="Send a tiny warmup request (top-k=1, n-last=1) before main call")
    args = ap.parse_args()

    reg = Path(args.registry)
    feats = Path(args.features_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = args.serving_url.rstrip("/") + "/invocations"

    if args.warmup:
        pairs_w = _read_registry_pairs(reg, top_k=1)
        if not pairs_w:
            raise SystemExit("Warmup skipped: no pairs in registry.json")
        df_w = _collect_features_from_pairs(feats, pairs_w, n_last=1)
        status, body = _post(url, _to_mlflow_dataframe_split(df_w), args.timeout_connect, args.timeout_read)
        (out_dir / "latest_warmup_response.txt").write_text(body, encoding="utf-8")
        if status != 200:
            print(f"[inference] warmup HTTP {status}, body saved to latest_warmup_response.txt")

    # Main request
    pairs = _read_registry_pairs(reg, args.top_k)
    if not pairs:
        raise SystemExit("No pairs in registry.json")

    df = _collect_features_from_pairs(feats, pairs, args.n_last)
    payload = _to_mlflow_dataframe_split(df)

    try:
        status, body = _post(url, payload, args.timeout_connect, args.timeout_read)
    except requests.exceptions.ReadTimeout:
        raise SystemExit(f"Read timeout after {args.timeout_read}s. Try --warmup and/or increase --timeout-read.")
    except Exception as e:
        raise SystemExit(f"Request failed: {e}")

    raw_path = out_dir / "latest_raw_response.txt"
    raw_path.write_text(body, encoding="utf-8")
    if status != 200:
        raise SystemExit(f"Serving returned HTTP {status}. Body saved to {raw_path}")

    # Try to parse to DataFrame
    try:
        obj = json.loads(body)
        preds = obj.get("predictions", obj)
        pred_df = pd.json_normalize(preds)
    except Exception:
        (out_dir / "latest_predictions.json").write_text(body, encoding="utf-8")
        print(f"[inference] saved plain response to {out_dir/'latest_predictions.json'}")
        return

    # Attach original pairs back by row alignment if lengths match
    if len(pred_df) == len(df):
        pred_df.insert(0, "pair", df["pair"].tolist())

    pred_path = out_dir / "latest_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    (out_dir / "latest_predictions.json").write_text(pred_df.to_json(orient="records"), encoding="utf-8")
    print(f"[inference] saved {len(pred_df)} rows to {pred_path}")


if __name__ == "__main__":
    main()
=======
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
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700
