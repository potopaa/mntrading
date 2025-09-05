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