import argparse
import json
import re
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import mlflow


def _sanitize_pair(p: str) -> str:
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


def _as_dataframe(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj)
        if arr.ndim == 1:
            return pd.DataFrame({"pred": arr})
        if arr.ndim == 2 and arr.shape[1] == 1:
            return pd.DataFrame({"pred": arr.ravel()})
        cols = [f"c{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
    try:
        return pd.json_normalize(obj)
    except Exception:
        return pd.DataFrame({"pred": [obj]})


def main():
    ap = argparse.ArgumentParser(description="Offline inference using MLflow pyfunc (no HTTP).")
    ap.add_argument("--model-uri", default="models:/mntrading_router/Production",
                    help="MLflow model URI, e.g. models:/mntrading_router/Production")
    ap.add_argument("--registry", default="/app/data/models/registry.json", help="Path to registry.json")
    ap.add_argument("--features-dir", default="/app/data/features/pairs", help="Root with per-pair features")
    ap.add_argument("--n-last", type=int, default=1, help="Take last N rows per pair")
    ap.add_argument("--top-k", type=int, default=20, help="Limit number of pairs from registry")
    ap.add_argument("--out", default="/app/data/signals", help="Directory to store predictions")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _read_registry_pairs(Path(args.registry), args.top_k)
    if not pairs:
        raise SystemExit("No pairs in registry.json")

    df = _collect_features_from_pairs(Path(args.features_dir), pairs, args.n_last)

    model = mlflow.pyfunc.load_model(args.model_uri)
    preds = model.predict(df)
    pred_df = _as_dataframe(preds)

    if len(pred_df) == len(df):
        pred_df.insert(0, "pair", df["pair"].tolist())

    pred_path = out_dir / "latest_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    (out_dir / "latest_predictions.json").write_text(pred_df.to_json(orient="records"), encoding="utf-8")

    print(f"[offline_inference] saved {len(pred_df)} rows to {pred_path}")


if __name__ == "__main__":
    main()