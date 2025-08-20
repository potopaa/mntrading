# scripts/build_registry.py
# All comments are in English by request.

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def _collect_candidates(summary: dict) -> List[Tuple[str, float, dict]]:
    """
    Extract (pair, sharpe, metrics) from summary.
    Supports summary['pairs'] as dict or list of dicts.
    """
    items: List[Tuple[str, float, dict]] = []
    pairs_obj = summary.get("pairs")

    if isinstance(pairs_obj, dict):
        for pair, data in pairs_obj.items():
            met = data.get("metrics", {}) if isinstance(data, dict) else {}
            sh = float(met.get("sharpe", 0.0))
            items.append((str(pair), sh, met))

    elif isinstance(pairs_obj, list):
        for it in pairs_obj:
            if not isinstance(it, dict):
                continue
            pair = it.get("pair") or it.get("name") or it.get("key")
            met = it.get("metrics", {})
            sh = float(met.get("sharpe", 0.0))
            if pair:
                items.append((str(pair), sh, met))

    else:
        # unknown format -> empty
        pass

    return items

def _find_model(models_dir: Path, pair: str) -> Optional[Path]:
    """
    Find latest model artifact for given pair under models_dir[/pairs]/<pair_key>.
    Accept extensions: .pkl, .joblib, .json
    """
    key = pair.replace("/", "_")
    roots = [models_dir / key, models_dir / "pairs" / key]
    root = next((r for r in roots if r.exists()), None)
    if not root:
        return None
    cands: List[Path] = []
    for ext in (".pkl", ".joblib", ".json"):
        cands.extend(sorted(root.rglob(f"*{ext}")))
    return cands[-1] if cands else None

def main():
    ap = argparse.ArgumentParser(description="Build a normalized registry.json from backtest summary.")
    ap.add_argument("--summary", required=True, type=Path, help="Path to backtest summary JSON (e.g. data/backtest_results/_summary.json)")
    ap.add_argument("--models-dir", required=False, type=Path, default=Path("data/models/pairs"), help="Directory to search for model artifacts")
    ap.add_argument("--out", required=True, type=Path, help="Output registry path (e.g. data/models/registry.json)")
    ap.add_argument("--top-k", type=int, default=20, help="How many pairs to keep")
    args = ap.parse_args()

    if not args.summary.exists():
        raise FileNotFoundError(f"Summary not found: {args.summary}")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    items = _collect_candidates(summary)
    if not items:
        raise RuntimeError("No pairs found in summary or unsupported summary format")

    # sort by sharpe descending and take top-k
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[: max(args.top_k, 1)]

    # build output
    out_pairs: List[Dict] = []
    for pair, sh, met in items:
        mpath = _find_model(args.models_dir, pair)
        out_pairs.append({
            "pair": pair,
            "model_path": str(mpath) if mpath else None,
            "metrics": met,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"pairs": out_pairs}, indent=2), encoding="utf-8")
    print(f"[build_registry] wrote {args.out} (pairs={len(out_pairs)})")

if __name__ == "__main__":
    main()
