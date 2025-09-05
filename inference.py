from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import click
import numpy as np
import pandas as pd


# ---------- utils: registry parsing ----------

@dataclass
class RegistryItem:
    pair: str
    model_path: Optional[Path] = None


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _find_model_for_pair(models_dir: Path, pair: str) -> Optional[Path]:
    key = pair.replace("/", "_")
    root = models_dir / key
    if not root.exists():
        alt = models_dir / "pairs" / key
        if alt.exists():
            root = alt
        else:
            return None
    best: Tuple[float, Optional[Path]] = (-1.0, None)
    for ext in (".pkl", ".joblib", ".json"):
        for p in root.rglob(f"*{ext}"):
            ts = p.stat().st_mtime
            if ts > best[0]:
                best = (ts, p)
    return best[1]


def _parse_registry(path: Optional[Path], models_dir: Path) -> List[RegistryItem]:

    out: List[RegistryItem] = []
    if path is None:
        return out
    obj = _load_json(path)
    pairs = obj.get("pairs")
    if pairs is None:
        raise ValueError(f"Unsupported or empty registry format: {path}")

    if isinstance(pairs, list):
        for it in pairs:
            if isinstance(it, str):
                out.append(RegistryItem(pair=it))
            elif isinstance(it, dict):
                p = it.get("pair") or it.get("name") or it.get("key")
                mp = it.get("model_path") or it.get("model") or it.get("path")
                if p:
                    out.append(RegistryItem(pair=str(p), model_path=Path(mp) if mp else None))
    if not out:
        raise ValueError(f"Unsupported or empty registry format: {path}")

    resolved: List[RegistryItem] = []
    for ri in out:
        if ri.model_path and ri.model_path.exists():
            resolved.append(ri)
            continue
        mp = _find_model_for_pair(models_dir, ri.pair)
        resolved.append(RegistryItem(pair=ri.pair, model_path=mp))
    return resolved


# ---------- utils: features manifest ----------

def _read_features_manifest(manifest_path: Path) -> Dict[str, Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pairs manifest not found: {manifest_path}")
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = obj.get("items") or []
    out: Dict[str, Path] = {}
    for it in items:
        pair = it.get("pair") or it.get("name") or it.get("key")
        path = it.get("path") or it.get("parquet") or it.get("file")
        if pair and path:
            out[str(pair)] = Path(str(path))
    return out


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    tcol = None
    for c in ("ts", "timestamp", "time", "date", "datetime"):
        if c in df.columns:
            tcol = c; break
    if tcol is None:
        raise ValueError("features parquet must contain time column (ts/timestamp/...)")
    if tcol != "ts":
        df = df.rename(columns={tcol: "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    zc = None
    for c in ("z", "zscore", "z_score", "spread_z"):
        if c in df.columns:
            zc = c; break
    if zc is None:
        raise ValueError("features parquet must contain z-score column (z/zscore/...)")
    if zc != "z":
        df = df.rename(columns={zc: "z"})
    df["z"] = pd.to_numeric(df["z"], errors="coerce").astype("float64")

    df = df.dropna(subset=["ts", "z"]).sort_values("ts").reset_index(drop=True)
    return df


# ---------- inference logic ----------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def _z_based_signals(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    z = df["z"].values
    side = np.where(z <= -thr, 1, np.where(z >= thr, -1, 0))
    proba = _sigmoid(np.abs(z) - float(thr))
    out = df.loc[np.abs(z) >= float(thr), ["ts"]].copy()
    out["side"] = side[np.abs(z) >= float(thr)]
    out["proba"] = proba[np.abs(z) >= float(thr)]
    return out

def _model_based_signals(df: pd.DataFrame, model_path: Path, thr: float) -> pd.DataFrame:
    if not model_path or not model_path.exists():
        return _z_based_signals(df, thr)
    return _z_based_signals(df, thr)


def _write_signals(out_dir: Path, pair: str, sig: pd.DataFrame) -> None:
    out_pair = out_dir / pair.replace("/", "_")
    out_pair.mkdir(parents=True, exist_ok=True)
    sig2 = sig.copy()
    sig2["pair"] = pair
    (out_pair / "signals.parquet").write_bytes(sig2.to_parquet(index=False))


@click.command()
@click.option("--registry", "registry_path", type=click.Path(path_type=Path), required=False, help="Path to models registry JSON.")
@click.option("--pairs-manifest", type=click.Path(path_type=Path), required=True, help="Features pairs manifest (data/features/pairs/_manifest.json).")
@click.option("--signals-from", type=click.Choice(["auto", "model", "z"]), default="auto", show_default=True)
@click.option("--proba-threshold", type=float, default=0.55, show_default=True)
@click.option("--model-dir", type=click.Path(path_type=Path), required=False, default=Path("data/models/pairs"))
@click.option("--n-last", type=int, default=1, show_default=True, help="How many last bars to generate signals for.")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), required=True, default=Path("data/signals"))
@click.option("--update", is_flag=True, help="Not used in this simplified version; kept for CLI compatibility.")
@click.option("--skip-flat", is_flag=True, help="Not used; kept for CLI compatibility.")
def main(registry_path: Optional[Path],
         pairs_manifest: Path,
         signals_from: str,
         proba_threshold: float,
         model_dir: Path,
         n_last: int,
         out_dir: Path,
         update: bool,
         skip_flat: bool):
    feats = _read_features_manifest(pairs_manifest)
    pairs_in_manifest = list(feats.keys())

    registry_items: List[RegistryItem] = []
    if registry_path and registry_path.exists():
        try:
            registry_items = _parse_registry(registry_path, model_dir)
        except Exception as e:
            print(f"[inference] registry parse failed ({e}); falling back to manifest pairs")
    if not registry_items:
        registry_items = [RegistryItem(pair=p) for p in pairs_in_manifest]

    mode = signals_from
    if mode == "auto":
        if any(ri.model_path for ri in registry_items):
            mode = "model"
        else:
            mode = "z"

    out_dir.mkdir(parents=True, exist_ok=True)

    produced = 0
    for ri in registry_items:
        pair = ri.pair
        p = feats.get(pair)
        if p is None or not p.exists():
            guess = Path("data/features/pairs") / pair.replace("/", "_") / "features.parquet"
            if guess.exists():
                p = guess
            else:
                print(f"[inference] skip {pair}: features not found")
                continue
        try:
            df = pd.read_parquet(p)
            df = _ensure_schema(df)
            if n_last > 0 and len(df) > n_last:
                df = df.iloc[-n_last:].copy()

            if mode == "model":
                sig = _model_based_signals(df, ri.model_path, proba_threshold)
            else:
                sig = _z_based_signals(df, proba_threshold)

            if not sig.empty:
                _write_signals(out_dir, pair, sig)
                produced += len(sig)
        except Exception as e:
            print(f"[inference] error on {pair}: {e}")

    print(f"[inference] done. signals rows={produced}, pairs={len(registry_items)}")


if __name__ == "__main__":
    main()