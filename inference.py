#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


def _pair_to_model_name(pair_key: str) -> str:
    # "ADA/USDT__DOT/USDT" -> "mntrading__ADA_USDT__DOT_USDT"
    return "mntrading__" + pair_key.replace("/", "_")


def _load_model_anyway(pair_key: str, run_id: Optional[str], model_version: Optional[str]) -> object:
    """
    Try: registry by (name, version) -> registry by stage 'Staging' -> runs:/<run_id>/...
    """
    name = _pair_to_model_name(pair_key)
    # 1) registry by version
    if model_version:
        target = f"models:/{name}/{model_version}"
        try:
            return mlflow.pyfunc.load_model(target)
        except Exception:
            pass
    # 2) registry by stage
    for stage in ("Staging", "Production"):
        try:
            return mlflow.pyfunc.load_model(f"models:/{name}/{stage}")
        except Exception:
            pass
    # 3) runs by common paths
    if run_id:
        candidates = ["model", "models", "sklearn-model", "xgb-model", "lgbm-model", ""]
        for sub in candidates:
            uri = f"runs:/{run_id}/{sub}" if sub else f"runs:/{run_id}"
            try:
                return mlflow.pyfunc.load_model(uri)
            except Exception:
                continue
    raise RuntimeError(f"cannot load model for {pair_key} (run_id={run_id}, version={model_version})")


def _parse_registry(path: Path) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # поддержка форматов: {"pairs": {...}} или просто {"pair": {...}} и т.д.
    pairs = obj.get("pairs") or obj
    if not isinstance(pairs, dict):
        raise ValueError(f"Unsupported or empty registry format: {path}")
    out = {}
    for pkey, info in pairs.items():
        if not isinstance(info, dict):
            continue
        run_id = str(info.get("run_id") or "").strip() or None
        model_version = str(info.get("model_version") or "").strip() or None
        out[pkey] = {"run_id": run_id, "model_version": model_version}
    return out


def _read_pairs_manifest(manifest_path: Path) -> Dict[str, str]:
    """
    Возвращает {pair_key: parquet_path}
    Ожидает форматы:
      {"items":[{"pair":"...","path":"..."}]}  или {"pairs":[{...}]}  или просто список [{...}]
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        entries = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("items"), list):
            entries = obj["items"]
        elif isinstance(obj.get("pairs"), list):
            entries = obj["pairs"]
        else:
            # берём первый list-поля в объекте
            lists = [v for v in obj.values() if isinstance(v, list)]
            entries = lists[0] if lists else []
    else:
        entries = []

    out = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        pair = e.get("pair")
        path = e.get("path")
        if pair and path:
            out[pair] = path
    return out


@click.command()
@click.option("--registry", "registry_path", type=click.Path(path_type=Path), required=True)
@click.option("--pairs-manifest", "manifest_path", type=click.Path(path_type=Path), required=True)
@click.option("--timeframe", type=str, default="5m", show_default=True)
@click.option("--limit", type=int, default=1000, show_default=True)
@click.option("--proba-threshold", type=float, default=0.55, show_default=True)
@click.option("--update", is_flag=True, help="Append to existing jsonl if exists")
@click.option("--n-last", type=int, default=1, show_default=True, help="How many last bars to score")
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("data/signals"), show_default=True)
def main(
    registry_path: Path,
    manifest_path: Path,
    timeframe: str,
    limit: int,
    proba_threshold: float,
    update: bool,
    n_last: int,
    out_dir: Path,
):
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = _parse_registry(registry_path)
    feat_map = _read_pairs_manifest(manifest_path)

    # какие пары реально сможем скорить
    pairs_to_score = []
    for pkey, meta in reg.items():
        if pkey not in feat_map:
            print(f"[warn] features parquet not found for pair '{pkey}': None")
            continue
        pairs_to_score.append((pkey, meta))

    if not pairs_to_score:
        print("[warn] no pairs to score (no features).")
        return

    # Загрузим последние фичи
    for idx, (pkey, meta) in enumerate(pairs_to_score, 1):
        fpath = feat_map[pkey]
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"[warn] cannot read features for '{pkey}' at '{fpath}': {e}")
            continue

        # последние n_last по времени
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
        sub = df.tail(n_last)

        # загрузка модели
        try:
            model = _load_model_anyway(pkey, meta.get("run_id"), meta.get("model_version"))
        except Exception as e:
            print(f"[warn] failed to load model for {pkey} (run_id={meta.get('run_id')}): {e}")
            continue

        # предсказание вероятностей (pyfunc: ожидает DataFrame фич)
        X = sub.drop(columns=[c for c in ("y",) if c in sub.columns], errors="ignore")
        try:
            proba = np.asarray(model.predict(X))
        except Exception:
            # некоторые pyfunc возвращают logits/proba разной формы
            yhat = model.predict(X)
            proba = np.asarray(yhat)

        # нормируем к [0,1] если надо
        if proba.ndim > 1 and proba.shape[1] >= 2:
            proba = proba[:, 1]
        elif proba.ndim == 1:
            proba = 1 / (1 + np.exp(-proba))  # на всякий (если logits)

        # собираем сигналы
        rows = []
        for ts, p in zip(sub.index, proba):
            sig = int(p >= float(proba_threshold)) * 2 - 1  # 1-> +1, 0-> -1 (или 0 — если хочешь без позы)
            rows.append({
                "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                "pair": pkey,
                "proba": float(p),
                "signal": int(sig if abs(p - 0.5) > 1e-6 else 0),
            })

        # записываем jsonl
        out_path = out_dir / f"{pkey.replace('/', '_')}.jsonl"
        mode = "a" if update and out_path.exists() else "w"
        with open(out_path, mode, encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")

        print(f"[ok] {pkey}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
