#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import click
import pandas as pd
import numpy as np


@dataclass
class OrderLeg:
    symbol: str
    side: str   # "buy" or "sell"
    qty: float
    price: float


@dataclass
class PairOrder:
    pair: str
    ts: str
    proba: float
    z: float
    weight: float
    notional_per_leg: float
    legs: List[OrderLeg]


def _ensure_dirs(path: str):
    os.makedirs(path, exist_ok=True)


def _load_signals_last(signals_dir: str, verbose: bool = False) -> pd.DataFrame:
    """
    Читает все *.jsonl, берёт ПОСЛЕДНЮЮ строку каждого файла.
    Ожидаемые поля: ts, pair, proba, signal, price, z, threshold
    """
    files = sorted(glob.glob(os.path.join(signals_dir, "*.jsonl")))
    if verbose:
        click.echo(f"[dbg] signals_dir={signals_dir}, files_found={len(files)}")
    rows = []
    for path in files:
        try:
            last = None
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        last = line
            if last:
                rec = json.loads(last)
                rows.append(rec)
        except Exception as e:
            click.echo(f"[warn] failed to read {path}: {e}")
    if not rows:
        return pd.DataFrame(columns=["pair", "ts", "proba", "signal", "price", "z", "threshold"])
    df = pd.DataFrame(rows)
    df["pair"] = df["pair"].astype(str)
    return df


def _load_pairs_manifest(path: str, verbose: bool = False) -> Dict[str, str]:
    if not os.path.exists(path):
        if verbose:
            click.echo(f"[dbg] pairs_manifest not found: {path} (using prices from signals)")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    for row in data.get("pairs", []):
        pkey = str(row.get("pair", "")).strip()
        ppath = str(row.get("path", "")).strip()
        if pkey and ppath:
            m[pkey] = ppath
    if verbose:
        click.echo(f"[dbg] manifest pairs loaded: {len(m)}")
    return m


def _parse_pair(pair_key: str) -> Tuple[str, str]:
    if "__" in pair_key:
        a, b = pair_key.split("__", 1)
        return a.strip(), b.strip()
    for sep in [",", "|", ";", " "]:
        if sep in pair_key:
            p = [t.strip() for t in pair_key.split(sep) if t.strip()]
            if len(p) == 2:
                return p[0], p[1]
    raise ValueError(f"Unrecognized pair key: {pair_key}")


def _get_prices_at_ts(pair_path: str, ts_iso: str) -> Tuple[float, float]:
    df = pd.read_parquet(pair_path)
    if "pa" not in df.columns or "pb" not in df.columns:
        nums = df.select_dtypes(include="number")
        if nums.shape[1] == 0:
            return np.nan, np.nan
        v = float(nums.iloc[-1, 0])
        return v, v
    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.to_datetime(ts_iso, utc=True)
        df2 = df.loc[:ts]
        row = df2.iloc[-1] if len(df2) else df.iloc[-1]
    else:
        row = df.iloc[-1]
    return float(row["pa"]), float(row["pb"])


def _weights_equal(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    return np.ones(n) / n


@click.command()
@click.option("--signals-dir", default="data/signals", show_default=True)
@click.option("--pairs-manifest", default="data/features/pairs/_manifest.json", show_default=True)
@click.option("--min-proba", default=0.55, type=float, show_default=True)
@click.option("--top-k", default=10, type=int, show_default=True)
@click.option("--scheme", type=click.Choice(["equal_weight"]), default="equal_weight", show_default=True)
@click.option("--equity", default=10000.0, type=float, show_default=True)
@click.option("--leverage", default=1.0, type=float, show_default=True)
@click.option("--out-dir", default="data/portfolio", show_default=True)
@click.option("--verbose/--no-verbose", default=True, show_default=True)
def main(signals_dir, pairs_manifest, min_proba, top_k, scheme, equity, leverage, out_dir, verbose):
    """
    Собираем последние сигналы и формируем ордера (лонг/шорт) для парного трейдинга.
    """
    _ensure_dirs(out_dir)

    sig = _load_signals_last(signals_dir, verbose=verbose)
    if verbose:
        click.echo(f"[dbg] latest-signal rows loaded: {len(sig)}")

    if sig.empty:
        click.echo("[warn] no signals found; nothing to aggregate")
        return

    # фильтр
    before = len(sig)
    sig["proba"] = pd.to_numeric(sig["proba"], errors="coerce")
    sig["signal"] = pd.to_numeric(sig["signal"], downcast="integer", errors="coerce")
    sig = sig[(sig["signal"].fillna(0).astype(int) != 0) & (sig["proba"].fillna(0.0) >= float(min_proba))].copy()
    after = len(sig)
    if verbose:
        click.echo(f"[dbg] filter: before={before}, after={after}, min_proba={min_proba}")

    if sig.empty:
        click.echo("[done] no qualifying signals (signal==0 or proba<threshold). Try lowering --min-proba.")
        return

    sig = sig.sort_values(["proba", "z"], ascending=[False, False])
    if top_k and top_k > 0:
        sig = sig.head(top_k)
    if verbose:
        click.echo(f"[dbg] selected pairs: {len(sig)} (top_k={top_k})")

    # веса
    n = len(sig)
    weights = _weights_equal(n)
    sig = sig.reset_index(drop=True)
    sig["weight"] = weights

    manifest = _load_pairs_manifest(pairs_manifest, verbose=verbose)

    orders: List[PairOrder] = []
    for _, row in sig.iterrows():
        pair_key = str(row["pair"])
        try:
            a, b = _parse_pair(pair_key)
        except Exception as e:
            click.echo(f"[warn] skip pair '{pair_key}': {e}")
            continue

        pa, pb = (row.get("price", np.nan), np.nan)
        ppath = manifest.get(pair_key)
        if ppath and os.path.exists(ppath):
            try:
                pa, pb = _get_prices_at_ts(ppath, str(row["ts"]))
            except Exception as e:
                if verbose:
                    click.echo(f"[dbg] price fallback for {pair_key}: {e}")

        w = float(row["weight"])
        notional_pair = float(equity) * float(leverage) * w
        notional_per_leg = notional_pair / 2.0

        sig_dir = int(row["signal"])
        if not np.isfinite(pa) or pa <= 0:
            pa = float(row.get("price", np.nan))
        if not np.isfinite(pb) or pb <= 0:
            pb = float(pa)

        qty_a = notional_per_leg / float(pa) if np.isfinite(pa) and pa > 0 else 0.0
        qty_b = notional_per_leg / float(pb) if np.isfinite(pb) and pb > 0 else 0.0

        if sig_dir == 1:
            legs = [OrderLeg(a, "buy", qty_a, float(pa)), OrderLeg(b, "sell", qty_b, float(pb))]
        else:
            legs = [OrderLeg(a, "sell", qty_a, float(pa)), OrderLeg(b, "buy", qty_b, float(pb))]

        orders.append(PairOrder(
            pair=pair_key, ts=str(row["ts"]), proba=float(row["proba"]), z=float(row.get("z", np.nan)),
            weight=w, notional_per_leg=notional_per_leg, legs=legs
        ))

    if not orders:
        click.echo("[done] no orders produced (nothing passed parsing/pricing)")
        return

    try:
        ts_latest = pd.to_datetime(sig["ts"]).max().strftime("%Y%m%d_%H%M%SZ")
    except Exception:
        ts_latest = "latest"

    out_json = os.path.join(out_dir, f"orders_{ts_latest}.json")
    out_csv = os.path.join(out_dir, f"orders_{ts_latest}.csv")

    payload = {
        "meta": {
            "equity": float(equity),
            "leverage": float(leverage),
            "scheme": scheme,
            "min_proba": float(min_proba),
            "top_k": int(top_k),
            "count": len(orders),
            "signals_dir": signals_dir,
            "pairs_manifest": pairs_manifest,
            "version": "1.2"
        },
        "orders": [
            {
                "pair": o.pair,
                "ts": o.ts,
                "proba": o.proba,
                "z": o.z,
                "weight": o.weight,
                "notional_per_leg": o.notional_per_leg,
                "legs": [{"symbol": l.symbol, "side": l.side, "qty": float(l.qty), "price": float(l.price)} for l in o.legs]
            }
            for o in orders
        ]
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # CSV по ногам
    rows_csv = []
    for o in orders:
        for l in o.legs:
            rows_csv.append({
                "ts": o.ts, "pair": o.pair, "proba": o.proba, "z": o.z, "weight": o.weight,
                "side": l.side, "symbol": l.symbol, "qty": float(l.qty), "price": float(l.price),
                "notional_leg": float(o.notional_per_leg)
            })
    pd.DataFrame(rows_csv).to_csv(out_csv, index=False, encoding="utf-8")

    click.echo(f"[ok] orders saved -> {out_json}")
    click.echo(f"[ok] orders saved -> {out_csv}")
    click.echo(f"[ok] total orders: {len(orders)}")
    click.echo("Preview (first 8):")
    prev = pd.DataFrame(rows_csv).head(8)
    with pd.option_context("display.max_columns", None, "display.width", 140):
        click.echo(prev.to_string(index=False))


if __name__ == "__main__":
    main()
