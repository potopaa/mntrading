#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate latest per-pair signals into hedged pair orders.

Usage (example):
    python portfolio/aggregate_signals.py \
        --signals-dir data/signals \
        --pairs-manifest data/features/pairs/_manifest.json \
        --min-proba 0.55 --top-k 10 \
        --scheme equal_weight --equity 10000 --leverage 1.0

Outputs (under data/portfolio by default directory layout):
    - orders_<UTC_ISO>.jsonl      # one JSON object per selected pair
    - latest_orders.json          # summary object with same content (array)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd


# ------------------------- Time/IO helpers -------------------------
def _utf8_stdio():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def now_utc() -> pd.Timestamp:
    """Return a tz-aware UTC Timestamp (safe across pandas versions)."""
    return pd.Timestamp.now(tz="UTC")


def _read_last_jsonl(path: Path) -> Optional[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            buf = bytearray()
            i = size - 1
            # read backwards until we hit a non-empty line
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
        return json.loads(line)
    except Exception:
        return None


def _normalize_pair_key(s: str) -> str:
    """Allow AAA_USDT__BBB_USDT or AAA/USDT__BBB/USDT."""
    if "__" in s and "/" not in s:
        a, b = s.split("__", 1)
        return a.replace("_", "/") + "__" + b.replace("_", "/")
    return s


def _split_pair(pair_key: str) -> Tuple[str, str]:
    if "__" not in pair_key:
        return pair_key, pair_key
    a, b = pair_key.split("__", 1)
    return a, b


# ------------------------- Models -------------------------
@dataclass
class PairQuote:
    ts: pd.Timestamp
    a_price: float
    b_price: float
    beta: float


# ------------------------ Loaders --------------------------
def load_latest_signals(signals_dir: Path) -> pd.DataFrame:
    """
    Read the last record from each *.jsonl under signals_dir.
    Returns DataFrame with columns: [ts, pair, a, b, signal, proba] (+ z if present)
    """
    files = sorted(signals_dir.glob("*.jsonl"))
    print(f"[dbg] signals_dir={signals_dir}, files_found={len(files)}")

    rows: List[dict] = []
    for p in files:
        obj = _read_last_jsonl(p)
        if not obj:
            continue
        rows.append(obj)

    if not rows:
        print("[warn] no signals found")
        return pd.DataFrame(columns=["ts", "pair", "a", "b", "signal", "proba"])

    df = pd.DataFrame(rows)
    # normalize ts
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        df["ts"] = now_utc()
    # normalize pair key
    if "pair" in df.columns:
        df["pair"] = df["pair"].astype(str).map(_normalize_pair_key)
    else:
        # derive from filenames as a fallback
        df["pair"] = [p.stem.replace("_", "/") for p in files[: len(df)]]

    # normalize side/signal
    if "signal" not in df.columns:
        side_map = {
            "long_spread": 1,
            "short_spread": -1,
            "long": 1,
            "short": -1,
            "buy": 1,
            "sell": -1,
            "flat": 0,
            "close": 0,
            "none": 0,
        }
        if "side" in df.columns:
            df["signal"] = (
                df["side"].astype(str).str.lower().map(side_map).fillna(0).astype(int)
            )
        else:
            df["signal"] = 0
    else:
        df["signal"] = pd.to_numeric(df["signal"], downcast="integer", errors="coerce").fillna(0).astype(int)

    # proba present?
    if "proba" not in df.columns:
        df["proba"] = 0.5
    df["proba"] = pd.to_numeric(df["proba"], errors="coerce").fillna(0.5).clip(0.0, 1.0)

    # optionally keep z
    if "z" in df.columns:
        df["z"] = pd.to_numeric(df["z"], errors="coerce")

    # a/b symbols
    if "a" not in df.columns or "b" not in df.columns:
        a_b = df["pair"].map(_split_pair)
        df["a"] = [x[0] for x in a_b]
        df["b"] = [x[1] for x in a_b]

    print(f"[dbg] latest-signal rows loaded: {len(df)}")
    return df[["ts", "pair", "a", "b", "signal", "proba"] + (["z"] if "z" in df.columns else [])]


def load_pair_quote(parquet_path: Path) -> Optional[PairQuote]:
    """
    Read last prices and beta from features parquet.
    Expected columns: ['ts','a_close','b_close','beta']
    """
    if not parquet_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path, columns=["ts", "a_close", "b_close", "beta"])
    except Exception:
        df = pd.read_parquet(parquet_path)
    cols = {c.lower(): c for c in df.columns}
    # normalize columns
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            df["ts"] = np.arange(len(df))
    if "a_close" not in cols and "a" in cols:
        df = df.rename(columns={cols["a"]: "a_close"})
    if "b_close" not in cols and "b" in cols:
        df = df.rename(columns={cols["b"]: "b_close"})

    df = df.dropna(subset=["ts"]).sort_values("ts")
    if df.empty:
        return None
    r = df.iloc[-1]
    ts = pd.to_datetime(r["ts"], utc=True, errors="coerce")
    a_price = float(r.get("a_close", np.nan))
    b_price = float(r.get("b_close", np.nan))
    beta = float(r.get("beta", np.nan))
    if not np.isfinite(a_price) or not np.isfinite(b_price):
        return None
    if not np.isfinite(beta):
        beta = 1.0
    return PairQuote(ts=ts, a_price=a_price, b_price=b_price, beta=beta)


def load_manifest_pairs_paths(manifest_path: Path) -> Dict[str, Path]:
    """
    Return {pair_key: features_parquet_path}
    """
    obj = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    mp: Dict[str, Path] = {}
    for it in obj.get("items", []):
        pk = _normalize_pair_key(str(it.get("pair")))
        path = Path(it.get("path", "")).resolve()
        if pk and path.exists():
            mp[pk] = path
    return mp


# -------------------- Allocation / Orders --------------------
def build_pair_orders_equal_weight(
    pair_key: str,
    signal: int,
    proba: float,
    pq: PairQuote,
    gross_budget: float,
) -> dict:
    """
    Create hedged pair order so that:
        units_a = q, units_b = beta * q
        q = gross_budget / (a_price + beta * b_price)
    Side:
        signal = +1 -> long spread:  long A, short (beta * B)
        signal = -1 -> short spread: short A, long  (beta * B)
    """
    a_sym, b_sym = _split_pair(pair_key)
    beta = abs(float(pq.beta)) if np.isfinite(pq.beta) else 1.0
    denom = pq.a_price + beta * pq.b_price
    if denom <= 0:
        denom = max(1e-6, abs(denom))
    q = float(gross_budget) / denom

    qty_a = q
    qty_b = beta * q

    # Sides
    if signal > 0:
        side_a = "buy"    # long A
        side_b = "sell"   # short B
    elif signal < 0:
        side_a = "sell"   # short A
        side_b = "buy"    # long B
    else:
        # flat: return a no-op order object
        return {
            "pair": pair_key,
            "side_spread": "flat",
            "legs": [],
            "gross_notional": 0.0,
            "proba": float(proba),
            "px": {"a": pq.a_price, "b": pq.b_price, "beta": beta},
        }

    leg_a_notional = qty_a * pq.a_price
    leg_b_notional = qty_b * pq.b_price
    gross = float(abs(leg_a_notional) + abs(leg_b_notional))

    return {
        "pair": pair_key,
        "side_spread": "long_spread" if signal > 0 else "short_spread",
        "proba": float(proba),
        "px": {"a": pq.a_price, "b": pq.b_price, "beta": beta},
        "gross_notional": gross,
        "legs": [
            {"symbol": a_sym, "side": side_a, "qty": float(qty_a), "price": float(pq.a_price),
             "notional": float(leg_a_notional)},
            {"symbol": b_sym, "side": side_b, "qty": float(qty_b), "price": float(pq.b_price),
             "notional": float(leg_b_notional)},
        ],
    }


# ---------------------------- CLI ----------------------------
@click.command()
@click.option("--signals-dir", required=True, type=click.Path(exists=True, file_okay=False),
              help="Directory with per-pair *.jsonl signals")
@click.option("--pairs-manifest", required=True, type=click.Path(exists=True, dir_okay=False),
              help="features manifest (data/features/pairs/_manifest.json)")
@click.option("--min-proba", default=0.55, show_default=True, type=float,
              help="Keep signals with proba >= this value")
@click.option("--top-k", default=10, show_default=True, type=int,
              help="Keep top-K strongest signals")
@click.option("--scheme", default="equal_weight", show_default=True, type=click.Choice(["equal_weight"]),
              help="Allocation scheme")
@click.option("--equity", default=10000.0, show_default=True, type=float,
              help="Account equity in quote currency")
@click.option("--leverage", default=1.0, show_default=True, type=float,
              help="Gross leverage (1.0 means gross = equity)")
@click.option("--out-dir", default="data/portfolio", show_default=True, type=click.Path(file_okay=False),
              help="Directory to write orders")
def main(
    signals_dir: str,
    pairs_manifest: str,
    min_proba: float,
    top_k: int,
    scheme: str,
    equity: float,
    leverage: float,
    out_dir: str,
):
    _utf8_stdio()

    signals_root = Path(signals_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Load latest signals
    sig = load_latest_signals(signals_root)
    if sig.empty:
        print("[warn] no signals to aggregate; exiting")
        return

    # 2) Filter & rank
    sig = sig[(sig["signal"] != 0) & (sig["proba"] >= float(min_proba))].copy()
    if sig.empty:
        print("[warn] no signals pass the min_proba filter; exiting")
        return

    sig["rank_score"] = sig["proba"].astype(float) * sig["signal"].abs()
    sig = sig.sort_values(["rank_score", "proba"], ascending=[False, False])
    if top_k > 0:
        sig = sig.head(int(top_k)).reset_index(drop=True)

    print(f"[ok] selected: {len(sig)} pairs (min_proba={min_proba}, top_k={top_k})")

    # 3) Read prices/beta
    paths = load_manifest_pairs_paths(Path(pairs_manifest))

    # 4) Build orders
    gross_capital = float(equity) * float(leverage)
    per_pair_gross = gross_capital / max(1, len(sig))
    orders: List[dict] = []
    skipped: List[Tuple[str, str]] = []

    for _, r in sig.iterrows():
        pk = _normalize_pair_key(str(r["pair"]))
        fpath = paths.get(pk)
        if not fpath:
            skipped.append((pk, "features parquet not found in manifest"))
            continue
        pq = load_pair_quote(fpath)
        if pq is None:
            skipped.append((pk, "cannot load latest price/beta"))
            continue

        if scheme == "equal_weight":
            od = build_pair_orders_equal_weight(
                pair_key=pk,
                signal=int(r["signal"]),
                proba=float(r["proba"]),
                pq=pq,
                gross_budget=per_pair_gross,
            )
        else:
            skipped.append((pk, f"unsupported scheme: {scheme}"))
            continue

        orders.append({
            "ts": now_utc().isoformat(),
            **od
        })

    if not orders:
        print("[warn] no orders produced")
        return

    # 5) Write outputs
    ts_token = now_utc().strftime("%Y%m%dT%H%M%SZ")
    out_jsonl = out_root / f"orders_{ts_token}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as w:
        for od in orders:
            w.write(json.dumps(od, ensure_ascii=False) + "\n")

    latest_path = out_root / "latest_orders.json"
    latest_path.write_text(json.dumps({"orders": orders}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {len(orders)} orders -> {out_jsonl}")
    print(f"[ok] summary -> {latest_path}")

    if skipped:
        for pk, msg in skipped[:10]:
            print(f"[skip] {pk}: {msg}")
        if len(skipped) > 10:
            print(f"[skip] ... and {len(skipped) - 10} more")


if __name__ == "__main__":
    main()
