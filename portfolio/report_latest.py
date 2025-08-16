#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
from typing import Dict, List, Tuple, Optional

import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _latest_orders_path(out_dir: str) -> Optional[str]:
    paths = glob.glob(os.path.join(out_dir, "orders_*.json"))
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def _read_orders(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pair_key_to_parquet_path(pair_key: str) -> str:
    safe = pair_key.replace("/", "_")
    return os.path.join("data", "backtest_results", f"{safe}.parquet")


def _metrics_from_equity(ret: pd.Series) -> Dict[str, float]:
    ret = ret.dropna().astype(float)
    if ret.empty:
        return {"sharpe": float("nan"), "maxdd": float("nan"), "cumret": float("nan")}
    equity = (1.0 + ret).cumprod()
    vol = ret.std(ddof=0)
    sharpe = float(np.sqrt(252) * ret.mean() / vol) if vol > 0 else float("nan")
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    maxdd = float(dd.min()) if len(dd) else float("nan")
    cumret = float(equity.iloc[-1] - 1.0)
    return {"sharpe": sharpe, "maxdd": maxdd, "cumret": cumret}


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _plot_equity(path_png: str, ret: pd.Series, title: str):
    ret = ret.dropna().astype(float)
    if ret.empty:
        return
    eq = (1.0 + ret).cumprod()
    plt.figure(figsize=(8, 3))
    eq.plot()
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("equity")
    plt.tight_layout()
    plt.savefig(path_png, dpi=120)
    plt.close()


@click.command()
@click.option("--orders-dir", default="data/portfolio", show_default=True, help="Где искать orders_*.json")
@click.option("--backtests-dir", default="data/backtest_results", show_default=True)
@click.option("--lookback-bars", default=2000, type=int, show_default=True, help="Сколько последних баров брать для отчёта")
@click.option("--out-root", default="data/portfolio", show_default=True, help="Куда сохранять отчёт")
def main(orders_dir: str, backtests_dir: str, lookback_bars: int, out_root: str):
    """
    Мини-отчёт:
      - читает последние orders_*.json
      - вытаскивает per-pair backtests из data/backtest_results/*.parquet
      - считает метрики и портфельную equity за lookback window
      - сохраняет summary.{json,csv} + картинки equity
    """
    orders_path = _latest_orders_path(orders_dir)
    if not orders_path:
        raise SystemExit("[err] orders_*.json не найден. Сначала запусти aggregate_signals.py")

    orders = _read_orders(orders_path)
    orders_list = orders.get("orders", [])
    if not orders_list:
        raise SystemExit("[err] В orders.json нет записей 'orders'")

    # Папка отчёта
    ts = os.path.splitext(os.path.basename(orders_path))[0].replace("orders_", "")
    out_dir = os.path.join(out_root, f"report_{ts}")
    _ensure_dir(out_dir)
    charts_dir = os.path.join(out_dir, "charts")
    _ensure_dir(charts_dir)

    # Собираем табличку по ордерам (pair-level)
    rows = []
    weights = {}
    for o in orders_list:
        pair = str(o.get("pair"))
        w = float(o.get("weight", 0.0))
        prob = float(o.get("proba", np.nan))
        z = float(o.get("z", np.nan))
        ts_iso = str(o.get("ts"))
        legs = o.get("legs", [])
        # сохраним веса для агрегации портфеля
        weights[pair] = w
        rows.append({
            "pair": pair, "ts": ts_iso, "weight": w,
            "proba": prob, "z": z,
            "notional_per_leg": float(o.get("notional_per_leg", np.nan)),
            "n_legs": len(legs),
            "legA": f"{legs[0]['side']} {legs[0]['symbol']}" if len(legs) >= 1 else "",
            "legB": f"{legs[1]['side']} {legs[1]['symbol']}" if len(legs) >= 2 else "",
        })
    df_orders = pd.DataFrame(rows).sort_values("proba", ascending=False).reset_index(drop=True)

    # Тянем бэктесты (если есть) и считаем метрики за окно
    pair_metrics: List[Dict] = []
    pair_returns: Dict[str, pd.Series] = {}
    for pair in df_orders["pair"]:
        pq = _pair_key_to_parquet_path(pair)
        if not os.path.exists(pq):
            pair_metrics.append({"pair": pair, "sharpe": np.nan, "maxdd": np.nan, "cumret": np.nan})
            continue
        df_bt = pd.read_parquet(pq)
        # ожидаем столбец 'returns' (как в нашем backtest.runner)
        if "returns" not in df_bt.columns:
            # fallback: если есть 'equity', восстановим лог-доходности
            if "equity" in df_bt.columns:
                eq = df_bt["equity"].astype(float)
                ret = eq.pct_change().fillna(0.0)
            else:
                pair_metrics.append({"pair": pair, "sharpe": np.nan, "maxdd": np.nan, "cumret": np.nan})
                continue
        else:
            ret = df_bt["returns"].astype(float)

        if lookback_bars and lookback_bars > 0:
            ret = ret.iloc[-lookback_bars:]

        pair_returns[pair] = ret.copy()
        m = _metrics_from_equity(ret)
        m["pair"] = pair
        pair_metrics.append(m)

        # картинка по паре
        _plot_equity(os.path.join(charts_dir, f"equity_{pair.replace('/','_')}.png"), ret, f"{pair} equity (last {len(ret)} bars)")

    df_metrics = pd.DataFrame(pair_metrics)

    # Портфельная equity: сумма взвешенных доходностей пар (где есть данные)
    # Нормируем веса на те пары, по которым есть ret
    valid_pairs = [p for p in df_orders["pair"] if p in pair_returns]
    if valid_pairs:
        w = np.array([weights[p] for p in valid_pairs], dtype=float)
        if w.sum() <= 0:
            w = np.ones_like(w) / len(w)
        else:
            w = w / w.sum()

        # выравниваем по времени (inner join по индексу)
        aligned = pd.concat([pair_returns[p] for p in valid_pairs], axis=1, join="inner")
        aligned.columns = valid_pairs
        port_ret = aligned.dot(w)
        _plot_equity(os.path.join(charts_dir, f"equity_portfolio.png"), port_ret, f"Portfolio equity (last {len(port_ret)} bars)")

        port_metrics = _metrics_from_equity(port_ret)
    else:
        port_ret = pd.Series(dtype=float)
        port_metrics = {"sharpe": np.nan, "maxdd": np.nan, "cumret": np.nan}

    # Сводная таблица (orders + metrics)
    df_summary = df_orders.merge(df_metrics, on="pair", how="left")

    # Сохраняем
    df_summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "orders_path": orders_path,
                "lookback_bars": int(lookback_bars),
                "pairs_count": int(len(df_orders)),
            },
            "portfolio_metrics": port_metrics,
            "pairs": df_summary.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)

    # Консольный вывод (короткая витрина)
    print(f"[ok] report saved -> {out_dir}")
    print("Portfolio metrics:", port_metrics)
    print("\nTop pairs:")
    show_cols = ["pair", "weight", "proba", "z", "sharpe", "maxdd", "cumret"]
    print(df_summary[show_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
