from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
<<<<<<< HEAD
=======

>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700
import click


def _read_json(p: Path) -> Any:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def _try_load_orders(orders_json: Path) -> List[Dict[str, Any]]:
    obj = _read_json(orders_json)
    if isinstance(obj, dict) and isinstance(obj.get("orders"), list):
        return list(obj["orders"])
    orders_path = obj.get("orders_path") if isinstance(obj, dict) else None
    if isinstance(orders_path, str):
        jp = Path(orders_path)
        if not jp.is_absolute():
            jp = orders_json.parent / jp
        return _read_jsonl(jp)
    sel = obj.get("selected") if isinstance(obj, dict) else None
    if isinstance(sel, list):
        return list(sel)
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported latest_orders.json format: {orders_json}")


def _load_backtest_metrics(summary_path: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if not summary_path:
        return {}
    obj = _read_json(summary_path)
    pairs = obj.get("pairs")
    out: Dict[str, Dict[str, float]] = {}
    if isinstance(pairs, dict):
        for pair, d in pairs.items():
            met = d.get("metrics", {}) if isinstance(d, dict) else {}
            if isinstance(met, dict):
                out[str(pair)] = {
                    "sharpe": float(met.get("sharpe", 0.0)),
                    "maxdd": float(met.get("maxdd", 0.0)),
                }
    elif isinstance(pairs, list):
        for it in pairs:
            if not isinstance(it, dict):
                continue
            pair = it.get("pair") or it.get("name") or it.get("key")
            met = it.get("metrics", {})
            if pair and isinstance(met, dict):
                out[str(pair)] = {
                    "sharpe": float(met.get("sharpe", 0.0)),
                    "maxdd": float(met.get("maxdd", 0.0)),
                }
    return out


def _load_champions(registry_path: Optional[Path]) -> List[Dict[str, Any]]:
<<<<<<< HEAD
=======
    """
    Accept registry formats:
      {'pairs': [{'pair': 'A__B', 'model_path': '...', 'metrics': {...}}, ...]}
      or similar lists under keys like 'items', 'selected', 'champions', 'models', 'registry'.
    """
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700
    if not registry_path or not registry_path.exists():
        return []
    obj = _read_json(registry_path)
    for key in ("pairs", "items", "selected", "champions", "models", "registry"):
        arr = obj.get(key)
        if isinstance(arr, list):
            out = []
            for it in arr:
                if not isinstance(it, dict):
                    continue
                pair = it.get("pair") or it.get("name") or it.get("key")
                model = it.get("model_path") or it.get("model") or it.get("path")
                met = it.get("metrics") or {}
                if pair:
                    out.append({"pair": str(pair), "model_path": model, "metrics": met})
            if out:
                return out
<<<<<<< HEAD
=======
    # mapping dict form {'A__B': {...}}
>>>>>>> 227f8359141ef32f8d3f3d29b3512f9332ccc700
    if isinstance(obj, dict) and any(isinstance(v, dict) for v in obj.values()):
        out = []
        for k, v in obj.items():
            if k in ("pairs", "items", "selected", "champions", "models", "registry"):
                continue
            if isinstance(v, dict):
                model = v.get("model_path") or v.get("model") or v.get("path")
                met = v.get("metrics") or {}
                out.append({"pair": str(k), "model_path": model, "metrics": met})
        return out
    return []


def _fmt_side(v: Any) -> str:
    if isinstance(v, (int, float)):
        if v > 0: return "LONG"
        if v < 0: return "SHORT"
        return "FLAT"
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("long", "buy", "1", "+1"): return "LONG"
        if s in ("short", "sell", "-1"): return "SHORT"
        return s.upper()
    return str(v)


def _render_markdown(orders: List[Dict[str, Any]],
                     metrics: Dict[str, Dict[str, float]],
                     champions: List[Dict[str, Any]]) -> str:
    dt = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines.append("# MNTrading — Latest Orders Report")
    lines.append("")
    lines.append(f"_Generated at: **{dt}**_")
    lines.append("")

    if champions:
        lines.append("## Champions")
        lines.append("")
        lines.append("| Pair | Model | Sharpe | MaxDD |")
        lines.append("|---|---|---:|---:|")
        for ch in champions:
            pair = ch.get("pair", "")
            model = ch.get("model_path")
            model_name = Path(model).name if model else ""
            met = ch.get("metrics") or {}
            sharpe = float(met.get("sharpe", metrics.get(pair, {}).get("sharpe", 0.0)))
            maxdd = float(met.get("maxdd", metrics.get(pair, {}).get("maxdd", 0.0)))
            lines.append(f"| {pair} | {model_name} | {sharpe:.3f} | {maxdd:.4f} |")
        lines.append("")

    lines.append("## Selected Orders")
    lines.append("")
    lines.append("| Pair | Side | Proba | Notes |")
    lines.append("|---|---:|---:|---|")
    for o in orders:
        pair = str(o.get("pair") or o.get("symbol") or o.get("pair_key") or "?")
        side = _fmt_side(o.get("side"))
        proba = o.get("proba")
        try:
            proba_s = f"{float(proba):.3f}" if proba is not None else ""
        except Exception:
            proba_s = str(proba) if proba is not None else ""
        note = o.get("note") or ""
        lines.append(f"| {pair} | {side} | {proba_s} | {note} |")

    if metrics:
        lines.append("")
        lines.append("## Backtest Metrics (per pair)")
        lines.append("")
        lines.append("| Pair | Sharpe | MaxDD |")
        lines.append("|---|---:|---:|")
        for pair, met in metrics.items():
            lines.append(f"| {pair} | {met.get('sharpe', 0.0):.3f} | {met.get('maxdd', 0.0):.4f} |")

    lines.append("")
    return "\n".join(lines)


@click.command()
@click.option("--orders-json", type=click.Path(path_type=Path), required=True, help="Path to latest_orders.json produced by aggregator.")
@click.option("--backtest-summary", type=click.Path(path_type=Path), required=False, help="Path to backtest _summary.json for metrics (optional).")
@click.option("--registry", type=click.Path(path_type=Path), required=False, help="Path to models registry.json (optional, to show champions).")
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True, help="Output Markdown file path.")
def main(orders_json: Path, backtest_summary: Optional[Path], registry: Optional[Path], out_path: Path):
    orders = _try_load_orders(orders_json)
    metrics = _load_backtest_metrics(backtest_summary) if backtest_summary else {}
    champions = _load_champions(registry) if registry else []

    md = _render_markdown(orders, metrics, champions)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    latest_alias = out_path.parent / "_latest_report.md"
    try:
        latest_alias.write_text(md, encoding="utf-8")
    except Exception:
        pass

    print(f"[report] wrote {out_path}")


if __name__ == "__main__":
    main()