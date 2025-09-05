from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SHARPE_KEYS = ["sharpe", "sharpe_ratio", "sr"]
MAXDD_KEYS = ["maxdd", "max_dd", "max_drawdown", "mdd", "drawdown"]
EQUITY_KEYS = [
    "equity_last", "equity", "equity_final", "equity_end", "equity_usd",
    "pnl", "total_pnl", "pnl_total", "cum_pnl",
    "return_total", "total_return", "ret", "cum_return", "cumulative_return",
]


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _from_synonyms(m: Dict[str, Any], names: List[str]) -> Optional[Any]:
    if not isinstance(m, dict):
        return None
    lower = {str(k).lower(): v for k, v in m.items()}
    for name in names:
        v = lower.get(name.lower())
        if v is not None:
            return v
    return None


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Backtest summary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Backtest summary must be a JSON object with 'pairs' field")
    return obj


def select_champions(
    summary_path: str = "/app/data/backtest_results/_summary.json",
    registry_out: str = "/app/data/models/registry.json",
    *,
    sharpe_min: float = 0.0,
    maxdd_max: float = 1.0,
    top_k: int = 20,
    **_: Any,
) -> Dict[str, Any]:
    summary = _load_summary(Path(summary_path))
    pairs_info: Dict[str, Any] = summary.get("pairs", {}) or {}

    risk_pass: List[Tuple[str, Optional[float], Optional[float], Optional[float], bool]] = []
    fallback: List[Tuple[str, float, Optional[float], bool]] = []

    for pair_key, rec in pairs_info.items():
        rec = rec or {}
        met = rec.get("metrics", {}) or {}

        sharpe = _safe_float(_from_synonyms(met, SHARPE_KEYS))
        maxdd = _safe_float(_from_synonyms(met, MAXDD_KEYS))
        equity_like = _safe_float(_from_synonyms(met, EQUITY_KEYS))
        oof_used = bool(rec.get("oof_used", False))

        risk_valid = (sharpe is not None) and (maxdd is not None)

        if risk_valid:
            if (sharpe >= float(sharpe_min)) and (maxdd <= float(maxdd_max)):
                risk_pass.append((pair_key, sharpe, maxdd, equity_like, oof_used))
        else:
            if equity_like is not None:
                fallback.append((pair_key, equity_like, maxdd, oof_used))

    risk_pass.sort(key=lambda t: (t[1], (t[3] if t[3] is not None else float("-inf"))), reverse=True)

    selected: List[Tuple[str, Optional[float], Optional[float], Optional[float], bool]] = []
    seen = set()
    for item in risk_pass:
        if item[0] in seen:
            continue
        selected.append(item)
        seen.add(item[0])
        if len(selected) >= int(top_k):
            break

    if len(selected) < int(top_k):
        fallback.sort(key=lambda t: t[1], reverse=True)
        for pk, eq_like, maxdd, oof_used in fallback:
            if pk in seen:
                continue
            selected.append((pk, None, maxdd, eq_like, oof_used))
            seen.add(pk)
            if len(selected) >= int(top_k):
                break

    registry: Dict[str, Any] = {
        "criteria": {
            "sharpe_min": float(sharpe_min),
            "maxdd_max": float(maxdd_max),
            "top_k": int(top_k),
            "summary_path": str(summary_path),
            "equity_keys": EQUITY_KEYS,
        },
        "pairs": [],
    }

    for rank, (pk, sharpe, maxdd, eq_like, oof_used) in enumerate(selected, start=1):
        registry["pairs"].append(
            {
                "pair": pk,
                "rank": rank,
                "metrics": {
                    "sharpe": float(sharpe) if sharpe is not None else None,
                    "maxdd": float(maxdd) if maxdd is not None else None,
                    "equity_like": float(eq_like) if eq_like is not None else None,
                    "oof_used": bool(oof_used),
                },
            }
        )

    out_p = Path(registry_out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


if __name__ == "__main__":
    import sys
    summary = sys.argv[1] if len(sys.argv) > 1 else "/app/data/backtest_results/_summary.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "/app/data/models/registry.json"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    res = select_champions(summary_path=summary, registry_out=out, top_k=k)
    print(f"[select] wrote {len(res.get('pairs', []))} pairs -> {out}")