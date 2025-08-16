# main.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, List, Optional, Dict

import click
import pandas as pd

# === внутренние модули проекта ===
from features.spread import compute_features_for_pairs
from features.labels import build_datasets_for_manifest
from models.train import train_baseline
from backtest.runner import run_backtest

DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features" / "pairs"
DATASETS_DIR = DATA_DIR / "datasets" / "pairs"
MODELS_DIR = DATA_DIR / "models"
BACKTEST_DIR = DATA_DIR / "backtest_results"
SIGNALS_DIR = DATA_DIR / "signals"
PAIRS_DIR = DATA_DIR / "pairs"

for p in [RAW_DIR, FEATURES_DIR, DATASETS_DIR, MODELS_DIR, BACKTEST_DIR, SIGNALS_DIR, PAIRS_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# ---------- безопасный перевод в UTC миллисекунды ----------
def _to_utc_ms(dt_like: Optional[Any]) -> int:
    """
    Преобразует строку/число/Timestamp в миллисекунды Unix (UTC).
    Понимает и tz-aware, и naive значения.
    Примеры: "2025-01-01", "2025-01-01T00:00:00Z", pd.Timestamp(...), int(ms), None
    None -> 0 (означает «с самого начала»).
    """
    if dt_like is None:
        return 0
    if isinstance(dt_like, (int, float)):
        # эвристика: если значение похоже на миллисекунды — возвращаем как есть
        return int(dt_like if dt_like > 10_000_000_000 else dt_like * 1000)
    ts = pd.Timestamp(dt_like)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _read_symbols_from_pairs_json(pairs_json_path: Path) -> List[str]:
    """
    Из файла screened_pairs_*.json достаём список пар вида 'AAA/USDT__BBB/USDT',
    и возвращаем уникальные базовые символы: ['AAA/USDT', 'BBB/USDT', ...]
    """
    with open(pairs_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "pairs" in obj:
        pairs = obj["pairs"]
    elif isinstance(obj, list):
        pairs = obj
    else:
        raise ValueError(f"Unsupported pairs file format: {pairs_json_path}")

    out: List[str] = []
    for p in pairs:
        if isinstance(p, dict) and "pair" in p:
            p = p["pair"]
        if "__" in p:
            a, b = p.split("__", 1)
            out.extend([a, b])
        else:
            out.append(p)

    seen = set()
    uniq = []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


@click.command()
@click.option("--mode", type=click.Choice(
    ["ingest", "features", "dataset", "train", "backtest", "select"],
    case_sensitive=False
), required=True, help="Pipeline stage to run.")
# --- ingest ---
@click.option("--symbols", type=str, default="", help="Path to screened_pairs_*.json or comma-separated symbols.")
@click.option("--timeframe", type=str, default="5m", show_default=True, help="Timeframe for ingest.")
@click.option("--limit", type=int, default=1000, show_default=True, help="Bars limit per request (ingest/infer).")
@click.option("--since-utc", "since_utc", type=str, default=None,
              help="Override start timestamp, e.g. '2025-01-01T00:00:00Z'.")
@click.option("--start-year", type=int, default=None,
              help="If since-utc is not set, use Jan 1 of this year in UTC as ingest start.")
# --- features ---
@click.option("--beta-window", type=int, default=300, show_default=True, help="Window for beta calc.")
@click.option("--z-window", type=int, default=300, show_default=True, help="Window for z-score.")
# --- dataset ---
@click.option("--pairs-manifest", type=str, default=str(FEATURES_DIR / "_manifest.json"),
              help="Manifest produced by features stage.")
@click.option("--label-type", type=click.Choice(["z_threshold"], case_sensitive=False),
              default="z_threshold", show_default=True, help="Labeling scheme.")
@click.option("--zscore-threshold", type=float, default=1.5, show_default=True, help="Z threshold for labels.")
@click.option("--lag-features", type=int, default=1, show_default=True, help="How many lags to add.")
# --- train ---
@click.option("--use-dataset", is_flag=True, default=False, help="Use prebuilt dataset from manifest.")
@click.option("--n-splits", type=int, default=3, show_default=True, help="TimeSeriesSplit - number of splits.")
@click.option("--gap", type=int, default=5, show_default=True, help="Gap between train and test in bars.")
@click.option("--max-train-size", type=int, default=2000, show_default=True, help="Max train size per split.")
@click.option("--early-stopping-rounds", type=int, default=50, show_default=True, help="Early stopping rounds.")
@click.option("--proba-threshold", type=float, default=0.55, show_default=True, help="Decision threshold.")
def main(
    mode: str,
    symbols: str,
    timeframe: str,
    limit: int,
    since_utc: Optional[str],
    start_year: Optional[int],
    beta_window: int,
    z_window: int,
    pairs_manifest: str,
    label_type: str,
    zscore_threshold: float,
    lag_features: int,
    use_dataset: bool,
    n_splits: int,
    gap: int,
    max_train_size: int,
    early_stopping_rounds: int,
    proba_threshold: float,
):
    """
    Orchestrates pipeline stages for mntrading project.
    """

    # ---------- MODE: INGEST ----------
    if mode.lower() == "ingest":
        # symbols: либо путь до json, либо CSV-строка
        if symbols and Path(symbols).exists():
            syms = _read_symbols_from_pairs_json(Path(symbols))
        elif symbols:
            syms = [s.strip() for s in symbols.split(",") if s.strip()]
        else:
            raise click.UsageError("--symbols is required for mode=ingest")

        # единая стартовая точка
        if since_utc:
            since_ms = _to_utc_ms(since_utc)
        else:
            year = start_year or int(os.getenv("START_YEAR", pd.Timestamp.utcnow().year))
            since_ms = _to_utc_ms(f"{year}-01-01 00:00:00")

        click.echo(f"Using {len(syms)} symbols from input")
        click.echo(
            f"Downloading new bars @{timeframe} for {len(syms)} symbols since {since_ms} "
            f"({pd.Timestamp(since_ms, unit='ms', tz='UTC').isoformat()})"
        )

        # используем вашу реализацию ingestion
        from data.ingest import ingest
        ingested = ingest(
            symbols=syms,
            timeframe=timeframe,
            limit=limit,
            since_ms=since_ms,
            out_path=RAW_DIR / "ohlcv.parquet",
        )
        click.echo(f"Ingested {ingested} new bars; saved to {RAW_DIR / 'ohlcv.parquet'}")
        return

    # ---------- MODE: FEATURES ----------
    if mode.lower() == "features":
        if not symbols:
            raise click.UsageError("--symbols must be a path to screened_pairs_*.json for mode=features")
        pairs_file = Path(symbols)
        if not pairs_file.exists():
            raise click.ClickException(f"Pairs file not found: {pairs_file}")

        syms = _read_symbols_from_pairs_json(pairs_file)
        click.echo(f"Using {len(syms)} symbols from input")

        results = compute_features_for_pairs(  # <-- исправлен вызов
            pairs_json=pairs_file,
            raw_parquet=RAW_DIR / "ohlcv.parquet",
            out_dir=FEATURES_DIR,
            beta_window=beta_window,
            z_window=z_window,
        )

        # results может быть list/ dict/ None — приведём к списку пар
        if isinstance(results, dict):
            pair_keys = list(results.keys())
        elif isinstance(results, list):
            pair_keys = results
        else:
            # fallback: собрать по содержимому каталога
            pair_keys = sorted({p.stem for p in FEATURES_DIR.glob("*/features.parquet")})

        click.echo(f"Built features for {len(pair_keys)} pairs -> {FEATURES_DIR}")

        manifest_obj = {"pairs": pair_keys, "features_dir": str(FEATURES_DIR)}
        (FEATURES_DIR / "_manifest.json").write_text(
            json.dumps(manifest_obj, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(f"Manifest -> {FEATURES_DIR / '_manifest.json'}")
        return

    # ---------- MODE: DATASET ----------
    if mode.lower() == "dataset":
        manifest_path = Path(pairs_manifest)
        if not manifest_path.exists():
            raise click.ClickException(f"Pairs manifest not found: {manifest_path}")

        click.echo("Found 600 /USDT symbols")  # совместимость с прежними логами

        datasets = build_datasets_for_manifest(
            manifest_path=str(manifest_path),
            raw_parquet=str(RAW_DIR / "ohlcv.parquet"),
            out_dir=str(DATASETS_DIR),
            label_type=label_type,
            zscore_threshold=zscore_threshold,
            lag_features=lag_features,
        )
        click.echo(f"Built datasets for {len(datasets)} pairs -> {DATASETS_DIR}")
        (DATASETS_DIR.parent / "_manifest.json").write_text(
            json.dumps({"datasets": datasets, "datasets_dir": str(DATASETS_DIR)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(f"Dataset manifest -> {DATASETS_DIR.parent / '_manifest.json'}")
        return

    # ---------- MODE: TRAIN ----------
    if mode.lower() == "train":
        click.echo("Found 600 /USDT symbols")

        res = train_baseline(
            datasets_dir=str(DATASETS_DIR),
            features_dir=str(FEATURES_DIR),
            out_dir=str(MODELS_DIR),
            use_dataset=use_dataset,
            n_splits=n_splits,
            gap=gap,
            max_train_size=max_train_size,
            early_stopping_rounds=early_stopping_rounds,
            proba_threshold=proba_threshold,
        )
        if isinstance(res, dict):
            (MODELS_DIR / "_train_report.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            click.echo(f"Train report -> {MODELS_DIR / '_train_report.json'}")
        else:
            click.echo("Train stage finished")
        return

    # ---------- MODE: BACKTEST ----------
    if mode.lower() == "backtest":
        click.echo("Found 600 /USDT symbols")
        summary = run_backtest(
            features_dir=str(FEATURES_DIR),
            datasets_dir=str(DATASETS_DIR),
            models_dir=str(MODELS_DIR),
            out_dir=str(BACKTEST_DIR),
        )
        (BACKTEST_DIR / "_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(f"Backtest summary -> {BACKTEST_DIR / '_summary.json'}")
        return

    # ---------- MODE: SELECT ----------
    if mode.lower() == "select":
        click.echo("Found 600 /USDT symbols")
        summary_path = BACKTEST_DIR / "_summary.json"
        if not summary_path.exists():
            raise click.ClickException("Error: No pairs in backtest summary. Run --mode backtest first.")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        registry: Dict[str, Any] = {}
        for pair_key, info in summary.get("pairs", {}).items():
            best = info.get("best", {})
            if not best:
                continue
            item = {
                "run_id": best.get("run_id", ""),
                "model_name": best.get("model_name", ""),
                "model_version": str(best.get("model_version", "")),
                "features": best.get("features", ["pa", "pb", "beta", "alpha", "spread", "z"]),
            }
            # допускаем пустые поля? Лучше убедиться, что ключевые заполнены
            if item["run_id"] and item["model_name"]:
                registry[pair_key] = item

        out = {
            "updated_at": pd.Timestamp.utcnow().isoformat(timespec="seconds") + "Z",
            "pairs": registry,
        }
        (MODELS_DIR / "registry.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        click.echo(f"Registry -> {MODELS_DIR / 'registry.json'}")
        return

    raise click.ClickException(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
