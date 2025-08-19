# MNTrading — end-to-end MLOps pipeline (MinIO + MLflow + FastAPI + Streamlit + Airflow)

> **Purpose in one line:** turn raw market data into **cointegrated pairs**, learn **mean-reversion signals**, and produce **live portfolio orders** with full observability (MLflow/MinIO) and operability (API/Streamlit/Airflow).

## Why this project exists
Modern quant research needs a reproducible path from **idea → data → model → money**. This repo packages that path for **pair trading** with a clean, testable pipeline and batteries-included tooling (Docker, S3/MinIO, MLflow, UI, API, DAGs).

---

## Conceptual flow (what & why at each step)

```text
(1) Ingest 1h universe (from 2025-01-01) ──▶ (2) Screen cointegration on 1h ──▶  (auto) Ingest 5m for selected symbols
                                                │                                         │
                                                │                                         ▼
                                                └───────▶ Pairs JSON (selected) ──▶ (3) Features on 5m (rolling OLS, spread, z)
                                                                                     │
                                                                                     ▼
                                     (4) Dataset (labels) ──▶ (5) Train (MLflow) ──▶ (6) Backtest
                                                                                     │
                                                                                     ▼
                                                  (7) Select champions ──▶ (8) Promote (production_map.json)
                                                                                     │
                                                                                     ▼
                                                 (9) Inference (live signals) ──▶ (10) Aggregate (portfolio orders)
                                                                                     │
                                                                                     ▼
                                                                              (11) Report
```

### 1) Ingest 1-hour universe (from a fixed date)
- **Why 1h?** Enough history to see durable relations, lower microstructure noise than 1–5m, cheaper to collect for a large universe.
- **Output:** `data/raw/ohlcv_1h.parquet` (+ meta).
- **Command (example):**
  ```bash
  docker compose exec app bash -lc "python /app/main.py --mode ingest --symbols 'BTC/USDT,ETH/USDT,SOL/USDT,...' --timeframe 1h --since-utc 2025-01-01T00:00:00Z --limit 5000"
  ```

### 2) Screen cointegration on 1h (Engle–Granger, filters…)
- **Goal:** find pairs whose **spread is stationary** (long-run equilibrium); correlation alone is insufficient.
- **Output:** `data/pairs/screened_pairs_*.json` (list of selected pairs + diagnostics).
- **Important:** In this project, **`screen` will also auto-ingest 5m** for the union of selected symbols (see next step).

### (auto) Ingest 5-minute data for the selected pairs
- **Why 5m?** More observations for modeling and tighter execution control; still tractable at scale.
- The `screen` step **automatically** ingests 5m from `2025-01-01T00:00:00Z` only for the symbols present in selected pairs.
- **Output:** `data/raw/ohlcv_5m.parquet` (+ meta).

### 3) Features on 5m (per pair)
- **Purpose:** transform prices into informative predictors for mean-reversion.
- **Key features:**
  - **Rolling OLS (β, α):** hedge ratio and intercept via Cov/Var with rolling windows.
  - **Spread:** `A − (β·B + α)`; **z-score** (normalized by rolling σ).
  - Optional: velocity/acceleration of z, half-life, spread volatility, liquidity/volume, trend/regime filters, lags.
- **Output:** per-pair `features.parquet` + master manifest `_manifest.json`.

### 4) Dataset (labels)
- **Purpose:** define a target for learning.
- **Strategies provided:**
  - `z_threshold`: a positive label if a stretched spread **reverts** towards mean within `horizon`.
  - `revert_direction`: predict the direction/magnitude of reversion.
- **Output:** datasets under `data/datasets/pairs`, manifest for training.

### 5) Train baseline models
- **Purpose:** validate signal existence and calibrate thresholds.
- **Models:** Logistic Regression / RandomForest / LightGBM (baselines).
- **Validation:** proper time-series CV (n_splits, gap, max_train_size), metrics (AUC/PR, ROC), feature importances.
- **Observability:** **MLflow** logs all params/metrics/artifacts; artifacts stored in **MinIO**.
- **Output:** trained models under `data/models`, MLflow runs under `http://localhost:5000`.

### 6) Backtest
- **Purpose:** convert scores to trades and measure **PnL, Sharpe, max drawdown, turnover**, robustness across time.
- **Signals:** either model-based (`signals_from=model`) or rule-based (`signals_from=z`), with fees and realistic constraints.
- **Output:** `data/backtest_results/_summary.json`, equity curves and per-pair stats.

### 7) Select champions
- **Purpose:** pick robust pairs/models by rules (min Sharpe, max DD, stability…).
- **Output:** `data/models/registry.json` (champion roster).

### 8) Promote
- **Purpose:** freeze a production configuration.
- **Output:** `data/models/production_map.json` (what gets used in live inference).

### 9) Inference (live signals)
- **Purpose:** run the production map on the latest 5m bars and produce **signals** (entries/exits/sizes/confidence).
- **Output:** `data/signals/*.jsonl`, optional links to MLflow.

### 10) Aggregate (portfolio)
- **Purpose:** merge per-pair signals into a **portfolio** with risk caps (top-K, per-symbol caps, exposure/vol controls).
- **Output:** `data/portfolio/latest_orders.json` — actionable targets/weights/orders.

### 11) Report
- **Purpose:** human-readable summary (top pairs, PnL/stats, equity curves, risk diagnostics, recommendations).
- **Output:** `data/portfolio/_latest_report.md` (or HTML/PNG in your customization).

---

## Repository layout (key pieces)

- `main.py` — single CLI that orchestrates all steps:
  - `ingest`, `screen` (auto-ingest 5m), `features`, `dataset`, `train`, `backtest`,
    `select`, `promote`, `inference`, `aggregate`, `report`.
- `features/spread.py` — **rolling OLS β/α**, spread & z-score (and helpers).
- `features/labels.py` — dataset builder and labelers.
- `models/train.py` — baselines + time-series CV + MLflow logging.
- `backtest/runner.py` — backtest engine; `models/select.py` — champion selection.
- `portfolio/aggregate_signals.py`, `portfolio/report_latest.py` — portfolio and reporting.
- `api/server.py` — FastAPI: healthcheck + short/full cycle triggers.
- `ui/streamlit_app.py` — buttons for Short/Full cycles, tables/plots, links to MLflow.
- `scripts/upload_to_minio.py` — upload datasets/features/models to MinIO (S3).
- `dags/mntrading_dataset_upload.py` — Airflow DAG: build dataset → upload to MinIO.

> Windows note: run everything under **WSL2** or Linux containers. Airflow does not support native Windows execution.

---

## Quickstart (Docker)

1) Build & run infrastructure and apps:
```bash
docker compose build --no-cache
docker compose up -d minio setup-minio mlflow api app streamlit
```

2) Open UIs:
- MinIO console: <http://localhost:9001>
- MLflow UI:     <http://localhost:5000>
- FastAPI docs:  <http://localhost:8000/docs>
- Streamlit UI:  <http://localhost:8501>

3) Recommended pipeline (aligned with the logic above):

```bash
# (1) Ingest 1h universe from 2025-01-01
docker compose exec app bash -lc "python /app/main.py --mode ingest --symbols 'BTC/USDT,ETH/USDT,SOL/USDT,...' --timeframe 1h --since-utc 2025-01-01T00:00:00Z --limit 5000"

# (2) Screen cointegration on 1h + (auto) ingest 5m for selected symbols
docker compose exec app bash -lc "python /app/main.py --mode screen"

# (3) Features on 5m for selected pairs
docker compose exec app bash -lc "python /app/main.py --mode features --symbols '/app/data/pairs/screened_pairs_*.json' --beta-window 1000 --z-window 300"

# (4) Dataset
docker compose exec app bash -lc "python /app/main.py --mode dataset --label-type z_threshold --zscore-threshold 1.5 --lag-features 10 --horizon 3"

# (5) Train
docker compose exec app bash -lc 'python /app/main.py --mode train --use-dataset --n-splits 5 --gap 24 --proba-threshold 0.55'

# (6) Backtest
docker compose exec app bash -lc 'python /app/main.py --mode backtest --signals-from auto --proba-threshold 0.55 --fee-rate 0.0005'

# (7) Select
docker compose exec app bash -lc 'python /app/main.py --mode select'

# (8) Promote
docker compose exec app bash -lc 'python /app/main.py --mode promote'

# (9) Inference (live signals)
docker compose exec app bash -lc 'python /app/main.py --mode inference --registry-in /app/data/models/production_map.json --update'

# (10) Aggregate (portfolio orders)
docker compose exec app bash -lc 'python /app/main.py --mode aggregate --top-k 10 --proba-threshold 0.55'

# (11) Report
docker compose exec app bash -lc 'python /app/main.py --mode report'
```

4) Upload artifacts to MinIO (example: datasets):
```bash
docker compose exec app bash -lc "python /app/scripts/upload_to_minio.py --src /app/data/datasets --prefix datasets/"
```

---

## API (FastAPI)
- `GET /health` — service liveness
- `POST /run/short_cycle` — trigger a short pipeline in one call
- `GET /run/short_cycle_now` — alias for the short pipeline (demo)
- `GET /artifacts` — list produced artifacts (optional)

## Streamlit UI
- **Buttons:** run **Short** or **Full** pipelines.
- **Artifacts/Plots:** datasets, backtest summary, equity curves, latest portfolio report.
- **Links:** jump to MLflow runs and MinIO paths.

---

## Data & artifacts
- Local: `data/*` (raw, features, datasets, models, backtest_results, signals, portfolio, report)
- MLflow artifacts: S3/MinIO bucket (`experiments/...`) — credentials and endpoint configured via `.env`.
- You can upload any folder to MinIO with `scripts/upload_to_minio.py`.

## Configuration
- `.env` should define MinIO and MLflow endpoints/credentials. Example:
  ```ini
  MINIO_ROOT_USER=admin
  MINIO_ROOT_PASSWORD=adminadmin
  MINIO_BUCKET=mlflow
  MINIO_PORT=9000
  MINIO_CONSOLE_PORT=9001

  MLFLOW_PORT=5000
  MLFLOW_TRACKING_URI=http://mlflow:5000
  MLFLOW_S3_ENDPOINT_URL=http://minio:9000
  AWS_ACCESS_KEY_ID=admin
  AWS_SECRET_ACCESS_KEY=adminadmin
  AWS_DEFAULT_REGION=us-east-1

  STREAMLIT_PORT=8501
  APP_PATH=ui/streamlit_app.py
  ```

---

## Notes & caveats
- Airflow works only under Linux/WSL2 containers; let the **scheduler** pick up DAGs — do not `python dag.py`.
- Ensure files use **LF** line endings (not CRLF) or configure Git accordingly.
- CCXT fallback is included for convenience; for production, implement `data_loader.loader.load_ohlcv_for_symbols`.

---

## Roadmap ideas
- Add XGBoost/CatBoost behind a feature flag.
- Extend backtest with borrow/fees/slippage models, position sizing by spread half-life.
- Add rolling window re-screen of cointegration; auto re-ingest 5m on roster changes.
- CI for lint/tests + precommit hooks for notebooks/plots.
