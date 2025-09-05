# MNTrading — End-to-End MLOps Pipeline for Crypto Pairs

---

## What the project does

MNTrading turns raw crypto OHLCV data into mean-reversion signals on cointegrated pairs, selects champions via backtests, registers models to MLflow model registry, builds a router model that ensembles the best pairs, serves it via the MLflow scoring server, produces portfolio orders and report. 

Lifecycle: ingest → screen → features → dataset → train → backtest → select → register → router → serve → inference → aggregate → report.

---

## Repository layout

- `$null`
- `.dockerignore`
- `.env`
- `.git/`
- `.gitattributes`
- `.gitignore`
- `.idea/`
- `.streamlit/`
- `Dockerfile`
- `airflow/`
- `api/` — api, server.py
- `backtest/` — runner.py
- `dags/` — mntrading_dag.py, mntrading_dataset_upload.py, mntrading_full_pipeline.py, mntrading_pipeline.py
- `data_loader/`
- `docker/`
- `docker-compose.airflow.yml`
- `docker-compose.override.yml`
- `docker-compose.serving.override.yml`
- `docker-compose.serving.yml`
- `docker-compose.streamlit.yml`
- `docker-compose.yml`
- `features/` — labels.py, spread.py
- `inference.py`
- `logs/`
- `main.py`
- `minio-data/`
- `mkdir/`
- `mlflow.db/`
- `mlruns/`
- `models/` — select.py, train.py
- `plugins/`
- `portfolio/` — aggregate_signals.py, report_latest.py
- `requirements.txt`
- `reset_workspace.ps1`
- `router_config.json`
- `screen_pairs.py`
- `scripts/` — build_registry.py, build_router_from_mlflow.py, build_serving_model.py, log_to_mlflow.py, offline_inference_router.py, register_models.py, serve_router.sh, serving_inference.py, upload_to_minio.py
- `ui/` — .streamlit, streamlit_app.py
- `utils/`

---

## Typical external ports 

- MLflow UI: 5500→5000
- MinIO/S3: 9000 (API), 9001 (console)
- Serving: 5001
- Streamlit: 8501
- FastAPI (ops): 8000
- Airflow webserver: 8080

---

# Pipeline Steps

### 1) Ingest 1‑hour universe (from a fixed date)
- **Why 1h?** Enough history to see durable relations, lower microstructure noise than 1–5m, cheaper to collect for a large universe.
- **Input:** exchange via CCXT (default: Binance spot), list of symbols.
- **Output:** `data/raw/ohlcv_1h.parquet` (+ meta).
- **Command (example):**
  ```bash
  docker compose exec app bash -lc "python /app/main.py --mode ingest     --symbols 'BTC/USDT,ETH/USDT,SOL/USDT'     --timeframe 1h --since-utc 2025-01-01T00:00:00Z --limit 5000"
  ```

### 2) Screen cointegration on 1h (Engle–Granger, filters)
- **Goal:** find pairs whose spread is stationary (long‑run equilibrium); correlation alone is insufficient.
- **Output:** `data/pairs/screened_pairs_*.json` (list of selected pairs + diagnostics).
- **Command:**
  ```bash
  docker compose exec app bash -lc "python /app/main.py --mode screen"
  ```

### (auto) Ingest 5‑minute data for the selected pairs
- **Why 5m?** More observations for modeling and tighter execution control; still tractable at scale.
- The `screen` step automatically ingests 5m from `2025‑01‑01T00:00:00Z` only for the symbols present in selected pairs.
- **Output:** `data/raw/ohlcv_5m.parquet` (+ meta). 

### 3) Build features on 5m (per pair)
- **Purpose:** transform prices into informative predictors for mean‑reversion.
- **Key features:**
  - **Rolling OLS (β, α):** hedge ratio and intercept via Cov/Var with rolling windows.
  - **Spread:** `A − (β·B + α)`; **z‑score** (normalized by rolling σ).
  - Optional: velocity/acceleration of z, half‑life, spread volatility, liquidity/volume, trend/regime filters, lags.
- **Output:** per‑pair `features.parquet` under `data/features/pairs` + master manifest `_manifest.json`.
- **Command:**
  ```bash
  docker compose exec app bash -lc "python /app/main.py --mode features     --symbols /app/data/pairs/<LATEST_SCREENED_PAIRS.json>     --beta-window 1000 --z-window 300"
  ```

### 4) Dataset (labels)
- **Purpose:** define a target for learning.
- **Strategies provided:**
  - `z_threshold`: a positive label if a stretched spread reverts towards mean within `horizon`.
  - `revert_direction`: predict the direction/magnitude of reversion.
- **Output:** datasets under `data/datasets/pairs`, manifest for training.
- **Command:**
  ```bash
  docker compose exec app bash -lc "python /app/main.py --mode dataset     --label-type z_threshold --zscore-threshold 1.2     --lag-features 10 --horizon 3"
  ```

### 5) Train baseline models
- **Purpose:** validate signal existence and calibrate thresholds.
- **Models:** per‑pair classifiers (scikit‑learn baselines).
- **Validation:** proper time‑series CV (`n_splits`, `gap`, `max_train_size`), metrics (AUC/PR, ROC), feature importances.
- **Observability:** **MLflow** logs all params/metrics/artifacts; artifacts stored in **MinIO**.
- **Output:** trained models under `data/models`, MLflow runs under `http://localhost:5500` (proxied to MLflow 5000).
- **Command:**
  ```bash
  docker compose exec app bash -lc "python /app/main.py --mode train     --use-dataset --n-splits 5 --gap 24 --proba-threshold 0.50"
  ```

### 6) Backtest
- **Purpose:** convert scores to trades and measure PnL, Sharpe, max drawdown, turnover, robustness across time.
- **Signals:** either model‑based (`--signals-from model`) or rule‑based (`--signals-from z`), with fees and realistic constraints.
- **Output:** `data/backtest_results/_summary.json`, equity curves and per‑pair stats.
- **Command:**
  ```bash
  docker compose exec app bash -lc "python /app/main.py --mode backtest     --signals-from auto --proba-threshold 0.50 --fee-rate 0.0005"
  ```

### 7) Select champions
- **Purpose:** pick robust pairs/models by rules (min Sharpe, max DD, stability…).
- **Output:** `data/models/registry.json` (champion roster).
- **Recommended command:**
  ```bash
  docker compose exec app bash -lc "python /app/main.py --mode select \
    --top-k 50 --sharpe-min 0.20 --maxdd-max 50"
  ```
  > Without explicit thresholds you may accidentally select weak/unstable pairs.

### 8) Register to MLflow and build Router
- **Register pairs (stage = Staging):**
  ```bash
  docker compose exec app bash -lc "python /app/scripts/register_models.py \
    --summary /app/data/backtest_results/_summary.json \
    --stage Staging --prefix mntrading_"
  ```
- **Build router from MLflow:**
  ```bash
  docker compose exec app bash -lc "python /app/scripts/build_router_from_mlflow.py \
    --prefix mntrading_ --pair-stage Staging --top-k 20 \
    --registered-name mntrading_router --router-stage Production \
    --experiment mntrading"
  ```

### 9) Serve the router (MLflow Scoring Server)
- **Output:** HTTP service on port `5001` inside Docker; port-mapped on host.
- **Command:**
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.serving.yml up -d serving
  # Health (prefer IPv4 loopback to avoid IPv6/proxy quirks)
  curl -4 http://127.0.0.1:5001/ping
  curl -4 http://127.0.0.1:5001/version
  ```
- **Cold start / timeouts:** If first `/invocations` is slow, increase the scoring server timeout via the override compose file:
  ```yaml
  # docker-compose.serving.override.yml
  services:
    serving:
      environment:
        MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT: "600"
  ```

### 10) Inference (live signals via serving)
- **Purpose:** score latest N bars and emit JSONL signals per pair.
- **Output:** `data/signals/*.jsonl` + `latest_raw_response.txt` (for troubleshooting).
- **Command (with warmup and larger timeout):**
  ```bash
  docker compose exec app bash -lc "python -u /app/scripts/serving_inference.py \
    --serving-url http://serving:5001 \
    --registry /app/data/models/registry.json \
    --features-dir /app/data/features/pairs \
    --n-last 1 --top-k 20 \
    --timeout-read 180 --warmup \
    --out /app/data/signals"
  ```

### 11) Aggregate (portfolio construction)
- **Purpose:** merge per-pair signals into a **portfolio** with risk caps (top-K, per-symbol caps, exposure/vol controls).
- **Output:** `data/portfolio/latest_orders.json` — actionable targets/weights/orders.
- **Command:**
  ```bash
  docker compose exec app bash -lc "python /app/main.py \
    --mode aggregate \
    --signals-dir /app/data/signals \
    --portfolio-dir /app/data/portfolio \
    --proba-threshold 0.50 --top-k 20"
  ```

### 12) Report
- **Purpose:** human-readable summary (top pairs, PnL/stats, equity curves, risk diagnostics, recommendations).
- **Output:** `data/portfolio/_latest_report.md` (or HTML/PNG in your customization).
- **Command:**
  ```bash
  docker compose exec app bash -lc "python /app/main.py \
    --mode report \
    --orders-json /app/data/portfolio/latest_orders.json \
    --summary-path /app/data/backtest_results/_summary.json \
    --registry-in /app/data/models/registry.json \
    --report-out /app/data/portfolio/_latest_report.md"
  ```

## Operational Notes (Streamlit & Airflow)
**Streamlit UI**
  ```bash
  docker compose -f docker-compose.yml -f docker-compose.streamlit.yml up -d streamlit
  ```
  UI: http://localhost:8501

**Airflow**
  ```bash
  docker compose -f docker-compose.airflow.yml up -d airflow-db airflow-webserver airflow-scheduler
  ```
- First startup initializes the DB; DAGs are under `dags/`. UI: http://localhost:8080

## Future Improvements

1. **Stronger Leakage Guards**
   - Ensure all features are built strictly from past data (proper shifting/rolling windows). Add automated tests that fail if any lookahead is detected.

2. **Stability of Cointegration**
   - Cointegration relationships drift. Schedule periodic re-screening and re-validation on rolling out-of-sample windows; track stability metrics in MLflow.

3. **Transaction Costs Beyond Fees**
   - Add configurable slippage and latency modeling to backtests to stress-test profitability under execution frictions.

4. **Champion Selection Guardrails**
   - Enforce explicit thresholds (e.g., `Sharpe ≥ 0.2`, `MaxDD ≤ 50`) in selection logic and DAG defaults to avoid weak pairs entering Production.

5. **Data Robustness**
   - Improve ingestion resiliency (retries, gap-filling) and dataset validation (missing OHLCV, delistings, time gaps) with explicit checks and alerts.