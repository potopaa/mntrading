from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "mntrading",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

V = Variable.get

mn_symbols = V("mn_symbols", default_var="BTC/USDT,ETH/USDT,SOL/USDT")
mn_timeframe = V("mn_timeframe", default_var="1h")
mn_limit_1h = V("mn_limit_1h", default_var="5000")
mn_limit_5m = V("mn_limit_5m", default_var="50000")

mn_beta_window = V("mn_beta_window", default_var="1000")
mn_z_window = V("mn_z_window", default_var="300")
mn_label_type = V("mn_label_type", default_var="z_threshold")
mn_zscore_threshold = V("mn_zscore_threshold", default_var="1.2")
mn_lag_features = V("mn_lag_features", default_var="10")
mn_horizon = V("mn_horizon", default_var="3")

mn_cv_splits = V("mn_cv_splits", default_var="5")
mn_gap = V("mn_gap", default_var="24")
mn_proba_threshold = V("mn_proba_threshold", default_var="0.50")
mn_fee_rate = V("mn_fee_rate", default_var="0.0005")
mn_top_k = V("mn_top_k", default_var="50")

mn_router_top_k = V("mn_router_top_k", default_var="20")
mn_mlflow_experiment = V("mn_mlflow_experiment", default_var="mntrading")
mn_registered_router_name = V("mn_registered_router_name", default_var="mntrading_router")
mn_registered_prefix = V("mn_registered_prefix", default_var="mntrading_")

with DAG(
    dag_id="mntrading_e2e",
    description="MNTrading end-to-end daily pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["mntrading", "mlops", "crypto"],
) as dag:
    start = EmptyOperator(task_id="start")

    # 1) SCREEN:
    screen = BashOperator(
        task_id="screen",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/main.py --mode screen"
        ),
    )

    # 2) FEATURES
    features = BashOperator(
        task_id="features",
        bash_command=(
            "set -euo pipefail\n"
            "SP=$(ls -1t /app/data/pairs/screened_pairs_*.json | head -n1)\n"
            'echo "using pairs file: $SP"\n'
            "python -u /app/main.py --mode features "
            "--symbols \"$SP\" "
            f"--beta-window {mn_beta_window} --z-window {mn_z_window}"
        ),
    )

    # 3) DATASET
    dataset = BashOperator(
        task_id="dataset",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/main.py --mode dataset "
            f"--label-type {mn_label_type} "
            f"--zscore-threshold {mn_zscore_threshold} "
            f"--lag-features {mn_lag_features} "
            f"--horizon {mn_horizon}"
        ),
    )

    # 4) TRAIN
    train = BashOperator(
        task_id="train",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/main.py --mode train --use-dataset "
            f"--n-splits {mn_cv_splits} --gap {mn_gap} "
            f"--proba-threshold {mn_proba_threshold}"
        ),
    )

    # 5) BACKTEST
    backtest = BashOperator(
        task_id="backtest",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/main.py --mode backtest "
            "--signals-from auto "
            f"--proba-threshold {mn_proba_threshold} "
            f"--fee-rate {mn_fee_rate}"
        ),
    )

    # 6) SELECT champions -> registry.json
    select = BashOperator(
        task_id="select",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/main.py --mode select "
            f"--top-k {mn_top_k}"
        ),
    )

    # 7) REGISTER pair models into MLflow
    register_pairs = BashOperator(
        task_id="register_pairs",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/scripts/register_models.py "
            "--registry /app/data/models/registry.json "
            f"--experiment {mn_mlflow_experiment} "
            "--stage Staging"
        ),
    )

    # 8) BUILD router from MLflow
    build_router = BashOperator(
        task_id="build_router",
        bash_command=(
            "set -euo pipefail\n"
            "chmod +x /app/scripts/build_router_from_mlflow.py || true\n"
            "python -u /app/scripts/build_router_from_mlflow.py "
            f"--prefix {mn_registered_prefix} "
            "--pair-stage Staging "
            f"--top-k {mn_router_top_k} "
            f"--registered-name {mn_registered_router_name} "
            "--router-stage Production "
            f"--experiment {mn_mlflow_experiment}"
        ),
    )

    # 9) OFFLINE inference with router -> /app/data/signals
    offline_infer = BashOperator(
        task_id="offline_inference_router",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/scripts/offline_inference_router.py "
            f"--model-uri models:/{mn_registered_router_name}/Production "
            "--registry /app/data/models/registry.json "
            "--features-dir /app/data/features/pairs "
            "--n-last 1 "
            f"--top-k {mn_router_top_k} "
            "--out /app/data/signals"
        ),
    )

    # 10) Aggregate portfolio
    aggregate = BashOperator(
        task_id="aggregate",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/main.py --mode aggregate "
            "--signals-dir /app/data/signals "
            "--portfolio-dir /app/data/portfolio "
            f"--proba-threshold {mn_proba_threshold} "
            f"--top-k {mn_router_top_k}"
        ),
    )

    # 11) Report
    report = BashOperator(
        task_id="report",
        bash_command=(
            "set -euo pipefail\n"
            "python -u /app/main.py --mode report "
            "--orders-json /app/data/portfolio/latest_orders.json "
            "--summary-path /app/data/backtest_results/_summary.json "
            "--registry-in /app/data/models/registry.json "
            "--report-out /app/data/portfolio/_latest_report.md"
        ),
    )

    end = EmptyOperator(task_id="end")

    # Wiring
    start >> screen >> features >> dataset >> train >> backtest >> select >> register_pairs >> build_router >> offline_infer >> aggregate >> report >> end
