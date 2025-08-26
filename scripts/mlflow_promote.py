import argparse, mlflow
from utils.mlflow_utils import promote_if_better

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--metric-key", default="oof_score")
    ap.add_argument("--direction", choices=["higher","lower"], default="higher")
    ap.add_argument("--improve", type=float, default=0.0)
    ap.add_argument("--run-id", help="if run_id, take in; else — last created")
    args = ap.parse_args()

    # Находим последнюю версию модели (без run-id — по времени)
    client = mlflow.tracking.MlflowClient()
    mvs = sorted(client.search_model_versions(f"name='{args.model_name}'"),
                 key=lambda v: int(v.version), reverse=True)
    if not mvs:
        raise SystemExit("No moodels for promotion")
    target = next((mv for mv in mvs if (not args.run_id) or (mv.run_id == args.run_id)), mvs[0])

    res = promote_if_better(args.model_name, target, args.metric_key, args.direction, args.improve)
    print(res)
