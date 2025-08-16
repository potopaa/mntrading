#!/usr/bin/env bash
# Copy the latest model from MLflow run into ml_model/
set -e
RUN_ID=$(cat data/raw/last_run_id.txt)
rm -rf ml_model/*
mkdir -p ml_model
cp -r mlruns/0/${RUN_ID}/artifacts/model/* ml_model/
echo "Model ${RUN_ID} copied to ml_model/"