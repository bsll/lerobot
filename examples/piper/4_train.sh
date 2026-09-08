#!/usr/bin/env bash
# Train an ACT policy on a Piper teleop dataset.

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-piper_demo}"
JOB_NAME="${JOB_NAME:-act_piper_demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/train_data/${DATASET_NAME}}"

lerobot-train \
  --dataset.repo_id="${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.push_to_hub=false \
  --output_dir="outputs/models/${JOB_NAME}" \
  --job_name="${JOB_NAME}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --batch_size=32 \
  --steps=30000 \
  --env_eval_freq=0 \
  --save_freq=5000 \
  --wandb.enable=true
