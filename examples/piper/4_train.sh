#!/usr/bin/env bash
# Train a policy on a Piper teleop dataset (ACT / SmolVLA / pi05, ...).

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-piper_demo}"
JOB_NAME="${JOB_NAME:-act_piper_demo}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/train_data/${DATASET_NAME}}"
POLICY_TYPE="${POLICY_TYPE:-act}"
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"

EXTRA_ARGS=()

# Large VLAs (pi0/pi05): load pretrained + LoRA + freeze most of the model.
# batch_size alone cannot fit full finetune of gemma_2b on a 24GB card.
if [[ "${POLICY_TYPE}" == "pi05" || "${POLICY_TYPE}" == "pi0" ]]; then
  PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"
  [[ "${POLICY_TYPE}" == "pi0" ]] && PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi0_base}"
  EXTRA_ARGS+=(
    --policy.pretrained_path="${PRETRAINED_PATH}"
    --peft.method_type="${PEFT_METHOD_TYPE:-LORA}"
    --policy.freeze_vision_encoder=true
    --policy.train_expert_only=true
    --policy.gradient_checkpointing=true
    --policy.dtype=bfloat16
  )
fi

lerobot-train \
  --dataset.repo_id="${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type="${POLICY_TYPE}" \
  --policy.push_to_hub=false \
  --output_dir="outputs/models/${JOB_NAME}" \
  --job_name="${JOB_NAME}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --accelerator.gradient_accumulation.steps="${GRADIENT_ACCUMULATION_STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --env_eval_freq=0 \
  --save_freq="${SAVE_FREQ}" \
  --wandb.enable="${WANDB_ENABLE}" \
  "${EXTRA_ARGS[@]}"
