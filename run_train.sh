#!/usr/bin/env bash
# Convenience wrapper for Piper policy training.
# Variables are consumed by examples/piper/4_train.sh.

set -euo pipefail

# export JOB_NAME=act_piper_demo POLICY_TYPE=act
# export JOB_NAME=act_piper_demo_smolvla POLICY_TYPE=smolvla

export JOB_NAME="${JOB_NAME:-act_piper_demo_pi05}"
export POLICY_TYPE="${POLICY_TYPE:-pi05}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export STEPS="${STEPS:-30000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
export PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"
export WANDB_ENABLE="${WANDB_ENABLE:-true}"

bash examples/piper/4_train.sh
