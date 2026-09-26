#!/usr/bin/env bash
# Convenience wrapper: train SmolVLA on the Piper local dataset.
# POLICY_TYPE is consumed by examples/piper/4_train.sh as --policy.type.

set -euo pipefail

#export JOB_NAME="${JOB_NAME:-act_piper_demo}"
#export POLICY_TYPE="${POLICY_TYPE:-act}"
#export JOB_NAME="${JOB_NAME:-piper_grasp_smolvla}"
#export POLICY_TYPE="${POLICY_TYPE:-smolvla}"
#export JOB_NAME="${JOB_NAME:-act_piper_demo_pi05}"
#export POLICY_TYPE="${POLICY_TYPE:-pi05}"
#export PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"

export JOB_NAME="${JOB_NAME:-piper_grasp_smolvla}"
export POLICY_TYPE="${POLICY_TYPE:-smolvla}"
export DATASET_NAME="${DATASET_NAME:-rollout_piper_grasp_crab_eval}"
export RECORD="${RECORD:-false}"

rm -rf -- "train_data/${DATASET_NAME}"
bash examples/piper/5e_rollout_grasp_auto.sh
