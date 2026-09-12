#!/usr/bin/env bash
# Convenience wrapper: train SmolVLA on the Piper local dataset.
# POLICY_TYPE is consumed by examples/piper/4_train.sh as --policy.type.

set -euo pipefail

export JOB_NAME="${JOB_NAME:-act_piper_demo}"
export POLICY_TYPE="${POLICY_TYPE:-act}"
#export JOB_NAME="${JOB_NAME:-act_piper_demo_smolvla}"
#export POLICY_TYPE="${POLICY_TYPE:-smolvla}"
#export JOB_NAME="${JOB_NAME:-act_piper_demo_pi05}"
#export POLICY_TYPE="${POLICY_TYPE:-pi05}"
#export PRETRAINED_PATH="${PRETRAINED_PATH:-lerobot/pi05_base}"

bash examples/piper/5_rollout.sh
