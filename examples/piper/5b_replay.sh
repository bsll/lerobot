#!/usr/bin/env bash
# Replay a recorded episode on the Piper follower (open-loop joint playback).
#
# This does NOT use the trained policy — it plays actions from the dataset.
# Useful to sanity-check recording / robot mapping before policy rollout.
#
# IMPORTANT: PC must be allowed to command the follower (same as 5_rollout.sh).

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-piper_demo}"
CAN_PORT="${CAN_PORT:-can0}"
EPISODE="${EPISODE:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/train_data/${DATASET_NAME}}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Dataset not found: ${DATASET_ROOT}"
  exit 1
fi

echo "Replaying episode ${EPISODE} from ${DATASET_ROOT}"

lerobot-replay \
  --robot.type=piper_follower \
  --robot.port="${CAN_PORT}" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --dataset.repo_id="${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.episode="${EPISODE}"
