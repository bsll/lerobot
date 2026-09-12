#!/usr/bin/env bash
# Single-CAN / hardware leader-follower recording for AgileX Piper.
#
# Use this when master-slave is already configured on the arms (上位机) and both
# arms share one USB-CAN (e.g. can0). LeRobot only listens on the follower and
# stores its joint state as the dataset action — it does NOT send commands.
#
# Bring up CAN first, e.g.:
#   sudo ip link set can0 down
#   sudo ip link set can0 type can bitrate 1000000
#   sudo ip link set can0 up
#
# Data is saved locally under train_data/<DATASET_NAME>/ (no Hub upload).
# Override with DATASET_ROOT=/path/to/dir if needed.

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-piper_demo}"
CAN_PORT="${CAN_PORT:-can0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/train_data/${DATASET_NAME}}"

mkdir -p "$(dirname "${DATASET_ROOT}")"

lerobot-record \
  --direct_record=true \
  --robot.type=piper_follower \
  --robot.port="${CAN_PORT}" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --robot.cameras="{
    front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, backend: V4L2},
    wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30, backend: V4L2}
  }" \
  --dataset.repo_id="${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.push_to_hub=false \
  --dataset.no_stamp=true \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick up the pen and put it in the bowl." \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --dataset.fps=30 \
  --display_data=true \
  --play_sounds=false

