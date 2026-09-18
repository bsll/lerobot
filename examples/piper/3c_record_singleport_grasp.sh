#!/usr/bin/env bash
# Single-CAN / hardware leader-follower recording for AgileX Piper, with YOLO OBB
# grasp-target pose appended to observation.state.
#
# Before each episode:
#   1) run MODEL_PATH on DETECT_CAMERA
#   2) keep the highest-confidence detection
#   3) write normalized (cx, cy, w, h, angle) into observation.state for every frame
#   4) overlay "GRASP THIS" on the live view only (not saved to dataset video)
#
# Action remains follower joint state only (--direct_record); grasp dims do not
# enter the action vector.
#
# Bring up CAN first, e.g.:
#   sudo ip link set can0 down
#   sudo ip link set can0 type can bitrate 1000000
#   sudo ip link set can0 up
#
# Requires: uv pip install ultralytics  (and an OBB best.pt)
#
# Usage (from repo root):
#   export DATASET_NAME=piper_grasp_demo
#   export MODEL_PATH=/path/to/best.pt   # default: $REPO_ROOT/best.pt
#   bash examples/piper/3c_record_singleport_grasp.sh
# Continue an existing local dataset:
#   RESUME=true NUM_EPISODES=50 bash examples/piper/3c_record_singleport_grasp.sh

set -euo pipefail

DATASET_NAME="${DATASET_NAME:-piper_grasp_demo}"
CAN_PORT="${CAN_PORT:-can0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/train_data/${DATASET_NAME}}"
RESUME="${RESUME:-true}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/best.pt}"
DETECT_CAMERA="${DETECT_CAMERA:-front}"
DETECT_CONF="${DETECT_CONF:-0.5}"
DETECT_DEVICE="${DETECT_DEVICE:-auto}"
DETECT_NUM_FRAMES="${DETECT_NUM_FRAMES:-15}"
# Same ROI / conf defaults as collect_yolo_images.sh.
DETECT_MIN_CX_RATIO="${DETECT_MIN_CX_RATIO:-0.47}"
DETECT_MAX_CX_RATIO="${DETECT_MAX_CX_RATIO:-0.83}"
DETECT_MIN_CY_RATIO="${DETECT_MIN_CY_RATIO:-0.12}"
DETECT_MAX_CY_RATIO="${DETECT_MAX_CY_RATIO:-0.55}"

NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-10}"
TASK="${TASK:-Pick up a crab stick from the red tray and put it into the plate.}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: YOLO weights not found at: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH=/path/to/best.pt (OBB format)." >&2
  exit 1
fi

mkdir -p "$(dirname "${DATASET_ROOT}")"

echo "Recording Piper (direct_record) with grasp-target state injection"
echo "  dataset        : ${DATASET_ROOT}"
echo "  resume         : ${RESUME}"
echo "  episodes       : ${NUM_EPISODES} (this session)"
echo "  detect camera  : ${DETECT_CAMERA}"
echo "  detect ROI     : cx=[${DETECT_MIN_CX_RATIO},${DETECT_MAX_CX_RATIO}) cy=[${DETECT_MIN_CY_RATIO},${DETECT_MAX_CY_RATIO})"
echo "  model          : ${MODEL_PATH}"
echo

lerobot-record \
  --direct_record=true \
  --resume="${RESUME}" \
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
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${TASK}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.fps=30 \
  --display_data=true \
  --play_sounds=false \
  --object_detection.enabled=true \
  --object_detection.model_path="${MODEL_PATH}" \
  --object_detection.camera_key="${DETECT_CAMERA}" \
  --object_detection.conf="${DETECT_CONF}" \
  --object_detection.device="${DETECT_DEVICE}" \
  --object_detection.num_detect_frames="${DETECT_NUM_FRAMES}" \
  --object_detection.min_cx_ratio="${DETECT_MIN_CX_RATIO}" \
  --object_detection.max_cx_ratio="${DETECT_MAX_CX_RATIO}" \
  --object_detection.min_cy_ratio="${DETECT_MIN_CY_RATIO}" \
  --object_detection.max_cy_ratio="${DETECT_MAX_CY_RATIO}"
