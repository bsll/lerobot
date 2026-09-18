#!/usr/bin/env bash
# Run a trained policy on AgileX Piper with the same YOLO grasp-target state
# injection used during 3c recording (observation.state must match training dims).
#
# IMPORTANT (single-CAN hardware leader-follower):
#   Inference sends joint commands to the follower. Disable hardware master-slave
#   before running.
#
# Usage:
#   export POLICY_PATH=outputs/models/.../pretrained_model
#   export MODEL_PATH=/path/to/best.pt
#   bash examples/piper/5c_rollout_grasp.sh

set -euo pipefail

JOB_NAME="${JOB_NAME:-act_piper_grasp_demo}"
CAN_PORT="${CAN_PORT:-can0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-100000}"
POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/outputs/models/${JOB_NAME}/checkpoints/${CHECKPOINT_STEP}/pretrained_model}"
TASK="${TASK:-Pick up the pen and put it in the bowl.}"
DURATION_S="${DURATION_S:-60}"
DEVICE="${DEVICE:-cuda}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/best.pt}"
DETECT_CAMERA="${DETECT_CAMERA:-front}"
DETECT_CONF="${DETECT_CONF:-0.2}"
DETECT_DEVICE="${DETECT_DEVICE:-auto}"
DETECT_NUM_FRAMES="${DETECT_NUM_FRAMES:-15}"
# Same ROI / conf defaults as collect_yolo_images.sh.
DETECT_MIN_CX_RATIO="${DETECT_MIN_CX_RATIO:-0.47}"
DETECT_MAX_CX_RATIO="${DETECT_MAX_CX_RATIO:-0.83}"
DETECT_MIN_CY_RATIO="${DETECT_MIN_CY_RATIO:-0.12}"
DETECT_MAX_CY_RATIO="${DETECT_MAX_CY_RATIO:-0.55}"

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Policy not found: ${POLICY_PATH}" >&2
  echo "Set POLICY_PATH or CHECKPOINT_STEP to an existing checkpoint." >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: YOLO weights not found at: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH=/path/to/best.pt (must match recording)." >&2
  exit 1
fi

echo "Using policy: ${POLICY_PATH}"
echo "Task: ${TASK}"
echo "Duration: ${DURATION_S}s  CAN: ${CAN_PORT}"
echo "Grasp detect: camera=${DETECT_CAMERA} model=${MODEL_PATH}"

lerobot-rollout --strategy.type=base --policy.path="${POLICY_PATH}" \
  --robot.type=piper_follower \
  --robot.port="${CAN_PORT}" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --robot.calibrate_on_connect=false \
  --robot.cameras="{
    front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30, backend: V4L2},
    wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30, backend: V4L2}
  }" \
  --fps=30 \
  --duration="${DURATION_S}" \
  --task="${TASK}" \
  --device="${DEVICE}" \
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
