#!/usr/bin/env bash
# Episodic policy rollout on AgileX Piper with YOLO grasp-target state injection.
#
# Like 3c recording + 5c inference, but uses --strategy.type=episodic so you get
# lerobot-record-style keyboard controls during rollout:
#   → / n  — end current episode or reset phase early
#   ← / r  — discard episode and re-record
#   Esc / q — stop session
#
# Each episode: policy drives the follower; grasp target refreshed via YOLO.
# Between episodes: robot returns to startup pose (no teleop on single-CAN Piper).
# Startup pose = arm position when you launch this script (NOT SDK parking origin).
# Place the arm in your recording start pose before running; scripts use
# --robot.calibrate_on_connect=false (same idea as 3c --direct_record).
# Rollout frames are saved under train_data/rollout_<name>/ (repo_id must start
# with rollout_).
#
# Defaults to Real-Time Chunking (--inference.type=rtc) for slow VLAs (SmolVLA /
# Pi0 / Pi0.5). Override with INFERENCE_TYPE=sync for one-call-per-tick.
#
# IMPORTANT: disable hardware master-slave before running — PC sends commands.
#
# Usage:
#   export JOB_NAME=piper_grasp_demo_smolvla
#   export CHECKPOINT_STEP=030000
#   export MODEL_PATH=/path/to/best.pt
#   bash examples/piper/5d_rollout_grasp_episodic.sh

set -euo pipefail

JOB_NAME="${JOB_NAME:-piper_grasp_demo_smolvla}"
CAN_PORT="${CAN_PORT:-can0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-100000}"
POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/outputs/models/${JOB_NAME}/checkpoints/${CHECKPOINT_STEP}/pretrained_model}"

DATASET_NAME="${DATASET_NAME:-rollout_piper_grasp_smolvla_local_test22}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/train_data/${DATASET_NAME}}"
RESUME="${RESUME:-false}"

TASK="${TASK:-Pick up a crab stick from the red tray and put it into the plate.}"
NUM_EPISODES="${NUM_EPISODES:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-1}"
DEVICE="${DEVICE:-cuda}"

# RTC (Real-Time Chunking) — async inference for slow VLAs
INFERENCE_TYPE="${INFERENCE_TYPE:-sync}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-15}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
RTC_QUEUE_THRESHOLD="${RTC_QUEUE_THRESHOLD:-50}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/best.pt}"
DETECT_CAMERA="${DETECT_CAMERA:-front}"
DETECT_CONF="${DETECT_CONF:-0.5}"
DETECT_DEVICE="${DETECT_DEVICE:-auto}"
DETECT_NUM_FRAMES="${DETECT_NUM_FRAMES:-15}"
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
  echo "Set MODEL_PATH=/path/to/best.pt (must match 3c recording)." >&2
  exit 1
fi

if [[ "${DATASET_NAME}" != rollout_* ]]; then
  echo "ERROR: episodic rollout dataset repo_id must start with rollout_ (got: ${DATASET_NAME})" >&2
  exit 1
fi

mkdir -p "$(dirname "${DATASET_ROOT}")"

INFERENCE_ARGS=(--inference.type="${INFERENCE_TYPE}")
if [[ "${INFERENCE_TYPE}" == "rtc" ]]; then
  INFERENCE_ARGS+=(
    --inference.rtc.execution_horizon="${RTC_EXECUTION_HORIZON}"
    --inference.rtc.max_guidance_weight="${RTC_MAX_GUIDANCE_WEIGHT}"
    --inference.queue_threshold="${RTC_QUEUE_THRESHOLD}"
  )
fi

echo "Episodic grasp rollout on Piper"
echo "  policy         : ${POLICY_PATH}"
echo "  dataset        : ${DATASET_ROOT}"
echo "  resume         : ${RESUME}"
echo "  episodes       : ${NUM_EPISODES} x ${EPISODE_TIME_S}s (+ ${RESET_TIME_S}s reset)"
echo "  task           : ${TASK}"
echo "  CAN            : ${CAN_PORT}"
echo "  inference      : ${INFERENCE_TYPE}"
if [[ "${INFERENCE_TYPE}" == "rtc" ]]; then
  echo "  RTC            : horizon=${RTC_EXECUTION_HORIZON} guidance=${RTC_MAX_GUIDANCE_WEIGHT} queue_threshold=${RTC_QUEUE_THRESHOLD}"
fi
echo "  detect         : camera=${DETECT_CAMERA} model=${MODEL_PATH}"
echo "  detect ROI     : cx=[${DETECT_MIN_CX_RATIO},${DETECT_MAX_CX_RATIO}) cy=[${DETECT_MIN_CY_RATIO},${DETECT_MAX_CY_RATIO})"
echo "  keys           : → next  ← re-record  Esc quit (terminal focus required)"
echo

lerobot-rollout \
  --strategy.type=episodic \
  --policy.path="${POLICY_PATH}" \
  "${INFERENCE_ARGS[@]}" \
  --resume="${RESUME}" \
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
  --task="${TASK}" \
  --device="${DEVICE}" \
  --display_data=true \
  --play_sounds=false \
  --dataset.repo_id="${DATASET_NAME}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.push_to_hub=false \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${TASK}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
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
