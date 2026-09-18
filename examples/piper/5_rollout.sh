#!/usr/bin/env bash
# Run a trained ACT policy on AgileX Piper (true inference / rollout).
#
# IMPORTANT (single-CAN hardware leader-follower):
#   Recording used --direct_record (PC only listened). Inference MUST send
#   joint commands to the follower. Before running this script:
#     1. Disable hardware master-slave / leave only the follower under PC control
#     2. Manually move the arm to your task start pose (same as recording)
#     3. Keep the workspace clear
#
# Scripts pass --robot.calibrate_on_connect=false so connect() does NOT parking()
# to SDK origin; episodic → reset returns to the pose captured at launch.
#
# Bring up CAN first, e.g.:
#   sudo ip link set can0 down
#   sudo ip link set can0 type can bitrate 1000000
#   sudo ip link set can0 up
#
# Usage:
#   export POLICY_PATH=outputs/models/act_piper_demo/checkpoints/030000/pretrained_model
#   bash examples/piper/5_rollout.sh

set -euo pipefail

JOB_NAME="${JOB_NAME:-act_piper_demo}"
CAN_PORT="${CAN_PORT:-can0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-030000}"
POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/outputs/models/${JOB_NAME}/checkpoints/${CHECKPOINT_STEP}/pretrained_model}"
echo $POLICY_PATH
TASK="${TASK:-Pick up the pen and put it in the bowl.}"
DURATION_S="${DURATION_S:-60}"
DEVICE="${DEVICE:-cuda}"

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "Policy not found: ${POLICY_PATH}"
  echo "Set POLICY_PATH or CHECKPOINT_STEP to an existing checkpoint under outputs/models/${JOB_NAME}/checkpoints/"
  exit 1
fi

echo "Using policy: ${POLICY_PATH}"
echo "Task: ${TASK}"
echo "Duration: ${DURATION_S}s  CAN: ${CAN_PORT}"

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

lerobot-rollout --strategy.type=base --policy.path="${POLICY_PATH}"  --robot.type=piper_follower \
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
  --play_sounds=false
