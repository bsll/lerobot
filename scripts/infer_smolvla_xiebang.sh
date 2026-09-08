#!/usr/bin/env bash
# 真机推理：用训练好的 SmolVLA 在 SO-101 上跑蟹棒抓取任务。
# 与 scripts/record_xiebang_grasp.sh / train_smolvla_xiebang.sh 对齐：
#   - front/wrist 相机 + rename_map → camera1/camera2
#   - 每 episode 开始前 YOLO OBB 注入 grasp_target 到 observation.state
#
# Usage (from repo root):
#   ./scripts/infer_smolvla_xiebang.sh
#   POLICY_PATH=outputs/train/smolvla/pnp_xiebang_grasp_state/checkpoints/050000/pretrained_model \
#     ./scripts/infer_smolvla_xiebang.sh
#   STRATEGY=episodic NUM_EPISODES=10 ./scripts/infer_smolvla_xiebang.sh
#   INFERENCE=rtc DURATION=60 ./scripts/infer_smolvla_xiebang.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -x .tools/bin/uv ]]; then
  export PATH="$ROOT/.tools/bin:$PATH"
fi

##############################
# Policy / 任务
##############################
POLICY_PATH="${POLICY_PATH:-${ROOT}/outputs/train/smolvla/pnp_xiebang_grasp_state/checkpoints/last/pretrained_model}"
DEVICE="${DEVICE:-cuda}"
# base: 只跑策略不存盘；episodic: 按 episode 录评测数据（repo 名需 rollout_ 前缀）
STRATEGY="${STRATEGY:-base}"
# sync: 每控制周期同步推理；rtc: 后台 chunk 推理（慢 VLA 更稳）
INFERENCE="${INFERENCE:-sync}"
DURATION="${DURATION:-30}"          # base 策略：单次运行秒数；0=一直跑到 Ctrl-C
FPS="${FPS:-30}"
INTERACTIVE="${INTERACTIVE:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

TASK="${TASK:-Pick up a crab stick from the red tray and put it into the brown round bowl.}"
RENAME_MAP="${RENAME_MAP:-{\"observation.images.front\":\"observation.images.camera1\",\"observation.images.wrist\":\"observation.images.camera2\"}}"

##############################
# Robot / Teleop（与录制脚本一致）
##############################
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
TELEOP_PORT="${TELEOP_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-R12252702}"
TELEOP_ID="${TELEOP_ID:-R07252702}"
# episodic 重置阶段可用 teleop；base 默认不连 teleop
USE_TELEOP="${USE_TELEOP:-auto}"    # auto | true | false

##############################
# YOLO grasp target（与录制一致）
##############################
MODEL_PATH="${MODEL_PATH:-$ROOT/best.pt}"
DETECT_CAMERA="${DETECT_CAMERA:-front}"
DETECT_CONF="${DETECT_CONF:-0.25}"
DETECT_DEVICE="${DETECT_DEVICE:-auto}"
DETECT_MIN_CX_RATIO="${DETECT_MIN_CX_RATIO:-0.55}"
DETECT_MAX_CX_RATIO="${DETECT_MAX_CX_RATIO:-0.85}"
DETECT_MIN_CY_RATIO="${DETECT_MIN_CY_RATIO:-0.22}"
DETECT_MAX_CY_RATIO="${DETECT_MAX_CY_RATIO:-0.80}"

##############################
# Episodic 录制（仅 STRATEGY=episodic）
##############################
REPO_ID="${REPO_ID:-bsll/rollout_eval_xiebang_grasp}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/${REPO_ID}}"
NUM_EPISODES="${NUM_EPISODES:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-6}"
RESUME="${RESUME:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

##############################
# RTC 可选参数（INFERENCE=rtc）
##############################
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10.0}"

##############################
# 校验
##############################
if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "ERROR: policy checkpoint not found: ${POLICY_PATH}" >&2
  echo "Set POLICY_PATH=.../checkpoints/<step>/pretrained_model" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: YOLO weights not found: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH=/path/to/best.pt" >&2
  exit 1
fi

if [[ "${USE_TELEOP}" == "auto" ]]; then
  if [[ "${STRATEGY}" == "episodic" || "${STRATEGY}" == "dagger" ]]; then
    USE_TELEOP=true
  else
    USE_TELEOP=false
  fi
fi

##############################
# 组装命令
##############################
CMD=(
  lerobot-rollout
  --strategy.type="${STRATEGY}"
  --inference.type="${INFERENCE}"
  --policy.path="${POLICY_PATH}"
  --device="${DEVICE}"
  --fps="${FPS}"
  --duration="${DURATION}"
  --task="${TASK}"
  --interactive="${INTERACTIVE}"
  --display_data="${DISPLAY_DATA}"
  --rename_map="${RENAME_MAP}"
  --robot.type=so101_follower
  --robot.port="${ROBOT_PORT}"
  --robot.id="${ROBOT_ID}"
  --robot.disable_torque_on_disconnect=true
  --robot.cameras="{
    wrist: {type: opencv, index_or_path: 0, width: 640, height: 360, fps: 30},
    front: {type: opencv, index_or_path: 2, width: 640, height: 360, fps: 30}
  }"
  --object_detection.enabled=true
  --object_detection.model_path="${MODEL_PATH}"
  --object_detection.camera_key="${DETECT_CAMERA}"
  --object_detection.conf="${DETECT_CONF}"
  --object_detection.device="${DETECT_DEVICE}"
  --object_detection.min_cx_ratio="${DETECT_MIN_CX_RATIO}"
  --object_detection.max_cx_ratio="${DETECT_MAX_CX_RATIO}"
  --object_detection.min_cy_ratio="${DETECT_MIN_CY_RATIO}"
  --object_detection.max_cy_ratio="${DETECT_MAX_CY_RATIO}"
)

if [[ "${USE_TELEOP}" == "true" ]]; then
  CMD+=(
    --teleop.type=so101_leader
    --teleop.port="${TELEOP_PORT}"
    --teleop.id="${TELEOP_ID}"
  )
fi

if [[ "${INFERENCE}" == "rtc" ]]; then
  CMD+=(
    --inference.rtc.execution_horizon="${RTC_EXECUTION_HORIZON}"
    --inference.rtc.max_guidance_weight="${RTC_MAX_GUIDANCE_WEIGHT}"
  )
fi

if [[ "${STRATEGY}" == "episodic" ]]; then
  CMD+=(
    --dataset.repo_id="${REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.single_task="${TASK}"
    --dataset.num_episodes="${NUM_EPISODES}"
    --dataset.episode_time_s="${EPISODE_TIME_S}"
    --dataset.reset_time_s="${RESET_TIME_S}"
    --dataset.streaming_encoding=true
    --dataset.rgb_encoder.vcodec=h264
    --dataset.encoder_threads=4
    --dataset.push_to_hub="${PUSH_TO_HUB}"
    --resume="${RESUME}"
  )
fi

echo "========== SmolVLA infer (xiebang) =========="
echo "policy   : ${POLICY_PATH}"
echo "strategy : ${STRATEGY}  inference: ${INFERENCE}"
echo "duration : ${DURATION}s  fps: ${FPS}"
echo "task     : ${TASK}"
echo "rename   : ${RENAME_MAP}"
echo "detect   : ${DETECT_CAMERA}  model: ${MODEL_PATH}"
if [[ "${STRATEGY}" == "episodic" ]]; then
  echo "dataset  : ${REPO_ID}  episodes: ${NUM_EPISODES}"
fi
echo "============================================"
printf '%q ' "${CMD[@]}"
echo

exec "${CMD[@]}"
