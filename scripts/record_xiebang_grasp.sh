#!/usr/bin/env bash
# Record SO-101 crab-stick pick-and-place data with YOLO grasp-target injection.
#
# Before each episode:
#   1) run best.pt on the selected camera
#   2) keep the highest-confidence detection
#   3) write normalized (cx, cy, w, h, angle) into observation.state
#   4) overlay "GRASP THIS" on the live camera view (display only; not saved)
#
# Usage (from repo root):
#   ./scripts/record_xiebang_grasp.sh
#   RESUME=true ./scripts/record_xiebang_grasp.sh
#   NUM_EPISODES=20 ./scripts/record_xiebang_grasp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [[ -x .tools/bin/uv ]]; then
  export PATH="$ROOT/.tools/bin:$PATH"
fi

# ---- editable knobs ---------------------------------------------------------
REPO_ID="${REPO_ID:-bsll/pnp_xiebang_grasp_state}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot/${REPO_ID}}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-6}"
RESUME="${RESUME:-false}"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
TELEOP_PORT="${TELEOP_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-R12252702}"
TELEOP_ID="${TELEOP_ID:-R07252702}"

# YOLO OBB weights (default: repo-root best.pt)
MODEL_PATH="${MODEL_PATH:-$ROOT/best.pt}"
# Prefer the fixed top/front camera for recommending which crab stick to grasp.
DETECT_CAMERA="${DETECT_CAMERA:-front}"
DETECT_CONF="${DETECT_CONF:-0.25}"
DETECT_DEVICE="${DETECT_DEVICE:-auto}"
# Keep only detections whose center falls inside the red tray ROI
# (normalized ratios relative to front camera 640x360).
# Tune these if the blue ROI overlay does not tightly cover the red tray.
DETECT_MIN_CX_RATIO="${DETECT_MIN_CX_RATIO:-0.55}"
DETECT_MAX_CX_RATIO="${DETECT_MAX_CX_RATIO:-0.85}"
DETECT_MIN_CY_RATIO="${DETECT_MIN_CY_RATIO:-0.22}"
DETECT_MAX_CY_RATIO="${DETECT_MAX_CY_RATIO:-0.80}"

TASK="${TASK:-Pick up a crab stick from the red tray and put it into the brown round bowl.}"
# -----------------------------------------------------------------------------

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: YOLO weights not found at: $MODEL_PATH" >&2
  echo "Set MODEL_PATH=/path/to/best.pt" >&2
  exit 1
fi

CMD=(
  lerobot-record
  --robot.type=so101_follower
  --robot.port="$ROBOT_PORT"
  --robot.id="$ROBOT_ID"
  --robot.disable_torque_on_disconnect=true
  --robot.cameras="{
    wrist: {type: opencv, index_or_path: 0, width: 640, height: 360, fps: 30},
    front: {type: opencv, index_or_path: 2, width: 640, height: 360, fps: 30}
  }"
  --teleop.type=so101_leader
  --teleop.port="$TELEOP_PORT"
  --teleop.id="$TELEOP_ID"
  --display_data=true
  --dataset.repo_id="$REPO_ID"
  --dataset.root="$DATASET_ROOT"
  --dataset.single_task="$TASK"
  --dataset.num_episodes="$NUM_EPISODES"
  --dataset.episode_time_s="$EPISODE_TIME_S"
  --dataset.reset_time_s="$RESET_TIME_S"
  --dataset.streaming_encoding=true
  --dataset.rgb_encoder.vcodec=h264
  --dataset.encoder_threads=4
  --dataset.push_to_hub=false
  --resume="$RESUME"
  --object_detection.enabled=true
  --object_detection.model_path="$MODEL_PATH"
  --object_detection.camera_key="$DETECT_CAMERA"
  --object_detection.conf="$DETECT_CONF"
  --object_detection.device="$DETECT_DEVICE"
  --object_detection.min_cx_ratio="$DETECT_MIN_CX_RATIO"
  --object_detection.max_cx_ratio="$DETECT_MAX_CX_RATIO"
  --object_detection.min_cy_ratio="$DETECT_MIN_CY_RATIO"
  --object_detection.max_cy_ratio="$DETECT_MAX_CY_RATIO"
)

echo "Recording with grasp-target state injection"
echo "  repo_id        : $REPO_ID"
echo "  root           : $DATASET_ROOT"
echo "  episodes       : $NUM_EPISODES"
echo "  resume         : $RESUME"
echo "  detect camera  : $DETECT_CAMERA"
echo "  detect ROI     : cx=[$DETECT_MIN_CX_RATIO,$DETECT_MAX_CX_RATIO) cy=[$DETECT_MIN_CY_RATIO,$DETECT_MAX_CY_RATIO)"
echo "  model          : $MODEL_PATH"
echo

exec "${CMD[@]}"
