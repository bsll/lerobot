#!/usr/bin/env bash
# Capture still images for YOLO training — one target at a time (like 3c grasp).
#
# Preview: best.pt shows ONLY the highest-conf detection inside ROI (≥ DETECT_CONF).
# Press n after removing that stick; when none left in ROI, Space saves a clean JPEG.
#
# Keys (preview window must have focus):
#   n / →   — current stick taken away → countdown → show next best
#   Space/s — save ONLY when no targets left in ROI
#   r       — re-detect while empty
#   Esc / q — quit
#
# Usage:
#   bash examples/piper/collect_yolo_images.sh
#   DETECT_CONF=0.5 bash examples/piper/collect_yolo_images.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/train_data/yolo_dataset}"
CAMERA="${CAMERA:-/dev/video4}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
RESET_TIME_S="${RESET_TIME_S:-5}"
CLASS_NAME="${CLASS_NAME:-target}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/best.pt}"
DETECT_CONF="${DETECT_CONF:-0.5}"
DETECT_EVERY="${DETECT_EVERY:-2}"
DETECT_DEVICE="${DETECT_DEVICE:-auto}"
# Same defaults as 3c_record_singleport_grasp.sh
DETECT_MIN_CX_RATIO="${DETECT_MIN_CX_RATIO:-0.47}"
DETECT_MAX_CX_RATIO="${DETECT_MAX_CX_RATIO:-0.83}"
DETECT_MIN_CY_RATIO="${DETECT_MIN_CY_RATIO:-0.12}"
DETECT_MAX_CY_RATIO="${DETECT_MAX_CY_RATIO:-0.55}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: YOLO weights not found at: ${MODEL_PATH}" >&2
  echo "Set MODEL_PATH=/path/to/best.pt" >&2
  exit 1
fi

echo "YOLO image capture (one target at a time)"
echo "  output     : ${OUTPUT_DIR}/images"
echo "  camera     : ${CAMERA}  ${WIDTH}x${HEIGHT}@${FPS}"
echo "  model      : ${MODEL_PATH}  (show best conf≥${DETECT_CONF})"
echo "  ROI        : cx=[${DETECT_MIN_CX_RATIO},${DETECT_MAX_CX_RATIO}) cy=[${DETECT_MIN_CY_RATIO},${DETECT_MAX_CY_RATIO})"
echo "  reset wait : ${RESET_TIME_S}s after n / after save"
echo "  keys       : n=next after remove | Space=save when empty | Esc=quit"
echo

uv run python "${REPO_ROOT}/examples/piper/collect_yolo_images.py" \
  --output-dir "${OUTPUT_DIR}" \
  --camera "${CAMERA}" \
  --width "${WIDTH}" \
  --height "${HEIGHT}" \
  --fps "${FPS}" \
  --reset-time-s "${RESET_TIME_S}" \
  --class-name "${CLASS_NAME}" \
  --model-path "${MODEL_PATH}" \
  --detect-conf "${DETECT_CONF}" \
  --detect-every "${DETECT_EVERY}" \
  --detect-device "${DETECT_DEVICE}" \
  --min-cx-ratio "${DETECT_MIN_CX_RATIO}" \
  --max-cx-ratio "${DETECT_MAX_CX_RATIO}" \
  --min-cy-ratio "${DETECT_MIN_CY_RATIO}" \
  --max-cy-ratio "${DETECT_MAX_CY_RATIO}"
