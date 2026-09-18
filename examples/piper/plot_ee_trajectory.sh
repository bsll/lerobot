#!/usr/bin/env bash
# Plot Piper EE XYZ trajectories for one dataset episode (state vs action).
#
# Usage:
#   EPISODE=0 bash examples/piper/plot_ee_trajectory.sh
#   DATASET_NAME=piper_grasp_demo EPISODE=12 bash examples/piper/plot_ee_trajectory.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_NAME="${DATASET_NAME:-piper_grasp_demo}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/train_data/${DATASET_NAME}}"
EPISODE="${EPISODE:-0}"
OUT="${OUT:-${REPO_ROOT}/outputs/ee_traj/${DATASET_NAME}_ep$(printf '%03d' "${EPISODE}").png}"

EXTRA=()
if [[ "${SHOW:-false}" == "true" ]]; then
  EXTRA+=(--show)
fi
if [[ "${SAVE_NPY:-false}" == "true" ]]; then
  EXTRA+=(--save-npy)
fi

uv run python "${REPO_ROOT}/examples/piper/plot_ee_trajectory.py" \
  --dataset-root "${DATASET_ROOT}" \
  --repo-id "${DATASET_NAME}" \
  --episode "${EPISODE}" \
  --output "${OUT}" \
  "${EXTRA[@]}"
