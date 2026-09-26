#!/usr/bin/env bash
# Auto multi-grasp episodic rollout on Piper.
#
# Remembers the joint pose at launch. After the arm leaves that pose and later
# remains stable inside the configured end-pose region, ends the episode,
# returns to the launch pose, re-runs YOLO, and starts the next grasp.
#
# Still supports:
#   → / n  — end early anyway
#   ← / r  — discard & re-record
#   Esc / q — quit
#
# Usage:
#   # Place the arm in the ready / start pose, then:
#   export JOB_NAME=piper_grasp_smolvla
#   export NUM_EPISODES=30
#   bash examples/piper/5e_rollout_grasp_auto.sh
#   # Run without saving trajectory frames/videos:
#   RECORD=false bash examples/piper/5e_rollout_grasp_auto.sh
#
# Tune completion detection (end-region settle):
#   AUTO_NEXT_MOTION_TOL=2.5   # higher = tolerate more jitter, fewer stable RESETs
#   AUTO_NEXT_MOTION_HOLD_S=0.2  # wall-clock seconds of continuous low motion
#   AUTO_NEXT_MOTION_TOL=3 AUTO_NEXT_MOTION_HOLD_S=0.1 bash examples/piper/5e_rollout_grasp_auto.sh
# The default end pose below comes from successful rollout episodes. Recalculate
# it if the task layout, calibration, or desired final arm pose changes.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export DATASET_NAME="${DATASET_NAME:-rollout_piper_grasp_auto_nosave}"
export NUM_EPISODES="${NUM_EPISODES:-30}"
export EPISODE_TIME_S="${EPISODE_TIME_S:-20}"
export RESET_TIME_S="${RESET_TIME_S:-0}"
export RESET_TO_INITIAL="${RESET_TO_INITIAL:-true}"
export RECORD="${RECORD:-true}"

export AUTO_NEXT_GRASP="${AUTO_NEXT_GRASP:-true}"
export AUTO_NEXT_MIN_EPISODE_S="${AUTO_NEXT_MIN_EPISODE_S:-5}"
export AUTO_NEXT_SETTLE_S="${AUTO_NEXT_SETTLE_S:-0.2}"
export AUTO_NEXT_LEAVE_TOL="${AUTO_NEXT_LEAVE_TOL:-55}"
export AUTO_NEXT_END_POSITION="${AUTO_NEXT_END_POSITION:-[6.17,-25.01,58.10,26.66,95.73,-33.78,37.01]}"
export AUTO_NEXT_END_TOL="${AUTO_NEXT_END_TOL:-30}"
export AUTO_NEXT_MOTION_TOL="${AUTO_NEXT_MOTION_TOL:-2.5}"
export AUTO_NEXT_MOTION_HOLD_S="${AUTO_NEXT_MOTION_HOLD_S:-0.2}"
export AUTO_NEXT_LOG_INTERVAL_S="${AUTO_NEXT_LOG_INTERVAL_S:-1}"
export DETECT_NUM_FRAMES="${DETECT_NUM_FRAMES:-5}"

exec bash "${REPO_ROOT}/examples/piper/5d_rollout_grasp_episodic.sh"
