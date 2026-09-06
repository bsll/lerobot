#!/usr/bin/env bash
# Auto-calibrate SO-101 follower and/or leader (JoyandAI-style limit exploration).
#
# Usage:
#   ./scripts/auto_calibrate_so101.sh robot
#   ./scripts/auto_calibrate_so101.sh tele
#   ./scripts/auto_calibrate_so101.sh both
#
# Env overrides: ROBOT_PORT, TELEOP_PORT, ROBOT_ID, TELEOP_ID, YES=true

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

AUTO_CALIB="${ROOT}/.venv/bin/lerobot-auto-calibrate"
if [[ ! -x "$AUTO_CALIB" ]]; then
  AUTO_CALIB="lerobot-auto-calibrate"
fi

TARGET="${1:-}"
case "$TARGET" in
  robot|tele|both) ;;
  *)
    echo "Usage: $0 robot|tele|both" >&2
    exit 1
    ;;
esac

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
TELEOP_PORT="${TELEOP_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-R12252702}"
TELEOP_ID="${TELEOP_ID:-R07252702}"
YES_FLAG=()
if [[ "${YES:-false}" == "true" ]]; then
  YES_FLAG=(--yes=true)
fi

run_robot() {
  echo "=== Auto-calibrating follower (robot) on $ROBOT_PORT id=$ROBOT_ID ==="
  "$AUTO_CALIB" \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id="$ROBOT_ID" \
    "${YES_FLAG[@]}"
}

run_tele() {
  echo "=== Auto-calibrating leader (teleop) on $TELEOP_PORT id=$TELEOP_ID ==="
  "$AUTO_CALIB" \
    --teleop.type=so101_leader \
    --teleop.port="$TELEOP_PORT" \
    --teleop.id="$TELEOP_ID" \
    "${YES_FLAG[@]}"
}

case "$TARGET" in
  robot) run_robot ;;
  tele) run_tele ;;
  both)
    run_robot
    run_tele
    ;;
esac
