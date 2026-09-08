#!/usr/bin/env bash
# Leader-follower teleoperation for AgileX Piper.
# Edit camera device paths before running.

set -euo pipefail

lerobot-teleoperate \
  --robot.type=piper_follower \
  --robot.port=can_follower \
  --robot.id=follower \
  --robot.cameras="{
    front: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 30},
    wrist: {type: opencv, index_or_path: /dev/video10, width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=piper_leader \
  --teleop.port=can_leader \
  --teleop.id=leader \
  --display_data=true
