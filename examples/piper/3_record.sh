#!/usr/bin/env bash
# Dual-port Piper leader-follower dataset recording.
# Replace dataset.repo_id / cameras / task before a long session.

set -euo pipefail

HF_USER="${HF_USER:-your_HF_id}"
DATASET_NAME="${DATASET_NAME:-piper_demo}"

lerobot-record \
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
  --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
  --dataset.num_episodes=10 \
  --dataset.single_task="Put the object in the cup." \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=10 \
  --display_data=true
