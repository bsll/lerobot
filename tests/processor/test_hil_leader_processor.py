#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    InterventionActionProcessorStep,
    LeaderHandoverProcessorStep,
    ResolveLeaderActionProcessorStep,
    create_transition,
)
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.processor.pipeline import ProcessorStep
from lerobot.teleoperators.utils import TeleopEvents


class _MockTeleopWithFeedback:
    def __init__(self) -> None:
        self.feedback_calls: list[dict] = []

    def send_feedback(self, feedback: dict) -> None:
        self.feedback_calls.append(feedback)


class _MockIKStep(ProcessorStep):
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition[TransitionKey.ACTION] = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        return transition

    def transform_features(self, features):
        return features


def test_intervention_action_converts_leader_joint_dict():
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex"]
    step = InterventionActionProcessorStep(use_gripper=False, motor_names=motor_names)

    transition = create_transition(
        action=torch.zeros(4),
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={
            TELEOP_ACTION_KEY: {
                "shoulder_pan.pos": 1.0,
                "shoulder_lift.pos": 2.0,
                "elbow_flex.pos": 3.0,
            }
        },
    )

    result = step(transition)
    torch.testing.assert_close(result[TransitionKey.ACTION], torch.tensor([1.0, 2.0, 3.0]))


def test_resolve_leader_action_bypasses_ik_for_joint_commands():
    motor_names = ["j1", "j2", "j3"]
    ik_step = _MockIKStep()
    step = ResolveLeaderActionProcessorStep(motor_names=motor_names, ik_steps=[ik_step])

    joint_action = torch.tensor([1.0, 2.0, 3.0])
    transition = create_transition(action=joint_action)

    result = step(transition)
    torch.testing.assert_close(result[TransitionKey.ACTION], joint_action)


def test_resolve_leader_action_runs_ik_for_ee_delta():
    motor_names = ["j1", "j2", "j3"]
    ik_step = _MockIKStep()
    step = ResolveLeaderActionProcessorStep(motor_names=motor_names, ik_steps=[ik_step])

    ee_delta = torch.tensor([0.1, 0.2, 0.3, 1.0])
    transition = create_transition(action=ee_delta)

    result = step(transition)
    torch.testing.assert_close(
        result[TransitionKey.ACTION], torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    )


def test_leader_handover_sends_feedback_when_intervention_starts():
    teleop = _MockTeleopWithFeedback()
    step = LeaderHandoverProcessorStep(teleop_device=teleop)

    follower_pose = {"shoulder_pan.pos": 10.0, "shoulder_lift.pos": 20.0}
    transition = create_transition(
        info={TeleopEvents.IS_INTERVENTION: True},
        complementary_data={"raw_joint_positions": follower_pose},
    )

    step(transition)
    assert teleop.feedback_calls == [follower_pose]

    # Still in intervention: no additional handover
    step(transition)
    assert len(teleop.feedback_calls) == 1

    # Leaving and re-entering intervention triggers another handover
    step(create_transition(info={TeleopEvents.IS_INTERVENTION: False}))
    step(transition)
    assert len(teleop.feedback_calls) == 2
