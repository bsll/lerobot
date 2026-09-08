#!/usr/bin/env python

# Copyright 2025 WeGo-Robotics Inc. EDU team. All rights reserved.
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

from lerobot.motors import Motor, MotorCalibration, MotorNormMode

MODEL_RESOLUTION_TABLE = {
    "AGILEX-M": 4096,
    "AGILEX-S": 4096,
}

# Parking pose in normalized units (RANGE_M100_100 for joints, RANGE_0_100 for gripper).
INITIALIZE_POSITION = {
    "joint1": 0.0,
    "joint2": -100.0,
    "joint3": 100.0,
    "joint4": 0.0,
    "joint5": 35.0,
    "joint6": 0.0,
    "gripper": 0.0,
}

# Joint encoder ranges used by Piper SDK (milli-degrees / gripper units).
DEFAULT_CALIBRATION: dict[str, MotorCalibration] = {
    "joint1": MotorCalibration(1, 0, 0, -150000, 150000),
    "joint2": MotorCalibration(2, 0, 0, 0, 180000),
    "joint3": MotorCalibration(3, 0, 0, -170000, 0),
    "joint4": MotorCalibration(4, 0, 0, -100000, 100000),
    "joint5": MotorCalibration(5, 0, 0, -65000, 65000),
    "joint6": MotorCalibration(6, 0, 0, -120000, 120000),
    "gripper": MotorCalibration(7, 0, 0, 0, 68000),
}


def make_default_motors() -> dict[str, Motor]:
    """Build the standard Piper joint/gripper motor map."""
    return {
        "joint1": Motor(1, "AGILEX-M", MotorNormMode.RANGE_M100_100),
        "joint2": Motor(2, "AGILEX-M", MotorNormMode.RANGE_M100_100),
        "joint3": Motor(3, "AGILEX-M", MotorNormMode.RANGE_M100_100),
        "joint4": Motor(4, "AGILEX-S", MotorNormMode.RANGE_M100_100),
        "joint5": Motor(5, "AGILEX-S", MotorNormMode.RANGE_M100_100),
        "joint6": Motor(6, "AGILEX-S", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(7, "AGILEX-S", MotorNormMode.RANGE_0_100),
    }
