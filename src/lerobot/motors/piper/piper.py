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

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.motors_bus import MotorsBusBase, Value
from lerobot.utils.import_utils import _piper_sdk_available, require_package

from .port_handler import PortHandler
from .tables import INITIALIZE_POSITION, MODEL_RESOLUTION_TABLE

if TYPE_CHECKING or _piper_sdk_available:
    from piper_sdk import C_PiperInterface_V2
else:
    C_PiperInterface_V2 = object  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


class PiperMotorsBus(MotorsBusBase):
    """AgileX Piper arm bus backed by ``piper_sdk`` over SocketCAN."""

    apply_drive_mode = False

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        id: str | None = None,
    ):
        require_package("piper_sdk", extra="piper")
        super().__init__(port, motors, calibration)

        self.port_handler = PortHandler()
        self.id = id or port
        self._is_connected = False
        self.piper = C_PiperInterface_V2(port)
        logger.info("%s : %s is selected.", self.id, port)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, handshake: bool = True) -> None:
        self.port_handler.setupPort(self.piper)
        if not self.port_handler.openPort():
            raise ConnectionError(f"Failed to open Piper CAN port '{self.port}' for '{self.id}'.")
        self._is_connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque and self._is_connected:
            self.parking()
            self.piper.DisablePiper()
        self.port_handler.closePort()
        self._is_connected = False

    def read(self, data_name: str, motor: str) -> Value:
        del data_name
        return self.get_action().get(motor, 0.0)

    def write(self, data_name: str, motor: str, value: Value) -> None:
        del data_name
        current = self.get_action()
        current[motor] = float(value)
        self.set_action(current, is_conv=True)

    def sync_read(self, data_name: str, motors: str | list[str] | None = None) -> dict[str, Value]:
        del data_name
        pos = self.get_action()
        if motors is None:
            return pos
        if isinstance(motors, str):
            motors = [motors]
        return {m: pos[m] for m in motors if m in pos}

    def sync_write(self, data_name: str, values: dict[str, Value]) -> None:
        del data_name
        self.set_action(values, is_conv=True)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        del motors
        retry = num_retry if num_retry > 0 else 50
        while not self.piper.EnablePiper() and retry:
            retry -= 1
            time.sleep(0.1)
        logger.info("%s", self.piper.GetArmEnableStatus())
        if not retry:
            logger.warning("%s enable_torque timed out, continuing anyway", self.id)
        else:
            logger.info("%s torque on.", self.id)

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        del motors, num_retry
        self.piper.DisablePiper()

    def read_calibration(self) -> dict[str, MotorCalibration]:
        return self.calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration], cache: bool = True) -> None:
        if cache:
            self.calibration = calibration_dict

    @property
    def is_calibrated(self) -> bool:
        return True

    def clear_gripper(self) -> None:
        self.piper.GripperCtrl(0, 1000, 0x03, 0)

    def parking(self) -> None:
        timeout = 100
        self.set_action(INITIALIZE_POSITION, is_conv=True)
        time.sleep(0.1)
        status = self.piper.GetArmStatus()
        while status.arm_status.motion_status and timeout:
            self.set_action(INITIALIZE_POSITION, is_conv=True)
            time.sleep(0.1)
            status = self.piper.GetArmStatus()
            timeout -= 1

    def set_slave(self) -> None:
        self.piper.MasterSlaveConfig(0xFC, 0, 0, 0)

    def set_master(self) -> None:
        self.piper.MasterSlaveConfig(0xFA, 0, 0, 0)

    def get_action(self) -> dict[str, float]:
        msg_joint = self.piper.GetArmJointMsgs()
        msg_gripr = self.piper.GetArmGripperMsgs()
        raw = {
            "joint1": float(msg_joint.joint_state.joint_1),
            "joint2": float(msg_joint.joint_state.joint_2),
            "joint3": float(msg_joint.joint_state.joint_3),
            "joint4": float(msg_joint.joint_state.joint_4),
            "joint5": float(msg_joint.joint_state.joint_5),
            "joint6": float(msg_joint.joint_state.joint_6),
            "gripper": float(msg_gripr.gripper_state.grippers_angle),
        }
        return self._normalize(raw)

    def get_control(self) -> dict[str, float]:
        msg_joint = self.piper.GetArmJointCtrl()
        msg_gripr = self.piper.GetArmGripperCtrl()
        raw = {
            "joint1": float(msg_joint.joint_ctrl.joint_1),
            "joint2": float(msg_joint.joint_ctrl.joint_2),
            "joint3": float(msg_joint.joint_ctrl.joint_3),
            "joint4": float(msg_joint.joint_ctrl.joint_4),
            "joint5": float(msg_joint.joint_ctrl.joint_5),
            "joint6": float(msg_joint.joint_ctrl.joint_6),
            "gripper": float(msg_gripr.gripper_ctrl.grippers_angle),
        }
        return self._normalize(raw)

    def set_action(self, action: dict[str, Any], is_conv: bool = True) -> dict[str, float]:
        action_denormalized = self._unnormalize(action) if is_conv else action

        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(
            int(action_denormalized["joint1"]),
            int(action_denormalized["joint2"]),
            int(action_denormalized["joint3"]),
            int(action_denormalized["joint4"]),
            int(action_denormalized["joint5"]),
            int(action_denormalized["joint6"]),
        )
        self.piper.GripperCtrl(abs(int(action_denormalized["gripper"])), 1000, 0x03, 0)
        return self.get_control()

    def _normalize(self, ids_values: dict[str, float]) -> dict[str, float]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        normalized_values: dict[str, float] = {}
        for motor, val in ids_values.items():
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            bounded_val = min(max_, max(min_, val))
            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                norm = (((bounded_val - min_) / (max_ - min_)) * 200) - 100
                normalized_values[motor] = -norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                norm = ((bounded_val - min_) / (max_ - min_)) * 100
                normalized_values[motor] = 100 - norm if drive_mode else norm
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = MODEL_RESOLUTION_TABLE[self.motors[motor].model] - 1
                normalized_values[motor] = (val - mid) * 360 / max_res
            else:
                raise NotImplementedError(self.motors[motor].norm_mode)

        return normalized_values

    def _unnormalize(self, ids_values: dict[str, float]) -> dict[str, int]:
        if not self.calibration:
            raise RuntimeError(f"{self} has no calibration registered.")

        unnormalized_values: dict[str, int] = {}
        for motor, val in ids_values.items():
            min_ = self.calibration[motor].range_min
            max_ = self.calibration[motor].range_max
            drive_mode = self.apply_drive_mode and self.calibration[motor].drive_mode
            if max_ == min_:
                raise ValueError(f"Invalid calibration for motor '{motor}': min and max are equal.")

            if self.motors[motor].norm_mode is MotorNormMode.RANGE_M100_100:
                val = -val if drive_mode else val
                bounded_val = min(100.0, max(-100.0, float(val)))
                unnormalized_values[motor] = int(((bounded_val + 100) / 200) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.RANGE_0_100:
                val = 100 - val if drive_mode else val
                bounded_val = min(100.0, max(0.0, float(val)))
                unnormalized_values[motor] = int((bounded_val / 100) * (max_ - min_) + min_)
            elif self.motors[motor].norm_mode is MotorNormMode.DEGREES:
                mid = (min_ + max_) / 2
                max_res = MODEL_RESOLUTION_TABLE[self.motors[motor].model] - 1
                unnormalized_values[motor] = int((float(val) * max_res / 360) + mid)
            else:
                raise NotImplementedError(self.motors[motor].norm_mode)

        return unnormalized_values
