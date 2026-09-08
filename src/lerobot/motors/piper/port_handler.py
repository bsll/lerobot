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

"""Minimal CAN port lifecycle wrapper for piper_sdk.

Absorbed from wego_piper / lerobot_robot_piper so Piper support only depends on
``piper_sdk``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from lerobot.utils.import_utils import _piper_sdk_available

if TYPE_CHECKING or _piper_sdk_available:
    from piper_sdk import C_PiperInterface_V2
else:
    C_PiperInterface_V2 = object  # type: ignore[misc, assignment]

DEFAULT_BAUDRATE = 1_000_000


class PortHandler:
    def __init__(self) -> None:
        self.baudrate = DEFAULT_BAUDRATE
        self.packet_start_time = 0.0
        self.packet_timeout = 0.0
        self.tx_time_per_byte = 0.0
        self.is_open = False
        self.is_using = False
        self.port_name = ""
        self.ser: C_PiperInterface_V2 | None = None

    def openPort(self) -> bool:  # noqa: N802
        assert self.ser is not None
        self.ser.ConnectPort()
        self.is_open = bool(self.ser.get_connect_status())
        return self.is_open

    def closePort(self) -> None:  # noqa: N802
        if self.ser is None:
            self.is_open = False
            return
        self.ser.DisconnectPort()
        self.is_open = bool(self.ser.get_connect_status())

    def clearPort(self) -> None:  # noqa: N802
        pass

    def setupPort(self, controller: C_PiperInterface_V2) -> bool:  # noqa: N802
        if self.ser is not None:
            self.closePort()

        self.port_name = controller.GetCanName()
        self.ser = controller
        self.tx_time_per_byte = (1000.0 / self.baudrate) * 10.0
        return True

    def getPortName(self) -> str:  # noqa: N802
        if self.ser is None:
            return self.port_name
        return self.ser.GetCanName()

    def getBaudRate(self) -> int:  # noqa: N802
        return self.baudrate

    def getCurrentTime(self) -> float:  # noqa: N802
        return round(time.time() * 1_000_000_000) / 1_000_000.0
