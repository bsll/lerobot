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

"""SO leader arm + keyboard episode controls for HIL-SERL."""

import logging
import time
from queue import Queue
from typing import Any

from lerobot.lerobot_types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .config_so_leader import SOLeaderKeyboardTeleopConfig
from .so_leader import SOLeader

logger = logging.getLogger(__name__)

keyboard = None
if _pynput_available:
    try:
        from pynput import keyboard
    except Exception as exc:
        logging.info("Could not import pynput keyboard backend: %s", exc)


class LeaderKeyboardTeleop(Teleoperator):
    """Leader-arm teleoperation with keyboard-only episode / intervention controls.

    The leader arm provides joint-space actions during human intervention. The keyboard
    handles HIL-SERL events (toggle intervention, success, failure, rerecord).

    Keyboard mappings:
        - Space: toggle human intervention on/off
        - s: mark episode success
        - Esc: mark episode failure
        - r: rerecord episode
    """

    config_class = SOLeaderKeyboardTeleopConfig
    name = "so_leader_keyboard"

    def __init__(self, config: SOLeaderKeyboardTeleopConfig):
        super().__init__(config)
        self.config = config
        self.leader = SOLeader(config)
        self._intervention_active = False
        self._misc_keys_queue: Queue[str] = Queue()
        self._keyboard_listener = None

    @property
    def action_features(self) -> dict[str, type]:
        return self.leader.action_features

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.leader.feedback_features

    @property
    def is_connected(self) -> bool:
        return self.leader.is_connected

    @property
    def is_calibrated(self) -> bool:
        return self.leader.is_calibrated

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        require_package("pynput", extra="pynput-dep")
        self.leader.connect(calibrate=calibrate)

        if keyboard is not None and pynput_can_capture():
            self._keyboard_listener = keyboard.Listener(
                on_press=self._on_press,
            )
            self._keyboard_listener.start()
            logger.info(
                "Leader+keyboard teleop ready. Space=toggle intervention, s=success, Esc=failure, r=rerecord."
            )
        else:
            logger.warning(
                "Keyboard events are unavailable in this environment. Connect an X11/Wayland session "
                "with pynput support, or grant macOS Accessibility permissions."
            )
            self._keyboard_listener = None

    def calibrate(self) -> None:
        self.leader.calibrate()

    def configure(self) -> None:
        self.leader.configure()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return self.leader.get_action()

    def get_teleop_events(self) -> dict[str, Any]:
        terminate_episode = False
        success = False
        rerecord_episode = False

        while not self._misc_keys_queue.empty():
            key = self._misc_keys_queue.get_nowait()
            if key == "s":
                success = True
                terminate_episode = True
            elif key == "r":
                rerecord_episode = True
                terminate_episode = True
            elif key == "esc":
                terminate_episode = True
                success = False

        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_active,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        self.leader.send_feedback(feedback)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        self.leader.disconnect()

    def _on_press(self, key) -> None:
        if keyboard is None:
            return

        if key == keyboard.Key.space:
            self._intervention_active = not self._intervention_active
            state = "ON" if self._intervention_active else "OFF"
            logger.info("Human intervention toggled %s", state)
            return

        if key == keyboard.Key.esc:
            self._misc_keys_queue.put("esc")
            return

        if hasattr(key, "char") and key.char is not None:
            char = key.char.lower()
            if char in {"s", "r"}:
                self._misc_keys_queue.put(char)
