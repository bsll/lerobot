# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Automatically calibrate an SO-series Feetech arm by exploring joint limits.

Ported from JoyandAI/lerobot auto-calibration. Unlike ``lerobot-calibrate``
(manual middle-pose + hand-moved ranges), this command drives each joint to
its mechanical stops and writes ``homing_offset`` / ``range_min`` / ``range_max``.

Requires: ``pip install 'lerobot[hardware]'`` (or ``lerobot[feetech]``).

Examples:

```shell
# Follower (robot)
lerobot-auto-calibrate \\
    --robot.type=so101_follower \\
    --robot.port=/dev/ttyACM0 \\
    --robot.id=my_follower \\
    --try_torque=600 \\
    --max_torque=1000

# Leader (teleop)
lerobot-auto-calibrate \\
    --teleop.type=so101_leader \\
    --teleop.port=/dev/ttyACM1 \\
    --teleop.id=my_leader \\
    --try_torque=400 \\
    --max_torque=500
```

Clear the workspace before running. Use ``--yes`` to skip the confirmation prompt.
"""

import logging
from dataclasses import asdict, dataclass, fields
from pprint import pformat

import draccus

from lerobot.motors.auto_calibrate import (
    AutoCalibrateConfig,
    auto_calibrate_connected_device,
    set_calibration_paused,
)
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so_leader,
    make_teleoperator_from_config,
    omx_leader,
    so_leader,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class AutoCalibratePipelineConfig:
    """CLI config: choose either ``--robot.*`` or ``--teleop.*``, plus torque params."""

    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    # Skip the interactive "Press ENTER" confirmation.
    yes: bool = False

    # Auto-calibration parameters (same defaults as AutoCalibrateConfig).
    try_torque: int = 400
    max_torque: int = 600
    wrist_roll_torque_limit: int = 300
    gripper_torque_limit: int = 300
    shoulder_pan_midpoint_adjustment: int = -150
    torque_step: int = 50
    explore_velocity: int = 600
    wait_time_s: float = 0.3
    velocity_threshold: int = 4
    position_tolerance: int = 4000

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot (exactly one).")


# Sensible defaults for SO-101 follower vs leader when the user does not override.
ROBOT_DEFAULTS = {
    "try_torque": 600,
    "max_torque": 1000,
    "explore_velocity": 800,
    "wait_time_s": 0.5,
}
TELEOP_DEFAULTS = {
    "try_torque": 400,
    "max_torque": 500,
    "explore_velocity": 600,
    "wait_time_s": 0.5,
}


def _apply_device_defaults(cfg: AutoCalibratePipelineConfig) -> AutoCalibrateConfig:
    """Build the core auto-calibrate params, filling device-specific defaults."""
    defaults = ROBOT_DEFAULTS if cfg.robot is not None else TELEOP_DEFAULTS
    base = AutoCalibrateConfig()
    kwargs = {}
    for field_info in fields(AutoCalibrateConfig):
        field_name = field_info.name
        current = getattr(cfg, field_name)
        if field_name in defaults and current == getattr(base, field_name):
            kwargs[field_name] = defaults[field_name]
        else:
            kwargs[field_name] = current
    return AutoCalibrateConfig(**kwargs)


@draccus.wrap()
def auto_calibrate(cfg: AutoCalibratePipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    set_calibration_paused(False)

    if isinstance(cfg.robot, RobotConfig):
        device = make_robot_from_config(cfg.robot)
    else:
        device = make_teleoperator_from_config(cfg.teleop)

    auto_cfg = _apply_device_defaults(cfg)

    logging.info("Connecting device (calibrate=False)...")
    device.connect(calibrate=False)

    try:
        logging.info(
            "Starting auto calibration. Clear the workspace — joints will move to mechanical stops."
        )
        if not cfg.yes:
            input("Press ENTER to continue (or Ctrl+C to abort)...")

        result = auto_calibrate_connected_device(device, auto_cfg)
        logging.info("Auto calibration finished successfully.")
        if result.calibration_path is not None:
            logging.info("Calibration file saved to: %s", result.calibration_path)

        for motor_name, calib in result.calibration_dict.items():
            logging.info(
                "  %s: id=%s offset=%s range=[%s, %s]",
                motor_name,
                calib.id,
                calib.homing_offset,
                calib.range_min,
                calib.range_max,
            )
    finally:
        device.disconnect()


def main():
    register_third_party_plugins()
    auto_calibrate()


if __name__ == "__main__":
    main()
