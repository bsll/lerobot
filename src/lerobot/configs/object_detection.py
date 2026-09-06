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

"""Optional YOLO-based grasp-target detection used during recording and rollout."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GraspObjectDetectionConfig:
    """Detect a grasp target (e.g. crab stick) and append it to ``observation.state``.

    Disabled by default. Enable from the CLI with::

        --object_detection.enabled=true \\
        --object_detection.model_path=best.pt \\
        --object_detection.camera_key=front

    Before each episode (record) or grasp attempt (rollout), YOLO runs once.
    The highest-confidence detection is recommended to the operator / policy and
    its normalized OBB ``(cx, cy, w, h, angle)`` is written into every frame's state:

    - ``cx, w`` / image width
    - ``cy, h`` / image height
    - ``angle`` / ``pi``
    """

    # Run YOLO detection and append the best target to observation.state.
    enabled: bool = False
    # Path to an Ultralytics YOLO weights file (OBB models supported).
    model_path: str | Path = "best.pt"
    # Observation camera key to run on. ``None`` picks the first RGB image key.
    camera_key: str | None = None
    # Minimum detection confidence.
    conf: float = 0.25
    # Inference device: ``"cpu"``, ``"cuda"``, ``"cuda:0"``, or ``"auto"``.
    device: str = "auto"
    # Keep detections whose center falls inside this normalized ROI rectangle:
    #   cx in [min_cx_ratio * width,  max_cx_ratio * width)
    #   cy in [min_cy_ratio * height, max_cy_ratio * height)
    # Example for the red tray on the right of a 640x360 panoramic view:
    #   min_cx_ratio=0.52 max_cx_ratio=0.95 min_cy_ratio=0.18 max_cy_ratio=0.75
    min_cx_ratio: float = 0.0
    max_cx_ratio: float = 1.0
    min_cy_ratio: float = 0.0
    max_cy_ratio: float = 1.0

    def __post_init__(self):
        if not (0.0 <= self.min_cx_ratio <= self.max_cx_ratio <= 1.0):
            raise ValueError(
                "object_detection requires 0 <= min_cx_ratio <= max_cx_ratio <= 1, "
                f"got min_cx_ratio={self.min_cx_ratio}, max_cx_ratio={self.max_cx_ratio}"
            )
        if not (0.0 <= self.min_cy_ratio <= self.max_cy_ratio <= 1.0):
            raise ValueError(
                "object_detection requires 0 <= min_cy_ratio <= max_cy_ratio <= 1, "
                f"got min_cy_ratio={self.min_cy_ratio}, max_cy_ratio={self.max_cy_ratio}"
            )
