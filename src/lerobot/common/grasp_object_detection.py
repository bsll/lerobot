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

"""YOLO (OBB) helpers for selecting a grasp target and storing it in state.

Workflow
--------
1. Before each episode / grasp attempt, run detection on a camera frame.
2. Keep the highest-confidence target as the recommended grasp.
3. Append that target's normalized OBB (cx, cy, w, h, angle) to ``observation.state`` for
   every recorded / inferred frame until the next refresh.

   Normalization uses image size: ``cx/w /= width``, ``cy/h /= height``, ``angle /= pi``.
"""

from __future__ import annotations

import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.configs.object_detection import GraspObjectDetectionConfig
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.import_utils import _ultralytics_available

if TYPE_CHECKING or _ultralytics_available:
    from ultralytics import YOLO
else:
    YOLO = None

logger = logging.getLogger(__name__)

# Flattened into ``observation.state`` after joint positions.
GRASP_TARGET_STATE_NAMES: tuple[str, ...] = (
    "grasp_target.cx",
    "grasp_target.cy",
    "grasp_target.w",
    "grasp_target.h",
    "grasp_target.angle",
)


@dataclass(frozen=True)
class GraspObjectDetection:
    """One oriented detection in image pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    # OBB center / size / rotation (radians, Ultralytics convention).
    cx: float
    cy: float
    width: float
    height: float
    angle: float
    # Four corner points as [[x, y], ...] in image coordinates.
    corners: tuple[tuple[float, float], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
            "corners": [list(pt) for pt in self.corners],
        }

    def to_state_values(self, image_hw: tuple[int, int]) -> dict[str, float]:
        """Map this detection to normalized flat keys for ``observation.state``.

        Normalization:
        - ``cx, w`` divided by image width
        - ``cy, h`` divided by image height
        - ``angle`` divided by ``pi`` (radians → roughly ``[-1, 1]`` / ``[0, 1]``)
        """
        image_h, image_w = image_hw
        if image_h <= 0 or image_w <= 0:
            raise ValueError(f"Invalid image shape for normalization: hw={image_hw}")
        return {
            "grasp_target.cx": float(self.cx) / float(image_w),
            "grasp_target.cy": float(self.cy) / float(image_h),
            "grasp_target.w": float(self.width) / float(image_w),
            "grasp_target.h": float(self.height) / float(image_h),
            "grasp_target.angle": float(self.angle) / float(np.pi),
        }


def empty_grasp_target_state_values() -> dict[str, float]:
    """Zero placeholder used when no detection is available."""
    return {name: 0.0 for name in GRASP_TARGET_STATE_NAMES}


def select_best_detection(
    detections: list[GraspObjectDetection],
) -> GraspObjectDetection | None:
    """Return the detection with the highest confidence, or ``None`` if empty."""
    if not detections:
        return None
    return max(detections, key=lambda d: d.confidence)


def select_camera_image(
    observation: dict[str, Any], camera_key: str | None = None
) -> tuple[str, NDArray[Any]]:
    """Return ``(camera_key, HxWx3 uint8 image)`` from a robot observation."""
    if camera_key is not None:
        if camera_key not in observation:
            available = sorted(observation.keys())
            raise KeyError(
                f"Camera key {camera_key!r} not found in observation. Available keys: {available}"
            )
        image = observation[camera_key]
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"Observation[{camera_key!r}] is not an HxWx3 image "
                f"(got type={type(image)}, shape={getattr(image, 'shape', None)})."
            )
        return camera_key, image

    rgb_keys = sorted(
        key
        for key, value in observation.items()
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3
    )
    if rgb_keys:
        key = rgb_keys[0]
        return key, observation[key]

    raise ValueError(
        "No RGB camera image found in observation. "
        f"Available keys: {sorted(observation.keys())}. "
        "Set --object_detection.camera_key=<name> explicitly."
    )


def extend_observation_state_features(features: dict[str, dict]) -> dict[str, dict]:
    """Append grasp-target names to ``observation.state`` in a dataset feature dict."""
    features = deepcopy(features)
    state_ft = features.get(OBS_STATE)
    if state_ft is None:
        features[OBS_STATE] = {
            "dtype": "float32",
            "shape": (len(GRASP_TARGET_STATE_NAMES),),
            "names": list(GRASP_TARGET_STATE_NAMES),
        }
        return features

    names = list(state_ft.get("names") or [])
    for name in GRASP_TARGET_STATE_NAMES:
        if name not in names:
            names.append(name)
    state_ft["names"] = names
    state_ft["shape"] = (len(names),)
    features[OBS_STATE] = state_ft
    return features


def extend_hw_observation_features(observation_features: dict[str, Any]) -> dict[str, Any]:
    """Add grasp-target float keys to robot-style observation feature specs."""
    extended = dict(observation_features)
    for name in GRASP_TARGET_STATE_NAMES:
        extended[name] = float
    return extended


class GraspObjectDetector:
    """Thin wrapper around an Ultralytics YOLO OBB model."""

    def __init__(self, cfg: GraspObjectDetectionConfig):
        if not cfg.enabled:
            raise ValueError("GraspObjectDetector requires object_detection.enabled=true.")
        if not _ultralytics_available or YOLO is None:
            raise ImportError(
                "ultralytics is required for --object_detection.enabled=true. "
                "Install it with: uv pip install ultralytics"
            )

        model_path = Path(cfg.model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at {model_path.resolve()}. "
                "Pass --object_detection.model_path=/path/to/best.pt"
            )

        self.cfg = cfg
        self.model_path = model_path
        self.model = YOLO(str(model_path))
        self.last_camera_key: str | None = None
        self.last_detections: list[GraspObjectDetection] = []
        logger.info(
            "Loaded grasp-object YOLO model from %s (task=%s, conf=%s, rgb→bgr=%s)",
            model_path,
            self.model.task,
            cfg.conf,
            cfg.convert_rgb_to_bgr,
        )

    def detect_image(self, image: NDArray[Any]) -> list[GraspObjectDetection]:
        """Run detection on an HxWx3 robot image (RGB by default).

        When ``cfg.convert_rgb_to_bgr`` is True (default), converts to BGR before
        YOLO so inference matches ``collect_yolo_images`` (OpenCV VideoCapture).
        """
        device = None if self.cfg.device == "auto" else self.cfg.device
        infer = image
        if self.cfg.convert_rgb_to_bgr:
            infer = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        result = self.model.predict(infer, conf=self.cfg.conf, device=device, verbose=False)[0]
        detections: list[GraspObjectDetection] = []

        obb = result.obb
        if obb is None or len(obb) == 0:
            self.last_detections = detections
            return detections

        xywhr = obb.xywhr.cpu().numpy()
        corners = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        classes = obb.cls.cpu().numpy().astype(int)
        names = result.names

        for i in range(len(obb)):
            class_id = int(classes[i])
            cx, cy, width, height, angle = map(float, xywhr[i])
            corner_pts = tuple((float(x), float(y)) for x, y in corners[i])
            detections.append(
                GraspObjectDetection(
                    class_id=class_id,
                    class_name=str(names.get(class_id, class_id)),
                    confidence=float(confs[i]),
                    cx=cx,
                    cy=cy,
                    width=width,
                    height=height,
                    angle=angle,
                    corners=corner_pts,
                )
            )

        image_h = float(image.shape[0])
        image_w = float(image.shape[1])
        x_min = self.cfg.min_cx_ratio * image_w
        x_max = self.cfg.max_cx_ratio * image_w
        y_min = self.cfg.min_cy_ratio * image_h
        y_max = self.cfg.max_cy_ratio * image_h
        roi_active = (
            self.cfg.min_cx_ratio > 0.0
            or self.cfg.max_cx_ratio < 1.0
            or self.cfg.min_cy_ratio > 0.0
            or self.cfg.max_cy_ratio < 1.0
        )
        if roi_active:
            before = len(detections)
            detections = [
                d for d in detections if x_min <= d.cx < x_max and y_min <= d.cy < y_max
            ]
            logger.debug(
                "ROI filter (cx,cy) in [%.1f,%.1f)x[%.1f,%.1f) kept %d/%d detections",
                x_min,
                x_max,
                y_min,
                y_max,
                len(detections),
                before,
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        self.last_detections = detections
        return detections

    def detect_observation(self, observation: dict[str, Any]) -> list[GraspObjectDetection]:
        camera_key, image = select_camera_image(observation, self.cfg.camera_key)
        self.last_camera_key = camera_key
        return self.detect_image(image)


class GraspTargetTracker:
    """Keeps the current recommended grasp target for recording / inference."""

    def __init__(self, detector: GraspObjectDetector):
        self.detector = detector
        self.selected: GraspObjectDetection | None = None
        self.state_values: dict[str, float] = empty_grasp_target_state_values()
        self.last_image_hw: tuple[int, int] | None = None

    def refresh(
        self,
        observation: dict[str, Any] | None = None,
        *,
        get_observation: Callable[[], dict[str, Any]] | None = None,
    ) -> GraspObjectDetection | None:
        """Detect over one or more frames; keep the highest-confidence target.

        Prefer ``get_observation`` so ``num_detect_frames`` consecutive grabs can
        advance the camera buffer. If only ``observation`` is passed, a single
        frame is used (backward compatible).
        """
        if get_observation is None and observation is None:
            raise ValueError("refresh() requires observation=... or get_observation=...")

        n = max(1, int(self.detector.cfg.num_detect_frames))
        interval = float(self.detector.cfg.detect_frame_interval_s)
        if get_observation is None:
            n = 1

            def get_observation() -> dict[str, Any]:
                assert observation is not None
                return observation

        best: GraspObjectDetection | None = None
        last_obs: dict[str, Any] | None = None
        per_frame_conf: list[float] = []

        for i in range(n):
            obs = get_observation()
            last_obs = obs
            camera_key, image = select_camera_image(obs, self.detector.cfg.camera_key)
            self.detector.last_camera_key = camera_key
            self.last_image_hw = (int(image.shape[0]), int(image.shape[1]))

            detections = self.detector.detect_image(image)
            frame_best = select_best_detection(detections)
            per_frame_conf.append(frame_best.confidence if frame_best is not None else 0.0)
            if frame_best is not None and (best is None or frame_best.confidence > best.confidence):
                best = frame_best

            if i + 1 < n and interval > 0.0:
                time.sleep(interval)

        assert last_obs is not None
        self.selected = best
        if best is None:
            logger.warning(
                "No grasp target detected on camera %r over %d frame(s) "
                "(per-frame max conf=%s) — writing zeros into state.",
                self.detector.last_camera_key,
                n,
                [round(c, 3) for c in per_frame_conf],
            )
            self.state_values = empty_grasp_target_state_values()
            return None

        assert self.last_image_hw is not None
        self.state_values = best.to_state_values(self.last_image_hw)
        logger.info(
            "Selected grasp target on camera %r over %d frame(s) "
            "(per-frame max conf=%s): %s conf=%.2f "
            "pixel_center=(%.1f, %.1f) pixel_size=(%.1fx%.1f) angle=%.3frad | "
            "normalized state=%s",
            self.detector.last_camera_key,
            n,
            [round(c, 3) for c in per_frame_conf],
            best.class_name,
            best.confidence,
            best.cx,
            best.cy,
            best.width,
            best.height,
            best.angle,
            {k: round(v, 4) for k, v in self.state_values.items()},
        )
        return best

    def inject(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return a shallow copy of *observation* with grasp-target state keys added."""
        merged = dict(observation)
        merged.update(self.state_values)
        return merged

    def annotate_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Return observation with the selected grasp target drawn on the detect camera.

        Intended for live display only — do not feed the annotated image into the dataset.
        """
        camera_key = self.detector.last_camera_key
        if camera_key is None or camera_key not in observation:
            return observation
        image = observation[camera_key]
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            return observation

        annotated = draw_grasp_guidance(
            image,
            selected=self.selected,
            roi=(
                self.detector.cfg.min_cx_ratio,
                self.detector.cfg.min_cy_ratio,
                self.detector.cfg.max_cx_ratio,
                self.detector.cfg.max_cy_ratio,
            ),
        )
        merged = dict(observation)
        merged[camera_key] = annotated
        return merged


def draw_grasp_guidance(
    image: NDArray[Any],
    selected: GraspObjectDetection | None,
    others: list[GraspObjectDetection] | None = None,
    label: str = "GRASP THIS",
    roi: tuple[float, float, float, float] | None = None,
) -> NDArray[Any]:
    """Draw minimal OBB guidance on a camera frame (RGB or BGR HxWx3).

    Shows the ROI rectangle (when set), plus the selected target box and confidence.
    ``others`` / ``label`` are kept for call-site compatibility but ignored.
    """
    del others, label
    import cv2

    canvas = np.ascontiguousarray(image.copy())
    h, w = canvas.shape[:2]

    if roi is not None:
        min_cx, min_cy, max_cx, max_cy = roi
        x1, y1 = int(min_cx * w), int(min_cy * h)
        x2, y2 = int(max_cx * w) - 1, int(max_cy * h) - 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 180, 255), 1)

    if selected is None:
        return canvas

    color = (0, 255, 80)
    pts = np.asarray(selected.corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)

    text = f"{selected.confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    text_thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, text_thickness)
    cx_i, cy_i = int(round(selected.cx)), int(round(selected.cy))
    x = max(0, min(cx_i - tw // 2, canvas.shape[1] - tw - 1))
    y = max(th + 2, min(cy_i - 8, canvas.shape[0] - baseline - 1))
    cv2.putText(canvas, text, (x, y), font, scale, color, text_thickness, cv2.LINE_AA)

    return canvas


def format_best_detection_log(
    detection: GraspObjectDetection | None, camera_key: str | None = None
) -> str:
    """Human-readable summary of the selected grasp target."""
    cam = camera_key or "?"
    if detection is None:
        return f"No grasp target selected on camera '{cam}'."
    return (
        f"Best grasp target on camera '{cam}': {detection.class_name} "
        f"conf={detection.confidence:.2f} center=({detection.cx:.1f}, {detection.cy:.1f}) "
        f"size=({detection.width:.1f}x{detection.height:.1f}) angle={detection.angle:.3f}rad"
    )


def make_grasp_target_tracker(cfg: GraspObjectDetectionConfig) -> GraspTargetTracker | None:
    """Return a tracker when detection is enabled, otherwise ``None``."""
    if not cfg.enabled:
        return None
    return GraspTargetTracker(GraspObjectDetector(cfg))
