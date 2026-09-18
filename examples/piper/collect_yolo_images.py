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

"""Capture still images for YOLO training — one target at a time (like 3c grasp).

Workflow (simulates recording)
------------------------------
1. Run existing OBB weights; show **only the highest-confidence** detection (≥ conf).
2. Physically remove that crab stick, then press **n** / **→**.
3. After a short reset countdown, re-detect and show the next best target.
4. When **no** detection remains, prompt to **Space** / **s** save the clean frame.
5. After save, another reset to rearrange a new scene; repeat. **Esc** / **q** quit.

Usage::

    bash examples/piper/collect_yolo_images.sh
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np


class Phase(Enum):
    SHOW = auto()  # one target highlighted
    REMOVE = auto()  # countdown while user removes current target
    EMPTY = auto()  # no dets — prompt save
    ARRANGE = auto()  # countdown after save to set up a new scene


@dataclass(frozen=True)
class Det:
    conf: float
    class_name: str
    corners: np.ndarray  # (4, 2)
    cx: float
    cy: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=repo_root / "train_data" / "yolo_dataset")
    p.add_argument("--camera", type=str, default="/dev/video4")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--reset-time-s",
        type=float,
        default=5.0,
        help="Countdown after 'next' (remove) and after save (rearrange).",
    )
    p.add_argument("--model-path", type=Path, default=repo_root / "best.pt")
    p.add_argument("--detect-conf", type=float, default=0.5, help="Min conf; only the best above this is shown.")
    p.add_argument("--detect-every", type=int, default=2)
    p.add_argument("--detect-device", type=str, default="auto")
    # Same ROI as 3c_record_singleport_grasp.sh (center must fall inside).
    p.add_argument("--min-cx-ratio", type=float, default=0.47)
    p.add_argument("--max-cx-ratio", type=float, default=0.83)
    p.add_argument("--min-cy-ratio", type=float, default=0.12)
    p.add_argument("--max-cy-ratio", type=float, default=0.55)
    p.add_argument("--class-name", type=str, default="target")
    p.add_argument("--jpeg-quality", type=int, default=95)
    return p.parse_args()


def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise ConnectionError(f"Failed to open camera {device} with V4L2 backend.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(10):
        cap.read()
        time.sleep(0.05)
    print(
        f"Camera {device} opened: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.1f} fps"
    )
    return cap


def load_yolo(model_path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics is required. Install with: uv pip install ultralytics") from e
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO weights not found: {model_path.resolve()}")
    model = YOLO(str(model_path))
    print(f"Loaded YOLO from {model_path} (task={model.task})")
    return model


def run_detect(
    model,
    frame_bgr: np.ndarray,
    conf: float,
    device: str,
    *,
    min_cx_ratio: float,
    max_cx_ratio: float,
    min_cy_ratio: float,
    max_cy_ratio: float,
) -> list[Det]:
    device_arg = None if device == "auto" else device
    result = model.predict(frame_bgr, conf=conf, device=device_arg, verbose=False)[0]
    dets: list[Det] = []
    obb = result.obb
    if obb is None or len(obb) == 0:
        return dets
    xywhr = obb.xywhr.cpu().numpy()
    corners = obb.xyxyxyxy.cpu().numpy()
    confs = obb.conf.cpu().numpy()
    classes = obb.cls.cpu().numpy().astype(int)
    names = result.names or {}
    image_h, image_w = float(frame_bgr.shape[0]), float(frame_bgr.shape[1])
    x_min, x_max = min_cx_ratio * image_w, max_cx_ratio * image_w
    y_min, y_max = min_cy_ratio * image_h, max_cy_ratio * image_h
    roi_active = (
        min_cx_ratio > 0.0 or max_cx_ratio < 1.0 or min_cy_ratio > 0.0 or max_cy_ratio < 1.0
    )
    for i in range(len(obb)):
        c = float(confs[i])
        if c < conf:
            continue
        class_id = int(classes[i])
        cx, cy = float(xywhr[i][0]), float(xywhr[i][1])
        if roi_active and not (x_min <= cx < x_max and y_min <= cy < y_max):
            continue
        dets.append(
            Det(
                conf=c,
                class_name=str(names.get(class_id, class_id)),
                corners=corners[i].reshape(4, 2),
                cx=cx,
                cy=cy,
            )
        )
    dets.sort(key=lambda d: d.conf, reverse=True)
    return dets


def select_best(dets: list[Det]) -> Det | None:
    return dets[0] if dets else None


def draw_roi(
    frame_bgr: np.ndarray,
    min_cx_ratio: float,
    max_cx_ratio: float,
    min_cy_ratio: float,
    max_cy_ratio: float,
) -> np.ndarray:
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    x1, y1 = int(min_cx_ratio * w), int(min_cy_ratio * h)
    x2, y2 = int(max_cx_ratio * w) - 1, int(max_cy_ratio * h) - 1
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 180, 255), 1)
    return canvas


def draw_one(frame_bgr: np.ndarray, det: Det | None, *, label_prefix: str = "GRASP THIS") -> np.ndarray:
    canvas = frame_bgr.copy()
    if det is None:
        return canvas
    pts = np.asarray(det.corners, dtype=np.int32).reshape(-1, 1, 2)
    color = (0, 255, 80)
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
    text = f"{label_prefix}  {det.class_name} {det.conf:.2f}"
    cx_i, cy_i = int(round(det.cx)), int(round(det.cy))
    cv2.putText(
        canvas,
        text,
        (max(0, cx_i - 80), max(20, cy_i - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def draw_hud(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = frame_bgr.copy()
    y = 28
    for line in lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 220, 80), 2, cv2.LINE_AA)
        y += 26
    return canvas


def next_image_index(images_dir: Path) -> int:
    existing = list(images_dir.glob("img_*.jpg")) + list(images_dir.glob("img_*.png"))
    if not existing:
        return 0
    indices: list[int] = []
    for path in existing:
        try:
            indices.append(int(path.stem.split("_", 1)[1].split("_", 1)[0]))
        except (IndexError, ValueError):
            continue
    return (max(indices) + 1) if indices else 0


def save_frame(images_dir: Path, idx: int, frame: np.ndarray, jpeg_quality: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = images_dir / f"img_{idx:06d}_{stamp}.jpg"
    ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError(f"Failed to write {path}")
    return path


def main() -> None:
    args = parse_args()
    out_root = args.output_dir.expanduser().resolve()
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    classes_path = out_root / "classes.txt"
    if not classes_path.exists():
        classes_path.write_text(f"{args.class_name}\n", encoding="utf-8")

    model = load_yolo(Path(args.model_path).expanduser())
    idx = next_image_index(images_dir)
    cap = open_camera(args.camera, args.width, args.height, args.fps)
    window = "YOLO capture — one target at a time"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    print(f"Saving to {images_dir}")
    print(f"Show only best det with conf ≥ {args.detect_conf:.2f}")
    print(
        f"ROI (center): cx=[{args.min_cx_ratio},{args.max_cx_ratio}) "
        f"cy=[{args.min_cy_ratio},{args.max_cy_ratio})"
    )
    print("Keys: n/→ = taken away → next | Space = save when empty | Esc/q = quit")

    roi_kw = dict(
        min_cx_ratio=args.min_cx_ratio,
        max_cx_ratio=args.max_cx_ratio,
        min_cy_ratio=args.min_cy_ratio,
        max_cy_ratio=args.max_cy_ratio,
    )

    def detect(frame_bgr: np.ndarray) -> list[Det]:
        return run_detect(model, frame_bgr, args.detect_conf, args.detect_device, **roi_kw)

    def decorate(frame_bgr: np.ndarray, det: Det | None = None) -> np.ndarray:
        view = draw_roi(frame_bgr, **roi_kw)
        if det is not None:
            view = draw_one(view, det)
        return view

    phase = Phase.SHOW
    current: Det | None = None
    phase_deadline = 0.0
    frame_i = 0
    taken_count = 0  # how many "next" in this scene round

    # Initial detect
    ok, boot = cap.read()
    if ok and boot is not None:
        current = select_best(detect(boot))
        if current is None:
            phase = Phase.EMPTY
            print("No target at start — rearrange or Space to save empty frame.")
        else:
            print(f"Current target conf={current.conf:.2f}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            now = time.perf_counter()

            # --- phase transitions on timer ---
            if phase in (Phase.REMOVE, Phase.ARRANGE) and now >= phase_deadline:
                dets = detect(frame)
                current = select_best(dets)
                if current is None:
                    phase = Phase.EMPTY
                    print("No more targets — press Space/s to save, or r to re-detect.")
                else:
                    phase = Phase.SHOW
                    print(f"Next target conf={current.conf:.2f}  (taken so far this round: {taken_count})")

            # Refresh current box lightly while SHOW (object may move)
            if phase == Phase.SHOW and frame_i % max(1, args.detect_every) == 0:
                dets = detect(frame)
                current = select_best(dets)
                if current is None:
                    phase = Phase.EMPTY
                    print("No more targets — press Space/s to save, or r to re-detect.")

            frame_i += 1

            # --- draw ---
            if phase == Phase.SHOW:
                view = decorate(frame, current)
                view = draw_hud(
                    view,
                    [
                        f"SHOW one target  conf≥{args.detect_conf:.2f}  ROI  saved={idx}  taken={taken_count}",
                        "Remove THIS stick, then press n / →",
                        "Esc / q = quit",
                    ],
                )
            elif phase == Phase.REMOVE:
                remaining = max(0.0, phase_deadline - now)
                view = draw_hud(
                    decorate(frame),
                    [
                        f"REMOVE current stick — {remaining:0.1f}s",
                        "Then next best target will appear",
                        "Esc / q = quit",
                    ],
                )
            elif phase == Phase.EMPTY:
                view = draw_hud(
                    decorate(frame),
                    [
                        f"NO TARGET in ROI (conf≥{args.detect_conf:.2f})  —  press Space / s to SAVE",
                        "r = re-detect without saving",
                        f"saved so far: {idx}   Esc / q = quit",
                    ],
                )
                cv2.rectangle(view, (0, view.shape[0] - 50), (view.shape[1], view.shape[0]), (0, 80, 0), -1)
                cv2.putText(
                    view,
                    "EMPTY — Space/s SAVE clean frame",
                    (20, view.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )
            else:  # ARRANGE
                remaining = max(0.0, phase_deadline - now)
                view = draw_hud(
                    decorate(frame),
                    [
                        f"ARRANGE new scene — {remaining:0.1f}s",
                        "Put crab sticks back, then auto-detect",
                        "Esc / q = quit",
                    ],
                )

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("Quit.")
                break

            # n / Enter / right-arrow = taken away → remove countdown
            if phase == Phase.SHOW and key in (ord("n"), ord("N"), ord("\r"), ord("\n"), 83, 3):
                taken_count += 1
                phase = Phase.REMOVE
                phase_deadline = time.perf_counter() + args.reset_time_s
                current = None
                print(f"Marked taken #{taken_count} — remove it within {args.reset_time_s:.0f}s")

            if phase == Phase.EMPTY and key in (ord(" "), ord("s"), ord("S")):
                path = save_frame(images_dir, idx, frame, args.jpeg_quality)
                print(f"Saved {path}")
                idx += 1
                taken_count = 0
                phase = Phase.ARRANGE
                phase_deadline = time.perf_counter() + args.reset_time_s
                current = None
                print(f"Arrange a new scene within {args.reset_time_s:.0f}s")

            if phase == Phase.EMPTY and key in (ord("r"), ord("R")):
                dets = detect(frame)
                current = select_best(dets)
                if current is None:
                    print("Still no target in ROI.")
                else:
                    phase = Phase.SHOW
                    print(f"Re-detected target conf={current.conf:.2f}")

            # Ignore Space during SHOW — only save when empty (as requested)
            if phase == Phase.SHOW and key in (ord(" "), ord("s"), ord("S")):
                print("还有目标：先按 n 拿走当前框；没有目标后才会提示保存。")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Done. Images in {images_dir} (next index {idx}).")


if __name__ == "__main__":
    main()
