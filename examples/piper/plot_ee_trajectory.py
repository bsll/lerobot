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

"""Plot Piper end-effector XYZ trajectories for one LeRobot episode.

Takes ``observation.state`` and ``action`` joint positions (motor-normalized),
runs Piper SDK forward kinematics, and plots EE XYZ (mm) over time + in 3D.

Usage (from repo root)::

    uv run python examples/piper/plot_ee_trajectory.py \\
      --dataset-root train_data/piper_grasp_demo \\
      --episode 0

    # or via shell wrapper
    EPISODE=0 bash examples/piper/plot_ee_trajectory.sh
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from lerobot.datasets import LeRobotDataset
from lerobot.motors.piper.tables import DEFAULT_CALIBRATION
from lerobot.utils.import_utils import require_package

JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
# piper_sdk: degrees -> radians divisor used in GetFK path
_DEG_TO_RAD_DIV = 180.0 / math.pi


def _unnormalize_joint(norm: float, motor: str) -> float:
    """Map RANGE_M100_100 motor value back to Piper millidegrees."""
    cal = DEFAULT_CALIBRATION[motor]
    mn, mx = float(cal.range_min), float(cal.range_max)
    bounded = min(100.0, max(-100.0, float(norm)))
    return ((bounded + 100.0) / 200.0) * (mx - mn) + mn


def joints_norm_to_radians(joint_norms: np.ndarray) -> list[float]:
    """Convert 6 motor-normalized joints → radians for ``C_PiperForwardKinematics``."""
    if joint_norms.shape[-1] < 6:
        raise ValueError(f"Need at least 6 joint dims, got shape {joint_norms.shape}")
    milli = [_unnormalize_joint(float(joint_norms[i]), JOINT_NAMES[i]) for i in range(6)]
    return [m / (1000.0 * _DEG_TO_RAD_DIV) for m in milli]


def joints_to_ee_xyz_mm(joint_norms: np.ndarray, fk) -> np.ndarray:
    """(T, >=6) motor-normalized joints → (T, 3) EE XYZ in mm (link-6)."""
    out = np.zeros((len(joint_norms), 3), dtype=np.float64)
    for t, q in enumerate(joint_norms):
        pose = fk.CalFK(joints_norm_to_radians(q))[-1]  # [x,y,z,r,p,y]
        out[t] = pose[:3]
    return out


def _as_numpy(x) -> np.ndarray:
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def load_episode_arrays(
    dataset_root: Path,
    repo_id: str,
    episode: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(timestamps_s, state_joints Tx6, action_joints Tx6, fps)``."""
    ds = LeRobotDataset(repo_id, root=dataset_root, episodes=[episode])
    fps = float(ds.fps)
    states, actions, timestamps = [], [], []
    for i in range(ds.num_frames):
        item = ds[i]
        state = _as_numpy(item["observation.state"]).reshape(-1)
        action = _as_numpy(item["action"]).reshape(-1)
        states.append(state[:6])
        actions.append(action[:6])
        timestamps.append(float(_as_numpy(item["timestamp"]).reshape(-1)[0]))
    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(states, dtype=np.float64),
        np.asarray(actions, dtype=np.float64),
        fps,
    )


def plot_trajectories(
    t: np.ndarray,
    xyz_state: np.ndarray,
    xyz_action: np.ndarray,
    *,
    title: str,
    out_path: Path | None,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(title)

    # XYZ vs time
    labels = ("X", "Y", "Z")
    for i, lab in enumerate(labels):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.plot(t, xyz_state[:, i], label="observation.state", linewidth=1.5)
        ax.plot(t, xyz_action[:, i], label="action", linewidth=1.2, alpha=0.85, linestyle="--")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"{lab} (mm)")
        ax.set_title(f"EE {lab}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    # 3D path
    ax3 = fig.add_subplot(2, 3, (4, 6), projection="3d")
    ax3.plot(
        xyz_state[:, 0],
        xyz_state[:, 1],
        xyz_state[:, 2],
        label="observation.state",
        linewidth=1.5,
    )
    ax3.plot(
        xyz_action[:, 0],
        xyz_action[:, 1],
        xyz_action[:, 2],
        label="action",
        linewidth=1.2,
        alpha=0.85,
        linestyle="--",
    )
    ax3.scatter(*xyz_state[0], c="green", s=40, label="start (state)")
    ax3.scatter(*xyz_state[-1], c="red", s=40, label="end (state)")
    ax3.set_xlabel("X (mm)")
    ax3.set_ylabel("Y (mm)")
    ax3.set_zlabel("Z (mm)")
    ax3.set_title("EE path (base frame)")
    ax3.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved figure → {out_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "train_data" / "piper_grasp_demo",
        help="Local dataset root (contains meta/ + data/).",
    )
    p.add_argument("--repo-id", type=str, default=None, help="Dataset repo_id (default: dataset-root name).")
    p.add_argument("--episode", type=int, default=0, help="Episode index to plot.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (default: outputs/ee_traj/<repo>_ep<N>.png).",
    )
    p.add_argument("--show", action="store_true", help="Open an interactive matplotlib window.")
    p.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save xyz_state.npy / xyz_action.npy next to the PNG.",
    )
    return p.parse_args()


def main() -> None:
    require_package("piper_sdk", extra="piper")
    from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

    args = parse_args()
    root: Path = args.dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    repo_id = args.repo_id or root.name

    print(f"Loading {repo_id} episode {args.episode} from {root}")
    t, state_q, action_q, fps = load_episode_arrays(root, repo_id, args.episode)
    print(f"  frames={len(t)}  fps={fps:.1f}  duration={t[-1] - t[0]:.2f}s")

    fk = C_PiperForwardKinematics(dh_is_offset=0x01)
    xyz_state = joints_to_ee_xyz_mm(state_q, fk)
    xyz_action = joints_to_ee_xyz_mm(action_q, fk)

    delta = np.linalg.norm(xyz_state - xyz_action, axis=1)
    print(
        f"  EE |state-action| mm: mean={delta.mean():.2f}  max={delta.max():.2f}  "
        f"(direct_record demos are often ~0)"
    )
    print(
        f"  state EE Z mm: min={xyz_state[:, 2].min():.1f}  max={xyz_state[:, 2].max():.1f}  "
        f"end={xyz_state[-1, 2]:.1f}"
    )

    out = args.output
    if out is None:
        out = Path("outputs/ee_traj") / f"{repo_id}_ep{args.episode:03d}.png"

    plot_trajectories(
        t,
        xyz_state,
        xyz_action,
        title=f"{repo_id}  episode {args.episode}  (Piper FK, XYZ mm)",
        out_path=out,
        show=args.show,
    )

    if args.save_npy:
        npy_dir = out.parent
        np.save(npy_dir / f"{repo_id}_ep{args.episode:03d}_xyz_state.npy", xyz_state)
        np.save(npy_dir / f"{repo_id}_ep{args.episode:03d}_xyz_action.npy", xyz_action)
        np.save(npy_dir / f"{repo_id}_ep{args.episode:03d}_t.npy", t)
        print(f"Saved npy arrays under {npy_dir.resolve()}")


if __name__ == "__main__":
    main()
