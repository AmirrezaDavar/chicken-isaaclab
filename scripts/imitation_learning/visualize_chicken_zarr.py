#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Visualize low-dimensional and RGB data stored in a chicken replay_buffer.zarr.

Outputs:
  plots/dataset_overview.png
  plots/episode_000000_lowdim.png
  plots/episode_000000_camera_sheet.png

Example:
  python scripts/imitation_learning/visualize_chicken_zarr.py \
      --zarr_path ./data/chicken_rgb_state/replay_buffer.zarr \
      --episode 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr


ARM_LABELS = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
ACTION_LABELS = ["a_pan", "a_lift", "a_elbow", "a_w1", "a_w2", "a_w3", "L_jaw", "R_jaw"]


parser = argparse.ArgumentParser(description="Plot chicken zarr low-dim curves and RGB camera frames.")
parser.add_argument("--zarr_path", type=Path, default=Path("./data/chicken_rgb_state/replay_buffer.zarr"))
parser.add_argument("--episode", type=int, default=0)
parser.add_argument("--out_dir", type=Path, default=None)
parser.add_argument("--image_key", type=str, default="camera_rgb")
parser.add_argument("--num_frames", type=int, default=12, help="Frames to show in the camera contact sheet.")
parser.add_argument("--show", action="store_true", help="Also open interactive plot windows.")
args = parser.parse_args()


def require_key(group, key: str) -> None:
    if key not in group:
        raise SystemExit(f"Missing data/{key} in {args.zarr_path}")


def episode_bounds(episode_ends: np.ndarray, episode: int) -> tuple[int, int]:
    if episode < 0 or episode >= len(episode_ends):
        raise SystemExit(f"Episode {episode} does not exist. Dataset has {len(episode_ends)} episodes.")
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    end = int(episode_ends[episode])
    return start, end


def episode_time(data, start: int, end: int) -> np.ndarray:
    if "timestamp" in data:
        ts = data["timestamp"][start:end, 0]
        if len(ts) and ts[-1] > 0:
            return ts
    return np.arange(end - start, dtype=np.float32) * 0.02


def save_or_show(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] {path}")
    if args.show:
        fig.show()
    plt.close(fig)


def plot_overview(data, episode_ends: np.ndarray, out_dir: Path) -> None:
    starts = np.concatenate([[0], episode_ends[:-1]])
    lengths = episode_ends - starts
    ep_ids = np.arange(len(episode_ends))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Chicken dataset overview: {len(episode_ends)} episodes, {int(episode_ends[-1])} steps")

    axes[0, 0].bar(ep_ids, lengths, color="tab:blue")
    axes[0, 0].set_title("Episode lengths")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].set_ylabel("steps")
    axes[0, 0].grid(True, axis="y", alpha=0.3)

    if "stage" in data:
        lifted_frac = []
        for ep in ep_ids:
            start, end = episode_bounds(episode_ends, int(ep))
            lifted_frac.append(float(np.mean(data["stage"][start:end, 0])))
        axes[0, 1].bar(ep_ids, lifted_frac, color="tab:green")
        axes[0, 1].set_title("Lifted-stage fraction")
        axes[0, 1].set_ylim(0, 1.05)
    else:
        axes[0, 1].axis("off")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].grid(True, axis="y", alpha=0.3)

    if "robot_joint" in data:
        joints = data["robot_joint"][:]
        for i, label in enumerate(ARM_LABELS):
            axes[1, 0].hist(np.rad2deg(joints[:, i]), bins=50, alpha=0.55, label=label)
        axes[1, 0].set_title("Joint position distribution")
        axes[1, 0].set_xlabel("degrees")
        axes[1, 0].legend(ncol=3, fontsize=8)
    else:
        axes[1, 0].axis("off")
    axes[1, 0].grid(True, axis="y", alpha=0.3)

    require_key(data, "action")
    action = data["action"][:]
    axes[1, 1].plot(action[:, 6], label="left jaw cmd")
    axes[1, 1].plot(action[:, 7], label="right jaw cmd")
    axes[1, 1].set_title("Jaw action commands across dataset")
    axes[1, 1].set_xlabel("global step")
    axes[1, 1].set_ylim(-1.2, 1.2)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    save_or_show(fig, out_dir / "dataset_overview.png")


def plot_episode_lowdim(data, episode_ends: np.ndarray, episode: int, out_dir: Path) -> None:
    start, end = episode_bounds(episode_ends, episode)
    t = episode_time(data, start, end)

    require_key(data, "action")
    require_key(data, "state")
    action = data["action"][start:end]

    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    fig.suptitle(f"Episode {episode:06d}: low-dimensional signals ({end - start} steps)")

    if "robot_joint" in data:
        joints = data["robot_joint"][start:end]
    else:
        joints = data["state"][start:end, 0:6]
    for i, label in enumerate(ARM_LABELS):
        axes[0, 0].plot(t, np.rad2deg(joints[:, i]), label=label)
    axes[0, 0].set_title("Robot joint positions")
    axes[0, 0].set_ylabel("deg")
    axes[0, 0].legend(ncol=3, fontsize=8)

    if "robot_joint_vel" in data:
        joint_vel = data["robot_joint_vel"][start:end]
    else:
        joint_vel = data["state"][start:end, 6:12]
    for i, label in enumerate(ARM_LABELS):
        axes[0, 1].plot(t, np.rad2deg(joint_vel[:, i]), label=label)
    axes[0, 1].set_title("Robot joint velocities")
    axes[0, 1].set_ylabel("deg/s")
    axes[0, 1].legend(ncol=3, fontsize=8)

    if "robot_eef_pose" in data:
        eef = data["robot_eef_pose"][start:end]
    else:
        eef = data["state"][start:end, 12:18]
    for i, label in enumerate(["x", "y", "z"]):
        axes[1, 0].plot(t, eef[:, i], label=label)
    axes[1, 0].set_title("End-effector position")
    axes[1, 0].set_ylabel("m")
    axes[1, 0].legend(fontsize=8)

    for i, label in enumerate(["roll", "pitch", "yaw"]):
        axes[1, 1].plot(t, np.rad2deg(eef[:, 3 + i]), label=label)
    axes[1, 1].set_title("End-effector orientation")
    axes[1, 1].set_ylabel("deg")
    axes[1, 1].legend(fontsize=8)

    for i, label in enumerate(ARM_LABELS):
        axes[2, 0].plot(t, np.rad2deg(action[:, i]), label=label)
    axes[2, 0].set_title("Action: arm joint targets")
    axes[2, 0].set_xlabel("time (s)")
    axes[2, 0].set_ylabel("deg")
    axes[2, 0].legend(ncol=3, fontsize=8)

    axes[2, 1].step(t, action[:, 6], where="post", label=ACTION_LABELS[6])
    axes[2, 1].step(t, action[:, 7], where="post", label=ACTION_LABELS[7])
    if "left_jaw" in data and "right_jaw" in data:
        axes[2, 1].plot(t, data["left_jaw"][start:end, 0], alpha=0.7, label="left jaw state")
        axes[2, 1].plot(t, data["right_jaw"][start:end, 0], alpha=0.7, label="right jaw state")
    axes[2, 1].set_title("Gripper commands and states")
    axes[2, 1].set_xlabel("time (s)")
    axes[2, 1].set_ylim(-1.2, 1.2)
    axes[2, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_or_show(fig, out_dir / f"episode_{episode:06d}_lowdim.png")


def plot_camera_sheet(data, episode_ends: np.ndarray, episode: int, out_dir: Path) -> None:
    if args.image_key not in data:
        print(f"[plot] data/{args.image_key} not found; skipping camera sheet.")
        return

    start, end = episode_bounds(episode_ends, episode)
    t = episode_time(data, start, end)
    n = max(1, min(args.num_frames, end - start))
    frame_idxs = np.linspace(start, end - 1, n, dtype=int)

    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"Episode {episode:06d}: {args.image_key} samples")

    for ax, idx in zip(axes, frame_idxs):
        frame = data[args.image_key][idx]
        if frame.dtype != np.uint8:
            frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
        ax.imshow(frame)
        ax.set_title(f"step {idx - start}, t={t[idx - start]:.2f}s")
        ax.axis("off")
    for ax in axes[len(frame_idxs):]:
        ax.axis("off")

    fig.tight_layout()
    save_or_show(fig, out_dir / f"episode_{episode:06d}_camera_sheet.png")


def main() -> None:
    if not args.zarr_path.exists():
        raise SystemExit(f"Missing zarr dataset: {args.zarr_path}")

    root = zarr.open_group(str(args.zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]
    if "episode_ends" not in meta:
        raise SystemExit(f"Missing meta/episode_ends in {args.zarr_path}")

    episode_ends = meta["episode_ends"][:]
    if len(episode_ends) == 0:
        raise SystemExit("Dataset has no episodes.")

    out_dir = args.out_dir or args.zarr_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.zarr_path}")
    print(f"Episodes: {len(episode_ends)}")
    print(f"Arrays: {sorted(data.keys())}")

    plot_overview(data, episode_ends, out_dir)
    plot_episode_lowdim(data, episode_ends, args.episode, out_dir)
    plot_camera_sheet(data, episode_ends, args.episode, out_dir)


if __name__ == "__main__":
    main()
