#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Export RGB review videos from a collected chicken replay_buffer.zarr.

Use this after teleoperation collection instead of encoding MP4s during the
real-time GELLO loop.

Example:
  python scripts/imitation_learning/export_chicken_zarr_videos.py \
      --zarr_path ./data/chicken_rgb_state/replay_buffer.zarr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

try:
    import cv2
except ImportError as exc:
    raise SystemExit("opencv-python is required to export videos.") from exc


parser = argparse.ArgumentParser(description="Export per-episode MP4 videos from chicken zarr camera frames.")
parser.add_argument("--zarr_path", type=Path, default=Path("./data/chicken_rgb_state/replay_buffer.zarr"))
parser.add_argument("--image_key", type=str, default="camera_rgb")
parser.add_argument("--out_dir", type=Path, default=None)
parser.add_argument("--fps", type=int, default=30)
args = parser.parse_args()


def episode_bounds(episode_ends: np.ndarray, episode: int) -> tuple[int, int]:
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    end = int(episode_ends[episode])
    return start, end


def main() -> None:
    if not args.zarr_path.exists():
        raise SystemExit(f"Missing zarr dataset: {args.zarr_path}")

    root = zarr.open_group(str(args.zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]

    if args.image_key not in data:
        raise SystemExit(f"Missing data/{args.image_key} in {args.zarr_path}")
    if "episode_ends" not in meta:
        raise SystemExit(f"Missing meta/episode_ends in {args.zarr_path}")

    images = data[args.image_key]
    episode_ends = meta["episode_ends"][:]
    out_dir = args.out_dir or args.zarr_path.parent / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w = images.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for episode in range(len(episode_ends)):
        start, end = episode_bounds(episode_ends, episode)
        path = out_dir / f"episode_{episode:06d}.mp4"
        writer = cv2.VideoWriter(str(path), fourcc, args.fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {path}")
        for idx in range(start, end):
            frame = images[idx]
            if frame.dtype != np.uint8:
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[video] {path}  frames={end - start} fps={args.fps}")


if __name__ == "__main__":
    main()
