#!/usr/bin/env python3
"""Inspect a diffusion-policy replay_buffer.zarr collected from Isaac Lab.

Example:
  python scripts/imitation_learning/inspect_dp_zarr.py \
      --zarr_path ./data/replay_buffer.zarr
"""

import argparse
from pathlib import Path

import numpy as np
import zarr


parser = argparse.ArgumentParser(description="Inspect chicken imitation-learning zarr datasets.")
parser.add_argument("--zarr_path", type=str, default="./data/replay_buffer.zarr")
parser.add_argument("--image_key", type=str, default="camera_rgb")
parser.add_argument("--sample_episode", type=int, default=0)
args = parser.parse_args()


def episode_bounds(episode_ends: np.ndarray, episode: int) -> tuple[int, int]:
    if episode < 0 or episode >= len(episode_ends):
        raise SystemExit(f"Episode {episode} does not exist. Dataset has {len(episode_ends)} episodes.")
    start = 0 if episode == 0 else int(episode_ends[episode - 1])
    end = int(episode_ends[episode])
    return start, end


def main() -> None:
    zarr_path = Path(args.zarr_path)
    if not zarr_path.exists():
        raise SystemExit(f"Missing zarr dataset: {zarr_path}")

    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]
    meta = root["meta"]

    if "episode_ends" not in meta:
        raise SystemExit("Dataset is missing meta/episode_ends.")

    episode_ends = meta["episode_ends"][:]
    total_steps = int(episode_ends[-1]) if len(episode_ends) else 0
    print(f"Dataset: {zarr_path}")
    print(f"Episodes: {len(episode_ends)}")
    print(f"Steps:    {total_steps}")
    print()

    print("Arrays:")
    for key in sorted(data.keys()):
        arr = data[key]
        length_ok = arr.shape[0] == total_steps
        status = "OK" if length_ok else f"LENGTH MISMATCH expected {total_steps}"
        print(f"  data/{key:<20} shape={arr.shape!s:<24} dtype={arr.dtype} chunks={arr.chunks}  {status}")
    print()

    required = {"action": 8, "state": 20}
    for key, dim in required.items():
        if key not in data:
            raise SystemExit(f"Missing required data/{key} array.")
        if data[key].shape != (total_steps, dim):
            raise SystemExit(f"data/{key} should have shape ({total_steps}, {dim}), got {data[key].shape}.")

    if args.image_key in data:
        image_shape = data[args.image_key].shape[1:]
        if len(image_shape) != 3 or image_shape[-1] != 3:
            raise SystemExit(f"data/{args.image_key} should be HWC RGB images, got per-frame shape {image_shape}.")
        print("Shape meta for ChicGrasp / diffusion-policy:")
        print("shape_meta:")
        print("  obs:")
        print(f"    {args.image_key}:")
        print(f"      shape: [3, {image_shape[0]}, {image_shape[1]}]")
        print("      type: rgb")
        print("    state:")
        print("      shape: [20]")
        print("      type: low_dim")
        print("  action:")
        print("    shape: [8]")
    else:
        print(f"No data/{args.image_key} array found. This dataset is low-dimensional only.")

    if len(episode_ends):
        start, end = episode_bounds(episode_ends, args.sample_episode)
        print()
        print(f"Episode {args.sample_episode}: steps [{start}, {end}) length={end - start}")
        if "stage" in data:
            lifted = bool(np.max(data["stage"][start:end, 0]) > 0.5)
            print(f"Lifted stage reached: {lifted}")


if __name__ == "__main__":
    main()
