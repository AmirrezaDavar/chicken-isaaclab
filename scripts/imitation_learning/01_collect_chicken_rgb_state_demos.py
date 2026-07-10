#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Collect chicken imitation demos with both low-dimensional state and RGB camera frames.

This is a small, opinionated wrapper around collect_isaac_demos.py. It keeps
camera storage enabled so the resulting zarr can train high-dimensional
Diffusion Policy models.

Example:
  cd /home/wanglab22/3_chicken-isaaclab
  ./env_isaacsim/bin/python scripts/imitation_learning/01_collect_chicken_rgb_state_demos.py \
      --out_dir ./data/chicken_rgb_state \
      --num_demos 50 \
      --episode_steps 300 \
      --preview_stride 5
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _has_arg(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def main() -> None:
    if _has_arg("--no_zarr_images"):
        raise SystemExit(
            "RGB+state collection needs camera images in zarr. "
            "Remove --no_zarr_images."
        )

    script = Path(__file__).with_name("collect_isaac_demos.py")

    # Good defaults for high-dimensional DP. User-provided args still win.
    if not _has_arg("--out_dir"):
        sys.argv.extend(["--out_dir", "./data/chicken_rgb_state"])
    if not _has_arg("--image_key"):
        sys.argv.extend(["--image_key", "camera_rgb"])
    if not _has_arg("--preview_stride"):
        sys.argv.extend(["--preview_stride", "5"])

    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
