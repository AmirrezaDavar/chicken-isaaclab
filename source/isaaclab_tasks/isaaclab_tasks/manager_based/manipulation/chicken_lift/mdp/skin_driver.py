# SPDX-License-Identifier: BSD-3-Clause

"""Startup hook for the rendered chicken skin driver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_DRIVER_MODULE_NAME = "_isaac_lab_chicken_skin_driver"


def _load_driver(driver_path: Path):
    if _DRIVER_MODULE_NAME in sys.modules:
        return sys.modules[_DRIVER_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_DRIVER_MODULE_NAME, driver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load chicken skin driver from {driver_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_DRIVER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def start_chicken_skin_driver(env: ManagerBasedRLEnv, env_ids, max_envs: int = 32) -> None:
    """Start the USD skin-following driver for small rendered chicken runs.

    The hidden rigid bodies are the source of physical truth. This hook only
    updates the visible UsdSkel mesh so rendered/teleop sessions keep the
    continuous no-gap skin. Large RL batches skip it to avoid per-frame USD
    animation edits across thousands of cloned environments.
    """

    if max_envs is not None and env.num_envs > max_envs:
        print(
            f"Chicken skin physics driver skipped for {env.num_envs} envs "
            f"(max_envs={max_envs}). Physics still runs on the hidden bodies."
        )
        return

    try:
        from isaaclab_assets.robots.chicken import CHICKEN_SKIN_DRIVER_PY

        driver_path = Path(CHICKEN_SKIN_DRIVER_PY)
        if not driver_path.exists():
            raise FileNotFoundError(driver_path)
        driver = _load_driver(driver_path)
        driver.start()
    except Exception as exc:
        print(f"Warning: could not start chicken skin physics driver: {exc}")
