# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command and reward helpers for the Class Humanoid squat task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class UniformPelvisHeightCommandCfg(CommandTermCfg):
    class_type: type = None
    resampling_time_range: tuple[float, float] = (1.5, 3.0)
    min_height: float = 0.62
    max_height: float = 0.75
    asset_name: str = "robot"

    def __post_init__(self):
        self.class_type = UniformPelvisHeightCommand


class UniformPelvisHeightCommand(CommandTerm):
    cfg: UniformPelvisHeightCommandCfg

    def __init__(self, cfg: UniformPelvisHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.asset_name]
        self._command = torch.full((self.num_envs, 1), (cfg.min_height + cfg.max_height) * 0.5, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["height_error"] = torch.abs(self.asset.data.root_pos_w[:, 2] - self._command[:, 0])

    def _resample_command(self, env_ids: Sequence[int]):
        self._command[env_ids, 0] = self._command[env_ids, 0].uniform_(self.cfg.min_height, self.cfg.max_height)

    def _update_command(self):
        pass


def pelvis_height_error_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    target_h = env.command_manager.get_command(command_name)[:, 0]
    return torch.square(asset.data.root_pos_w[:, 2] - target_h)
