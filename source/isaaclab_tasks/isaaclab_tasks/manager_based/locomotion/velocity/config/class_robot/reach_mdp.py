# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command and reward helpers for Class Humanoid reach-depth tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _estimate_target_pos_from_depth(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    data_type: str,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    """Estimate a coarse target position in base frame from the nearest depth pixel."""
    sensor = env.scene.sensors[sensor_name]
    depth = sensor.data.output[data_type].squeeze(-1)
    num_envs, height, width = depth.shape
    device = depth.device

    valid = torch.isfinite(depth) & (depth > min_depth)
    large_val = torch.full_like(depth, max_depth + 10.0)
    masked_depth = torch.where(valid, depth, large_val)

    flat = masked_depth.reshape(num_envs, -1)
    flat_idx = torch.argmin(flat, dim=1)
    min_depth_vals = flat[torch.arange(num_envs, device=device), flat_idx]
    min_depth_vals = torch.clamp(min_depth_vals, min=min_depth, max=max_depth)

    u = (flat_idx % width).float()
    v = (flat_idx // width).float()
    u_norm = (u / max(float(width - 1), 1.0)) - 0.5
    v_norm = (v / max(float(height - 1), 1.0)) - 0.5

    x = min_depth_vals
    y = -u_norm * min_depth_vals
    z = -v_norm * min_depth_vals

    estimate = torch.stack((x, y, z), dim=1)
    estimate[~torch.isfinite(estimate)] = 0.0
    return estimate


@configclass
class DepthTargetPosCommandCfg(CommandTermCfg):
    class_type: type = None
    resampling_time_range: tuple[float, float] = (0.5, 0.5)
    sensor_name: str = "depth_camera"
    data_type: str = "distance_to_image_plane"
    min_depth: float = 0.20
    max_depth: float = 2.00
    smooth_factor: float = 0.8

    def __post_init__(self):
        self.class_type = DepthTargetPosCommand


class DepthTargetPosCommand(CommandTerm):
    cfg: DepthTargetPosCommandCfg

    def __init__(self, cfg: DepthTargetPosCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self.metrics["depth_target_norm"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["depth_target_norm"] = torch.norm(self._command, dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        self._update_command()

    def _update_command(self):
        est = _estimate_target_pos_from_depth(
            self._env, self.cfg.sensor_name, self.cfg.data_type, self.cfg.min_depth, self.cfg.max_depth
        )
        alpha = torch.clamp(torch.tensor(self.cfg.smooth_factor, device=self.device), 0.0, 1.0)
        self._command = alpha * self._command + (1.0 - alpha) * est


def end_effector_object_distance(
    env: ManagerBasedRLEnv,
    target_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Wrist_Right_1"]),
) -> torch.Tensor:
    """Distance between the configured end effector and target object center."""
    robot = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_name]
    ee_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    target_pos = target.data.root_pos_w[:, :3]
    return torch.norm(ee_pos - target_pos, dim=1)


def end_effector_object_reward_tanh(
    env: ManagerBasedRLEnv,
    target_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Wrist_Right_1"]),
) -> torch.Tensor:
    dist = end_effector_object_distance(env, target_name, asset_cfg)
    return 1.0 - torch.tanh(dist / std)


def target_pos_base(
    env: ManagerBasedRLEnv,
    target_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ground-truth target position in robot base frame."""
    robot: RigidObject = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_name]
    rel_pos_w = target.data.root_pos_w[:, :3] - robot.data.root_pos_w
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)


def target_pos_base_from_depth(
    env: ManagerBasedRLEnv,
    sensor_name: str = "depth_camera",
    data_type: str = "distance_to_image_plane",
    min_depth: float = 0.20,
    max_depth: float = 2.00,
) -> torch.Tensor:
    """Depth-only target estimate in base frame."""
    return _estimate_target_pos_from_depth(env, sensor_name, data_type, min_depth, max_depth)
