# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared MDP helpers for Class Humanoid tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_rpy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base orientation in roll/pitch/yaw."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack((roll, pitch, yaw), dim=1)


def binary_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Binary contact indicators for the configured bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    contacts = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return contacts.float()


def illegal_contact_below_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    max_height: float,
) -> torch.Tensor:
    """Terminate when a monitored body has significant contact while near the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    contact_mask = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    near_ground_mask = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] < max_height
    return torch.any(contact_mask & near_ground_mask, dim=1)


def torso_tilt_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Equivalent to flat-orientation cost, kept explicit for task configs."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize horizontal base drift for in-place behaviors."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


def base_ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize yaw spinning for in-place behaviors."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_w[:, 2])


def semantic_signed_joint_pos(
    env: ManagerBasedRLEnv,
    joint_signs: Sequence[float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return joint positions multiplied by task-level semantic signs."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_signs_tensor = torch.as_tensor(joint_signs, device=env.device, dtype=joint_pos.dtype)
    if joint_signs_tensor.numel() != len(asset_cfg.joint_ids):
        raise ValueError(
            f"Expected {len(asset_cfg.joint_ids)} semantic joint signs for {asset_cfg.joint_names}, "
            f"received {joint_signs_tensor.numel()}."
        )
    return joint_pos * joint_signs_tensor.unsqueeze(0)


def resolve_articulation_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[int]:
    """Resolve joint ids from a scene entity config into a concrete Python list."""
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return list(range(asset.num_joints))
    if isinstance(joint_ids, slice):
        return list(range(asset.num_joints))[joint_ids]
    if isinstance(joint_ids, torch.Tensor):
        return joint_ids.tolist()
    return list(joint_ids)


def class_humanoid_joint_semantic_signs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return semantic sign multipliers for Class Humanoid joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_articulation_joint_ids(asset, asset_cfg)
    joint_names = [asset.data.joint_names[i] for i in joint_ids]
    joint_signs = [-1.0 if name.startswith("Right_") else 1.0 for name in joint_names]
    return torch.tensor(joint_signs, device=env.device, dtype=asset.data.joint_pos.dtype)


def class_humanoid_joint_pos_rel_semantic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions relative to default, converted into Class Humanoid semantic signs."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_articulation_joint_ids(asset, asset_cfg)
    joint_signs = class_humanoid_joint_semantic_signs(env, asset_cfg)
    joint_pos_rel = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return joint_pos_rel * joint_signs.unsqueeze(0)


def class_humanoid_joint_vel_rel_semantic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocities relative to default, converted into Class Humanoid semantic signs."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_articulation_joint_ids(asset, asset_cfg)
    joint_signs = class_humanoid_joint_semantic_signs(env, asset_cfg)
    joint_vel_rel = asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]
    return joint_vel_rel * joint_signs.unsqueeze(0)


def class_humanoid_knee_flexion(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
) -> torch.Tensor:
    """Return knee flexion where positive always means bending the knee."""
    return semantic_signed_joint_pos(env, joint_signs=(1.0, -1.0), asset_cfg=asset_cfg)


def class_humanoid_hip_pitch_flexion(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch_RS04", "Right_Hip_Pitch_RS04"]),
) -> torch.Tensor:
    """Return hip-pitch flexion where positive always means lifting the thigh forward."""
    return semantic_signed_joint_pos(env, joint_signs=(1.0, -1.0), asset_cfg=asset_cfg)
