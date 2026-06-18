# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def gripper_close_near_object(
    env: ManagerBasedRLEnv,
    approach_std: float,
    gripper_open_val: float,
    gripper_close_val: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["PrismaticJoint.*"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the gripper jaws when the EE is close to the object.

    Returns proximity_to_object * gripper_closed_fraction, so the agent
    is rewarded proportionally for closing around the cube.  This breaks the
    local minimum where the arm hovers near the cube with an open gripper.

    Args:
        approach_std: Tanh kernel width for EE-object proximity (m).
        gripper_open_val:  Joint position when gripper is fully open (e.g. 0.0).
        gripper_close_val: Joint position when gripper is fully closed (e.g. -0.0093).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object_: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # ── proximity of EE to object ──────────────────────────────────────────
    cube_pos_w = object_.data.root_pos_w           # (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]  # (num_envs, 3)
    dist = torch.norm(cube_pos_w - ee_w, dim=1)   # (num_envs,)
    proximity = 1.0 - torch.tanh(dist / approach_std)

    # ── fraction by which each prismatic jaw is closed ─────────────────────
    # joint_pos[:,joint_ids]: 0.0 = open, gripper_close_val (negative) = closed
    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]  # (num_envs, n_jaws)
    closed_frac = torch.clamp(
        (gripper_pos - gripper_open_val) / (gripper_close_val - gripper_open_val),
        0.0, 1.0,
    ).mean(dim=1)  # (num_envs,)

    return proximity * closed_frac
