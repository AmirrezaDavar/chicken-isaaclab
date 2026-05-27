# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""MDP helpers for the shoot-ball-into-target task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_pos_base(
    env: ManagerBasedRLEnv,
    ball_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball position expressed in the robot base frame (3-D)."""
    robot: RigidObject = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_name]
    rel_pos_w = ball.data.root_pos_w[:, :3] - robot.data.root_pos_w
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)


def ball_to_goal_distance(
    env: ManagerBasedRLEnv,
    ball_name: str,
    goal_name: str = "goal_marker",
) -> torch.Tensor:
    """Horizontal distance from ball to the goal marker (per-env, correct for all envs)."""
    ball: RigidObject = env.scene[ball_name]
    goal: RigidObject = env.scene[goal_name]
    # Use XY only — goal is on the floor, ball floats above it
    ball_xy = ball.data.root_pos_w[:, :2]
    goal_xy = goal.data.root_pos_w[:, :2]
    return torch.norm(ball_xy - goal_xy, dim=1)


def ball_velocity_toward_goal(
    env: ManagerBasedRLEnv,
    ball_name: str,
    goal_name: str = "goal_marker",
) -> torch.Tensor:
    """Component of ball velocity pointing toward the goal (positive = moving toward goal)."""
    ball: RigidObject = env.scene[ball_name]
    goal: RigidObject = env.scene[goal_name]
    ball_xy = ball.data.root_pos_w[:, :2]
    goal_xy = goal.data.root_pos_w[:, :2]
    direction = goal_xy - ball_xy
    dist = torch.norm(direction, dim=1, keepdim=True).clamp(min=1e-6)
    unit_dir = direction / dist
    ball_vel_xy = ball.data.root_lin_vel_w[:, :2]
    return (ball_vel_xy * unit_dir).sum(dim=1)


def ball_at_goal_bonus(
    env: ManagerBasedRLEnv,
    ball_name: str,
    goal_name: str = "goal_marker",
    threshold: float = 0.35,
) -> torch.Tensor:
    """Binary bonus when the ball reaches the goal area."""
    return (ball_to_goal_distance(env, ball_name, goal_name) < threshold).float()


def goal_pos_base(
    env: ManagerBasedRLEnv,
    goal_name: str = "goal_marker",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Goal marker position expressed in the robot base frame (3-D)."""
    robot: RigidObject = env.scene[asset_cfg.name]
    goal: RigidObject = env.scene[goal_name]
    rel_pos_w = goal.data.root_pos_w[:, :3] - robot.data.root_pos_w
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)


def arm_to_ball_distance(
    env: ManagerBasedRLEnv,
    ball_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Wrist_Right_1"]),
) -> torch.Tensor:
    """Distance from right wrist to ball centre."""
    robot: RigidObject = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_name]
    ee_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.norm(ee_pos - ball.data.root_pos_w[:, :3], dim=1)
