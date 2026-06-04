# SPDX-License-Identifier: BSD-3-Clause

"""Reward and observation functions for sequential two-jaw chicken grasping.

Behaviour the rewards guide:
  Phase 1 – EE approaches the chicken; left jaw (PrismaticJoint1/2) closes on
             the left leg.
  Phase 2 – Once the left jaw is gripping the left leg (gate signal), the right
             jaw (PrismaticJoint3/4) closes on the right leg.
  Phase 3 – With at least the left leg secured, the arm lifts the chicken.

Sequential gating is soft: the Phase-2/3 reward terms are scaled by
`left_grasped` (a [0,1] float = left-jaw-closure × left-leg-proximity), so
gradients are always non-zero and PPO can learn the whole sequence end-to-end.

Body / joint indices are resolved once on the first call and cached by object
identity, so there is no per-step overhead from find_bodies / find_joints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Internal caches (resolved once per articulation instance)
# ---------------------------------------------------------------------------

_BODY_CACHE: dict[tuple, int] = {}
_JOINT_CACHE: dict[tuple, list[int]] = {}


def _body_idx(art: Articulation, name: str) -> int:
    key = (id(art), name)
    if key not in _BODY_CACHE:
        ids, _ = art.find_bodies([name])
        _BODY_CACHE[key] = ids[0]
    return _BODY_CACHE[key]


def _joint_ids(art: Articulation, names: tuple[str, ...]) -> list[int]:
    key = (id(art), names)
    if key not in _JOINT_CACHE:
        ids, _ = art.find_joints(list(names))
        _JOINT_CACHE[key] = ids
    return _JOINT_CACHE[key]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Full prismatic travel from open (0) to closed (−9.3 mm)
_PRISM_TRAVEL: float = 0.0093


def _jaw_closure(robot: Articulation, joint_names: tuple[str, ...]) -> torch.Tensor:
    """Returns a [0, 1] closure ratio for the given prismatic jaw joints.
    0 = fully open, 1 = fully closed."""
    ids = _joint_ids(robot, joint_names)
    # joint_pos for prismatic joints: 0 = open, -_PRISM_TRAVEL = closed
    return (-robot.data.joint_pos[:, ids].mean(dim=1) / _PRISM_TRAVEL).clamp(0.0, 1.0)


_LEFT_JAW = ("PrismaticJoint1", "PrismaticJoint2")
_RIGHT_JAW = ("PrismaticJoint3", "PrismaticJoint4")


# ---------------------------------------------------------------------------
# Phase 1 – left jaw on left leg
# ---------------------------------------------------------------------------


def left_jaw_closing_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Continuous reward [0, 1] for how closed the left jaw is.
    Encourages the policy to close the left jaw rather than leaving it open."""
    robot: Articulation = env.scene["robot"]
    return _jaw_closure(robot, _LEFT_JAW)


def left_leg_grasped_reward(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward = proximity-to-left-leg × left-jaw-closure.

    Peaks at 1.0 when the EE is directly on the left leg and the jaw is fully
    closed.  This is the primary Phase-1 signal.
    """
    chicken: Articulation = env.scene["chicken"]
    robot: Articulation = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    left_leg_pos = chicken.data.body_pos_w[:, _body_idx(chicken, "left_leg"), :3]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    dist = torch.norm(left_leg_pos - ee_pos, dim=1)
    proximity = 1.0 - torch.tanh(dist / std)
    closure = _jaw_closure(robot, _LEFT_JAW)

    return proximity * closure


# ---------------------------------------------------------------------------
# Phase 2 – right jaw on right leg, gated on Phase 1
# ---------------------------------------------------------------------------


def right_jaw_gated_reward(
    env: ManagerBasedRLEnv,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for right jaw closing on the right leg, scaled by `left_grasped`.

    `left_grasped` = proximity-to-left-leg × left-jaw-closure  (soft [0,1] gate)

    This means Phase-2 rewards are only substantial once the left jaw is
    already gripping.  The policy learns the correct temporal ordering without
    any explicit finite-state machine.
    """
    chicken: Articulation = env.scene["chicken"]
    robot: Articulation = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]

    # ---- left-grasp gate ----
    left_leg_pos = chicken.data.body_pos_w[:, _body_idx(chicken, "left_leg"), :3]
    left_dist = torch.norm(left_leg_pos - ee_pos, dim=1)
    left_proximity = 1.0 - torch.tanh(left_dist / std)
    left_closure = _jaw_closure(robot, _LEFT_JAW)
    left_grasped = (left_proximity * left_closure)  # soft [0, 1]

    # ---- right jaw signal ----
    right_leg_pos = chicken.data.body_pos_w[:, _body_idx(chicken, "right_leg"), :3]
    right_dist = torch.norm(right_leg_pos - ee_pos, dim=1)
    right_proximity = 1.0 - torch.tanh(right_dist / std)
    right_closure = _jaw_closure(robot, _RIGHT_JAW)

    return left_grasped * right_proximity * right_closure


# ---------------------------------------------------------------------------
# Phase 3 – lift, gated on Phase 1 being established
# ---------------------------------------------------------------------------


def chicken_lifted_gated(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    gate_threshold: float = 0.3,
) -> torch.Tensor:
    """Sparse lift reward, enabled only once the left jaw is at least
    `gate_threshold` closed.

    Keeps the policy from learning to lift without grasping first.
    """
    chicken: Articulation = env.scene["chicken"]
    robot: Articulation = env.scene["robot"]

    left_closure = _jaw_closure(robot, _LEFT_JAW)
    # Hard gate so the lift signal is clean (no gradient through gate)
    gate = (left_closure >= gate_threshold).float()

    lifted = (chicken.data.root_pos_w[:, 2] > minimal_height).float()
    return gate * lifted


def chicken_goal_tracking_gated(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    gate_threshold: float = 0.3,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Goal-tracking reward (carry to target pose), enabled once left jaw is closed.

    Ensures the policy first learns to grasp before trying to carry.
    """
    from isaaclab.utils.math import combine_frame_transforms

    chicken: Articulation = env.scene["chicken"]
    robot: Articulation = env.scene[robot_cfg.name]

    left_closure = _jaw_closure(robot, _LEFT_JAW)
    gate = (left_closure >= gate_threshold).float()

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - chicken.data.root_pos_w, dim=1)

    in_range = (chicken.data.root_pos_w[:, 2] > minimal_height).float()
    return gate * in_range * (1.0 - torch.tanh(distance / std))


# ---------------------------------------------------------------------------
# Observation helper
# ---------------------------------------------------------------------------


def chicken_legs_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns left-leg and right-leg positions in the robot root frame.

    Output shape: (num_envs, 6)  — [left_x, left_y, left_z, right_x, right_y, right_z]

    Gives the policy explicit leg-position targets so it can learn to position
    each jaw over the correct leg.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    chicken: Articulation = env.scene["chicken"]

    left_idx = _body_idx(chicken, "left_leg")
    right_idx = _body_idx(chicken, "right_leg")

    left_pos_w = chicken.data.body_pos_w[:, left_idx, :3]
    right_pos_w = chicken.data.body_pos_w[:, right_idx, :3]

    left_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, left_pos_w)
    right_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, right_pos_w)

    return torch.cat([left_pos_b, right_pos_b], dim=-1)


def chicken_orientation(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Chicken root orientation as a quaternion (w, x, y, z) expressed in the
    robot root frame.  Shape: (num_envs, 4).

    Why this matters: the chicken can land on the table rotated arbitrarily.
    Without knowing its orientation, the policy cannot tell which side the left
    leg vs the right leg is on — the XYZ leg positions alone are ambiguous when
    the chicken is upside-down or sideways.
    """
    from isaaclab.utils.math import quat_mul, quat_inv

    robot: Articulation = env.scene[robot_cfg.name]
    chicken: Articulation = env.scene["chicken"]

    # q_rel = q_robot^{-1} ⊗ q_chicken  →  chicken orientation in robot frame
    q_rel = quat_mul(quat_inv(robot.data.root_quat_w), chicken.data.root_quat_w)
    return q_rel  # (N, 4)


def chicken_root_velocity(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Chicken root linear velocity in world frame.  Shape: (num_envs, 3).

    Why this matters: a successfully grasped chicken moves with the arm — its
    velocity becomes non-zero as the robot lifts.  A failed grasp (fingers
    slipped) leaves the chicken stationary.  The policy can use this signal
    to confirm whether its current grasp is holding before it tries to lift.
    """
    chicken: Articulation = env.scene["chicken"]
    return chicken.data.root_lin_vel_w  # (N, 3)
