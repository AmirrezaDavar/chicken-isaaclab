# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Class Humanoid task configuration helpers."""

from __future__ import annotations

import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg

from . import common_mdp

from isaaclab_assets import CLASS_HUMANOID_CFG  # isort: skip

BASE_BODY_NAME = "base_link"
FOOT_BODY_NAMES = ["Foot_Left_1", "Foot_Right_1"]
KNEE_JOINT_NAMES = ["Left_Knee_RS04", "Right_Knee_RS04"]
HIP_PITCH_JOINT_NAMES = ["Left_Hip_Pitch_RS04", "Right_Hip_Pitch_RS04"]
HIP_YOKE_BODY_NAMES = ["HipYoke_Left_1", "HipYoke_Right_1"]
RIGHT_ARM_JOINT_NAMES = [
    "Right_Shoulder_Pitch_RS03",
    "Right_Shoulder_Roll_RS03",
    "Right_Shoulder_Yaw_RS02",
    "Right_Elbow_RS02",
    "Right_Wrist_RS00",
]
RIGHT_WRIST_BODY_NAME = "Wrist_Right_1"
NON_FOOT_FALL_CONTACT_BODY_NAMES = [
    "base_link",
    "Head_1",
    "Hip_1",
    "HipYoke_.*",
    "UpperThigh_.*",
    "LowerThigh_.*",
    "Shin_.*",
]


@configclass
class ClassHumanoidTaskSceneCfg(MySceneCfg):
    """Default flat scene for Class Humanoid skill-style tasks."""

    pass


@configclass
class ClassHumanoidTaskObservationsCfg:
    """Common low-dimensional observation space for non-walking Class Humanoid tasks."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_rpy = ObsTerm(func=common_mdp.base_rpy, noise=Unoise(n_min=-0.02, n_max=0.02))
        joint_pos = ObsTerm(
            func=common_mdp.class_humanoid_joint_pos_rel_semantic,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=common_mdp.class_humanoid_joint_vel_rel_semantic,
            noise=Unoise(n_min=-0.15, n_max=0.15),
        )
        foot_contacts = ObsTerm(
            func=common_mdp.binary_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ClassHumanoidTaskTerminationsCfg:
    """Common geometric/contact termination conditions for non-walking tasks."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(15.0)})
    root_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.58})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY_NAME), "threshold": 5.0},
    )


@configclass
class ClassHumanoidTaskRewardsCfg:
    """Common stability and smoothness rewards for non-walking tasks."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    torso_tilt_l2 = RewTerm(func=common_mdp.torso_tilt_l2, weight=-2.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.0e-4)


def configure_class_humanoid_flat_scene(cfg) -> None:
    """Install the Class Humanoid asset and use a flat terrain scene."""
    cfg.scene.robot = CLASS_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.scene.height_scanner = None
    cfg.curriculum.terrain_levels = None


def configure_class_humanoid_task_defaults(cfg, action_scale: float = 0.45) -> None:
    """Apply shared reset, randomization, action, and viewer defaults."""
    cfg.actions.joint_pos.scale = action_scale
    cfg.events.push_robot = None
    cfg.events.base_external_force_torque.params["asset_cfg"].body_names = [BASE_BODY_NAME]
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.reset_base.params = {
        "pose_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10), "z": (0.05, 0.10), "yaw": (-0.20, 0.20)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    cfg.viewer.asset_name = "robot"
    cfg.viewer.body_name = BASE_BODY_NAME
    cfg.viewer.eye = (3.0, -1.6, 1.8)
    cfg.viewer.lookat = (0.0, 0.0, 0.9)
