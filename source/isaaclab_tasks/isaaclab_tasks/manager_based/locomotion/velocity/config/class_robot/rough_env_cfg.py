# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import CLASS_HUMANOID_CFG  # isort: skip


def class_humanoid_joint_semantic_signs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return semantic sign multipliers for class humanoid joints.

    Convention:
    - left-side joints keep their raw sign
    - right-side joints are multiplied by -1
    - center / unmatched joints keep +1
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_articulation_joint_ids(asset, asset_cfg)
    joint_names = [asset.data.joint_names[i] for i in joint_ids]
    joint_signs = [-1.0 if name.startswith("Right_") else 1.0 for name in joint_names]
    return torch.tensor(joint_signs, device=env.device, dtype=asset.data.joint_pos.dtype)


def class_humanoid_joint_pos_rel_semantic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions relative to default, converted into class-humanoid semantic signs."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_articulation_joint_ids(asset, asset_cfg)
    joint_signs = class_humanoid_joint_semantic_signs(env, asset_cfg)
    joint_pos_rel = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return joint_pos_rel * joint_signs.unsqueeze(0)


def class_humanoid_joint_vel_rel_semantic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocities relative to default, converted into class-humanoid semantic signs."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = resolve_articulation_joint_ids(asset, asset_cfg)
    joint_signs = class_humanoid_joint_semantic_signs(env, asset_cfg)
    joint_vel_rel = asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]
    return joint_vel_rel * joint_signs.unsqueeze(0)

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


@configclass
class ClassHumanoidRewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot_.*"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Foot_.*"),
        },
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_Ankle_RS00")},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw_RS03", ".*_Hip_Roll_RS03"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_RS02", ".*_Wrist_RS00"])
        },
    )
    joint_deviation_torso = None


@configclass
class ClassHumanoidRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Submission config for the final rough walking task based on opt3 only."""

    rewards: ClassHumanoidRewards = ClassHumanoidRewards()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = CLASS_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Use semantic left/right joint conventions in policy observations for this humanoid.
        self.observations.policy.joint_pos = ObsTerm(
            func=class_humanoid_joint_pos_rel_semantic,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=class_humanoid_joint_vel_rel_semantic,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Viewer defaults for camera tracking overrides.
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "base_link"

        # Terminations: opt3 only.
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base_link",
            "Hip_1",
            "Head_1",
            "HipYoke_.*",
            "Shoulder_.*",
            "UpBicep_.*",
            "LowBicep_.*",
            "Forearm_.*",
            "Wrist_.*",
            "UpperThigh_.*",
            "LowerThigh_.*",
        ]


@configclass
class ClassHumanoidRoughEnvCfg_PLAY(ClassHumanoidRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
