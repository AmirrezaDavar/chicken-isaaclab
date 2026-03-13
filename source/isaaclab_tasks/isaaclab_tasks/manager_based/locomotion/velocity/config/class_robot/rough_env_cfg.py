# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.class_robot.primitive_mdp as primitive_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from isaaclab_assets import CLASS_HUMANOID_CFG  # isort: skip
# from isaaclab_assets import CLASS_HUMANOID_USD_CFG as CLASS_HUMANOID_CFG  # isort: skip


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
    # Keep the original behavior (disabled) in the base config.
    undesired_contacts = None
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
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_Ankle_RS00")},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
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
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7

        # Use semantic left/right joint conventions in policy observations for this humanoid.
        self.observations.policy.joint_pos = ObsTerm(
            func=primitive_mdp.class_humanoid_joint_pos_rel_semantic,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        self.observations.policy.joint_vel = ObsTerm(
            func=primitive_mdp.class_humanoid_joint_vel_rel_semantic,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Viewer defaults for camera tracking overrides.
        # NOTE:
        # configclass update enforces exact runtime types. If these stay None, CLI overrides
        # like `env.viewer.asset_name=robot` fail with NoneType mismatch.
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "base_link"

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"


@configclass
class ClassHumanoidRewardsWithUndesiredContacts(ClassHumanoidRewards):
    """Option 1: Penalize non-foot contacts instead of allowing them freely."""

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "base_link",
                    "Head_1",
                    "Hip_1",
                    "HipYoke_.*",
                    "UpperThigh_.*",
                    "LowerThigh_.*",
                    "Shin_.*",
                    "Shoulder_.*",
                    "UpBicep_.*",
                    "LowBicep_.*",
                    "Forearm_.*",
                    "Wrist_.*",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class ClassHumanoidRoughEnvCfgContactPenalty(ClassHumanoidRoughEnvCfg):
    """Option 1 env: Enable non-foot contact penalty."""

    rewards: ClassHumanoidRewardsWithUndesiredContacts = ClassHumanoidRewardsWithUndesiredContacts()


@configclass
class ClassHumanoidRoughEnvCfgBadOrientation(ClassHumanoidRoughEnvCfg):
    """Option 2 env: Add orientation-based termination."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.base_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class ClassHumanoidRoughEnvCfgExtendedBaseContact(ClassHumanoidRoughEnvCfg):
    """Option 3 env: Expand contact-termination bodies beyond base_link."""

    def __post_init__(self):
        super().__post_init__()
        # NOTE:
        # body_names must match existing USD link names.
        # Unmatched regex entries cause a ValueError at config parsing time.
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
class ClassHumanoidRoughEnvCfgFootLift(ClassHumanoidRoughEnvCfgExtendedBaseContact):
    """Option 4 env: Encourage clearer stepping while keeping extended base-contact termination."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.55
        self.rewards.feet_slide.weight = -0.35
        self.rewards.action_rate_l2.weight = -0.003
        self.actions.joint_pos.scale = 0.6


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
