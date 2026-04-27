# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stepping task configuration for the Class Humanoid robot."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import common_mdp, step_mdp
from .common import (
    BASE_BODY_NAME,
    FOOT_BODY_NAMES,
    HIP_PITCH_JOINT_NAMES,
    HIP_YOKE_BODY_NAMES,
    KNEE_JOINT_NAMES,
    ClassHumanoidTaskObservationsCfg,
    ClassHumanoidTaskRewardsCfg,
    ClassHumanoidTaskSceneCfg,
    ClassHumanoidTaskTerminationsCfg,
    configure_class_humanoid_flat_scene,
    configure_class_humanoid_task_defaults,
)


@configclass
class StepCommandsCfg:
    """Command interface for in-place stepping."""

    swing_foot = step_mdp.AlternatingFootCommandCfg(
        resampling_time_range=(0.45, 0.75),
        start_with_right=True,
        debug_vis=False,
    )
    target_foot_pos_xy = step_mdp.StepTargetFootCommandCfg(
        resampling_time_range=(1.0e6, 1.0e6),
        x_range=(0.0, 0.0),
        y_abs_range=(0.08, 0.14),
        swing_command_name="swing_foot",
        debug_vis=False,
    )


@configclass
class StepRewardsCfg(ClassHumanoidTaskRewardsCfg):
    step_target_reward = RewTerm(
        func=step_mdp.selected_foot_step_reward_tanh,
        weight=3.0,
        params={
            "swing_command_name": "swing_foot",
            "target_command_name": "target_foot_pos_xy",
            "std": 0.06,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    swing_knee_angle_range = RewTerm(
        func=step_mdp.selected_swing_knee_angle_range_reward,
        weight=0.8,
        params={
            "swing_command_name": "swing_foot",
            "min_angle": 0.60,
            "max_angle": 1.30,
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=KNEE_JOINT_NAMES),
        },
    )
    support_knee_straight = RewTerm(
        func=step_mdp.support_knee_straight_reward,
        weight=1.5,
        params={
            "swing_command_name": "swing_foot",
            "max_angle": 0.35,
            "std": 0.10,
            "asset_cfg": SceneEntityCfg("robot", joint_names=KNEE_JOINT_NAMES),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.2,
        params={
            "command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 0.45,
        },
    )
    alternating_contact_pattern = RewTerm(
        func=step_mdp.alternating_contact_pattern_reward,
        weight=2.0,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    swing_foot_contact = RewTerm(
        func=step_mdp.swing_foot_contact_penalty,
        weight=-1.0,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    support_foot_slip = RewTerm(
        func=step_mdp.support_foot_slip_penalty,
        weight=-0.6,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    feet_lateral_order = RewTerm(
        func=step_mdp.feet_lateral_order_penalty,
        weight=-1.0,
        params={
            "min_separation": 0.04,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    base_lin_vel_xy = RewTerm(func=common_mdp.base_lin_vel_xy_l2, weight=-0.3)
    base_ang_vel_z = RewTerm(func=common_mdp.base_ang_vel_z_l2, weight=-0.2)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw_RS03", ".*_Hip_Roll_RS03"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_RS02", ".*_Wrist_RS00"])
        },
    )


@configclass
class StepAlternatingRewardsCfg(ClassHumanoidTaskRewardsCfg):
    """Stronger alternating-contact rewards with minimum swing-knee flexion."""

    step_target_reward = RewTerm(
        func=step_mdp.selected_foot_step_reward_tanh,
        weight=4.0,
        params={
            "swing_command_name": "swing_foot",
            "target_command_name": "target_foot_pos_xy",
            "std": 0.06,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    swing_knee_min_flex = RewTerm(
        func=step_mdp.selected_swing_knee_min_flex_reward,
        weight=1.2,
        params={
            "swing_command_name": "swing_foot",
            "start_angle": 0.0,
            "target_angle": 0.60,
            "asset_cfg": SceneEntityCfg("robot", joint_names=KNEE_JOINT_NAMES),
        },
    )
    support_knee_straight = RewTerm(
        func=step_mdp.support_knee_straight_reward,
        weight=1.0,
        params={
            "swing_command_name": "swing_foot",
            "max_angle": 0.35,
            "std": 0.10,
            "asset_cfg": SceneEntityCfg("robot", joint_names=KNEE_JOINT_NAMES),
        },
    )
    swing_leg_shortening = RewTerm(
        func=step_mdp.swing_leg_shortening_reward,
        weight=1.0,
        params={
            "swing_command_name": "swing_foot",
            "target_margin": 0.08,
            "std": 0.04,
            "hip_cfg": SceneEntityCfg("robot", body_names=HIP_YOKE_BODY_NAMES),
            "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=3.0,
        params={
            "command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 0.45,
        },
    )
    alternating_contact_pattern = RewTerm(
        func=step_mdp.alternating_contact_pattern_reward,
        weight=3.0,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    swing_foot_contact = RewTerm(
        func=step_mdp.swing_foot_contact_penalty,
        weight=-1.5,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    support_foot_slip = RewTerm(
        func=step_mdp.support_foot_slip_penalty,
        weight=-0.3,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            "threshold": 1.0,
        },
    )
    feet_lateral_order = RewTerm(
        func=step_mdp.feet_lateral_order_penalty,
        weight=-1.2,
        params={
            "min_separation": 0.04,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    base_lin_vel_xy = RewTerm(func=common_mdp.base_lin_vel_xy_l2, weight=-0.3)
    base_ang_vel_z = RewTerm(func=common_mdp.base_ang_vel_z_l2, weight=-0.2)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw_RS03", ".*_Hip_Roll_RS03"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_RS02", ".*_Wrist_RS00"])
        },
    )


@configclass
class StepAllRewardsCfg(StepAlternatingRewardsCfg):
    """Minimum-flex shaping plus the target-range swing-knee reward."""

    swing_knee_angle_range = RewTerm(
        func=step_mdp.selected_swing_knee_angle_range_reward,
        weight=0.4,
        params={
            "swing_command_name": "swing_foot",
            "min_angle": 0.60,
            "max_angle": 1.30,
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=KNEE_JOINT_NAMES),
        },
    )


@configclass
class StepShapingRewardsCfg(StepAllRewardsCfg):
    """All step rewards plus clearance and anti-compensation shaping."""

    swing_foot_base_clearance = RewTerm(
        func=step_mdp.swing_foot_base_clearance_reward,
        weight=1.0,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "start_height": 0.03,
            "target_height": 0.10,
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    swing_support_height_difference = RewTerm(
        func=step_mdp.swing_support_height_difference_reward,
        weight=0.8,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "start_height_diff": 0.02,
            "target_height_diff": 0.08,
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )
    hip_only_swing = RewTerm(
        func=step_mdp.hip_only_swing_penalty,
        weight=-0.8,
        params={
            "swing_command_name": "swing_foot",
            "hip_threshold": 0.35,
            "hip_saturation": 0.80,
            "knee_threshold": 0.30,
            "knee_floor": 0.0,
            "hip_cfg": SceneEntityCfg("robot", joint_names=HIP_PITCH_JOINT_NAMES),
            "knee_cfg": SceneEntityCfg("robot", joint_names=KNEE_JOINT_NAMES),
        },
    )


@configclass
class ClassHumanoidStepEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Task: in-place stepping with alternating swing-foot placement targets."""

    scene: ClassHumanoidTaskSceneCfg = ClassHumanoidTaskSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ClassHumanoidTaskObservationsCfg = ClassHumanoidTaskObservationsCfg()
    commands: StepCommandsCfg = StepCommandsCfg()
    rewards: StepRewardsCfg = StepRewardsCfg()
    terminations: ClassHumanoidTaskTerminationsCfg = ClassHumanoidTaskTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_class_humanoid_flat_scene(self)
        configure_class_humanoid_task_defaults(self)

        self.episode_length_s = 10.0
        self.observations.policy.swing_foot = ObsTerm(func=mdp.generated_commands, params={"command_name": "swing_foot"})
        self.observations.policy.target_foot_pos_xy = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_foot_pos_xy"}
        )
        self.observations.policy.base_xy_from_reset = ObsTerm(
            func=step_mdp.BaseXYFromResetObservation,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        self.rewards.base_reset_position = RewTerm(
            func=step_mdp.BaseResetPositionPenalty,
            weight=-3.0,
            params={"deadband": 0.06, "std": 0.10, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.base_reset_outward_vel = RewTerm(
            func=step_mdp.BaseResetOutwardVelocityPenalty,
            weight=-1.5,
            params={
                "deadband": 0.03,
                "distance_scale": 0.10,
                "vel_scale": 0.20,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        self.rewards.feet_midpoint_reset = RewTerm(
            func=step_mdp.FeetMidpointResetPenalty,
            weight=-4.0,
            params={
                "deadband": 0.05,
                "std": 0.08,
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
            },
        )

        self.terminations.bad_orientation = None
        self.terminations.root_too_low = None
        self.terminations.base_contact.func = mdp.illegal_contact
        self.terminations.base_contact.params = {
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_BODY_NAME),
            "threshold": 1.0,
        }
        self.events.reset_base.params = {
            "pose_range": {"x": (-1.5, -1.5), "y": (1.5, 1.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }


@configclass
class ClassHumanoidStepAltEnvCfg(ClassHumanoidStepEnvCfg):
    """Step variant with stronger alternating-contact rewards."""

    rewards: StepAlternatingRewardsCfg = StepAlternatingRewardsCfg()


@configclass
class ClassHumanoidStepAllEnvCfg(ClassHumanoidStepEnvCfg):
    """Step variant combining minimum-flex and target-range knee shaping."""

    rewards: StepAllRewardsCfg = StepAllRewardsCfg()


@configclass
class ClassHumanoidStepShapingEnvCfg(ClassHumanoidStepEnvCfg):
    """Most shaped step variant."""

    rewards: StepShapingRewardsCfg = StepShapingRewardsCfg()


@configclass
class ClassHumanoidStepGeomTermEnvCfg(ClassHumanoidStepEnvCfg):
    """Step variant with geometric fall checks instead of broad illegal-contact termination."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
        self.terminations.root_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.55})
        self.terminations.base_contact = None
