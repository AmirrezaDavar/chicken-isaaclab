# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Primitive-skill environments for Task E (Squat, Step, Reach-with-depth)."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.config.class_robot.primitive_mdp as primitive_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, MySceneCfg

from isaaclab_assets import CLASS_HUMANOID_CFG  # isort: skip


@configclass
class PrimitiveObservationsCfg:
    """Common low-dimensional observation space used by all primitive tasks."""

    @configclass
    class PolicyCfg(ObsGroup):
        # base state
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_rpy = ObsTerm(func=primitive_mdp.base_rpy, noise=Unoise(n_min=-0.02, n_max=0.02))
        # proprioception
        joint_pos = ObsTerm(func=primitive_mdp.class_humanoid_joint_pos_rel_semantic, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=primitive_mdp.class_humanoid_joint_vel_rel_semantic, noise=Unoise(n_min=-0.15, n_max=0.15))
        # contact + previous action
        foot_contacts = ObsTerm(
            func=primitive_mdp.binary_contacts,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class PrimitiveTerminationsCfg:
    """Common termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(15.0)})
    root_too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.58})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 5.0},
    )


@configclass
class PrimitiveCommonRewardsCfg:
    """Stability and smoothness rewards shared by all primitive tasks."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    torso_tilt_l2 = RewTerm(func=primitive_mdp.torso_tilt_l2, weight=-2.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.0e-4)


@configclass
class SquatCommandsCfg:
    """Command interface for squatting."""

    target_pelvis_height = primitive_mdp.UniformPelvisHeightCommandCfg(
        min_height=0.62,
        max_height=0.75,
        resampling_time_range=(1.5, 2.5),
        debug_vis=False,
    )


@configclass
class SquatRewardsCfg(PrimitiveCommonRewardsCfg):
    pelvis_height_tracking = RewTerm(
        func=primitive_mdp.pelvis_height_error_l2, weight=-8.0, params={"command_name": "target_pelvis_height"}
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
        },
    )


@configclass
class StepCommandsCfg:
    """Command interface for in-place stepping."""

    swing_foot = primitive_mdp.AlternatingFootCommandCfg(
        resampling_time_range=(0.45, 0.75),
        start_with_right=True,
        debug_vis=False,
    )
    target_foot_pos_xy = primitive_mdp.StepTargetFootCommandCfg(
        # This command is resampled when the swing-leg phase changes, not on an independent timer.
        resampling_time_range=(1.0e6, 1.0e6),
        x_range=(0.02, 0.10),
        y_abs_range=(0.08, 0.14),
        swing_command_name="swing_foot",
        debug_vis=False,
    )


@configclass
class StepRewardsCfg(PrimitiveCommonRewardsCfg):
    step_target_reward = RewTerm(
        func=primitive_mdp.selected_foot_step_reward_tanh,
        weight=5.0,
        params={
            "swing_command_name": "swing_foot",
            "target_command_name": "target_foot_pos_xy",
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
        },
    )
    swing_knee_angle_range = RewTerm(
        func=primitive_mdp.selected_swing_knee_angle_range_reward,
        weight=0.8,
        params={
            "swing_command_name": "swing_foot",
            "min_angle": 0.60,
            "max_angle": 1.30,
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.2,
        params={
            "command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "threshold": 0.45,
        },
    )
    support_foot_slip = RewTerm(
        func=primitive_mdp.support_foot_slip_penalty,
        weight=-0.6,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "threshold": 1.0,
        },
    )
    feet_lateral_order = RewTerm(
        func=primitive_mdp.feet_lateral_order_penalty,
        weight=-1.0,
        params={
            "min_separation": 0.04,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
        },
    )
    base_lin_vel_xy = RewTerm(func=primitive_mdp.base_lin_vel_xy_l2, weight=-1.0)
    base_ang_vel_z = RewTerm(func=primitive_mdp.base_ang_vel_z_l2, weight=-0.2)
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
class StepAlternatingRewardsCfg(PrimitiveCommonRewardsCfg):
    """Option: strongly reward alternating contact with a minimum-flex swing-knee objective."""

    step_target_reward = RewTerm(
        func=primitive_mdp.selected_foot_step_reward_tanh,
        weight=5.0,
        params={
            "swing_command_name": "swing_foot",
            "target_command_name": "target_foot_pos_xy",
            "std": 0.06,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
        },
    )
    swing_knee_min_flex = RewTerm(
        func=primitive_mdp.selected_swing_knee_min_flex_reward,
        weight=1.2,
        params={
            "swing_command_name": "swing_foot",
            "start_angle": 0.0,
            "target_angle": 0.60,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
        },
    )
    swing_leg_shortening = RewTerm(
        func=primitive_mdp.swing_leg_shortening_reward,
        weight=1.0,
        params={
            "swing_command_name": "swing_foot",
            "target_margin": 0.08,
            "std": 0.04,
            "hip_cfg": SceneEntityCfg("robot", body_names=["HipYoke_Left_1", "HipYoke_Right_1"]),
            "foot_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "threshold": 0.45,
        },
    )
    alternating_contact_pattern = RewTerm(
        func=primitive_mdp.alternating_contact_pattern_reward,
        weight=3.0,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "threshold": 1.0,
        },
    )
    swing_foot_contact = RewTerm(
        func=primitive_mdp.swing_foot_contact_penalty,
        weight=-1.5,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "threshold": 1.0,
        },
    )
    support_foot_slip = RewTerm(
        func=primitive_mdp.support_foot_slip_penalty,
        weight=-0.3,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "threshold": 1.0,
        },
    )
    feet_lateral_order = RewTerm(
        func=primitive_mdp.feet_lateral_order_penalty,
        weight=-1.2,
        params={
            "min_separation": 0.04,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
        },
    )
    base_lin_vel_xy = RewTerm(func=primitive_mdp.base_lin_vel_xy_l2, weight=-0.6)
    base_ang_vel_z = RewTerm(func=primitive_mdp.base_ang_vel_z_l2, weight=-0.2)
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
    """Option: combine minimum-flex shaping with the original target-range knee reward."""

    swing_knee_angle_range = RewTerm(
        func=primitive_mdp.selected_swing_knee_angle_range_reward,
        weight=0.4,
        params={
            "swing_command_name": "swing_foot",
            "min_angle": 0.60,
            "max_angle": 1.30,
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
        },
    )


@configclass
class StepShapingRewardsCfg(StepAllRewardsCfg):
    """Option: add clearance and anti-hip-only shaping on top of the all-in step rewards."""

    swing_foot_base_clearance = RewTerm(
        func=primitive_mdp.swing_foot_base_clearance_reward,
        weight=1.0,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "start_height": 0.03,
            "target_height": 0.10,
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
        },
    )
    swing_support_height_difference = RewTerm(
        func=primitive_mdp.swing_support_height_difference_reward,
        weight=0.8,
        params={
            "swing_command_name": "swing_foot",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Foot_Left_1", "Foot_Right_1"]),
            "start_height_diff": 0.02,
            "target_height_diff": 0.08,
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
        },
    )
    hip_only_swing = RewTerm(
        func=primitive_mdp.hip_only_swing_penalty,
        weight=-0.8,
        params={
            "swing_command_name": "swing_foot",
            "hip_threshold": 0.35,
            "hip_saturation": 0.80,
            "knee_threshold": 0.30,
            "knee_floor": 0.0,
            "hip_cfg": SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch_RS04", "Right_Hip_Pitch_RS04"]),
            "knee_cfg": SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
        },
    )


@configclass
class ReachDepthCommandsCfg:
    """Command interface for depth-based reaching."""

    target_pos_base_from_depth = primitive_mdp.DepthTargetPosCommandCfg(
        sensor_name="depth_camera",
        data_type="distance_to_image_plane",
        min_depth=0.20,
        max_depth=2.00,
        smooth_factor=0.75,
        resampling_time_range=(0.5, 0.5),
        debug_vis=False,
    )


@configclass
class ReachDepthRewardsCfg(PrimitiveCommonRewardsCfg):
    reach_target = RewTerm(
        func=primitive_mdp.end_effector_object_reward_tanh,
        weight=8.0,
        params={"target_name": "target", "std": 0.05, "asset_cfg": SceneEntityCfg("robot", body_names=["Wrist_Right_1"])},
    )
    arm_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Right_Shoulder_Pitch_RS03",
                    "Right_Shoulder_Roll_RS03",
                    "Right_Shoulder_Yaw_RS02",
                    "Right_Elbow_RS02",
                    "Right_Wrist_RS00",
                ],
            )
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link", "Head_1", "Hip_1", "HipYoke_.*", "UpperThigh_.*", "LowerThigh_.*", "Shin_.*"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class ClassHumanoidPrimitiveBaseEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Common base for primitive-skill tasks."""

    scene: MySceneCfg = MySceneCfg(num_envs=2048, env_spacing=2.5)
    observations: PrimitiveObservationsCfg = PrimitiveObservationsCfg()
    rewards: PrimitiveCommonRewardsCfg = PrimitiveCommonRewardsCfg()
    terminations: PrimitiveTerminationsCfg = PrimitiveTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Scene: robot + flat terrain.
        self.scene.robot = CLASS_HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # Action defaults for primitives.
        self.actions.joint_pos.scale = 0.45

        # Randomization knobs.
        self.events.push_robot = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base_link"]
        # Keep startup stable first; re-enable mass randomization after primitive policies learn to stand.
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.reset_base.params = {
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
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

        # Viewer defaults.
        self.viewer.asset_name = "robot"
        self.viewer.body_name = "base_link"
        self.viewer.eye = (3.0, -1.6, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.9)


@configclass
class ClassHumanoidPrimitiveSquatEnvCfg(ClassHumanoidPrimitiveBaseEnvCfg):
    """Task: Squatting."""

    commands: SquatCommandsCfg = SquatCommandsCfg()
    rewards: SquatRewardsCfg = SquatRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Base-contact is too sensitive for this setup; use geometric fall checks.
        self.terminations.base_contact = None
        self.terminations.bad_orientation.params["limit_angle"] = 0.8
        self.terminations.root_too_low.params["minimum_height"] = 0.55
        self.episode_length_s = 8.0
        self.actions.joint_pos.scale = 0.35
        self.observations.policy.target_pelvis_height = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_pelvis_height"}
        )


@configclass
class ClassHumanoidPrimitiveStepEnvCfg(ClassHumanoidPrimitiveBaseEnvCfg):
    """Task: In-place stepping with alternating swing-foot placement targets."""

    commands: StepCommandsCfg = StepCommandsCfg()
    rewards: StepRewardsCfg = StepRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0

        # Make step commands explicit in the policy observations.
        self.observations.policy.swing_foot = ObsTerm(func=mdp.generated_commands, params={"command_name": "swing_foot"})
        self.observations.policy.target_foot_pos_xy = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_foot_pos_xy"}
        )

        # Match walking-style failure handling for fall detection.
        self.terminations.bad_orientation = None
        self.terminations.root_too_low = None
        self.terminations.base_contact.params["threshold"] = 1.0
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

        # Reflect walking Option 3: expand contact-termination bodies beyond base_link.
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
class ClassHumanoidPrimitiveStepAltEnvCfg(ClassHumanoidPrimitiveStepEnvCfg):
    """Option variant: stronger alternating-contact rewards with minimum swing-knee flexion."""

    rewards: StepAlternatingRewardsCfg = StepAlternatingRewardsCfg()


@configclass
class ClassHumanoidPrimitiveStepAllEnvCfg(ClassHumanoidPrimitiveStepEnvCfg):
    """Option variant: combine minimum-flex and target-range swing-knee shaping."""

    rewards: StepAllRewardsCfg = StepAllRewardsCfg()


@configclass
class ClassHumanoidPrimitiveStepShapingEnvCfg(ClassHumanoidPrimitiveStepEnvCfg):
    """Option variant: add clearance and anti-compensation shaping to the all-in step rewards."""

    rewards: StepShapingRewardsCfg = StepShapingRewardsCfg()


@configclass
class ClassHumanoidPrimitiveReachSceneCfg(MySceneCfg):
    """Scene extension for depth-based reaching."""

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.95)),
        spawn=sim_utils.SphereCfg(
            radius=0.045,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.15)),
        ),
    )

    depth_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/DepthCamera",
        update_latest_camera_pose=True,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.24, 0.0, 0.22), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=20.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.15, 3.0),
        ),
        width=64,
        height=64,
    )


@configclass
class ClassHumanoidPrimitiveReachDepthEnvCfg(ClassHumanoidPrimitiveBaseEnvCfg):
    """Task: Reaching with depth perception."""

    scene: ClassHumanoidPrimitiveReachSceneCfg = ClassHumanoidPrimitiveReachSceneCfg(num_envs=1536, env_spacing=2.5)
    commands: ReachDepthCommandsCfg = ReachDepthCommandsCfg()
    rewards: ReachDepthRewardsCfg = ReachDepthRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Base-contact is too sensitive for this setup; use geometric fall checks.
        self.terminations.base_contact = None
        self.terminations.bad_orientation.params["limit_angle"] = 0.8
        self.terminations.root_too_low.params["minimum_height"] = 0.55
        self.episode_length_s = 8.0

        # Right-arm-only control for reaching.
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "Right_Shoulder_Pitch_RS03",
                "Right_Shoulder_Roll_RS03",
                "Right_Shoulder_Yaw_RS02",
                "Right_Elbow_RS02",
                "Right_Wrist_RS00",
            ],
            scale=0.35,
            use_default_offset=True,
        )

        # Depth-estimated target position goes directly into policy observation.
        self.observations.policy.target_pos_base_from_depth = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_pos_base_from_depth"}
        )

        # Randomize target position in front of the robot at reset.
        self.events.reset_target = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("target"),
                "pose_range": {"x": (0.45, 0.70), "y": (-0.22, 0.22), "z": (0.82, 1.02)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
