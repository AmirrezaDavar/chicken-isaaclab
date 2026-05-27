# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Depth-based reaching task configuration for the Class Humanoid robot."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import reach_mdp
from .common import (
    NON_FOOT_FALL_CONTACT_BODY_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_WRIST_BODY_NAME,
    LEFT_ARM_JOINT_NAMES,
    LEFT_WRIST_BODY_NAME,
    ClassHumanoidTaskObservationsCfg,
    ClassHumanoidTaskRewardsCfg,
    ClassHumanoidTaskSceneCfg,
    ClassHumanoidTaskTerminationsCfg,
    configure_class_humanoid_flat_scene,
    configure_class_humanoid_task_defaults,
)


@configclass
class ReachDepthCommandsCfg:
    """Command interface for depth-based reaching."""

    target_pos_base_from_depth = reach_mdp.DepthTargetPosCommandCfg(
        sensor_name="depth_camera",
        data_type="distance_to_image_plane",
        min_depth=0.20,
        max_depth=2.00,
        smooth_factor=0.75,
        resampling_time_range=(0.5, 0.5),
        debug_vis=False,
    )


@configclass
class ReachDepthRewardsCfg(ClassHumanoidTaskRewardsCfg):
    reach_target = RewTerm(
        func=reach_mdp.end_effector_object_reward_tanh,
        weight=8.0,
        params={
            "target_name": "target",
            "std": 0.08,
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_WRIST_BODY_NAME]),
        },
    )
    reach_distance = RewTerm(
        func=reach_mdp.end_effector_object_distance,
        weight=-0.5,
        params={
            "target_name": "target",
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_WRIST_BODY_NAME]),
        },
    )
    arm_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINT_NAMES)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=NON_FOOT_FALL_CONTACT_BODY_NAMES),
            "threshold": 1.0,
        },
    )


@configclass
class ClassHumanoidReachSceneCfg(ClassHumanoidTaskSceneCfg):
    """Scene extension for depth-based reaching."""

    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.40, -0.08, 0.66)),
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
        offset=TiledCameraCfg.OffsetCfg(pos=(0.24, 0.0, 0.22), rot=(0.924, 0.0, -0.383, 0.0), convention="world"),
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
class ClassHumanoidReachDepthEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Task: reaching with a depth-derived target observation."""

    scene: ClassHumanoidReachSceneCfg = ClassHumanoidReachSceneCfg(num_envs=1536, env_spacing=2.5)
    observations: ClassHumanoidTaskObservationsCfg = ClassHumanoidTaskObservationsCfg()
    commands: ReachDepthCommandsCfg = ReachDepthCommandsCfg()
    rewards: ReachDepthRewardsCfg = ReachDepthRewardsCfg()
    terminations: ClassHumanoidTaskTerminationsCfg = ClassHumanoidTaskTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_class_humanoid_flat_scene(self)
        configure_class_humanoid_task_defaults(self, action_scale=0.5)

        self.terminations.base_contact = None
        self.terminations.bad_orientation.params["limit_angle"] = 0.8
        self.terminations.root_too_low.params["minimum_height"] = 0.55
        self.episode_length_s = 8.0
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )
        self.observations.policy.target_pos_base_from_depth = ObsTerm(
            func=reach_mdp.target_pos_base,
            params={"target_name": "target"},
        )
        self.events.reset_target = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("target"),
                "pose_range": {"x": (0.30, 0.50), "y": (-0.15, 0.0), "z": (0.60, 0.72)},
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


@configclass
class ClassHumanoidReachDepthCameraEnvCfg(ClassHumanoidReachDepthEnvCfg):
    """Reach task where the policy observes the target via the depth camera.

    The DepthTargetPosCommand reads the depth image each step, finds the nearest
    object (the red target sphere), and produces a smoothed 3-D position estimate
    in the robot base frame. That estimate — NOT the ground-truth position — is
    what the policy receives as input. Training requires --enable_cameras.
    """

    def __post_init__(self):
        super().__post_init__()

        # Fix the base so the policy only needs to learn arm control, not balance
        self.scene.robot.spawn.fix_base_link = True

        # Replace ground-truth target position with depth-camera-derived estimate.
        # mdp.generated_commands reads the output of DepthTargetPosCommand,
        # which is already temporally smoothed (smooth_factor=0.75).
        self.observations.policy.target_pos_base_from_depth = ObsTerm(
            func=base_mdp.generated_commands,
            params={"command_name": "target_pos_base_from_depth"},
        )

        # Fewer envs — camera rendering is expensive
        self.scene.num_envs = 512


@configclass
class ClassHumanoidReachLeftFixedEnvCfg(ClassHumanoidReachDepthEnvCfg):
    """Left-arm reaching with a fixed base — robot floats in place, only arm moves."""

    def __post_init__(self):
        super().__post_init__()

        # Fix the robot base so balance is not required
        self.scene.robot.spawn.fix_base_link = True

        # Switch to left arm joints
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )

        # Reward uses left wrist
        self.rewards.reach_target.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[LEFT_WRIST_BODY_NAME]
        )
        self.rewards.reach_distance.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[LEFT_WRIST_BODY_NAME]
        )
        self.rewards.arm_joint_deviation.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEFT_ARM_JOINT_NAMES
        )

        # Target spawns on the left side (positive y)
        self.events.reset_target = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("target"),
                "pose_range": {"x": (0.30, 0.50), "y": (0.05, 0.20), "z": (0.60, 0.72)},
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            },
        )
