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

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import reach_mdp
from .common import (
    NON_FOOT_FALL_CONTACT_BODY_NAMES,
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_WRIST_BODY_NAME,
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
            "std": 0.05,
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_WRIST_BODY_NAME]),
        },
    )
    arm_joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
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
        configure_class_humanoid_task_defaults(self, action_scale=0.35)

        self.terminations.base_contact = None
        self.terminations.bad_orientation.params["limit_angle"] = 0.8
        self.terminations.root_too_low.params["minimum_height"] = 0.55
        self.episode_length_s = 8.0
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINT_NAMES,
            scale=0.35,
            use_default_offset=True,
        )
        self.observations.policy.target_pos_base_from_depth = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_pos_base_from_depth"}
        )
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
