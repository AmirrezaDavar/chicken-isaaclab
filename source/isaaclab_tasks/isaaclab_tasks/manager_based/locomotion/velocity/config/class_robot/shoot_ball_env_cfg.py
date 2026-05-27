# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task F.2 — Shoot a Ball into a Target: arm-push floating ball to goal."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import shoot_ball_mdp
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
class ShootBallRewardsCfg(ClassHumanoidTaskRewardsCfg):
    # Pull right wrist toward ball
    arm_to_ball = RewTerm(
        func=shoot_ball_mdp.arm_to_ball_distance,
        weight=-2.0,
        params={
            "ball_name": "ball",
            "asset_cfg": SceneEntityCfg("robot", body_names=[RIGHT_WRIST_BODY_NAME]),
        },
    )
    # Dense shaping: push ball toward goal (uses per-env goal_marker position)
    ball_to_goal = RewTerm(
        func=shoot_ball_mdp.ball_to_goal_distance,
        weight=-1.5,
        params={"ball_name": "ball", "goal_name": "goal_marker"},
    )
    # Key shaping: reward ball velocity in the goal direction
    ball_velocity_toward_goal = RewTerm(
        func=shoot_ball_mdp.ball_velocity_toward_goal,
        weight=3.0,
        params={"ball_name": "ball", "goal_name": "goal_marker"},
    )
    # Large bonus when ball reaches goal
    ball_at_goal = RewTerm(
        func=shoot_ball_mdp.ball_at_goal_bonus,
        weight=25.0,
        params={"ball_name": "ball", "goal_name": "goal_marker", "threshold": 0.25},
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
class ShootBallObservationsCfg(ClassHumanoidTaskObservationsCfg):
    @configclass
    class PolicyCfg(ClassHumanoidTaskObservationsCfg.PolicyCfg):
        # Ball position in robot base frame
        ball_pos_base = ObsTerm(
            func=shoot_ball_mdp.ball_pos_base,
            params={"ball_name": "ball"},
        )
        # Goal position in robot base frame (tells robot where to push the ball)
        goal_pos_base = ObsTerm(
            func=shoot_ball_mdp.goal_pos_base,
            params={"goal_name": "goal_marker"},
        )

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ShootBallSceneCfg(ClassHumanoidTaskSceneCfg):
    # Orange ball — floats at wrist height, close to robot
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, -0.10, 0.50)),
        spawn=sim_utils.SphereCfg(
            radius=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                linear_damping=0.02,
                angular_damping=0.02,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
        ),
    )

    # Green goal disc — 1.2 m in front at arm height so ball can reach it while floating
    goal_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.20, 0.0, 0.50)),
        spawn=sim_utils.SphereCfg(
            radius=0.20,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.85, 0.1)),
        ),
    )


@configclass
class ClassHumanoidShootBallEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Task F.2 — Arm-push a floating ball 1.2 m into a goal sphere.

    Reward structure:
      - arm_to_ball (-): reach the ball with the right wrist
      - ball_to_goal (-): dense distance gradient toward goal
      - ball_velocity_toward_goal (+): reward ball speed in goal direction
      - ball_at_goal (+): large bonus when ball enters goal radius
    """

    scene: ShootBallSceneCfg = ShootBallSceneCfg(num_envs=1536, env_spacing=2.5)
    observations: ShootBallObservationsCfg = ShootBallObservationsCfg()
    rewards: ShootBallRewardsCfg = ShootBallRewardsCfg()
    terminations: ClassHumanoidTaskTerminationsCfg = ClassHumanoidTaskTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_class_humanoid_flat_scene(self)
        configure_class_humanoid_task_defaults(self, action_scale=0.5)

        self.terminations.base_contact = None
        self.terminations.bad_orientation.params["limit_angle"] = 1.4
        self.terminations.root_too_low.params["minimum_height"] = 0.35
        self.episode_length_s = 12.0

        # Control right arm joints for swiping the ball
        self.actions.joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINT_NAMES,
            scale=0.5,
            use_default_offset=True,
        )

        # Observe ball position in base frame
        self.observations.policy.ball_pos_base = ObsTerm(
            func=shoot_ball_mdp.ball_pos_base,
            params={"ball_name": "ball", "asset_cfg": SceneEntityCfg("robot")},
        )

        # Randomise ball start position each episode
        self.events.reset_ball = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "pose_range": {"x": (0.28, 0.42), "y": (-0.12, -0.02), "z": (0.42, 0.58)},
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            },
        )

        self.viewer.asset_name = "robot"
        self.viewer.eye = (3.0, -1.6, 1.8)
        self.viewer.lookat = (0.0, 0.0, 0.9)
