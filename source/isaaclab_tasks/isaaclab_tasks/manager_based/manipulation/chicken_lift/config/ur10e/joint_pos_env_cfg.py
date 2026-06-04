# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + custom 2-jaw parallel gripper — sequential left-then-right leg grasping.

The custom gripper (1_fixed.usda) has 4 prismatic joints split into two jaws:
  Left jaw  – PrismaticJoint1 + PrismaticJoint2  (grasps the chicken's left leg)
  Right jaw – PrismaticJoint3 + PrismaticJoint4  (grasps the chicken's right leg)

Training behaviour (shaped by SequentialGraspRewardsCfg):
  1. EE approaches the chicken.
  2. Policy closes the LEFT jaw on the left leg — rewarded by left_leg_grasped.
  3. Once the left jaw is gripping, right_jaw_gated becomes active and the
     policy closes the RIGHT jaw on the right leg.
  4. With at least the left leg secured, lifting_gated fires and the robot
     raises the chicken off the table.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.chicken_lift.chicken_lift_env_cfg import (
    ChickenSequentialGraspEnvCfg,
)

from isaaclab_assets.robots.universal_robots import UR10e_CUSTOM_GRIPPER_CFG  # isort: skip
from isaaclab_assets.robots.chicken import CHICKEN_CARCASS_CFG  # isort: skip

# Prismatic joint travel: 0 = open, −9.3 mm = fully closed
_OPEN = 0.0
_CLOSE = -0.0093


@configclass
class UR10eChickenLiftEnvCfg(ChickenSequentialGraspEnvCfg):
    """UR10e with custom 2-jaw gripper, sequential left-then-right leg grasp."""

    def __post_init__(self):
        super().__post_init__()

        # ---- Robot -------------------------------------------------------
        self.scene.robot = UR10e_CUSTOM_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -1.5708,
                    "elbow_joint": 1.5708,
                    "wrist_1_joint": -1.5708,
                    "wrist_2_joint": -1.5708,
                    "wrist_3_joint": 0.0,
                    "PrismaticJoint1": _OPEN,
                    "PrismaticJoint2": _OPEN,
                    "PrismaticJoint3": _OPEN,
                    "PrismaticJoint4": _OPEN,
                },
            ),
        )

        # Override gripper actuators: higher stiffness + damping for stable grasping
        self.scene.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["PrismaticJoint.*"],
            effort_limit_sim=40.0,
            velocity_limit_sim=0.15,
            stiffness=3000.0,
            damping=250.0,
            friction=0.0,
            armature=0.0,
        )

        # ---- Chicken object -----------------------------------------------
        self.scene.chicken = CHICKEN_CARCASS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Chicken"
        )

        # ---- End-effector frame (centre of gripper palm) -----------------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur10e/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur10e/wrist_3_link",
                    name="end_effector",
                    # 18 cm offset along wrist z to reach the gripper palm centre
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.18]),
                ),
            ],
        )

        # ---- Actions: 6-DOF arm + independent left/right jaw --------------
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=0.5,
            use_default_offset=True,
        )

        # Left jaw: PrismaticJoint1 + PrismaticJoint2
        # Policy uses this to grab the LEFT leg first
        self.actions.gripper_left_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint1", "PrismaticJoint2"],
            open_command_expr={
                "PrismaticJoint1": _OPEN,
                "PrismaticJoint2": _OPEN,
            },
            close_command_expr={
                "PrismaticJoint1": _CLOSE,
                "PrismaticJoint2": _CLOSE,
            },
        )

        # Right jaw: PrismaticJoint3 + PrismaticJoint4
        # Policy closes this AFTER the left jaw is gripping
        self.actions.gripper_right_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint3", "PrismaticJoint4"],
            open_command_expr={
                "PrismaticJoint3": _OPEN,
                "PrismaticJoint4": _OPEN,
            },
            close_command_expr={
                "PrismaticJoint3": _CLOSE,
                "PrismaticJoint4": _CLOSE,
            },
        )

        # ---- Command target (carry-goal visualisation off by default) -----
        self.commands.object_pose.body_name = "wrist_3_link"


@configclass
class UR10eChickenLiftEnvCfg_PLAY(UR10eChickenLiftEnvCfg):
    """Small scene for play / evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
