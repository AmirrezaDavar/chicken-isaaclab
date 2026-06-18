# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + custom gripper chicken lift with IK Cartesian control.

Action space: 8D
  [0:6]  6D relative EEF pose delta (IK) — from SpaceMouse translation + rotation
  [6]    Left-jaw binary  (PrismaticJoint1 + PrismaticJoint2)
  [7]    Right-jaw binary (PrismaticJoint3 + PrismaticJoint4)
"""

import isaaclab.envs.mdp as mdp
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.chicken_lift.chicken_lift_env_cfg import (
    SequentialActionsCfg,
)
from . import joint_pos_env_cfg

_GRIPPER_OPEN  =  0.0
_GRIPPER_CLOSE = -0.0093


@configclass
class UR10eCustomGripperChickenLiftIKRelEnvCfg(joint_pos_env_cfg.UR10eCustomGripperChickenLiftEnvCfg):
    """Chicken lift with IK Cartesian arm + independent left/right jaw control (8D action)."""

    def __post_init__(self):
        super().__post_init__()

        # ── Arm: 6D relative EEF pose (IK) ──────────────────────────────────
        # DLS IK maps [dx,dy,dz,rx,ry,rz] delta to joint velocities.
        # scale=0.5: smooth teleoperation; body_offset=18cm = gripper centre.
        arm_ik = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
            ),
            scale=0.5,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.18]),
        )

        # ── Gripper: independent left and right jaw ───────────────────────────
        left_jaw = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint1", "PrismaticJoint2"],
            open_command_expr={"PrismaticJoint1": _GRIPPER_OPEN, "PrismaticJoint2": _GRIPPER_OPEN},
            close_command_expr={"PrismaticJoint1": _GRIPPER_CLOSE, "PrismaticJoint2": _GRIPPER_CLOSE},
        )
        right_jaw = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint3", "PrismaticJoint4"],
            open_command_expr={"PrismaticJoint3": _GRIPPER_OPEN, "PrismaticJoint4": _GRIPPER_OPEN},
            close_command_expr={"PrismaticJoint3": _GRIPPER_CLOSE, "PrismaticJoint4": _GRIPPER_CLOSE},
        )

        # Replace unified ActionsCfg with SequentialActionsCfg (8D total)
        self.actions = SequentialActionsCfg(
            arm_action=arm_ik,
            gripper_left_action=left_jaw,
            gripper_right_action=right_jaw,
        )

        # Single env for teleoperation
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5


@configclass
class UR10eCustomGripperChickenLiftIKRelEnvCfg_PLAY(UR10eCustomGripperChickenLiftIKRelEnvCfg):
    """Play variant with corruption disabled."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
