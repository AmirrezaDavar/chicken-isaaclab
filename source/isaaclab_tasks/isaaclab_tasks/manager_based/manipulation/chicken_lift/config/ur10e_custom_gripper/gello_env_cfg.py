# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + custom gripper chicken lift with GELLO joint-position teleoperation.

Action space: 8D
  [0:6]  Absolute arm joint angles (radians) — directly from GELLO Dynamixel readings
  [6]    Left-jaw binary  (+1 = open, -1 = close) — driven by GELLO gripper trigger
  [7]    Right-jaw binary (+1 = open, -1 = close) — same trigger, both jaws move together
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.chicken_lift.chicken_lift_env_cfg import (
    SequentialActionsCfg,
)
from . import joint_pos_env_cfg

CAM_H = 480
CAM_W = 640

_GRIPPER_OPEN  =  0.0
_GRIPPER_CLOSE = -0.0093   # physical hard-stop (meters)
# Command target pushed 5 mm past the hard-stop so the spring is compressed even
# when the chicken leg blocks the jaw before it reaches the real limit.
_GRIPPER_CLOSE_CMD = -0.015


@configclass
class UR10eCustomGripperChickenLiftGelloEnvCfg(joint_pos_env_cfg.UR10eCustomGripperChickenLiftEnvCfg):
    """Chicken lift env for GELLO teleoperation (8D action).

    Arm receives absolute joint angles directly (use_default_offset=False, scale=1.0).
    Both gripper jaws are controlled by the single GELLO trigger as independent
    binary channels so the action format matches the SpaceMouse/IK collection schema.
    """

    def __post_init__(self):
        super().__post_init__()

        # Set robot reset position to match GELLO's natural home pose so that
        # env.reset() starts close to where the arm will be snapped to.
        # Values from diagnostic (Δ≈0 screenshots): pan=-11.7°, lift=-106.1°,
        # elbow=-90.6°, wrist1=-81.2° (=278.8°-360°), wrist2=+92.4°, wrist3=+11.5°.
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "shoulder_pan_joint":  -0.205,
                "shoulder_lift_joint": -1.852,
                "elbow_joint":         -1.582,
                "wrist_1_joint":       -1.417,
                "wrist_2_joint":        1.612,
                "wrist_3_joint":        0.200,
                "PrismaticJoint1":      0.0,
                "PrismaticJoint2":      0.0,
                "PrismaticJoint3":      0.0,
                "PrismaticJoint4":      0.0,
            },
        )

        # Stronger gripper actuator for firm grasping.
        # stiffness 20000 N/m  →  ~66 N at 3.3 mm partial block (was ~8 N at 2500 N/m).
        # effort_limit 200 N   →  removes the 35 N cap that was letting the leg slip.
        self.scene.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["PrismaticJoint.*"],
            effort_limit_sim=200.0,
            velocity_limit_sim=0.2,
            stiffness=20000.0,
            damping=500.0,
            friction=0.0,
            armature=0.0,
        )

        # GELLO commands absolute joint angles — bypass RL offset/scale
        arm_joints = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            scale=1.0,
            use_default_offset=False,
        )

        left_jaw = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint1", "PrismaticJoint2"],
            open_command_expr={"PrismaticJoint1": _GRIPPER_OPEN,      "PrismaticJoint2": _GRIPPER_OPEN},
            close_command_expr={"PrismaticJoint1": _GRIPPER_CLOSE_CMD, "PrismaticJoint2": _GRIPPER_CLOSE_CMD},
        )

        right_jaw = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint3", "PrismaticJoint4"],
            open_command_expr={"PrismaticJoint3": _GRIPPER_OPEN,      "PrismaticJoint4": _GRIPPER_OPEN},
            close_command_expr={"PrismaticJoint3": _GRIPPER_CLOSE_CMD, "PrismaticJoint4": _GRIPPER_CLOSE_CMD},
        )

        self.actions = SequentialActionsCfg(
            arm_action=arm_joints,
            gripper_left_action=left_jaw,
            gripper_right_action=right_jaw,
        )

        # ── Gripper-mounted RealSense camera ─────────────────────────────────
        # Mounted on wrist_3_link, looking along the tool axis toward the chicken.
        # Adjust pos/rot if the view is wrong — open the live popup to verify.
        #   pos: (x, y, z) offset in the wrist_3_link frame (z = along tool axis)
        #   rot: quaternion (w,x,y,z) in ROS convention (camera +Z = forward)
        self.scene.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur10e/wrist_3_link/realsense",
            update_period=0,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=40.0,   # ~80° horizontal FOV (like RealSense D435)
                clipping_range=(0.02, 10.0),
            ),
            width=CAM_W,
            height=CAM_H,
            offset=CameraCfg.OffsetCfg(
                # Values from Isaac Sim Property panel (wrist_3_link local frame):
                #   Translate: (0.0, -0.1, 0.1)
                #   Orient Euler XYZ: (125°, 0°, 180°)
                # Quaternion (0,0,-0.462,0.887) = Euler(125°,0°,180°) corrected
                # for Isaac Lab's internal ROS→USD convention offset.
                pos=(0.0, -0.1, 0.1),
                rot=(0.0, 0.0, 0.462, 0.8875),
                convention="ros",
            ),
        )

        # Single environment for teleoperation
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5


@configclass
class UR10eCustomGripperChickenLiftGelloEnvCfg_PLAY(UR10eCustomGripperChickenLiftGelloEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False
