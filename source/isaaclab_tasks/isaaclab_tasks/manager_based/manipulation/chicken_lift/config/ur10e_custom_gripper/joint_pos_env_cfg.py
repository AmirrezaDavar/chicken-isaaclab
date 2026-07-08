# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + custom 4-jaw parallel gripper for chicken pick-and-place.

The gripper has 4 prismatic joints (PrismaticJoint1-4) with range [-9.3 mm, 0].
All 4 joints are driven together as a single binary open/close action.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.chicken_lift.chicken_lift_env_cfg import (
    ChickenLiftEnvCfg,
    TABLE_TOP_Z,
    make_chicken_table_init_state,
)

from isaaclab_assets.robots.universal_robots import (  # isort: skip
    UR10E_RAISER_CFG,
    UR10E_SHACKLE_CFG,
    UR10E_TABLE_CFG,
    UR10e_CUSTOM_GRIPPER_CFG,
)
from isaaclab_assets.robots.chicken import CHICKEN_CARCASS_CFG  # isort: skip

# Gripper joint limits (meters): open = 0, closed = -0.0093
_GRIPPER_OPEN = 0.0
_GRIPPER_CLOSE = -0.0093
_DIAGNOSTIC_CUBE_SIZE = 0.018


@configclass
class UR10eCustomGripperChickenLiftEnvCfg(ChickenLiftEnvCfg):
    """UR10e with custom 4-jaw parallel gripper lifting a chicken carcass."""

    def __post_init__(self):
        super().__post_init__()

        # ---- Raiser stand ------------------------------------------------
        # Spawn the stand as its own scene asset. The robot USD is kept
        # articulation-only, so Isaac Lab resets do not separate the stand
        # from the robot base.
        self.scene.robot_raiser = UR10E_RAISER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/RobotRaiser",
        )

        # ---- Table -------------------------------------------------------
        # The table is a kinematic scene asset. It is placed in front of the
        # robot, with its top at z=0.6205 m.
        self.scene.table = UR10E_TABLE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Table",
        )

        # ---- Shackle -----------------------------------------------------
        # The shackle is a kinematic scene asset, placed 1 m above the far end
        # of the table.
        self.scene.shackle = UR10E_SHACKLE_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Shackle",
        )

        # ---- Robot -------------------------------------------------------
        self.scene.robot = UR10e_CUSTOM_GRIPPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                # 0.63 m is the top of the UR10e raiser stand.
                pos=(0.0, 0.0, 0.63),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -1.5708,
                    "elbow_joint": 1.5708,
                    "wrist_1_joint": -1.5708,
                    "wrist_2_joint": -1.5708,
                    "wrist_3_joint": 0.0,
                    "PrismaticJoint1": _GRIPPER_OPEN,
                    "PrismaticJoint2": _GRIPPER_OPEN,
                    "PrismaticJoint3": _GRIPPER_OPEN,
                    "PrismaticJoint4": _GRIPPER_OPEN,
                },
            ),
        )

        # Override gripper actuators to be tighter for RL stability
        self.scene.robot.actuators["gripper"] = ImplicitActuatorCfg(
            joint_names_expr=["PrismaticJoint.*"],
            effort_limit_sim=25.0,
            velocity_limit_sim=0.2,
            stiffness=1800.0,
            damping=180.0,
            friction=0.0,
            armature=0.0,
        )

        # ---- Chicken object -----------------------------------------------
        self.scene.chicken = CHICKEN_CARCASS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Chicken",
            init_state=make_chicken_table_init_state(),
        )

        # ---- Diagnostic cube ----------------------------------------------
        # Small independent rigid object to test whether grasping issues come
        # from the chicken USD collision or from the gripper/teleop setup.
        self.scene.diagnostic_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/DiagnosticCube",
            spawn=sim_utils.CuboidCfg(
                size=(_DIAGNOSTIC_CUBE_SIZE, _DIAGNOSTIC_CUBE_SIZE, _DIAGNOSTIC_CUBE_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=2.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=2,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.002,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=0.8,
                    restitution=0.0,
                    friction_combine_mode="max",
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.45, 1.0)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-0.45, -0.22, TABLE_TOP_Z + _DIAGNOSTIC_CUBE_SIZE * 0.5 + 0.003),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # ---- End-effector frame at gripper fingertip ----------------------
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
                    # ~18 cm offset past wrist to approximate fingertip centre
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.18]),
                ),
            ],
        )

        # ---- Actions: 6-DOF arm (joint pos) + binary gripper --------------
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
        # All 4 prismatic joints commanded together as a single open/close signal
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["PrismaticJoint1", "PrismaticJoint2", "PrismaticJoint3", "PrismaticJoint4"],
            open_command_expr={
                "PrismaticJoint1": _GRIPPER_OPEN,
                "PrismaticJoint2": _GRIPPER_OPEN,
                "PrismaticJoint3": _GRIPPER_OPEN,
                "PrismaticJoint4": _GRIPPER_OPEN,
            },
            close_command_expr={
                "PrismaticJoint1": _GRIPPER_CLOSE,
                "PrismaticJoint2": _GRIPPER_CLOSE,
                "PrismaticJoint3": _GRIPPER_CLOSE,
                "PrismaticJoint4": _GRIPPER_CLOSE,
            },
        )

        # ---- Command target frame -----------------------------------------
        self.commands.object_pose.body_name = "wrist_3_link"


@configclass
class UR10eCustomGripperChickenLiftEnvCfg_PLAY(UR10eCustomGripperChickenLiftEnvCfg):
    """Smaller scene for visualisation / play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
