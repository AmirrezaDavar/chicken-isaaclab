# SPDX-License-Identifier: BSD-3-Clause

"""UR10e + Robotiq 2F-85 configuration for the chicken pick-and-place task.

This file wires together:
  * UR10e_ROBOTIQ_2F_85_CFG  – robot already defined in this project
  * CHICKEN_CARCASS_CFG      – passive articulation from my_assets
  * ChickenLiftEnvCfg        – base scene / MDP
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.chicken_lift.chicken_lift_env_cfg import ChickenLiftEnvCfg

from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG  # isort: skip
from isaaclab_assets.robots.chicken import CHICKEN_CARCASS_CFG  # isort: skip


@configclass
class UR10eChickenLiftEnvCfg(ChickenLiftEnvCfg):
    """UR10e + Robotiq 2F-85 lifting a chicken carcass."""

    def __post_init__(self):
        super().__post_init__()

        # ---- Robot -------------------------------------------------------
        self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=UR10e_ROBOTIQ_2F_85_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=1,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005, rest_offset=0.0
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={
                    # arm at a neutral overhead pose
                    "shoulder_pan_joint": 0.0,
                    "shoulder_lift_joint": -1.5708,
                    "elbow_joint": 1.5708,
                    "wrist_1_joint": -1.5708,
                    "wrist_2_joint": -1.5708,
                    "wrist_3_joint": 0.0,
                    # gripper open
                    "finger_joint": 0.0,
                },
            ),
        )

        # Override gripper actuators for better contact stability on soft objects
        self.scene.robot.actuators["gripper_drive"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=1.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
            armature=0.0,
        )
        self.scene.robot.actuators["gripper_finger"] = ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=10.0,
            stiffness=10.0,
            damping=0.05,
            friction=0.0,
            armature=0.0,
        )

        # ---- Chicken object -----------------------------------------------
        self.scene.chicken = CHICKEN_CARCASS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Chicken"
        )

        # ---- End-effector frame tracking gripper tip ----------------------
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_3_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.13]),
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
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            open_command_expr={"finger_joint": 0.0},
            close_command_expr={"finger_joint": 0.8},
        )

        # ---- Command target frame -----------------------------------------
        self.commands.object_pose.body_name = "wrist_3_link"


@configclass
class UR10eChickenLiftEnvCfg_PLAY(UR10eChickenLiftEnvCfg):
    """Smaller scene for visualisation / play."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
