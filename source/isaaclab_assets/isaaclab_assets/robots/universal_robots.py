# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Configuration for the Universal Robots.

The following configuration parameters are available:

* :obj:`UR10_CFG`: The UR10 arm without a gripper.
* :obj:`UR10E_ROBOTIQ_GRIPPER_CFG`: The UR10E arm with Robotiq_2f_140 gripper.
* :obj:`UR10e_ROBOTIQ_2F_85_CFG`: The UR10E arm with Robotiq 2F-85 gripper.
* :obj:`UR10e_CUSTOM_GRIPPER_CFG`: The UR10e arm with the custom 4-jaw parallel gripper from
  Universal_Robots_ROS2_Description/urdf/1_fixed.usda.

Reference: https://github.com/ros-industrial/universal_robot
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# Absolute path to the local USDA for UR10e + custom gripper.
# universal_robots.py lives at source/isaaclab_assets/isaaclab_assets/robots/
# so 5 dirname calls reach the repo root.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
_UR10E_CUSTOM_GRIPPER_USD = os.path.join(
    _REPO_ROOT, "Universal_Robots_ROS2_Description", "urdf", "1_fixed.usda"
)

##
# Configuration
##

UR10_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/UR10/ur10_instanceable.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

UR10e_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 3.141592653589793,
            "shoulder_lift_joint": -1.5707963267948966,
            "elbow_joint": 1.5707963267948966,
            "wrist_1_joint": -1.5707963267948966,
            "wrist_2_joint": -1.5707963267948966,
            "wrist_3_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    actuators={
        # 'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.6636085,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64101615,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39387691,
            friction=0.0,
            armature=0.0,
        ),
    },
)

"""Configuration of UR-10 arm using implicit actuator models."""

UR10_LONG_SUCTION_CFG = UR10_CFG.copy()
UR10_LONG_SUCTION_CFG.spawn.usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10/ur10.usd"
UR10_LONG_SUCTION_CFG.spawn.variants = {"Gripper": "Long_Suction"}
UR10_LONG_SUCTION_CFG.spawn.rigid_props.disable_gravity = True
UR10_LONG_SUCTION_CFG.init_state.joint_pos = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5707,
    "elbow_joint": 1.5707,
    "wrist_1_joint": -1.5707,
    "wrist_2_joint": 1.5707,
    "wrist_3_joint": 0.0,
}

"""Configuration of UR10 arm with long suction gripper."""

UR10_SHORT_SUCTION_CFG = UR10_LONG_SUCTION_CFG.copy()
UR10_SHORT_SUCTION_CFG.spawn.variants = {"Gripper": "Short_Suction"}

"""Configuration of UR10 arm with short suction gripper."""

UR10e_ROBOTIQ_GRIPPER_CFG = UR10e_CFG.copy()
"""Configuration of UR10e arm with Robotiq_2f_140 gripper."""
UR10e_ROBOTIQ_GRIPPER_CFG.spawn.variants = {"Gripper": "Robotiq_2f_140"}
UR10e_ROBOTIQ_GRIPPER_CFG.spawn.rigid_props.disable_gravity = True
UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos["finger_joint"] = 0.0
UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_inner_finger_pad_joint"] = 0.0
UR10e_ROBOTIQ_GRIPPER_CFG.init_state.joint_pos[".*_outer_.*_joint"] = 0.0
# the major actuator joint for gripper
UR10e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)
# the auxiliary actuator joint for gripper
UR10e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)
# the passive joints for gripper
UR10e_ROBOTIQ_GRIPPER_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_pad_joint", ".*_outer_finger_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)


UR10e_ROBOTIQ_2F_85_CFG = UR10e_CFG.copy()
"""Configuration of UR-10E arm with Robotiq_2f_140 gripper."""
UR10e_ROBOTIQ_2F_85_CFG.spawn.variants = {"Gripper": "Robotiq_2f_85"}
UR10e_ROBOTIQ_2F_85_CFG.spawn.rigid_props.disable_gravity = True
UR10e_ROBOTIQ_2F_85_CFG.init_state.joint_pos["finger_joint"] = 0.0
UR10e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_inner_finger_joint"] = 0.0
UR10e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_inner_finger_knuckle_joint"] = 0.0
UR10e_ROBOTIQ_2F_85_CFG.init_state.joint_pos[".*_outer_.*_joint"] = 0.0
# the major actuator joint for gripper
UR10e_ROBOTIQ_2F_85_CFG.actuators["gripper_drive"] = ImplicitActuatorCfg(
    joint_names_expr=["finger_joint"],  # "right_outer_knuckle_joint" is its mimic joint
    effort_limit_sim=10.0,
    velocity_limit_sim=1.0,
    stiffness=11.25,
    damping=0.1,
    friction=0.0,
    armature=0.0,
)
# enable the gripper to grasp in a parallel manner
UR10e_ROBOTIQ_2F_85_CFG.actuators["gripper_finger"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.2,
    damping=0.001,
    friction=0.0,
    armature=0.0,
)
# set PD to zero for passive joints in close-loop gripper
UR10e_ROBOTIQ_2F_85_CFG.actuators["gripper_passive"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_inner_finger_knuckle_joint", "right_outer_knuckle_joint"],
    effort_limit_sim=1.0,
    velocity_limit_sim=1.0,
    stiffness=0.0,
    damping=0.0,
    friction=0.0,
    armature=0.0,
)

"""Configuration of UR-10E arm with Robotiq 2F-85 gripper."""


UR10e_CUSTOM_GRIPPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_UR10E_CUSTOM_GRIPPER_USD,
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
        activate_contact_sensors=False,
    ),
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
            # Gripper open: all prismatic joints at 0 (upper limit)
            "PrismaticJoint1": 0.0,
            "PrismaticJoint2": 0.0,
            "PrismaticJoint3": 0.0,
            "PrismaticJoint4": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=1320.0,
            damping=72.66,
            friction=0.0,
            armature=0.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=600.0,
            damping=34.64,
            friction=0.0,
            armature=0.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=216.0,
            damping=29.39,
            friction=0.0,
            armature=0.0,
        ),
        # 4 parallel prismatic fingers — controlled together as open/close
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["PrismaticJoint.*"],
            effort_limit_sim=35.0,
            velocity_limit_sim=0.5,
            stiffness=2500.0,
            damping=120.0,
            friction=0.0,
            armature=0.0,
        ),
    },
)
"""UR10e arm with custom 4-jaw parallel gripper (PrismaticJoint1-4, range [−9.3 mm, 0])."""
