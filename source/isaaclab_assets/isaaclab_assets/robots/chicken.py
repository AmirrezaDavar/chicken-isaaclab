# SPDX-License-Identifier: BSD-3-Clause

"""ArticulationCfg for the chicken carcass pick-and-place object.

The chicken is treated as a compliant passive articulation: its leg and wing
joints have just enough spring/damping to keep thin limbs from sagging through
the table, while still responding naturally to contact forces during grasping.
Joint poses are randomised at each episode reset via the environment event
system.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CHICKEN_CARCASS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="my_assets/chicken/chicken/chicken_with_collisions.usda",
        scale=(0.25, 0.25, 0.25),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.70, 0.20, 0.05),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_hip": 0.0,
            "right_hip": 0.0,
            "left_shoulder": 0.0,
            "right_shoulder": 0.0,
        },
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["left_hip", "right_hip"],
            stiffness=2.0,
            damping=8.0,
            effort_limit_sim=1.5,
        ),
        "wings": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder", "right_shoulder"],
            stiffness=0.5,
            damping=5.0,
            effort_limit_sim=0.5,
        ),
    },
)
"""Chicken carcass (original model) with compliant passive revolute joints."""


CHICKEN_BALANCE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="my_assets/chicken/chicken/chicken_with_collisions.usda",
        scale=(0.25, 0.25, 0.25),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.001,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Torso spawned ~0.15 m above ground (scaled down from 0.5 m)
        pos=(0.0, 0.0, 0.15),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            # Slight forward lean on both hips gives a stable base of support
            "left_hip": -0.2,
            "right_hip": 0.2,
            "left_shoulder": 0.0,
            "right_shoulder": 0.0,
        },
    ),
    actuators={
        # Legs: scaled down effort limits to match smaller body mass
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["left_hip", "right_hip"],
            stiffness=20.0,
            damping=1.0,
            effort_limit_sim=5.0,
        ),
        # Wings: used as balance arms
        "wings": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder", "right_shoulder"],
            stiffness=5.0,
            damping=0.5,
            effort_limit_sim=2.0,
        ),
    },
)
"""Chicken with active actuators for balance RL training."""
