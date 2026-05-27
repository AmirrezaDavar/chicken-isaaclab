# SPDX-License-Identifier: BSD-3-Clause

"""ArticulationCfg for the chicken carcass pick-and-place object.

The chicken is treated as a passive articulation: its leg and wing joints are
free-floating (zero stiffness, light damping) so they respond naturally to
contact forces during grasping.  Joint poses are randomised at each episode
reset via the environment event system.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

CHICKEN_CARCASS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="my_assets/chicken/chicken/chicken.usd",
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
            rest_offset=0.005,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.45, 0.0, 0.15),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "left_hip": 0.0,
            "right_hip": 0.0,
            "left_shoulder": 0.0,
            "right_shoulder": 0.0,
        },
    ),
    # Passive joints: zero stiffness so legs/wings respond to contact freely.
    # Small damping prevents unphysical oscillation.
    actuators={
        "passive": ImplicitActuatorCfg(
            joint_names_expr=["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
            stiffness=0.0,
            damping=5.0,
            effort_limit_sim=0.0,
        ),
    },
)
"""Chicken carcass as a passive articulation for pick-and-place RL."""


CHICKEN_BALANCE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="my_assets/chicken/chicken/chicken.usd",
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
            rest_offset=0.005,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Torso spawned ~0.5 m above ground so legs reach the floor on first step
        pos=(0.0, 0.0, 0.5),
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
        # Legs: higher stiffness for load-bearing balance control
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["left_hip", "right_hip"],
            stiffness=80.0,
            damping=4.0,
            effort_limit_sim=50.0,
        ),
        # Wings: lower stiffness, used as balance arms
        "wings": ImplicitActuatorCfg(
            joint_names_expr=["left_shoulder", "right_shoulder"],
            stiffness=20.0,
            damping=2.0,
            effort_limit_sim=20.0,
        ),
    },
)
"""Chicken with active actuators for balance RL training."""
