"""Configuration for custom class humanoid asset."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

CLASS_HUMANOID_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="my_assets/humanoid_tuned.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "Left_Hip_Pitch_RS04": math.radians(0.0),
            "Left_Hip_Roll_RS03": math.radians(0.0),
            "Left_Hip_Yaw_RS03": math.radians(0.0),
            "Left_Knee_RS04": math.radians(0.0),
            "Left_Ankle_RS00": math.radians(0.0),
            "Right_Hip_Pitch_RS04": math.radians(0.0),
            "Right_Hip_Roll_RS03": math.radians(0.0),
            "Right_Hip_Yaw_RS03": math.radians(0.0),
            "Right_Knee_RS04": math.radians(0.0),
            "Right_Ankle_RS00": math.radians(0.0),
            "Left_Shoulder_Pitch_RS03": math.radians(0.0),
            "Left_Shoulder_Roll_RS03": math.radians(0.0),
            "Left_Shoulder_Yaw_RS02": math.radians(0.0),
            "Left_Elbow_RS02": math.radians(0.0),
            "Left_Wrist_RS00": math.radians(0.0),
            "Right_Shoulder_Pitch_RS03": math.radians(0.0),
            "Right_Shoulder_Roll_RS03": math.radians(0.0),
            "Right_Shoulder_Yaw_RS02": math.radians(0.0),
            "Right_Elbow_RS02": math.radians(0.0),
            "Right_Wrist_RS00": math.radians(0.0),
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch_RS04",
                ".*_Hip_Roll_RS03",
                ".*_Hip_Yaw_RS03",
                ".*_Knee_RS04",
            ],
            effort_limit_sim=300.0,
            stiffness={
                "Left_Hip_Pitch_RS04": 2113.80,
                "Left_Hip_Roll_RS03": 1936.74,
                "Left_Hip_Yaw_RS03": 1827.95,
                "Left_Knee_RS04": 956.65,
                "Right_Hip_Pitch_RS04": 2113.69,
                "Right_Hip_Roll_RS03": 1983.14,
                "Right_Hip_Yaw_RS03": 1821.22,
                "Right_Knee_RS04": 979.47,
            },
            damping={
                "Left_Hip_Pitch_RS04": 0.846,
                "Left_Hip_Roll_RS03": 0.775,
                "Left_Hip_Yaw_RS03": 0.731,
                "Left_Knee_RS04": 0.383,
                "Right_Hip_Pitch_RS04": 0.845,
                "Right_Hip_Roll_RS03": 0.793,
                "Right_Hip_Yaw_RS03": 0.728,
                "Right_Knee_RS04": 0.392,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_RS00"],
            effort_limit_sim=120.0,
            stiffness={
                "Left_Ankle_RS00": 269.36,
                "Right_Ankle_RS00": 267.43,
            },
            damping={
                "Left_Ankle_RS00": 0.108,
                "Right_Ankle_RS00": 0.107,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch_RS03",
                ".*_Shoulder_Roll_RS03",
                ".*_Shoulder_Yaw_RS02",
                ".*_Elbow_RS02",
                ".*_Wrist_RS00",
            ],
            effort_limit_sim=180.0,
            stiffness={
                "Left_Shoulder_Pitch_RS03": 358.12,
                "Left_Shoulder_Roll_RS03": 209.35,
                "Left_Shoulder_Yaw_RS02": 187.04,
                "Left_Elbow_RS02": 49.34,
                "Left_Wrist_RS00": 4.93,
                "Right_Shoulder_Pitch_RS03": 358.80,
                "Right_Shoulder_Roll_RS03": 209.75,
                "Right_Shoulder_Yaw_RS02": 176.27,
                "Right_Elbow_RS02": 49.22,
                "Right_Wrist_RS00": 4.92,
            },
            damping={
                "Left_Shoulder_Pitch_RS03": 0.143,
                "Left_Shoulder_Roll_RS03": 0.084,
                "Left_Shoulder_Yaw_RS02": 0.075,
                "Left_Elbow_RS02": 0.020,
                "Left_Wrist_RS00": 0.002,
                "Right_Shoulder_Pitch_RS03": 0.144,
                "Right_Shoulder_Roll_RS03": 0.084,
                "Right_Shoulder_Yaw_RS02": 0.071,
                "Right_Elbow_RS02": 0.020,
                "Right_Wrist_RS00": 0.002,
            },
        ),
    },
)

# USD-authored dynamics/drive values are preserved.
# Use this when the robot already behaves well in USD and you want
# explicit Python-side values synchronized from USD dump.
CLASS_HUMANOID_USD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="my_assets/humanoid_moveable_default_params.usd",
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "Left_Hip_Pitch_RS04": math.radians(0.0),
            "Left_Hip_Roll_RS03": math.radians(0.0),
            "Left_Hip_Yaw_RS03": math.radians(0.0),
            "Left_Knee_RS04": math.radians(0.0),
            "Left_Ankle_RS00": math.radians(0.0),
            "Right_Hip_Pitch_RS04": math.radians(0.0),
            "Right_Hip_Roll_RS03": math.radians(0.0),
            "Right_Hip_Yaw_RS03": math.radians(0.0),
            "Right_Knee_RS04": math.radians(0.0),
            "Right_Ankle_RS00": math.radians(0.0),
            "Left_Shoulder_Pitch_RS03": math.radians(0.0),
            "Left_Shoulder_Roll_RS03": math.radians(0.0),
            "Left_Shoulder_Yaw_RS02": math.radians(0.0),
            "Left_Elbow_RS02": math.radians(0.0),
            "Left_Wrist_RS00": math.radians(0.0),
            "Right_Shoulder_Pitch_RS03": math.radians(0.0),
            "Right_Shoulder_Roll_RS03": math.radians(0.0),
            "Right_Shoulder_Yaw_RS02": math.radians(0.0),
            "Right_Elbow_RS02": math.radians(0.0),
            "Right_Wrist_RS00": math.radians(0.0),
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Hip_Pitch_RS04",
                ".*_Hip_Roll_RS03",
                ".*_Hip_Yaw_RS03",
                ".*_Knee_RS04",
            ],
            effort_limit_sim=100.0,
            stiffness={
                "Left_Hip_Pitch_RS04": 2113.80224609375,
                "Left_Hip_Roll_RS03": 1936.7353515625,
                "Left_Hip_Yaw_RS03": 1827.953857421875,
                "Left_Knee_RS04": 956.6509399414062,
                "Right_Hip_Pitch_RS04": 2113.693359375,
                "Right_Hip_Roll_RS03": 1983.1376953125,
                "Right_Hip_Yaw_RS03": 1821.2181396484375,
                "Right_Knee_RS04": 979.4730224609375,
            },
            damping={
                "Left_Hip_Pitch_RS04": 0.8455208539962769,
                "Left_Hip_Roll_RS03": 0.7746941447257996,
                "Left_Hip_Yaw_RS03": 0.7311815619468689,
                "Left_Knee_RS04": 0.3826603591442108,
                "Right_Hip_Pitch_RS04": 0.845477283000946,
                "Right_Hip_Roll_RS03": 0.793255090713501,
                "Right_Hip_Yaw_RS03": 0.7284872531890869,
                "Right_Knee_RS04": 0.39178919792175293,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_Ankle_RS00"],
            effort_limit_sim=100.0,
            stiffness={
                "Left_Ankle_RS00": 269.3608093261719,
                "Right_Ankle_RS00": 267.42694091796875,
            },
            damping={
                "Left_Ankle_RS00": 0.10774432122707367,
                "Right_Ankle_RS00": 0.10697077959775925,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_Shoulder_Pitch_RS03",
                ".*_Shoulder_Roll_RS03",
                ".*_Shoulder_Yaw_RS02",
                ".*_Elbow_RS02",
                ".*_Wrist_RS00",
            ],
            effort_limit_sim=100.0,
            stiffness={
                "Left_Shoulder_Pitch_RS03": 358.12396240234375,
                "Left_Shoulder_Roll_RS03": 209.34674072265625,
                "Left_Shoulder_Yaw_RS02": 187.03817749023438,
                "Left_Elbow_RS02": 49.3369026184082,
                "Left_Wrist_RS00": 4.926261901855469,
                "Right_Shoulder_Pitch_RS03": 358.7972412109375,
                "Right_Shoulder_Roll_RS03": 209.75108337402344,
                "Right_Shoulder_Yaw_RS02": 176.27186584472656,
                "Right_Elbow_RS02": 49.21586990356445,
                "Right_Wrist_RS00": 4.921940326690674,
            },
            damping={
                "Left_Shoulder_Pitch_RS03": 0.14324958622455597,
                "Left_Shoulder_Roll_RS03": 0.08373869955539703,
                "Left_Shoulder_Yaw_RS02": 0.07481527328491211,
                "Left_Elbow_RS02": 0.019734760746359825,
                "Left_Wrist_RS00": 0.0019705048762261868,
                "Right_Shoulder_Pitch_RS03": 0.14351889491081238,
                "Right_Shoulder_Roll_RS03": 0.08390042930841446,
                "Right_Shoulder_Yaw_RS02": 0.07050874829292297,
                "Right_Elbow_RS02": 0.019686346873641014,
                "Right_Wrist_RS00": 0.0019687761086970568,
            },
        ),
    },
)
