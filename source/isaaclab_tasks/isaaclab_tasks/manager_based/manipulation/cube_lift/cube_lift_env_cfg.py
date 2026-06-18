# SPDX-License-Identifier: BSD-3-Clause

"""Base environment config for UR10e + custom gripper lifting a rigid cube.

Simpler than the chicken task: rigid cube, no joint randomisation, standard
reach/lift/goal rewards.  Observations are 37-dim (vs 50 for chicken).
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp


@configclass
class CubeLiftSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # Flat platform for the cube to rest on (top surface at z = 0.05)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
    )

    # Cube rests on table top (table top at z=0.05, cube half-height=0.0075 → center at z=0.0575)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.0575),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.015, 0.015, 0.015),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.3, 0.3),
            pos_z=(0.2, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 6 arm + 4 gripper prismatic joints
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)           # 10
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)           # 10
        object_position = ObsTerm(                            # 3
            func=lift_mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("object")},
        )
        target_object_position = ObsTerm(                     # 7
            func=mdp.generated_commands,
            params={"command_name": "object_pose"},
        )
        actions = ObsTerm(func=mdp.last_action)               # 7
        # Total: 37

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    # Coarse approach: wide std gives gradient from ~30 cm so random exploration
    # can discover the reward. The lift bonus (×80) dominates once lifting occurs.
    reaching_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("object")},
        weight=2.0,
    )
    # Dominating lift bonus — must dwarf the approach reward to break the local min.
    lifting_object = RewTerm(
        func=lift_mdp.object_is_lifted,
        params={"minimal_height": 0.06, "object_cfg": SceneEntityCfg("object")},
        weight=80.0,
    )
    object_goal_tracking = RewTerm(
        func=lift_mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.12,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=30.0,
    )
    object_goal_tracking_fine = RewTerm(
        func=lift_mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.12,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=10.0,
    )
    # Close gripper when near the cube — breaks the hover-with-open-gripper local min.
    # Reward = proximity_to_cube * gripper_closed_fraction, so the agent must
    # both approach AND close to earn this signal.
    gripper_close_near_object = RewTerm(
        func=lift_mdp.gripper_close_near_object,
        params={
            "approach_std": 0.1,
            "gripper_open_val": 0.0,
            "gripper_close_val": -0.0093,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["PrismaticJoint.*"]),
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=10.0,
    )
    # Light regularisation — must stay small relative to the approach reward
    # so the policy doesn't learn to freeze in place to avoid the penalty.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 50_000_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 50_000_000},
    )


@configclass
class CubeLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: CubeLiftSceneCfg = CubeLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
