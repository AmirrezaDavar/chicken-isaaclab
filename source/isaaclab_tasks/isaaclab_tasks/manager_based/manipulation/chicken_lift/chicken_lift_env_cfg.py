# SPDX-License-Identifier: BSD-3-Clause

"""Base environment config for UR10e + Robotiq gripper lifting a chicken carcass.

Scene layout
------------
* robot  – UR10e with Robotiq gripper (set by concrete subclass)
* ee_frame – FrameTransformer tracking the gripper tip  (set by subclass)
* chicken – passive articulation; legs/wings randomised at every reset
* table  – Seattle Lab Table from Nucleus
* plane  – infinite ground plane
* light  – dome light

MDP
---
* observations : joint_pos, joint_vel, chicken torso pos in robot frame,
                 target pos (command), last action
* rewards      : reach chicken, lift chicken, track goal, action/vel penalty
* events       : reset chicken position (uniform on table) + joint randomisation
* terminations : timeout, chicken dropped below table
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

import isaaclab.envs.mdp as mdp


##
# Scene
##


@configclass
class ChickenLiftSceneCfg(InteractiveSceneCfg):
    """Scene: UR10e robot + passive chicken articulation on a table."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    # Chicken is an articulation so its leg/wing joints are simulated.
    chicken: ArticulationCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
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


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Target pose command for where the robot should carry the chicken."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # set by subclass (e.g. "wrist_3_link")
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
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
    """UR10e arm + gripper actions.  Filled in by subclass."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Policy observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        # 6 arm joints
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # chicken torso position in robot root frame
        object_position = ObsTerm(
            func=lift_mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("chicken")},
        )
        # desired carry position (3-D)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Randomise chicken position on the table surface.
    reset_chicken_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("chicken"),
        },
    )

    # Randomise chicken leg and wing joint angles at reset.
    # joints default to 0; offset [-1.0, 1.0] rad covers most of the ±1.57 range.
    randomise_chicken_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "chicken",
                joint_names=["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
            ),
        },
    )


@configclass
class RewardsCfg:
    """Shaped rewards for pick-and-place."""

    # Dense: approach the chicken
    reaching_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={
            "std": 0.1,
            "object_cfg": SceneEntityCfg("chicken"),
        },
        weight=1.0,
    )

    # Sparse: chicken torso above table
    lifting_object = RewTerm(
        func=lift_mdp.object_is_lifted,
        params={
            "minimal_height": 0.06,
            "object_cfg": SceneEntityCfg("chicken"),
        },
        weight=15.0,
    )

    # Dense: track carry goal once lifted
    object_goal_tracking = RewTerm(
        func=lift_mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": 0.06,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("chicken"),
        },
        weight=16.0,
    )

    object_goal_tracking_fine = RewTerm(
        func=lift_mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": 0.06,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("chicken"),
        },
        weight=5.0,
    )

    # Regularisation
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Episode terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # End episode if the chicken falls below the table edge
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": -0.05,
            "asset_cfg": SceneEntityCfg("chicken"),
        },
    )


@configclass
class CurriculumCfg:
    """Gradually increase regularisation penalty during training."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10_000},
    )


##
# Top-level env config
##


@configclass
class ChickenLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract base – concrete subclass must set robot, ee_frame, chicken, arm_action,
    gripper_action, and commands.object_pose.body_name."""

    scene: ChickenLiftSceneCfg = ChickenLiftSceneCfg(num_envs=4096, env_spacing=2.5)
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
