# SPDX-License-Identifier: BSD-3-Clause

"""Base environment config for UR10e + custom gripper lifting a chicken carcass.

Scene layout
------------
* robot   – UR10e with gripper (set by concrete subclass)
* table   – kinematic table in front of the robot
* shackle – kinematic shackle above the far end of the table
* ee_frame – FrameTransformer tracking the gripper tip (set by subclass)
* chicken  – passive articulation; legs/wings randomised at every reset
* plane   – infinite ground plane
* light   – dome light

Two reward suites are provided:
  RewardsCfg              – generic reach/lift/goal tracking (original).
  SequentialGraspRewardsCfg – phased rewards that guide the policy to first
                               close the left jaw on the left leg, then the
                               right jaw on the right leg, then lift.

Two action suites:
  ActionsCfg              – single unified gripper_action (original).
  SequentialActionsCfg    – independent gripper_left_action + gripper_right_action.
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
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
from isaaclab_tasks.manager_based.manipulation.chicken_lift import mdp as chicken_mdp

import isaaclab.envs.mdp as mdp


##
# Scene
##

TABLE_CENTER_X = -0.60
TABLE_CENTER_Y = 0.0
TABLE_TOP_Z = 0.6205
# Measured from the wrapped chicken asset after the 0.25 spawn scale.
CHICKEN_ROOT_TO_LOWEST_VISUAL_Z = 0.12635
CHICKEN_TABLE_CLEARANCE_Z = 0.005
CHICKEN_ROOT_ABOVE_TABLE_Z = CHICKEN_ROOT_TO_LOWEST_VISUAL_Z + CHICKEN_TABLE_CLEARANCE_Z
CHICKEN_SPAWN_Z = TABLE_TOP_Z + CHICKEN_ROOT_ABOVE_TABLE_Z
CHICKEN_LIFT_MIN_HEIGHT = CHICKEN_SPAWN_Z + 0.08
CHICKEN_DROP_MIN_HEIGHT = TABLE_TOP_Z - 0.08


def make_chicken_table_init_state() -> ArticulationCfg.InitialStateCfg:
    """Default chicken pose centered on the table used by the UR10e scenes."""

    return ArticulationCfg.InitialStateCfg(
        pos=(TABLE_CENTER_X, TABLE_CENTER_Y, CHICKEN_SPAWN_Z),
        rot=(0.707, 0.0, 0.0, 0.707),
        joint_pos={
            "left_hip": 0.0,
            "right_hip": 0.0,
            "left_shoulder": 0.0,
            "right_shoulder": 0.0,
        },
    )


@configclass
class ChickenLiftSceneCfg(InteractiveSceneCfg):
    """Scene: UR10e robot + passive chicken articulation on a table."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING

    chicken: ArticulationCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        # Keep the physical floor at world z=0. The chicken drop termination
        # also assumes this height, so moving the plane lower makes the object
        # fall through the visible grid and reset repeatedly.
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
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
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,  # set True to see the black goal-pose cylinder
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.3, 0.3),
            pos_z=(0.2, 0.5),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    """Single unified gripper action (original, for backward-compatible configs)."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg | None = None


@configclass
class SequentialActionsCfg:
    """Independent left-jaw and right-jaw actions for sequential grasping."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_left_action: mdp.BinaryJointPositionActionCfg = MISSING
    gripper_right_action: mdp.BinaryJointPositionActionCfg = MISSING


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Policy observations (works for both single-jaw and dual-jaw configs)."""

    @configclass
    class PolicyCfg(ObsGroup):
        # 6 arm + 4 gripper prismatic joints
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # chicken torso in robot root frame
        object_position = ObsTerm(
            func=lift_mdp.object_position_in_robot_root_frame,
            params={"object_cfg": SceneEntityCfg("chicken")},
        )
        # both leg positions in robot root frame (6D)
        chicken_legs = ObsTerm(
            func=chicken_mdp.chicken_legs_in_robot_frame,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        # chicken body orientation in robot frame (4D quaternion)
        # lets the policy distinguish left/right leg regardless of how the chicken landed
        chicken_orient = ObsTerm(
            func=chicken_mdp.chicken_orientation,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        # chicken root linear velocity (3D)
        # Clipped to ±5 m/s: physics glitches can spike velocity to 100+ m/s without this,
        # feeding a huge value into the network and causing value loss → inf.
        chicken_vel = ObsTerm(func=chicken_mdp.chicken_root_velocity, clip=(-5.0, 5.0))
        # desired carry position
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Reset events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_chicken_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # These offsets are added to the chicken's default root pose, which
            # the UR10e configs place slightly above the tabletop.
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.12, 0.12), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("chicken"),
        },
    )

    randomise_chicken_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.25, 0.25),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg(
                "chicken",
                joint_names=["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
            ),
        },
    )


# ---------------------------------------------------------------------------
# Rewards – original generic version
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Shaped rewards for generic pick-and-place (single unified gripper)."""

    reaching_object = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={"std": 0.1, "object_cfg": SceneEntityCfg("chicken")},
        weight=1.0,
    )

    lifting_object = RewTerm(
        func=lift_mdp.object_is_lifted,
        params={"minimal_height": CHICKEN_LIFT_MIN_HEIGHT, "object_cfg": SceneEntityCfg("chicken")},
        weight=15.0,
    )

    object_goal_tracking = RewTerm(
        func=lift_mdp.object_goal_distance,
        params={
            "std": 0.3,
            "minimal_height": CHICKEN_LIFT_MIN_HEIGHT,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("chicken"),
        },
        weight=16.0,
    )

    object_goal_tracking_fine = RewTerm(
        func=lift_mdp.object_goal_distance,
        params={
            "std": 0.05,
            "minimal_height": CHICKEN_LIFT_MIN_HEIGHT,
            "command_name": "object_pose",
            "object_cfg": SceneEntityCfg("chicken"),
        },
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Rewards – sequential two-jaw version
# ---------------------------------------------------------------------------


@configclass
class SequentialGraspRewardsCfg:
    """Phased rewards for sequential left-then-right jaw grasping.

    Reward flow:
      1. approaching_chicken  – dense EE-to-body approach signal (always on)
      2. left_jaw_closing     – continuous reward for closing the left jaw
      3. left_leg_grasped     – proximity-to-left-leg × left-jaw-closure
      4. right_jaw_gated      – right-jaw signal, scaled by left_grasped gate
      5. lifting_gated        – sparse lift reward, enabled once left jaw ≥ 30% closed
      6. goal_tracking_gated  – carry reward, enabled once left jaw ≥ 30% closed
      7. action_rate / joint_vel regularisation (curriculum-ramp to higher weights)
    """

    # Phase 0: keep the EE approaching the chicken body
    approaching_chicken = RewTerm(
        func=lift_mdp.object_ee_distance,
        params={"std": 0.12, "object_cfg": SceneEntityCfg("chicken")},
        weight=1.0,
    )

    # Phase 1a: reward the policy for closing the left jaw at all
    left_jaw_closing = RewTerm(
        func=chicken_mdp.left_jaw_closing_reward,
        weight=2.0,
    )

    # Phase 1b: proximity to left leg × left jaw closure → peak reward when grasping
    left_leg_grasped = RewTerm(
        func=chicken_mdp.left_leg_grasped_reward,
        params={"std": 0.06},
        weight=12.0,
    )

    # Phase 2: right jaw on right leg, gated by left_grasped
    right_jaw_gated = RewTerm(
        func=chicken_mdp.right_jaw_gated_reward,
        params={"std": 0.06},
        weight=10.0,
    )

    # Phase 3: lift the chicken — only rewarded once the left jaw is active
    lifting_gated = RewTerm(
        func=chicken_mdp.chicken_lifted_gated,
        params={"minimal_height": CHICKEN_LIFT_MIN_HEIGHT, "gate_threshold": 0.3},
        weight=20.0,
    )

    # Bonus: carry the chicken to the commanded pose (only useful after grasping)
    goal_tracking_gated = RewTerm(
        func=chicken_mdp.chicken_goal_tracking_gated,
        params={
            "std": 0.3,
            "minimal_height": CHICKEN_LIFT_MIN_HEIGHT,
            "command_name": "object_pose",
            "gate_threshold": 0.3,
        },
        weight=8.0,
    )

    # Regularisation (curriculum ramps these up during training)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Episode terminations."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": CHICKEN_DROP_MIN_HEIGHT, "asset_cfg": SceneEntityCfg("chicken")},
    )


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


@configclass
class CurriculumCfg:
    """Gradually increase regularisation penalty during training.

    num_steps is in environment steps.  With 1024 envs × 32 steps/iter = 32768 steps/iter,
    50_000_000 steps ≈ 1525 iterations — a gentle ramp across ~30% of training.
    Final weights are kept small (-1e-3) so they never dominate the task reward.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-3, "num_steps": 50_000_000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 50_000_000},
    )


##
# Top-level env configs
##


@configclass
class ChickenLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract base for original single-gripper chicken lift tasks."""

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


@configclass
class ChickenSequentialGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract base for sequential two-jaw grasping tasks.

    Subclass must set: scene.robot, scene.ee_frame, scene.chicken,
    actions.arm_action, actions.gripper_left_action,
    actions.gripper_right_action, commands.object_pose.body_name.
    """

    scene: ChickenLiftSceneCfg = ChickenLiftSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: SequentialActionsCfg = SequentialActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: SequentialGraspRewardsCfg = SequentialGraspRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0   # longer than generic — more phases to complete
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
