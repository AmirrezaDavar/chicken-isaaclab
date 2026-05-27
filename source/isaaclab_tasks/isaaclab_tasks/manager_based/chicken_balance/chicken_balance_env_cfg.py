# SPDX-License-Identifier: BSD-3-Clause

"""RL environment: teach the chicken to stand upright and balance on its legs.

The chicken is the robot.  Its 4 joints (left/right hip, left/right shoulder)
are actively controlled by the policy.  No locomotion goal — pure static balance.

Observation (17-D)
------------------
projected_gravity (3)  – how much the torso is tilting
base_ang_vel      (3)  – torso spin rate
base_lin_vel      (3)  – torso translation speed (penalised to stay still)
joint_pos_rel     (4)  – joint angles relative to default
joint_vel         (4)  – joint angular velocities

Rewards
-------
is_alive          +2.0  – constant bonus each step the chicken is not on the floor
flat_orientation  -2.0  – L2 penalty on projected gravity XY (tilting)
base_height       -1.0  – L2 penalty on height deviation from 0.35 m target
ang_vel_xy        -0.05 – penalise torso wobble
action_rate       -0.01 – penalise jerky joint commands

Termination
-----------
torso_too_low     – torso Z drops below 0.15 m (chicken has fallen)
time_out          – episode length exceeded
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp


##
# Scene
##


@configclass
class ChickenBalanceSceneCfg(InteractiveSceneCfg):
    """Flat ground + chicken.  No table, no robot arm."""

    # The chicken IS the robot in this task
    robot: ArticulationCfg = None  # set by subclass

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
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
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Gravity direction in the robot (torso) frame — tells agent how it's tilting
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        # Angular velocity of the torso
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        # Linear velocity of the torso
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")})
        # Joint angles relative to default pose
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # Joint velocities
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Joint position targets for all 4 chicken joints."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class EventCfg:
    """Randomise initial pose at each reset so the agent learns to recover."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Small random push to initial joint angles so the agent doesn't overfit to upright
    randomise_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Random external push every so often to test robustness (startup only during early training)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (0.0, 0.0),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    # Stay alive — biggest incentive to not fall
    is_alive = RewTerm(func=mdp.is_alive, weight=2.0)

    # Keep torso flat (upright): projected gravity XY should be ~0
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Keep torso at target height of 0.35 m (roughly where legs support the body)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalise rolling/pitching wobble
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Penalise jerky actions
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # Penalise excessive joint velocities
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if the torso hits the floor
    torso_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.15, "asset_cfg": SceneEntityCfg("robot")},
    )


##
# Top-level env
##


@configclass
class ChickenBalanceEnvCfg(ManagerBasedRLEnvCfg):
    scene: ChickenBalanceSceneCfg = ChickenBalanceSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005          # 200 Hz physics
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
