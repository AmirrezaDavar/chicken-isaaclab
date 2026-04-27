# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Squat task configuration for the Class Humanoid robot."""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import squat_mdp
from .common import (
    FOOT_BODY_NAMES,
    ClassHumanoidTaskObservationsCfg,
    ClassHumanoidTaskRewardsCfg,
    ClassHumanoidTaskSceneCfg,
    ClassHumanoidTaskTerminationsCfg,
    configure_class_humanoid_flat_scene,
    configure_class_humanoid_task_defaults,
)


@configclass
class SquatCommandsCfg:
    """Command interface for squatting."""

    target_pelvis_height = squat_mdp.UniformPelvisHeightCommandCfg(
        min_height=0.62,
        max_height=0.75,
        resampling_time_range=(1.5, 2.5),
        debug_vis=False,
    )


@configclass
class SquatRewardsCfg(ClassHumanoidTaskRewardsCfg):
    pelvis_height_tracking = RewTerm(
        func=squat_mdp.pelvis_height_error_l2,
        weight=-8.0,
        params={"command_name": "target_pelvis_height"},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES),
        },
    )


@configclass
class ClassHumanoidSquatEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Task: squatting in place with a commanded pelvis height."""

    scene: ClassHumanoidTaskSceneCfg = ClassHumanoidTaskSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ClassHumanoidTaskObservationsCfg = ClassHumanoidTaskObservationsCfg()
    commands: SquatCommandsCfg = SquatCommandsCfg()
    rewards: SquatRewardsCfg = SquatRewardsCfg()
    terminations: ClassHumanoidTaskTerminationsCfg = ClassHumanoidTaskTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        configure_class_humanoid_flat_scene(self)
        configure_class_humanoid_task_defaults(self, action_scale=0.35)

        self.terminations.base_contact = None
        self.terminations.bad_orientation.params["limit_angle"] = 0.8
        self.terminations.root_too_low.params["minimum_height"] = 0.55
        self.episode_length_s = 8.0
        self.observations.policy.target_pelvis_height = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "target_pelvis_height"}
        )
