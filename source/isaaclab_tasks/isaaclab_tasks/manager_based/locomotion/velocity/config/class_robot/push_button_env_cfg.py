# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Task F — Push a Button: composes depth sensing + arm reaching."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from .reach_env_cfg import ClassHumanoidReachDepthEnvCfg


@configclass
class ClassHumanoidPushButtonEnvCfg(ClassHumanoidReachDepthEnvCfg):
    """Task F.1 — Push a Button.

    Composes two primitives in sequence:
      1. Depth camera localises the button (observation: target_pos_base_from_depth).
      2. Right arm servo-tracks the button and presses it (reaching primitive).

    Domain randomisation adds varied friction and robot mass so the policy must be
    robust to different surface/payload conditions.
    """

    def __post_init__(self):
        super().__post_init__()

        # Randomise ground / robot body friction each reset
        self.events.randomize_friction = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.4, 1.2),
                "dynamic_friction_range": (0.3, 1.0),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 16,
            },
        )

        # Randomise robot base mass (+/- 3 kg) to simulate payload variation
        self.events.randomize_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (-3.0, 3.0),
                "operation": "add",
            },
        )
