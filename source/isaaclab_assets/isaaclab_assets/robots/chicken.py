# SPDX-License-Identifier: BSD-3-Clause

"""Asset config for the chicken carcass used in the GELLO teleop scene."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

# chicken.py lives at source/isaaclab_assets/isaaclab_assets/robots/, so five
# dirname calls reach the repository root.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    )
)
CHICKEN_CARCASS_USD = os.path.join(
    _REPO_ROOT, "my_assets", "chicken_3", "chicken_practical_real_physics_skinned.usda"
)
CHICKEN_SKIN_DRIVER_PY = os.path.join(
    _REPO_ROOT, "my_assets", "chicken_3", "drive_skinned_chicken_from_physics.py"
)


CHICKEN_CARCASS_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CHICKEN_CARCASS_USD,
        scale=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=2,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.001,
            rest_offset=0.0,
        ),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(
        pos=(-0.70, 0.20, 0.05),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
)
"""Skinned chicken USD as one rigid physical body."""
