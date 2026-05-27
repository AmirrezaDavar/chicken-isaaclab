# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.chicken_balance.chicken_balance_env_cfg import ChickenBalanceEnvCfg
from isaaclab_assets.robots.chicken import CHICKEN_BALANCE_CFG  # isort: skip


@configclass
class ChickenFlatBalanceEnvCfg(ChickenBalanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = CHICKEN_BALANCE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ChickenFlatBalanceEnvCfg_PLAY(ChickenFlatBalanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # No random pushes during play so you can watch it balance
        self.events.push_robot = None
