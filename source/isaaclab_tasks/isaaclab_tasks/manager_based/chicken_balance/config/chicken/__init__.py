# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Balance-Chicken-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ChickenFlatBalanceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChickenBalancePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Balance-Chicken-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:ChickenFlatBalanceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChickenBalancePPORunnerCfg",
    },
    disable_env_checker=True,
)
