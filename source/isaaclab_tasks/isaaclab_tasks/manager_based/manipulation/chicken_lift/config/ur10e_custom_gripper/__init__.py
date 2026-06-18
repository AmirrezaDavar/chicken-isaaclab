# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-Chicken-UR10e-CustomGripper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10eCustomGripperChickenLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChickenLiftCustomGripperPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Chicken-UR10e-CustomGripper-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10eCustomGripperChickenLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChickenLiftCustomGripperPPORunnerCfg",
    },
    disable_env_checker=True,
)

# IK Cartesian-control variants for teleoperation / demo collection
gym.register(
    id="Isaac-Lift-Chicken-UR10e-CustomGripper-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:UR10eCustomGripperChickenLiftIKRelEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChickenLiftCustomGripperPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Chicken-UR10e-CustomGripper-IK-Rel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ik_rel_env_cfg:UR10eCustomGripperChickenLiftIKRelEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChickenLiftCustomGripperPPORunnerCfg",
    },
    disable_env_checker=True,
)

# GELLO joint-position teleoperation variants
gym.register(
    id="Isaac-Lift-Chicken-UR10e-CustomGripper-GELLO-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gello_env_cfg:UR10eCustomGripperChickenLiftGelloEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChickenLiftCustomGripperPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Chicken-UR10e-CustomGripper-GELLO-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gello_env_cfg:UR10eCustomGripperChickenLiftGelloEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChickenLiftCustomGripperPPORunnerCfg",
    },
    disable_env_checker=True,
)
