# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ChickenLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32        # more steps → better advantage estimates for long horizon
    max_iterations = 5000         # sequential task needs more training than single-grasp
    save_interval = 100
    experiment_name = "ur10e_chicken_seq_grasp"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,   # running mean/std normalisation — prevents raw velocity spikes from diverging the network
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,             # slightly lower discount keeps value targets small and stable
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,      # tighter gradient clip catches early instability
    )
