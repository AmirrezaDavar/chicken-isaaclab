# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ClassHumanoidRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "class_humanoid_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ClassHumanoidFlatPPORunnerCfg(ClassHumanoidRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1000
        self.experiment_name = "class_humanoid_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]


@configclass
class ClassHumanoidTaskPPORunnerCfg(ClassHumanoidFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 2000
        self.experiment_name = "class_humanoid_task"
        self.policy.actor_hidden_dims = [256, 256, 128]
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.algorithm.entropy_coef = 0.005
        self.algorithm.learning_rate = 7.5e-4


@configclass
class ClassHumanoidSquatPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_squat"
        self.max_iterations = 1800


@configclass
class ClassHumanoidStepPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_step"
        self.max_iterations = 2200


@configclass
class ClassHumanoidStepAltPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_step"
        self.max_iterations = 2200


@configclass
class ClassHumanoidStepAllPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_step"
        self.max_iterations = 2200


@configclass
class ClassHumanoidStepShapingPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_step"
        self.max_iterations = 2200


@configclass
class ClassHumanoidReachDepthPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_reach_depth"
        self.max_iterations = 4000


@configclass
class ClassHumanoidPushButtonPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_push_button"
        self.max_iterations = 3000


@configclass
class ClassHumanoidShootBallPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_shoot_ball"
        self.max_iterations = 3000


@configclass
class ClassHumanoidReachDepthCameraPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_reach_depth_camera"
        self.max_iterations = 4000


@configclass
class ClassHumanoidReachLeftFixedPPORunnerCfg(ClassHumanoidTaskPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "class_humanoid_reach_left_fixed"
        self.max_iterations = 2600
