# Class Humanoid Flat Walking

## Which flat-walking variant is documented in detail

The flat-walking family has two registrations:

| Gym ID | Env config class | What changes |
| --- | --- | --- |
| `Isaac-Velocity-Flat-ClassHumanoid-v0` | `ClassHumanoidFlatEnvCfg` | Base flat-walking task |
| `Isaac-Velocity-Flat-ClassHumanoid-Play-v0` | `ClassHumanoidFlatEnvCfg_PLAY` | Play-mode configuration |

This document describes `Isaac-Velocity-Flat-ClassHumanoid-v0` in detail and summarizes the play-mode changes afterward.

## Registration and entry points

| Item | Value |
| --- | --- |
| Primary Gym ID documented here | `Isaac-Velocity-Flat-ClassHumanoid-v0` |
| Environment config class | `ClassHumanoidFlatEnvCfg` |
| RSL-RL runner config class | `ClassHumanoidFlatPPORunnerCfg` |
| SKRL config | `agents/skrl_flat_ppo_cfg.yaml` |
| Dedicated helper training script | None found in the repository |
| Default env count in env config | `4096` |
| Episode length | `20.0 s` |
| Policy step time | `0.02 s` |
| Steps per episode | `1000` |

## Task summary

Flat walking inherits the `class_humanoid` rough-walking task and then removes terrain roughness, terrain curriculum, and the height-scan observation. It keeps the same command-tracking locomotion objective but on a plane.

There is no explicit success flag or success-rate metric. The task remains reward-driven.

## Scene and terrain

| Item | Implemented behavior |
| --- | --- |
| Terrain type | `"plane"` |
| Terrain generator | Disabled |
| Terrain curriculum | Disabled |
| Height scanner | Removed |
| Height-scan observation | Removed |
| Contact sensor | Attached to `"{ENV_REGEX_NS}/Robot/.*"` with `history_length = 3`, `track_air_time = True` |
| Robot asset | `CLASS_HUMANOID_CFG` from `my_assets/humanoid_tuned.usd` |

All other inherited flat-walking behavior comes from `ClassHumanoidRoughEnvCfg` unless overridden here.

## Observation space

The flat policy observation group is concatenated and corruption is enabled in training.

| Term | Shape in code | Actual content | Noise / processing |
| --- | --- | --- | --- |
| `base_lin_vel` | 3 | Root linear velocity in the root frame | Uniform additive noise in `[-0.1, 0.1]` |
| `base_ang_vel` | 3 | Root angular velocity in the root frame | Uniform additive noise in `[-0.2, 0.2]` |
| `projected_gravity` | 3 | Gravity projected into the root frame | Uniform additive noise in `[-0.05, 0.05]` |
| `velocity_commands` | 3 | Generated `base_velocity` command | No extra noise |
| `joint_pos` | all robot joints | Joint positions relative to default, with semantic sign conversion for right-side joints | Uniform additive noise in `[-0.01, 0.01]` |
| `joint_vel` | all robot joints | Joint velocities relative to default, with the same semantic sign convention | Uniform additive noise in `[-1.5, 1.5]` |
| `actions` | matches action dimension | Last raw action | No extra noise |

Compared to rough walking:

- `height_scan` is removed entirely.

## Action space

| Item | Implemented behavior |
| --- | --- |
| Action type | `JointPositionActionCfg` |
| Controlled joints | `joint_names=[".*"]`, so all articulation joints resolved by the asset are controlled |
| Scale | `0.5` |
| Offset | Default joint positions from the articulation asset (`use_default_offset=True`) |
| Applied command | `processed_action = default_joint_pos + 0.5 * raw_action`, then `set_joint_position_target(...)` |
| Torque control | Not used |

## Command space

Flat walking inherits the same `base_velocity` command configuration from rough walking.

| Item | Implemented behavior |
| --- | --- |
| Command class | `UniformVelocityCommand` |
| Command dimension | 3 |
| Coordinate frame | Robot base frame |
| Resampling time | Exactly `10.0 s` |
| Linear-x range | `[0.0, 1.0] m/s` |
| Linear-y range | `[0.0, 0.0] m/s` |
| Heading-command mode | Enabled |
| Heading target range | `[-pi, pi]` |
| Angular command clamp | `[-1.0, 1.0] rad/s` |
| Standing-env probability | `0.02` |
| Heading-env probability | `1.0` |

As in rough walking:

- `lin_vel_y` is always zero.
- yaw command is computed from heading error for non-standing environments.
- standing environments have the full velocity command forced to zero.

Command-term metrics explicitly tracked by the code:

- `error_vel_xy`
- `error_vel_yaw`

## Reward function for `Isaac-Velocity-Flat-ClassHumanoid-v0`

Flat walking inherits the rough `ClassHumanoidRewards` and then changes the foot-air-time term.

| Reward term | Weight | Actual computation |
| --- | --- | --- |
| `track_lin_vel_xy_exp` | `+1.0` | Exponential tracking reward for linear XY velocity in a gravity-aligned yaw frame using `std=0.5` |
| `track_ang_vel_z_exp` | `+1.0` | Exponential tracking reward for yaw rate in world frame using `std=0.5` |
| `termination_penalty` | `-200.0` | Applies on non-timeout terminations only |
| `ang_vel_xy_l2` | `-0.05` | Penalizes root angular velocity in X and Y |
| `dof_torques_l2` | `0.0` | Present but disabled |
| `dof_acc_l2` | `-1.25e-7` | Penalizes joint accelerations |
| `action_rate_l2` | `-0.005` | Penalizes action changes |
| `feet_air_time` | `+1.0` | Uses `feet_air_time_positive_biped(...)` on `.*Foot_.*` with threshold `0.6` |
| `feet_slide` | `-0.25` | Penalizes horizontal foot speed while the foot is in contact |
| `flat_orientation_l2` | `-1.0` | Penalizes projected-gravity XY components |
| `dof_pos_limits` | `-1.0` | Penalizes ankle soft-limit violation magnitude on `.*_Ankle_RS00` |
| `joint_deviation_hip` | `-0.2` | Penalizes deviation from default on hip-yaw and hip-roll joints |
| `joint_deviation_arms` | `-0.2` | Penalizes deviation from default on shoulder, elbow, and wrist joints |

Explicitly disabled in flat walking:

- `lin_vel_z_l2`
- `undesired_contacts`
- `joint_deviation_torso`

Compared to the base rough task:

- `feet_air_time.weight` is increased from `0.25` to `1.0`
- `feet_air_time.threshold` is increased from `0.4` to `0.6`

## Success criteria

No explicit success criterion is implemented.

What the current code rewards instead:

- accurate tracking of commanded planar velocity
- upright orientation
- lower roll/pitch angular velocity
- more pronounced single-stance stepping through the biped foot-air-time term
- lower foot sliding
- fewer ankle-limit violations
- less off-task arm and hip deviation

## Failure conditions

For the base flat-walking task:

| Condition | Implemented rule |
| --- | --- |
| Timeout | Episode ends at `20.0 s` |
| Base-contact termination | `illegal_contact` on `base_link` with threshold `1.0` |
| Orientation termination | Not active |
| Root-height termination | Not active |

Unlike rough `FootLift`, flat walking does not expand the contact-termination body set.

## Reset conditions

Episodes reset on timeout or on the active termination condition.

At reset, the code applies:

| Reset component | Implemented behavior |
| --- | --- |
| Robot root position | Default root position plus sampled offsets: `x in [-0.5, 0.5]`, `y in [-0.5, 0.5]`, `z` offset omitted |
| Robot root orientation | Additional yaw sampled in `[-3.14, 3.14]` |
| Robot root velocity | All components reset to zero |
| Joint positions | Reset exactly to the default joint positions |
| Joint velocities | Reset exactly to zero |
| Interval pushes | Disabled |
| Base mass randomization | Disabled |
| Base COM randomization | Disabled |
| Reset-time force/torque event | Still present on `base_link`, but inherited force/torque ranges remain zero |

Because `CLASS_HUMANOID_CFG` sets the default root position to `(0.0, 0.0, 0.8)`, the actual flat-walking reset height is `0.8 m` above each environment origin.

## Evaluation metrics actually exposed by code

Explicit metrics from the velocity command term:

- `error_vel_xy`
- `error_vel_yaw`

Not implemented:

- No success counter
- No success-rate evaluator
- No thresholded "walking achieved" event

## Flat play-mode differences

`Isaac-Velocity-Flat-ClassHumanoid-Play-v0` inherits the base flat task and changes only the following:

| Item | Play-mode behavior |
| --- | --- |
| Number of environments | `50` |
| Environment spacing | `2.5` |
| Observation corruption | Disabled |
| Reset-time force/torque event | Disabled |
| Interval pushes | Disabled |

Unlike rough play mode, flat play mode does not change the command ranges or episode length in this file.

## Runner notes

RSL-RL flat-runner defaults:

- `max_iterations = 1000`
- `experiment_name = "class_humanoid_flat"`
- actor/critic hidden dims `[128, 128, 128]`

SKRL flat config is present in `agents/skrl_flat_ppo_cfg.yaml`.

## Source files

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/flat_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/rough_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/primitive_mdp.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/agents/skrl_flat_ppo_cfg.yaml`
- `source/isaaclab/isaaclab/envs/mdp/commands/commands_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py`
- `source/isaaclab/isaaclab/envs/mdp/observations.py`
- `source/isaaclab/isaaclab/envs/mdp/rewards.py`
- `source/isaaclab/isaaclab/envs/mdp/terminations.py`
- `source/isaaclab_assets/isaaclab_assets/robots/class_humanoid.py`
