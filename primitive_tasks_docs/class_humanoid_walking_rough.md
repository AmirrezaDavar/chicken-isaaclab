# Class Humanoid Rough Walking

## Which rough-walking variant is documented in detail

The rough-walking family has five training-oriented registrations plus one play registration:

| Gym ID | Env config class | What changes |
| --- | --- | --- |
| `Isaac-Velocity-Rough-ClassHumanoid-v0` | `ClassHumanoidRoughEnvCfg` | Base rough-walking task |
| `Isaac-Velocity-Rough-ClassHumanoid-ContactPenalty-v0` | `ClassHumanoidRoughEnvCfgContactPenalty` | Adds a non-foot contact penalty |
| `Isaac-Velocity-Rough-ClassHumanoid-BadOrientation-v0` | `ClassHumanoidRoughEnvCfgBadOrientation` | Adds an orientation-based termination |
| `Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0` | `ClassHumanoidRoughEnvCfgExtendedBaseContact` | Expands the base-contact termination bodies |
| `Isaac-Velocity-Rough-ClassHumanoid-FootLift-v0` | `ClassHumanoidRoughEnvCfgFootLift` | Inherits extended base-contact termination and adds the largest set of reward/action tweaks |
| `Isaac-Velocity-Rough-ClassHumanoid-Play-v0` | `ClassHumanoidRoughEnvCfg_PLAY` | Play-mode configuration, not a richer training objective |

No single rough variant combines every independent option. This document describes `Isaac-Velocity-Rough-ClassHumanoid-FootLift-v0` in detail because it changes the most training-time behavior, and then explicitly summarizes what the other variants add that it does not.

## Registration and entry points

| Item | Value |
| --- | --- |
| Primary Gym ID documented here | `Isaac-Velocity-Rough-ClassHumanoid-FootLift-v0` |
| Environment config class | `ClassHumanoidRoughEnvCfgFootLift` |
| RSL-RL runner config class | `ClassHumanoidRoughPPORunnerCfg` |
| SKRL config | `agents/skrl_rough_ppo_cfg.yaml` |
| Helper training script | `scripts/reinforcement_learning/rsl_rl/train_rough_walking.sh` |
| Helper recording script | `scripts/reinforcement_learning/rsl_rl/record_rough_walking_video.sh` |
| Default env count in env config | `4096` |
| Default env count in helper script | `2048` |
| Episode length | `20.0 s` |
| Policy step time | `0.02 s` |
| Steps per episode | `1000` |

Important implementation detail:

- The helper training script defaults to `TASK=Isaac-Velocity-Rough-ClassHumanoid-v0`, not the `FootLift` variant. It can still launch the other rough variants by overriding `TASK`.

## Task summary

This is a velocity-commanded locomotion task on procedurally generated rough terrain. The task uses whole-body joint-position targets and trains the policy to track a commanded planar base velocity while maintaining balance and foot behavior constraints.

There is no explicit success flag, no success-rate evaluator, and no terminal "goal reached" event. The task is expressed through command tracking rewards, locomotion-shaping rewards, reset logic, and contact-based failure conditions.

## Scene and terrain

| Item | Implemented behavior |
| --- | --- |
| Terrain type | `"generator"` |
| Terrain source | `ROUGH_TERRAINS_CFG` |
| Initial terrain level cap | `max_init_terrain_level = 5` |
| Terrain curriculum | Enabled in the base rough task through `terrain_levels_vel` |
| Height scanner | Enabled |
| Height-scanner anchor | Rebound from `base` to `base_link` in `ClassHumanoidRoughEnvCfg` |
| Height-scanner offset | `(0.0, 0.0, 20.0)` |
| Ray alignment | `"yaw"` |
| Ray grid size | `[1.6, 1.0] m` |
| Ray resolution | `0.1 m` |
| Ray count | `17 x 11 = 187` rays |
| Ray direction | Downward `(0.0, 0.0, -1.0)` |
| Contact sensor | Attached to `"{ENV_REGEX_NS}/Robot/.*"` with `history_length = 3`, `track_air_time = True` |
| Robot asset | `CLASS_HUMANOID_CFG` from `my_assets/humanoid_tuned.usd` |

## Observation space

The policy observation group is concatenated and observation corruption is enabled in training.

| Term | Shape in code | Actual content | Noise / processing |
| --- | --- | --- | --- |
| `base_lin_vel` | 3 | Root linear velocity in the root frame | Uniform additive noise in `[-0.1, 0.1]` |
| `base_ang_vel` | 3 | Root angular velocity in the root frame | Uniform additive noise in `[-0.2, 0.2]` |
| `projected_gravity` | 3 | Gravity projected into the root frame | Uniform additive noise in `[-0.05, 0.05]` |
| `velocity_commands` | 3 | Generated `base_velocity` command | No extra noise |
| `joint_pos` | all robot joints | Joint positions relative to default, with right-side joints sign-flipped into semantic coordinates | Uniform additive noise in `[-0.01, 0.01]` |
| `joint_vel` | all robot joints | Joint velocities relative to default, with the same semantic sign convention | Uniform additive noise in `[-1.5, 1.5]` |
| `actions` | matches action dimension | Last raw action | No extra noise |
| `height_scan` | 187 | Downward ray-cast height samples from the yaw-aligned scanner grid | Height values are `sensor_z - hit_z - 0.5`, then uniform noise in `[-0.1, 0.1]`, then clipped to `[-1.0, 1.0]` |

Notes:

- The semantic joint observation override is specific to this humanoid. Right-side joints are multiplied by `-1` so left/right pairs share a task-level sign convention.
- The height scan is present only in rough walking. The flat walking config removes it.

## Action space

For the detailed `FootLift` variant:

| Item | Implemented behavior |
| --- | --- |
| Action type | `JointPositionActionCfg` |
| Controlled joints | `joint_names=[".*"]`, so all articulation joints resolved by the asset are controlled |
| Scale | `0.6` in `FootLift` |
| Offset | Default joint positions from the articulation asset (`use_default_offset=True`) |
| Applied command | `processed_action = default_joint_pos + 0.6 * raw_action`, then `set_joint_position_target(...)` |
| Torque control | Not used |

Comparison note:

- The base rough variants use action scale `0.5`.
- Only `FootLift` changes that scale to `0.6`.

## Command space

The walking task uses one command term, `base_velocity`, configured as `UniformVelocityCommand`.

| Item | Implemented behavior |
| --- | --- |
| Command dimension | 3 |
| Coordinate frame | Robot base frame |
| Resampling time | Exactly `10.0 s` |
| Commanded linear-x range | `[0.0, 1.0] m/s` |
| Commanded linear-y range | `[0.0, 0.0] m/s` |
| Angular mode | Heading-command mode is enabled |
| Heading target range | `[-pi, pi]` |
| Angular command range clamp | `[-1.0, 1.0] rad/s` |
| Heading stiffness | `0.5` |
| Heading-env probability | `1.0` |
| Standing-env probability | `0.02` |
| Debug visualization | Enabled |

Actual command behavior:

- `lin_vel_x` is sampled uniformly from `[0.0, 1.0]`.
- `lin_vel_y` is always `0.0`, so lateral walking is not commanded.
- Because `heading_command=True` and `rel_heading_envs=1.0`, all non-standing environments compute `ang_vel_z` from heading error instead of using the initially sampled yaw rate directly.
- The heading-based yaw command is `clip(0.5 * wrap_to_pi(target_heading - current_heading), -1.0, 1.0)`.
- About 2% of environments are turned into standing environments at command resample time, and their entire velocity command is forced to zero.

Command-term metrics explicitly tracked by the code:

- `error_vel_xy`
- `error_vel_yaw`

These are accumulated per command interval and normalized by the maximum command duration in steps.

## Reward function for `Isaac-Velocity-Rough-ClassHumanoid-FootLift-v0`

All reward terms below are active together in the `FootLift` variant.

| Reward term | Weight | Actual computation |
| --- | --- | --- |
| `track_lin_vel_xy_exp` | `+1.0` | Exponential tracking reward for linear XY velocity in a gravity-aligned yaw frame using `std=0.5` |
| `track_ang_vel_z_exp` | `+1.0` | Exponential tracking reward for yaw rate in world frame using `std=0.5` |
| `termination_penalty` | `-200.0` | Applies on non-timeout terminations only |
| `ang_vel_xy_l2` | `-0.05` | Penalizes root angular velocity in X and Y |
| `dof_torques_l2` | `0.0` | Present but disabled |
| `dof_acc_l2` | `-1.25e-7` | Penalizes joint accelerations |
| `action_rate_l2` | `-0.003` | Penalizes action changes between consecutive steps |
| `feet_air_time` | `+0.75` | Uses `feet_air_time_positive_biped(...)` on `.*Foot_.*` with threshold `0.55` |
| `feet_slide` | `-0.35` | Penalizes horizontal foot speed while the foot is in contact |
| `flat_orientation_l2` | `-1.0` | Penalizes projected-gravity XY components |
| `dof_pos_limits` | `-1.0` | Penalizes ankle soft-limit violation magnitude on `.*_Ankle_RS00` |
| `joint_deviation_hip` | `-0.2` | Penalizes deviation from default on hip-yaw and hip-roll joints |
| `joint_deviation_arms` | `-0.2` | Penalizes deviation from default on shoulder, elbow, and wrist joints |

Explicitly disabled in this variant:

- `lin_vel_z_l2`
- `undesired_contacts`
- `joint_deviation_torso`

## Success criteria

No explicit success criterion is implemented.

What the current code rewards instead:

- accurate tracking of the commanded forward velocity and yaw behavior
- low roll/pitch angular velocity
- upright orientation
- longer single-stance behavior through the biped foot-air-time reward
- lower foot sliding
- fewer ankle-limit violations
- lower off-task arm and hip deviation

Because there is no success threshold or goal event, "successful walking" is only implicit in the reward composition and command metrics.

## Failure conditions

For the detailed `FootLift` rough variant:

| Condition | Implemented rule |
| --- | --- |
| Timeout | Episode ends at `20.0 s` |
| Base-contact termination | `illegal_contact` with threshold `1.0` on the following monitored bodies: `base_link`, `Hip_1`, `Head_1`, `HipYoke_.*`, `Shoulder_.*`, `UpBicep_.*`, `LowBicep_.*`, `Forearm_.*`, `Wrist_.*`, `UpperThigh_.*`, `LowerThigh_.*` |
| Orientation termination | Not active in `FootLift` |
| Root-height termination | Not active |

Important implementation detail:

- `FootLift` inherits `ExtendedBaseContact`, so its failure logic is stricter than the base rough task.
- The base rough task itself only monitors `base_link` for contact termination.

## Reset conditions

Episodes reset on timeout or on the active termination condition.

At reset, the code applies:

| Reset component | Implemented behavior |
| --- | --- |
| Robot root position | Default root position plus sampled offsets: `x in [-0.5, 0.5]`, `y in [-0.5, 0.5]`, `z` offset omitted, so no extra sampled `z` term |
| Robot root orientation | Additional yaw sampled in `[-3.14, 3.14]` |
| Robot root velocity | All components reset to zero |
| Joint positions | Reset exactly to the default joint positions via `position_range=(1.0, 1.0)` |
| Joint velocities | Reset exactly to zero |
| External push | Interval pushes disabled |
| Base mass randomization | Disabled |
| Base COM randomization | Disabled |
| Reset-time force/torque event | Still present on `base_link`, but inherited force/torque ranges remain zero |

Because `reset_root_state_uniform(...)` adds offsets to the default root state, and `CLASS_HUMANOID_CFG` sets the default root position to `(0.0, 0.0, 0.8)`, the actual rough-walking reset height is `0.8 m` above each environment origin.

## Evaluation metrics actually exposed by code

Explicit metrics from the velocity command term:

- `error_vel_xy`
- `error_vel_yaw`

Not implemented:

- No success counter
- No success-rate evaluator
- No thresholded "walking achieved" event

## Variant differences inside rough walking

| Variant | Difference relative to the base rough task |
| --- | --- |
| `Isaac-Velocity-Rough-ClassHumanoid-v0` | Base rough task with `base_link` contact termination only |
| `Isaac-Velocity-Rough-ClassHumanoid-ContactPenalty-v0` | Adds `undesired_contacts` reward penalty for `base_link`, head, torso/hips, thighs, shins, shoulders, upper/lower biceps, forearms, and wrists |
| `Isaac-Velocity-Rough-ClassHumanoid-BadOrientation-v0` | Adds an extra `bad_orientation(limit_angle=0.8)` termination on top of the base rough terminations |
| `Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0` | Expands the contact-termination body set to include torso, head, upper legs, and arms |
| `Isaac-Velocity-Rough-ClassHumanoid-FootLift-v0` | Inherits `ExtendedBaseContact` and changes `feet_air_time`, `feet_slide`, `action_rate_l2`, and `actions.joint_pos.scale` |
| `Isaac-Velocity-Rough-ClassHumanoid-Play-v0` | Uses a play configuration: `50` envs, `40 s` episodes, fixed `lin_vel_x=1.0`, `heading=0.0`, smaller terrain grid, no observation corruption, no reset force event, no interval pushing |

Important comparison note:

- `FootLift` is the richest single rough variant by amount of changed training behavior.
- `ContactPenalty` still adds one reward term that `FootLift` does not include.
- `BadOrientation` adds one termination that `FootLift` does not include.

## Training and runner notes

RSL-RL rough-runner defaults:

- `num_steps_per_env = 24`
- `max_iterations = 3000`
- `experiment_name = "class_humanoid_rough"`
- actor/critic hidden dims `[512, 256, 128]`

The helper rough training script currently defaults to:

- `NUM_ENVS = 2048`
- `MAX_ITERATIONS = 15000`
- `SEED = 42`
- `TASK = Isaac-Velocity-Rough-ClassHumanoid-v0`

SKRL rough config is also present in `agents/skrl_rough_ppo_cfg.yaml`.

## Source files

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/rough_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/common_mdp.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/agents/skrl_rough_ppo_cfg.yaml`
- `source/isaaclab/isaaclab/envs/mdp/commands/commands_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py`
- `source/isaaclab/isaaclab/envs/mdp/observations.py`
- `source/isaaclab/isaaclab/envs/mdp/rewards.py`
- `source/isaaclab/isaaclab/envs/mdp/terminations.py`
- `source/isaaclab_assets/isaaclab_assets/robots/class_humanoid.py`
