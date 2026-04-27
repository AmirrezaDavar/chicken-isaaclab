# Class Humanoid Primitive Step

## Which step variant is documented in detail

The step family has five registered environment IDs:

| Gym ID | Env config class | What changes |
| --- | --- | --- |
| `Isaac-Primitive-Step-v0` | `ClassHumanoidPrimitiveStepEnvCfg` | Base step task |
| `Isaac-Primitive-Step-Alt-v0` | `ClassHumanoidPrimitiveStepAltEnvCfg` | Replaces the reward set with `StepAlternatingRewardsCfg` |
| `Isaac-Primitive-Step-All-v0` | `ClassHumanoidPrimitiveStepAllEnvCfg` | Adds the swing-knee range reward on top of the Alt reward set |
| `Isaac-Primitive-Step-Shaping-v0` | `ClassHumanoidPrimitiveStepShapingEnvCfg` | Adds the most shaping terms and is the most feature-rich variant |
| `Isaac-Primitive-Step-GeomTerm-v0` | `ClassHumanoidPrimitiveStepGeomTermEnvCfg` | Uses geometric fall checks instead of base-contact termination |

This document describes `Isaac-Primitive-Step-Shaping-v0` in detail, because it is the most feature-rich step variant currently implemented. Differences from the other step variants are summarized near the end.

## Registration and entry points

| Item | Value |
| --- | --- |
| Primary Gym ID documented here | `Isaac-Primitive-Step-Shaping-v0` |
| Environment config class | `ClassHumanoidPrimitiveStepShapingEnvCfg` |
| PPO runner config class | `ClassHumanoidPrimitiveStepShapingPPORunnerCfg` |
| Helper training script | `scripts/reinforcement_learning/rsl_rl/train_primitive_step_shaping.sh` |
| Default env count in env config | `2048` |
| Default env count in helper script | `2048` |
| Episode length | `10.0 s` |
| Policy step time | `0.02 s` |
| Steps per episode | `500` |

## Task summary

This is an in-place stepping task. The code does not command forward walking. Instead, it alternates the swing foot and asks the selected foot to move toward a lateral target that stays directly under the pelvis in the forward/backward axis.

There is no explicit success condition, no success termination, and no success-rate metric. The task is implemented through alternating foot commands, target-foot commands, reward shaping, and failure/reset logic.

## Observation space

The policy observation group is concatenated and corruption is enabled.

| Term | Shape in code | Actual content | Noise / processing |
| --- | --- | --- | --- |
| `base_lin_vel` | 3 | Root linear velocity | Uniform additive noise in `[-0.05, 0.05]` |
| `base_ang_vel` | 3 | Root angular velocity | Uniform additive noise in `[-0.05, 0.05]` |
| `base_rpy` | 3 | Root roll, pitch, yaw | Uniform additive noise in `[-0.02, 0.02]` |
| `joint_pos` | all robot joints | Joint positions relative to default, with right-side joints sign-flipped into semantic coordinates | Uniform additive noise in `[-0.01, 0.01]` |
| `joint_vel` | all robot joints | Joint velocities relative to default, with the same semantic sign convention | Uniform additive noise in `[-0.15, 0.15]` |
| `foot_contacts` | 2 | Binary contact indicators for left and right feet | Contact threshold `1.0` |
| `actions` | matches action dimension | Last raw action | No extra noise |
| `swing_foot` | 1 | Current swing-foot command from the command manager | No extra noise |
| `target_foot_pos_xy` | 2 | Current target foot position command in the robot base frame | No extra noise |
| `base_xy_from_reset` | 2 | Base XY displacement from the episode-reset anchor in each environment frame | No extra noise |

Notes:

- The step tasks do not use a camera or any depth observation.
- `base_xy_from_reset` is not a built-in Isaac Lab term. It is a custom manager term that stores the base XY position at reset and returns the displacement from that anchor.
- The `actions` observation is the raw network output before scale and default-joint offset are applied.

## Action space

| Item | Implemented behavior |
| --- | --- |
| Action type | `JointPositionActionCfg` |
| Controlled joints | `joint_names=[".*"]`, so all resolved articulation joints are controlled |
| Scale | `0.45` |
| Offset | Default joint positions from the articulation asset (`use_default_offset=True`) |
| Applied command | `processed_action = default_joint_pos + 0.45 * raw_action`, then `set_joint_position_target(...)` |
| Torque control | Not used |

## Command space

Two command terms drive this task.

### 1. `swing_foot`

| Item | Implemented behavior |
| --- | --- |
| Command class | `AlternatingFootCommand` |
| Values | `+1.0` or `-1.0` |
| Initial phase | Starts with `+1.0` because `start_with_right=True` |
| Resampling time | Uniformly sampled in `[0.45, 0.75] s` |
| Reset behavior | On the first resample after reset, the phase is forced back to the configured default instead of toggling from stale episode state |
| Metric exposed by the command term | `swing_ratio_right = 1.0` when the command is positive, else `0.0` |

Actual leg selection rule:

- Positive command means the right foot is the selected swing foot.
- Non-positive command means the left foot is the selected swing foot.

### 2. `target_foot_pos_xy`

| Item | Implemented behavior |
| --- | --- |
| Command class | `StepTargetFootCommand` |
| Coordinate frame | Robot base frame |
| `x` range | Fixed to `0.0` |
| `y` magnitude range | Uniform in `[0.08, 0.14]` |
| Sign of `y` | Tied to the swing foot: right swing gives negative `y`, left swing gives positive `y` |
| Independent timer | Disabled in practice by `resampling_time_range=(1e6, 1e6)` |
| Actual update trigger | The target is re-sampled when the swing-foot phase changes |
| Metric exposed by the command term | `target_step_xy_norm = norm([x, y])` |

This means the step target is purely lateral. The code intentionally keeps the forward/backward target fixed at zero for "true in-place stepping."

## Reward function for `Isaac-Primitive-Step-Shaping-v0`

All terms below are active together in the shaping variant.

### Shared stability and smoothness terms

| Reward term | Weight | Actual computation |
| --- | --- | --- |
| `termination_penalty` | `-100.0` | Applies on non-timeout terminations only |
| `torso_tilt_l2` | `-2.0` | Sum of squared projected-gravity XY components |
| `action_rate_l2` | `-0.01` | Squared action change between consecutive steps |
| `joint_vel_l2` | `-2.0e-4` | Sum of squared joint velocities |

### Step-target and leg-phase terms

| Reward term | Weight | Actual computation |
| --- | --- | --- |
| `step_target_reward` | `+4.0` | `1 - tanh(step_error / 0.06)` for the currently selected swing foot, multiplied by `(1 - swing_contact) * support_contact` |
| `swing_knee_min_flex` | `+1.2` | Linearly ramps from 0 to 1 as selected swing-knee semantic flexion moves from `0.0` to `0.60 rad` |
| `swing_knee_angle_range` | `+0.4` | Returns 1 inside the semantic flexion range `[0.60, 1.30] rad`, then decays with `tanh` outside |
| `support_knee_straight` | `+1.0` | Rewards keeping the support knee at or below `0.35 rad` semantic flexion |
| `swing_leg_shortening` | `+1.0` | Rewards the swing leg being at least `0.08 m` shorter than the support leg, using hip-to-foot distance on each side |
| `feet_air_time` | `+3.0` | Uses `feet_air_time_positive_biped(...)`; because the command is always `+1` or `-1`, its internal "command magnitude > 0.1" gate is effectively always active |
| `alternating_contact_pattern` | `+3.0` | Gives 1 when the commanded swing foot is off the ground and the support foot is in contact |
| `swing_foot_contact` | `-1.5` | Penalizes the commanded swing foot being in contact |
| `support_foot_slip` | `-0.3` | Penalizes horizontal speed of the support foot while it is in contact |
| `feet_lateral_order` | `-1.2` | Penalizes the left foot no longer being sufficiently left of the right foot in the base frame |

### In-place behavior terms

| Reward term | Weight | Actual computation |
| --- | --- | --- |
| `base_lin_vel_xy` | `-0.3` | Penalizes squared world-frame base linear velocity in XY |
| `base_ang_vel_z` | `-0.2` | Penalizes squared world-frame yaw rate |
| `base_reset_position` | `-3.0` | Penalizes base XY drift from the reset anchor once it exceeds a `0.06 m` deadband |
| `base_reset_outward_vel` | `-1.5` | Penalizes only outward radial motion away from the reset anchor, not motion back toward it |
| `feet_midpoint_reset` | `-4.0` | Penalizes the feet midpoint drifting away from its reset anchor once it exceeds a `0.05 m` deadband |

### Posture-shaping terms

| Reward term | Weight | Actual computation |
| --- | --- | --- |
| `swing_foot_base_clearance` | `+1.0` | Rewards swing-foot height in the base frame rising from `0.03 m` to `0.10 m`, active only when swing foot is off-ground and support foot is planted |
| `swing_support_height_difference` | `+0.8` | Rewards the swing foot being `0.02 m` to `0.08 m` higher than the support foot in world frame, active only in a valid swing phase |
| `hip_only_swing` | `-0.8` | Penalizes large swing-hip flexion without enough swing-knee flexion |
| `joint_deviation_hip` | `-0.1` | Penalizes deviation from default for hip-yaw and hip-roll joints |
| `joint_deviation_arms` | `-0.05` | Penalizes deviation from default for shoulder, elbow, and wrist joints |

## Success criteria

No explicit success criterion is implemented.

What the shaping variant actually encourages:

- The commanded swing foot should leave the ground.
- The support foot should stay in contact.
- The swing foot should move toward the lateral target in the base frame.
- The swing leg should flex and shorten.
- The base and feet midpoint should stay near the reset anchor.
- The robot should avoid torso tilt, base drift, yaw spinning, and crossed feet.

These are reward preferences only. The code does not define a threshold such as "one successful step completed" or "target reached for N frames."

## Failure conditions

For the shaping variant, the failure logic is:

| Condition | Implemented rule |
| --- | --- |
| Timeout | Episode ends at `10.0 s` |
| Base contact | `illegal_contact` on `base_link` with threshold `1.0` |
| Bad orientation | Disabled |
| Root too low | Disabled |

This matters because the primitive base class starts with stricter geometric terminations, but the standard step-family configs replace them with contact-based fall detection. The `GeomTerm` variant is the exception and is described later.

## Reset conditions

Episodes reset on timeout or on the active failure condition above.

At reset, the code applies:

| Reset component | Implemented behavior |
| --- | --- |
| Robot root position | Default root position plus fixed `x = -1.5`, fixed `y = +1.5`, and zero `z` offset because `z` is not provided in the override dictionary |
| Robot root orientation | Additional yaw sampled in `[-3.14, 3.14]` |
| Robot root velocity | All components reset to zero |
| Joint positions | Reset exactly to the default joint positions |
| Joint velocities | Reset exactly to zero |
| Base reset anchor | Captured by `BaseXYFromResetObservation` and `BaseResetPositionPenalty` managers at reset |
| Feet-midpoint reset anchor | Captured by `FeetMidpointResetPenalty` at reset |

Because `reset_root_state_uniform(...)` adds offsets to the articulation default root state, and `CLASS_HUMANOID_CFG` sets the default root position to `(0.0, 0.0, 0.8)`, the step task actually resets the robot root to:

- `x = env_origin_x - 1.5`
- `y = env_origin_y + 1.5`
- `z = env_origin_z + 0.8`

## Evaluation metrics actually exposed by code

Explicit task-specific metrics defined in command terms:

- `swing_ratio_right`
- `target_step_xy_norm`

Additional quantities are computed inside reward helpers but are not explicitly registered as standalone metrics in the task config. Examples include:

- selected foot step error
- selected swing-knee flexion
- support-knee flexion
- base drift from reset
- feet-midpoint drift from reset

Not implemented:

- No success counter
- No step-completion counter
- No success-rate evaluator

## Variant differences inside the step family

The step variants differ only in reward composition and termination choice.

| Variant | Difference from the shaping variant |
| --- | --- |
| `Isaac-Primitive-Step-v0` | Uses the smaller `StepRewardsCfg`: lower step-target reward, range-based swing-knee reward instead of min-flex shaping, no leg-shortening reward, no base-clearance reward, no swing-support height-difference reward, no hip-only penalty |
| `Isaac-Primitive-Step-Alt-v0` | Uses `StepAlternatingRewardsCfg`: stronger alternating-contact shaping, minimum swing-knee flex reward, leg-shortening reward, but still no base-clearance or hip-only shaping |
| `Isaac-Primitive-Step-All-v0` | Uses `StepAllRewardsCfg`: same as Alt plus the swing-knee range reward |
| `Isaac-Primitive-Step-Shaping-v0` | Uses `StepShapingRewardsCfg`: same as All plus swing-foot base clearance, swing-vs-support height difference, and hip-only swing penalty |
| `Isaac-Primitive-Step-GeomTerm-v0` | Inherits the base step rewards and switches failure logic to `bad_orientation(limit=0.8)` and `root_height_below_minimum(0.55)`, with base-contact termination disabled |

## Training-script note

The PPO runner config for step variants sets `max_iterations = 2200`, but the helper step scripts currently pass `--max_iterations 15000` on the command line. Both facts are true in the current repository.

## Source files

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/primitive_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/primitive_mdp.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`
- `source/isaaclab/isaaclab/envs/mdp/terminations.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`
