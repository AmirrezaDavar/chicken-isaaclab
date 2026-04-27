# Class Humanoid Primitive Squat

## Registration and entry points

| Item | Value |
| --- | --- |
| Gym ID | `Isaac-Primitive-Squat-v0` |
| Environment config class | `ClassHumanoidPrimitiveSquatEnvCfg` |
| PPO runner config class | `ClassHumanoidPrimitiveSquatPPORunnerCfg` |
| Helper training script | `scripts/reinforcement_learning/rsl_rl/train_primitive_squat.sh` |
| Default env count in env config | `2048` |
| Default env count in helper script | `2048` |
| Episode length | `8.0 s` |
| Policy step time | `0.02 s` |
| Steps per episode | `400` |

## Task summary

This task is a flat-ground whole-body joint-position-control task for the `class_humanoid` robot. The code defines a scalar command named `target_pelvis_height`, but the reward and command metrics are actually computed from the robot root height (`asset.data.root_pos_w[:, 2]`), not from a separate pelvis body.

There is no explicit success flag, no success termination, and no task-specific evaluator. The task is implemented entirely through command generation, reward shaping, and termination/reset rules.

## Observation space

The policy observation group is concatenated and corruption is enabled.

| Term | Shape in code | Actual content | Noise / processing |
| --- | --- | --- | --- |
| `base_lin_vel` | 3 | Root linear velocity | Uniform additive noise in `[-0.05, 0.05]` |
| `base_ang_vel` | 3 | Root angular velocity | Uniform additive noise in `[-0.05, 0.05]` |
| `base_rpy` | 3 | Root roll, pitch, yaw from `root_quat_w` | Uniform additive noise in `[-0.02, 0.02]` |
| `joint_pos` | all robot joints | Joint positions relative to default, with right-side joints sign-flipped into a shared semantic convention | Uniform additive noise in `[-0.01, 0.01]` |
| `joint_vel` | all robot joints | Joint velocities relative to default, with right-side joints sign-flipped into the same semantic convention | Uniform additive noise in `[-0.15, 0.15]` |
| `foot_contacts` | 2 | Binary contact indicators for `Foot_Left_1` and `Foot_Right_1` | Contact threshold `1.0` |
| `actions` | matches action dimension | The last raw action sent to the environment | No extra noise |
| `target_pelvis_height` | 1 | Generated command value from the command manager | No extra noise |

Notes:

- The primitive observation config does not include any camera or depth input.
- `foot_contacts` comes from the shared contact sensor on the robot and checks only the two feet for this term.
- The `actions` observation is the raw action, not the offset-and-scaled processed joint target.

## Action space

| Item | Implemented behavior |
| --- | --- |
| Action type | `JointPositionActionCfg` |
| Controlled joints | `joint_names=[".*"]`, so all joints resolved by the articulation are controlled |
| Scale | `0.35` in the squat environment |
| Offset | Default joint positions from the articulation asset (`use_default_offset=True`) |
| Applied command | `processed_action = default_joint_pos + 0.35 * raw_action`, then `set_joint_position_target(...)` |
| Torque control | Not used |

The shipped `CLASS_HUMANOID_CFG` explicitly initializes 20 named joints in Python, and the squat task uses the all-joints action path rather than a restricted subset.

## Command space

| Command | Implemented behavior |
| --- | --- |
| `target_pelvis_height` | `UniformPelvisHeightCommand` |
| Value range | Uniformly sampled in `[0.62, 0.75]` |
| Resampling time | Uniformly sampled in `[1.5, 2.5] s` |
| Debug visualization | Disabled |
| Metric exposed by the command term | `height_error = abs(root_height - commanded_height)` |

Important implementation detail:

- Despite the command name, the code measures error using the robot root height, not a dedicated pelvis body height.

## Reward function

All reward terms are active at the same time.

| Reward term | Weight | Actual computation |
| --- | --- | --- |
| `termination_penalty` | `-100.0` | Applies when a non-timeout termination happens. Timeouts are excluded. |
| `torso_tilt_l2` | `-2.0` | Sum of squared `projected_gravity_b[:2]`. This penalizes torso tilt away from upright. |
| `action_rate_l2` | `-0.01` | Squared difference between current and previous action. |
| `joint_vel_l2` | `-2.0e-4` | Sum of squared joint velocities. |
| `pelvis_height_tracking` | `-8.0` | Negative squared error between root height and the `target_pelvis_height` command. |
| `feet_slide` | `-0.25` | Horizontal foot speed while the feet are in contact. |

## Success criteria

No explicit success criterion is implemented.

What the current code rewards instead:

- Small root-height error relative to the commanded height.
- Small torso tilt.
- Low action-rate change.
- Low joint velocity magnitude.
- Low foot sliding while in contact.

Because there is no success flag or thresholded success event, "success" in this task is only implicit in the reward structure.

## Failure conditions

| Condition | Implemented rule |
| --- | --- |
| Timeout | Episode ends at `8.0 s` |
| Bad orientation | Root orientation fails when the gravity-alignment angle exceeds `0.8 rad` |
| Root too low | Root height below `0.55 m` |
| Base contact | Disabled in the squat task |

The shared primitive base config originally defines a stricter orientation limit (`15 deg`), a root-height limit of `0.58`, and a base-contact termination, but the squat task overrides those settings to the values above.

## Reset conditions

Episodes reset when a termination or timeout occurs.

At reset, the code applies:

| Reset component | Implemented behavior |
| --- | --- |
| Robot root position | Default root position plus sampled offsets: `x in [-0.10, 0.10]`, `y in [-0.10, 0.10]`, `z in [0.05, 0.10]` |
| Robot root orientation | Additional yaw sampled in `[-0.20, 0.20]` |
| Robot root velocity | All components reset to zero |
| Joint positions | Reset exactly to the default joint positions via `position_range=(1.0, 1.0)` |
| Joint velocities | Reset exactly to zero |
| External push | None |

Because `reset_root_state_uniform(...)` adds sampled offsets to the articulation default root state, and `CLASS_HUMANOID_CFG` sets the default root position to `(0.0, 0.0, 0.8)`, the squat task actually resets the robot root height to `0.85` to `0.90` meters above each environment origin.

## Evaluation metrics actually exposed by code

Explicit task-specific metrics defined in this task family:

- `height_error` from the `target_pelvis_height` command term

Not implemented:

- No success counter
- No success-rate evaluator
- No dedicated evaluation episode mode
- No thresholded "correct squat" event

## Scene and environment details

| Item | Implemented behavior |
| --- | --- |
| Terrain | Flat plane |
| Terrain curriculum | Disabled |
| Height scanner | Disabled |
| Viewer target | `base_link` |
| Viewer eye | `(3.0, -1.6, 1.8)` |
| Viewer look-at | `(0.0, 0.0, 0.9)` |

## Source files

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/primitive_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/primitive_mdp.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py`
- `source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`
- `source/isaaclab/isaaclab/envs/mdp/terminations.py`
