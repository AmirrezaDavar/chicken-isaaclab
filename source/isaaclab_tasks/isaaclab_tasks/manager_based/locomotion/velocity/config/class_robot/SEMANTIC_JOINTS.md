# Class Humanoid Semantic Joint Notes

This note records the joint-sign conventions that matter for Class Humanoid task rewards.

## Why this exists

The Class Humanoid asset does not use one globally consistent "positive means flexion" convention across left/right joints.
That is acceptable for low-level position control, but it is unsafe for style rewards that interpret joint angles semantically.

Raw `joint_pos` is still fine for:
- `JointPositionActionCfg(..., use_default_offset=True)`
- raw proprioception observations
- `joint_deviation_l1`

Semantic helpers are required for:
- knee-flexion rewards
- minimum-flex knee rewards for early stepping exploration
- hip-flexion rewards
- ankle dorsiflexion / plantarflexion rewards
- any future "move like this joint should bend" reward

## Verified conventions

Verified by manual inspection during replay:

- `Left_Knee_RS04`
  - standing pose: `0 rad`
  - positive direction: knee flexion
- `Right_Knee_RS04`
  - standing pose: `0 rad`
  - negative direction: knee flexion

Inferred from the same mirrored joint convention and the `isaac_param_dump` limits:

- `Left_Hip_Pitch_RS04`
  - standing pose: `0 rad`
  - semantic positive direction used in task code: hip flexion / thigh lift
- `Right_Hip_Pitch_RS04`
  - standing pose: `0 rad`
  - semantic positive direction used in task code: hip flexion / thigh lift via sign flip

Verified from `isaac_param_dump/rigid_bodies.csv`:

- left leg bodies
  - `HipYoke_Left_1`
  - `UpperThigh_Left_1`
  - `LowerThigh_Left_1`
  - `Shin_Left_1`
  - `Foot_Left_1`
- right leg bodies
  - `HipYoke_Right_1`
  - `UpperThigh_Right_1`
  - `LowerThigh_Right_1`
  - `Shin_Right_1`
  - `Foot_Right_1`

## Current implementation

Code lives in:
- `common_mdp.py`
- `step_mdp.py` for step-specific semantic rewards

Helpers added:
- `semantic_signed_joint_pos(...)`
  - generic helper that multiplies raw joint positions by caller-provided semantic signs
- `class_humanoid_knee_flexion(...)`
  - returns `[left_knee_flex, right_knee_flex]`
  - both are positive when the knee is bending
- `class_humanoid_hip_pitch_flexion(...)`
  - returns `[left_hip_flex, right_hip_flex]`
  - both are positive when the thigh is flexing forward
- `selected_swing_knee_angle_range_reward(...)`
  - used by the default step task to shape the final preferred knee-bend window
- `selected_swing_knee_min_flex_reward(...)`
  - used by `Step-Alt` to provide reward signal from near-standing posture up to the target knee bend
  - saturates once the selected swing knee reaches the configured minimum useful flexion

Current semantic sign map in code:

- knee flexion
  - `Left_Knee_RS04`: `+1.0`
  - `Right_Knee_RS04`: `-1.0`

General observation convention now used for Class Humanoid tasks:

- joints whose names start with `Left_`: `+1.0`
- joints whose names start with `Right_`: `-1.0`
- unmatched / center joints: `+1.0`

This convention is applied to:
- `joint_pos` observations for class humanoid locomotion tasks
- `joint_vel` observations for class humanoid locomotion tasks
- semantic knee-flexion rewards in stepping

## Reward usage rule

If a reward cares about geometry or contact only, raw joint signs do not matter.

If a reward cares about a human-readable motion meaning such as:
- "bend the knee"
- "flex the hip"
- "keep the ankle neutral"

then do not read `asset.data.joint_pos` directly. Convert it through a semantic helper first.

## Where semantic observations are enabled

Semantic joint observations are enabled for the Class Humanoid locomotion tasks in:

- `rough_env_cfg.py`
- `flat_env_cfg.py` through inheritance from rough
- `squat_env_cfg.py`
- `step_env_cfg.py`
- `reach_env_cfg.py`

This means the policy now sees semantic joint position/velocity for:
- rough walking
- flat walking
- squat
- step
- step alt
- step all
- step shaping
- reach-depth

Additional step reward that now uses exact body names from `isaac_param_dump`:

- `swing_leg_shortening_reward`
  - compares `HipYoke_Left_1 -> Foot_Left_1` and `HipYoke_Right_1 -> Foot_Right_1`
  - rewards the commanded swing leg for being shorter than the support leg

## What is intentionally still raw

Action targets are still raw articulation joint targets.

Reason:
- changing action semantics is more invasive
- it risks breaking existing control behavior and old checkpoints
- observation/reward semanticization gives most of the symmetry benefit with much lower regression risk

## Next extensions

When hip-pitch or ankle rewards are added, follow the same pattern:

1. verify the sign in replay
2. add a semantic helper in `common_mdp.py`
3. use the helper in rewards instead of raw `joint_pos`

Do not change the USD / asset model just to fix sign conventions unless the whole project is being migrated.
For Task E, semantic conversion in task code is the lower-risk path.

Step task variants now differ only in knee-shaping strategy:

- `Isaac-Step-ClassHumanoid-v0`
  - original swing-knee target-range reward
- `Isaac-Step-ClassHumanoid-Alt-v0`
  - replaces the range reward with a minimum-flex reward so learning signal exists before the policy discovers a large bend
- `Isaac-Step-ClassHumanoid-All-v0`
  - keeps the `Alt` minimum-flex reward and adds the original range reward back as a secondary style term
- `Isaac-Step-ClassHumanoid-Shaping-v0`
  - keeps the `All` knee shaping and adds:
    - swing-foot clearance in base frame
    - swing/support foot height difference
    - hip-only swing compensation penalty
