# Class Humanoid Task Docs

This folder documents the `class_humanoid` task settings that are currently implemented in this repository.

Scope rules used for these docs:

- Only behavior that is directly implemented in the current code is described.
- No intended behavior is added if it is not enforced by code.
- Where the code has variants, the most feature-rich variant is documented in detail.
- If the code does not define an explicit success condition, that absence is stated directly.

Task docs:

- `class_humanoid_walking_rough.md`
- `class_humanoid_walking_flat.md`
- `class_humanoid_primitive_squat.md`
- `class_humanoid_primitive_step.md`
- `class_humanoid_primitive_reach_depth.md`

Primary source files:

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/common.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/common_mdp.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/squat_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/squat_mdp.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/step_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/step_mdp.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/reach_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/reach_mdp.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/isaaclab_assets/isaaclab_assets/robots/class_humanoid.py`

Common implementation facts shared by the primitive-task family:

| Item | Implemented behavior |
| --- | --- |
| Base environment class | Squat, step, and reach tasks inherit directly from `LocomotionVelocityRoughEnvCfg` and share defaults through `common.py`. |
| Simulation timing | `sim.dt = 0.005`, `decimation = 4`, so the policy acts every `0.02 s`. |
| Terrain | Flat plane terrain only. Terrain generator and terrain curriculum are disabled. |
| Robot asset | `CLASS_HUMANOID_CFG`, loaded from `my_assets/humanoid_tuned.usd`. |
| Contact sensing | A contact sensor is attached to `"{ENV_REGEX_NS}/Robot/.*"` with `history_length = 3` and `track_air_time = True`. |
| Height scanner | Disabled for primitive tasks. |
| Observation packaging | Primitive observations are concatenated into one policy vector and observation corruption is enabled. |
| Joint observation convention | Joint position and velocity observations use semantic signs: right-side joints are multiplied by `-1`, left-side joints stay positive. |
| Default action type | Joint position targets, not torques. The processed action is `default_joint_position + scale * raw_action` when `use_default_offset=True`. |
| Random pushes | Disabled. |
| Base mass randomization | Disabled. |
| Base COM randomization | Disabled. |
| External base force/torque reset event | Still present, but the inherited force and torque ranges remain zero, so it does not inject a disturbance. |

Registered primitive-task Gym IDs:

- `Isaac-Squat-ClassHumanoid-v0`
- `Isaac-Step-ClassHumanoid-v0`
- `Isaac-Step-ClassHumanoid-Alt-v0`
- `Isaac-Step-ClassHumanoid-All-v0`
- `Isaac-Step-ClassHumanoid-Shaping-v0`
- `Isaac-Step-ClassHumanoid-GeomTerm-v0`
- `Isaac-ReachDepth-ClassHumanoid-v0`

Registered walking-task Gym IDs:

- `Isaac-Velocity-Rough-ClassHumanoid-v0`
- `Isaac-Velocity-Rough-ClassHumanoid-ContactPenalty-v0`
- `Isaac-Velocity-Rough-ClassHumanoid-BadOrientation-v0`
- `Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0`
- `Isaac-Velocity-Rough-ClassHumanoid-FootLift-v0`
- `Isaac-Velocity-Rough-ClassHumanoid-Play-v0`
- `Isaac-Velocity-Flat-ClassHumanoid-v0`
- `Isaac-Velocity-Flat-ClassHumanoid-Play-v0`
