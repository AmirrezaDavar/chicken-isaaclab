# chicken-isaaclab

GPU-accelerated robot learning built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html).
This repo contains custom environments for manipulation, locomotion, and object interaction tasks — including chicken carcass handling, humanoid control, and robotic arm manipulation.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AmirrezaDavar/chicken-isaaclab.git
cd chicken-isaaclab
```

### 2. Activate the virtual environment

The project uses a pre-built venv at `env_isaacsim/`. Run this **every time you open a new terminal**:

```bash
source env_isaacsim/bin/activate
```

You should see `(isaaclab-uv-workspace)` in your prompt. All `./isaaclab.sh` commands require this.

> If `env_isaacsim/` is missing, follow the [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

---

## General Training Pattern

All tasks follow the same command structure:

```bash
# Train headless (fast, recommended for long runs)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs <N> --task <TASK_ID> +run_name=<your_run_name>

# Train with visualization (see the sim live)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --num_envs 64 --task <TASK_ID> +run_name=<your_run_name>

# Play / evaluate a checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --num_envs 32 --task <TASK_ID> \
  agent.resume=true "agent.load_run=.*<your_run_name>" \
  agent.load_checkpoint=model_XXXX.pt

# Record a video
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --headless --enable_cameras --video --video_length 400 \
  --num_envs 16 --task <TASK_ID> \
  agent.resume=true "agent.load_run=.*<your_run_name>" \
  agent.load_checkpoint=model_XXXX.pt

# Resume training from a checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs <N> --task <TASK_ID> \
  agent.resume=true "agent.load_run=.*<previous_run>" \
  agent.load_checkpoint=model_XXXX.pt +run_name=<new_run_name>
```

---

## Custom Environments

### Chicken Balance — `Isaac-Balance-Chicken-v0`

A chicken character learns to balance and locomote on flat terrain.

| | |
|---|---|
| Task ID (train) | `Isaac-Balance-Chicken-v0` |
| Task ID (play) | `Isaac-Balance-Chicken-Play-v0` |
| Log dir | `logs/rsl_rl/chicken_balance/` |
| Max iterations | 3000 |

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs 1532 --task Isaac-Balance-Chicken-v0 \
  +run_name=chicken_balance_v1
```

---

### Chicken Lift — Sequential Two-Jaw Grasping — `Isaac-Lift-Chicken-UR10e-v0`

A UR10e arm with a custom **2-jaw parallel gripper** learns to grasp a chicken carcass by its legs and lift it.
The robot uses the custom gripper from `Universal_Robots_ROS2_Description/urdf/1_fixed.usda`, which has 4 prismatic joints split into two independently controlled jaws:

- **Left jaw** — `PrismaticJoint1` + `PrismaticJoint2` → grasps the left leg
- **Right jaw** — `PrismaticJoint3` + `PrismaticJoint4` → grasps the right leg

**Sequential grasping behaviour learned during training:**
1. EE approaches the chicken
2. Left jaw closes on the left leg (rewarded by `left_leg_grasped`)
3. Once the left jaw is gripping, `right_jaw_gated` activates and the right jaw closes on the right leg
4. With at least the left leg secured, `lifting_gated` fires and the robot lifts the chicken

The action space is **8-dimensional**: 6 arm joints + 1 left-jaw binary + 1 right-jaw binary.

**Observation vector (~57 values):**

| Term | Dim | What it gives the policy |
|---|---|---|
| `joint_pos` | 10 | Arm + gripper joint positions |
| `joint_vel` | 10 | Arm + gripper joint velocities |
| `object_position` | 3 | Chicken torso XYZ in robot frame |
| `chicken_legs` | 6 | Left-leg XYZ + right-leg XYZ in robot frame |
| `chicken_orient` | 4 | Chicken orientation quaternion — tells which side each leg is on |
| `chicken_vel` | 3 | Chicken linear velocity — non-zero = gripper is holding |
| `target_object_position` | 7 | Commanded carry-goal pose |
| `actions` | 8 | Last commanded action |

**Asset dependency:** requires `Universal_Robots_ROS2_Description/urdf/` to be present locally (USD files are not in git).

| | |
|---|---|
| Task ID (train) | `Isaac-Lift-Chicken-UR10e-v0` |
| Task ID (play) | `Isaac-Lift-Chicken-UR10e-Play-v0` |
| Log dir | `logs/rsl_rl/ur10e_chicken_seq_grasp/` |
| Max iterations | 5000 |
| Episode length | 8 s |
| Network | 512 → 256 → 128 (actor & critic) |

```bash
# Train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs 1024 --task Isaac-Lift-Chicken-UR10e-v0 \
  +run_name=ur10e_seq_grasp_v1

# Play (after training)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --num_envs 16 --task Isaac-Lift-Chicken-UR10e-Play-v0 \
  agent.resume=true "agent.load_run=.*ur10e_seq_grasp_v1" \
  agent.load_checkpoint=model_5000.pt
```

---

### Chicken Lift with UR10e + Custom 4-Jaw Gripper (simultaneous) — `Isaac-Lift-Chicken-UR10e-CustomGripper-v0`

A UR10e arm with the same custom gripper, but all 4 prismatic joints commanded **together** as a single binary open/close action.
Use this task if you want simultaneous 4-jaw grasping rather than the sequential left-then-right approach.

| | |
|---|---|
| Task ID (train) | `Isaac-Lift-Chicken-UR10e-CustomGripper-v0` |
| Task ID (play) | `Isaac-Lift-Chicken-UR10e-CustomGripper-Play-v0` |
| Log dir | `logs/rsl_rl/ur10e_custom_gripper_chicken_lift/` |
| Max iterations | 2000 |

```bash
# Train
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs 512 --task Isaac-Lift-Chicken-UR10e-CustomGripper-v0 \
  +run_name=custom_gripper_v1

# Play
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --num_envs 16 --task Isaac-Lift-Chicken-UR10e-CustomGripper-Play-v0 \
  agent.resume=true "agent.load_run=.*custom_gripper_v1" \
  agent.load_checkpoint=model_2000.pt
```

---

### Class Humanoid Tasks

| Task ID | Description |
|---|---|
| `Isaac-ReachDepthCamera-ClassHumanoid-v0` | Humanoid reaches a target using depth camera |
| `Isaac-ShootBall-ClassHumanoid-v0` | Humanoid kicks/shoots a ball to a target |
| `Isaac-ReachLeft-Fixed-ClassHumanoid-v0` | Humanoid reaches with left arm to fixed target |

```bash
# Example: train shoot ball
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs 1536 --task Isaac-ShootBall-ClassHumanoid-v0 \
  +run_name=shoot_ball_v1
```

---

## Analysis & Plots

Two plot scripts are provided. Both are run **after training is complete**.

### Training Progress — `scripts/plot_training.py`

Reads RSL-RL tensorboard logs and generates a 9-panel training progress figure.
**Does not require Isaac Sim** — run with the plain venv Python.

```bash
source env_isaacsim/bin/activate

# Auto-picks the latest run for the sequential grasping experiment
python scripts/plot_training.py

# Specific experiment
python scripts/plot_training.py --experiment ur10e_chicken_seq_grasp

# Specific run directory
python scripts/plot_training.py \
  --log_dir logs/rsl_rl/ur10e_chicken_seq_grasp/<run_folder>
```

Output: `training_progress.png` saved inside the run folder.

**Panels:** mean reward, episode length, policy loss, value loss, entropy, learning rate, throughput (FPS), reward distribution.

---

### Rollout Observations — `scripts/reinforcement_learning/rsl_rl/plot_rollout.py`

Loads a checkpoint, runs N policy steps, and plots per-step observation trajectories.
**Requires Isaac Sim** — launch via `./isaaclab.sh -p`.

```bash
# Auto-loads latest checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/plot_rollout.py \
  --task Isaac-Lift-Chicken-UR10e-v0 \
  --num_steps 500

# Point to a specific checkpoint
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/plot_rollout.py \
  --task Isaac-Lift-Chicken-UR10e-v0 \
  --checkpoint logs/rsl_rl/ur10e_chicken_seq_grasp/<run_folder>/model_5000.pt \
  --num_steps 500
```

Output: `rollout_observations.png` saved next to the checkpoint.

**Panels:**
- **Chicken** — height (torso + left/right leg with lift threshold), XY position, speed
- **EE tracking** — EE vs chicken height, EE→chicken distance, step reward
- **Arm** — 6 joint angles, arm activity (Σ|Δθ|)
- **Gripper** — left jaw % closed, right jaw % closed, sequential phase timeline (left jaw → right jaw → lifted)

---

## TensorBoard (live monitoring during training)

```bash
tensorboard --logdir logs/rsl_rl/chicken_balance
tensorboard --logdir logs/rsl_rl/ur10e_chicken_seq_grasp
tensorboard --logdir logs/rsl_rl/ur10e_custom_gripper_chicken_lift
```

Open `http://localhost:6006` in your browser.

---

## Key Source Files

| File | Description |
|---|---|
| `source/isaaclab_assets/.../robots/chicken.py` | Chicken carcass asset configs (`CHICKEN_CARCASS_CFG`, `CHICKEN_BALANCE_CFG`) |
| `source/isaaclab_assets/.../robots/universal_robots.py` | All UR robot configs incl. `UR10e_CUSTOM_GRIPPER_CFG` (self-collisions enabled) |
| `source/isaaclab_tasks/.../chicken_lift/chicken_lift_env_cfg.py` | Base env + `SequentialGraspRewardsCfg` + `SequentialActionsCfg` + `ChickenSequentialGraspEnvCfg` |
| `source/isaaclab_tasks/.../chicken_lift/mdp/sequential_grasp_rewards.py` | Phased reward functions: `left_leg_grasped_reward`, `right_jaw_gated_reward`, `chicken_lifted_gated`, leg/orientation/velocity observations |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e/joint_pos_env_cfg.py` | UR10e + custom 2-jaw gripper, sequential grasping task |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e/agents/rsl_rl_ppo_cfg.py` | PPO config: 5000 iters, 512→256→128 network, γ=0.99 |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e_custom_gripper/joint_pos_env_cfg.py` | UR10e + custom 4-jaw (simultaneous) gripper config |
| `my_assets/chicken_2/chicken_carcass_2/chicken.urdf` | Chicken URDF (torso, left/right legs, left/right wings) |
| `Universal_Robots_ROS2_Description/urdf/1_fixed.usda` | UR10e + custom 2-jaw gripper scene USD (**not in git**) |
| `scripts/plot_training.py` | Post-training progress plots (no sim needed) |
| `scripts/reinforcement_learning/rsl_rl/plot_rollout.py` | Post-training rollout observation plots (needs sim) |

---

## Common Issues

**`ModuleNotFoundError: No module named 'isaaclab'`**
→ Activate the venv first: `source env_isaacsim/bin/activate`

**`[ERROR] Unable to find any Python executable at path: '.../env_isaacsim/bin/python'` (wrong path)**
→ The `activate` script has the old directory name hardcoded. Fix it once:
```bash
sed -i "s|VIRTUAL_ENV='.*env_isaacsim'|VIRTUAL_ENV='$(pwd)/env_isaacsim'|" env_isaacsim/bin/activate
source env_isaacsim/bin/activate
```

**`NVML_ERROR_LIB_RM_VERSION_MISMATCH` / CUDA errors**
→ Driver mismatch after a kernel update. Fix: `sudo reboot`

**`./isaaclab.sh: python: command not found`**
→ venv not activated. Run `source env_isaacsim/bin/activate` first.

**`+run_name` not working**
→ Make sure to prefix with `+`: `+run_name=my_run` (not `--run_name`)

**Robot links pass through each other (self-collision)**
→ `UR10e_CUSTOM_GRIPPER_CFG` now has `enabled_self_collisions=True` and solver iterations set to 16. If you still see penetration, increase `solver_position_iteration_count` further in `universal_robots.py`.
