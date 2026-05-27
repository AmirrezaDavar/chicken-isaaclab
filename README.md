# CSCE 50103 — Isaac Lab

GPU-accelerated robot learning built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html).
This repo contains custom environments for manipulation, locomotion, and object interaction tasks — including chicken carcass handling, humanoid control, and robotic arm manipulation.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AmirrezaDavar/CSCE50103-IsaacLab.git
cd CSCE50103-IsaacLab
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

### Chicken Lift with UR10e — `Isaac-Lift-Chicken-UR10e-v0`

A UR10e robotic arm with gripper learns to grasp and lift a chicken carcass from a table.

| | |
|---|---|
| Task ID (train) | `Isaac-Lift-Chicken-UR10e-v0` |
| Task ID (play) | `Isaac-Lift-Chicken-UR10e-Play-v0` |
| Log dir | `logs/rsl_rl/ur10e_chicken_lift/` |
| Max iterations | 2000 |

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs 1024 --task Isaac-Lift-Chicken-UR10e-v0 \
  +run_name=ur10e_lift_v1
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

## TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/chicken_balance
tensorboard --logdir logs/rsl_rl/ur10e_chicken_lift
```

Open `http://localhost:6006` in your browser.

---

## Key Source Files

| File | Description |
|---|---|
| `source/isaaclab_assets/.../robots/chicken.py` | Chicken carcass asset (joints, USD path) |
| `source/isaaclab_tasks/.../chicken_balance/chicken_balance_env_cfg.py` | Balance env rewards & observations |
| `source/isaaclab_tasks/.../chicken_balance/config/chicken/flat_env_cfg.py` | Flat terrain config |
| `source/isaaclab_tasks/.../chicken_balance/config/chicken/agents/rsl_rl_ppo_cfg.py` | PPO config for balance |
| `source/isaaclab_tasks/.../chicken_lift/chicken_lift_env_cfg.py` | Lift env rewards & observations |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e/joint_pos_env_cfg.py` | UR10e + gripper + table config |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e/agents/rsl_rl_ppo_cfg.py` | PPO config for lift |
| `my_assets/chicken/` | USD mesh files for the chicken |

---

## Common Issues

**`ModuleNotFoundError: No module named 'isaaclab'`**
→ Activate the venv first: `source env_isaacsim/bin/activate`

**`NVML_ERROR_LIB_RM_VERSION_MISMATCH` / CUDA errors**
→ Driver mismatch after a kernel update. Fix: `sudo reboot`

**`./isaaclab.sh: python: command not found`**
→ venv not activated. Run `source env_isaacsim/bin/activate` first.

**`+run_name` not working**
→ Make sure to prefix with `+`: `+run_name=my_run` (not `--run_name`)
