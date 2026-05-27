# Chicken Isaac Lab — CSCE 50103

Isaac Lab project with two chicken RL environments:

- **`Isaac-Balance-Chicken-v0`** — a chicken character learns to balance/locomote on flat terrain
- **`Isaac-Lift-Chicken-UR10e-v0`** — a UR10e robotic arm learns to pick up a chicken carcass

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AmirrezaDavar/CSCE50103-IsaacLab.git
cd CSCE50103-IsaacLab
```

### 2. Activate the virtual environment

The project uses a pre-built venv at `env_isaacsim/`. Activate it every time you open a new terminal:

```bash
source env_isaacsim/bin/activate
```

You should see `(isaaclab-uv-workspace)` in your prompt.

> **Note:** If `env_isaacsim/` is missing, the Isaac Sim Python environment needs to be set up separately — follow the [official Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

---

## Chicken Balance (`Isaac-Balance-Chicken-v0`)

A chicken character learns to stay balanced and walk on flat terrain using PPO.

### Train (headless, fast)

```bash
source env_isaacsim/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless \
  --num_envs 1532 \
  --task Isaac-Balance-Chicken-v0 \
  +run_name=chicken_balance_v1
```

### Train (with live visualization)

```bash
source env_isaacsim/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --num_envs 64 \
  --task Isaac-Balance-Chicken-v0 \
  +run_name=chicken_balance_v1
```

### Play / Evaluate a checkpoint

```bash
source env_isaacsim/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --num_envs 32 \
  --task Isaac-Balance-Chicken-Play-v0 \
  agent.resume=true \
  "agent.load_run=.*chicken_balance_v1" \
  agent.load_checkpoint=model_2999.pt
```

### Record a video

```bash
source env_isaacsim/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --headless --enable_cameras --video --video_length 400 \
  --num_envs 16 \
  --task Isaac-Balance-Chicken-Play-v0 \
  agent.resume=true \
  "agent.load_run=.*chicken_balance_v1" \
  agent.load_checkpoint=model_2999.pt
```

Logs and checkpoints are saved to `logs/rsl_rl/chicken_balance/`.

---

## Chicken Lift with UR10e (`Isaac-Lift-Chicken-UR10e-v0`)

A UR10e robotic arm learns to grasp and lift a chicken carcass using PPO.

### Train (headless, fast)

```bash
source env_isaacsim/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless \
  --num_envs 1024 \
  --task Isaac-Lift-Chicken-UR10e-v0 \
  +run_name=ur10e_lift_v1
```

### Train (with live visualization)

```bash
source env_isaacsim/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --num_envs 64 \
  --task Isaac-Lift-Chicken-UR10e-v0 \
  +run_name=ur10e_lift_v1
```

### Play / Evaluate a checkpoint

```bash
source env_isaacsim/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --num_envs 32 \
  --task Isaac-Lift-Chicken-UR10e-Play-v0 \
  agent.resume=true \
  "agent.load_run=.*ur10e_lift_v1" \
  agent.load_checkpoint=model_1999.pt
```

### Record a video

```bash
source env_isaacsim/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --headless --enable_cameras --video --video_length 400 \
  --num_envs 16 \
  --task Isaac-Lift-Chicken-UR10e-Play-v0 \
  agent.resume=true \
  "agent.load_run=.*ur10e_lift_v1" \
  agent.load_checkpoint=model_1999.pt
```

Logs and checkpoints are saved to `logs/rsl_rl/ur10e_chicken_lift/`.

---

## TensorBoard

Monitor training in real time:

```bash
tensorboard --logdir logs/rsl_rl/chicken_balance
# or
tensorboard --logdir logs/rsl_rl/ur10e_chicken_lift
```

Then open `http://localhost:6006` in your browser.

---

## Key Files

| File | Description |
|---|---|
| `source/isaaclab_assets/isaaclab_assets/robots/chicken.py` | Chicken asset config (USD path, joints) |
| `source/isaaclab_tasks/.../chicken_balance/chicken_balance_env_cfg.py` | Balance env rewards, observations |
| `source/isaaclab_tasks/.../chicken_balance/config/chicken/flat_env_cfg.py` | Flat terrain config |
| `source/isaaclab_tasks/.../chicken_balance/config/chicken/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters for balance |
| `source/isaaclab_tasks/.../chicken_lift/chicken_lift_env_cfg.py` | Lift env rewards, observations |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e/joint_pos_env_cfg.py` | UR10e joint-position config |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters for lift |
| `my_assets/chicken/` | USD mesh files for the chicken carcass |

---

## Common Issues

**`ModuleNotFoundError: No module named 'isaaclab'`**
→ You forgot to activate the venv: `source env_isaacsim/bin/activate`

**`NVML_ERROR_LIB_RM_VERSION_MISMATCH` / CUDA errors on startup**
→ NVIDIA driver mismatch after a kernel update. Fix: `sudo reboot`

**`./isaaclab.sh: python: command not found`**
→ venv not activated. Run `source env_isaacsim/bin/activate` first.
