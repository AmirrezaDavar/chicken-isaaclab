# ChicGrasp — Chicken Carcass Grasping with Isaac Lab + Diffusion Policy

GPU-accelerated robot grasping simulation using **NVIDIA Isaac Lab** (Isaac Sim 5.1).
A UR10e arm with a custom 4-jaw parallel gripper learns to pick up a chicken carcass by its legs
via GELLO teleoperation and diffusion policy imitation learning.

---

## Hardware & Software Requirements

- NVIDIA GPU (tested on RTX 4080)
- Ubuntu 22.04
- Isaac Sim 5.1 pre-installed at `env_isaacsim/`
- GELLO teleoperation device (Dynamixel-based, connected via USB-FTDI)

---

## Environment Setup

The project uses a pre-built venv at `env_isaacsim/`. Run this **every time you open a new terminal**:

```bash
source env_isaacsim/bin/activate
```

You should see `(isaaclab-uv-workspace)` in your prompt.

> **If `env_isaacsim/` is missing** after cloning, follow the
> [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
> and rebuild the venv.

> **If the venv activate script has a stale path** (cloned to a different location):
> ```bash
> sed -i "s|VIRTUAL_ENV='.*env_isaacsim'|VIRTUAL_ENV='$(pwd)/env_isaacsim'|" env_isaacsim/bin/activate
> source env_isaacsim/bin/activate
> ```

---

## Required Assets (not in git)

The robot and chicken USD files are too large for git. You need them locally:

| Asset | Path |
|---|---|
| Chicken USD | `my_assets/chicken/chicken/chicken.usd` |
| UR10e + gripper USD | `Universal_Robots_ROS2_Description/urdf/1_fixed.usda` |

Ask a team member for a copy or rebuild from URDF using the Isaac Sim URDF importer.

---

## Teleoperation Demo Collection (GELLO)

Collect demonstrations using the GELLO arm. Each episode is saved in
**diffusion-policy zarr format** with separate array folders per feature.

### Run

```bash
python scripts/imitation_learning/collect_isaac_demos.py \
    --out_dir /home/wanglab22/ChicGrasp/data/isaac_chicken \
    --num_demos 50
```

### Controls

| Key | Action |
|---|---|
| `C` | Start recording the current episode |
| `S` | Stop and **save** the episode |
| `Backspace` | **Discard** the current episode |
| `Q` | Quit |

Move the GELLO arm to control the UR10e. The GELLO trigger closes/opens all 4 gripper jaws.

### Output structure

```
/home/wanglab22/ChicGrasp/data/isaac_chicken/
  replay_buffer.zarr/
    data/
      action/             (T_total, 8)   arm joints(6) + left/right jaw binary(2)
      left_jaw/           (T_total, 1)   left-jaw closure [0=open, 1=closed]
      right_jaw/          (T_total, 1)   right-jaw closure [0=open, 1=closed]
      robot_eef_pose/     (T_total, 6)   ee pos(3) + euler(3) in robot frame
      robot_eef_pose_vel/ (T_total, 6)   ee lin_vel(3) + ang_vel(3)
      robot_joint/        (T_total, 6)   arm joint positions (rad)
      robot_joint_vel/    (T_total, 6)   arm joint velocities (rad/s)
      stage/              (T_total, 1)   0=reaching, 1=chicken lifted
      timestamp/          (T_total, 1)   seconds since episode start
    meta/
      episode_ends/       (N_episodes,)  cumulative step index at each episode boundary
  videos/
    episode_000000.mp4    wrist-camera recording per episode (for human review)
```

### Options

```bash
python scripts/imitation_learning/collect_isaac_demos.py \
    --out_dir ./data/isaac_chicken \
    --num_demos 50 \
    --episode_steps 300 \       # max steps before auto-save (default 300)
    --video_fps 30 \            # MP4 frame rate (default 30)
    --gello_port /dev/ttyUSB0 \ # auto-detected if omitted
    --diagnose                  # print GELLO↔sim joint table for calibration
```

> **If `pyarrow`/`zarr` are missing** in the Isaac Sim Python:
> ```bash
> env_isaacsim/bin/python -m pip install "zarr>=2.12,<3"
> ```

---

## Visualise Collected Data

Plot the low-dim zarr data **without Isaac Sim** (runs in the plain venv):

```bash
# Dataset overview — all episodes at once
python scripts/imitation_learning/plot_demos.py \
    --zarr_path /home/wanglab22/ChicGrasp/data/isaac_chicken/replay_buffer.zarr

# Single episode in detail (0-indexed)
python scripts/imitation_learning/plot_demos.py \
    --zarr_path /home/wanglab22/ChicGrasp/data/isaac_chicken/replay_buffer.zarr \
    --episode 0

# Save to PNG instead of interactive window
python scripts/imitation_learning/plot_demos.py \
    --zarr_path /home/wanglab22/ChicGrasp/data/isaac_chicken/replay_buffer.zarr \
    --save overview.png
```

**Overview mode** (no `--episode`):
- Episode lengths with success (green = chicken lifted) / fail (red) colours
- All arm joint trajectories overlaid across episodes
- Gripper closure across all episodes
- EE position & orientation
- Lifted fraction per episode
- Joint position histogram across all data

**Single episode mode** (`--episode N`):
- Arm joint positions & velocities
- EE position & orientation
- Jaw closure + stage timeline
- Arm action targets and gripper commands

---

## RL Training (Isaac Lab PPO)

Train the UR10e + custom gripper to grasp the chicken autonomously:

```bash
# Train headless (recommended — fastest)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --headless --num_envs 1024 \
    --task Isaac-Lift-Chicken-UR10e-CustomGripper-v0 \
    +run_name=custom_gripper_v1

# Play a trained policy
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --num_envs 16 \
    --task Isaac-Lift-Chicken-UR10e-CustomGripper-Play-v0 \
    agent.resume=true "agent.load_run=.*custom_gripper_v1" \
    agent.load_checkpoint=model_2000.pt

# Monitor training in TensorBoard
tensorboard --logdir logs/rsl_rl/ur10e_custom_gripper_chicken_lift
```

### Available task IDs

| Task | ID |
|---|---|
| UR10e + custom gripper (simultaneous jaws) | `Isaac-Lift-Chicken-UR10e-CustomGripper-v0` |
| UR10e + custom gripper (GELLO teleoperation) | `Isaac-Lift-Chicken-UR10e-CustomGripper-GELLO-v0` |
| UR10e sequential jaw grasping | `Isaac-Lift-Chicken-UR10e-v0` |
| Chicken balance locomotion | `Isaac-Balance-Chicken-v0` |

---

## Key Source Files

| File | Description |
|---|---|
| `scripts/imitation_learning/collect_isaac_demos.py` | GELLO demo collection → zarr format |
| `scripts/imitation_learning/plot_demos.py` | Visualise collected zarr data |
| `source/isaaclab_assets/.../robots/chicken.py` | Chicken asset configs |
| `source/isaaclab_tasks/.../chicken_lift/chicken_lift_env_cfg.py` | Base env, rewards, observations |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e_custom_gripper/gello_env_cfg.py` | GELLO teleoperation env + wrist camera config |
| `source/isaaclab_tasks/.../chicken_lift/config/ur10e_custom_gripper/joint_pos_env_cfg.py` | UR10e + 4-jaw gripper RL env |

---

## Common Issues

**`ModuleNotFoundError: No module named 'isaaclab'`**
→ Activate the venv: `source env_isaacsim/bin/activate`

**`NVML_ERROR_LIB_RM_VERSION_MISMATCH`**
→ Driver/kernel mismatch after an update. Fix: `sudo reboot`

**GELLO not detected**
→ Check USB connection. Verify with `ls /dev/serial/by-id/*FTDI*`. Pass the port explicitly with `--gello_port`.

**Camera shows wrong angle in Isaac Sim**
→ The wrist camera quaternion in `gello_env_cfg.py` is `rot=(0.0, 0.0, 0.462, 0.8875)` with `convention="ros"` — gives +125° pitch in the Isaac Sim Property panel.
