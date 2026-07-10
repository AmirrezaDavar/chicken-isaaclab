# Chicken IsaacLab Imitation Learning

This repository contains the IsaacLab simulation side of the chicken grasping imitation-learning project.

It provides:

- UR10e + custom gripper simulation in IsaacLab.
- Chicken asset and chicken lift task configuration.
- GELLO teleoperation data collection.
- Diffusion Policy zarr dataset inspection and visualization.
- In-simulation evaluation of trained ChicGrasp/Diffusion Policy checkpoints.

The training code lives in the sibling repository:

```text
/home/wanglab22/ChicGrasp-IsaacChicken
```

## Repository Roles

```text
/home/wanglab22/3_chicken-isaaclab
  Isaac simulation, assets, teleop collection, zarr tools, policy evaluation.

/home/wanglab22/ChicGrasp-IsaacChicken
  Diffusion Policy algorithms, training configs, dataset loaders, checkpoints.
```

Keep collected datasets, checkpoints, videos, and plots out of Git.

## Activate IsaacLab

From this repository:

```bash
cd /home/wanglab22/3_chicken-isaaclab
source env_isaacsim/bin/activate
```

Most Isaac scripts should be launched with:

```bash
./isaaclab.sh -p <script.py>
```

## Collect Demonstrations

Run:

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/01_collect_chicken_rgb_state_demos.py \
  --out_dir ./data/chicken_rgb_state \
  --num_demos 50 \
  --save_videos
```

Controls:

```text
C          start recording
S          finish and save current episode
Backspace  discard current episode
Q          quit
```

Notes:

- Episodes are unlimited by default.
- The camera preview shows a red recording dot and elapsed time while recording.
- The chicken is randomized in X/Y after each episode.
- If the chicken drops below the table, it is placed back on the table.
- Videos are saved under `data/chicken_rgb_state/videos/`.

Useful options:

```bash
--chicken_xy_range 0.08 0.12
--chicken_seed 123
--disable_chicken_drop_reset
--episode_steps 300
```

## Dataset Layout

Collection writes:

```text
data/chicken_rgb_state/
  replay_buffer.zarr/
    data/
      action
      state
      camera_rgb
      left_jaw
      right_jaw
      robot_eef_pose
      robot_eef_pose_vel
      robot_joint
      robot_joint_vel
      stage
      timestamp
    meta/
      episode_ends
  videos/
```

The important training arrays are:

```text
data/action      # (T, 8)
data/state       # (T, 20)
data/camera_rgb  # (T, H, W, 3)
```

## Inspect And Visualize Data

Inspect zarr structure:

```bash
python scripts/imitation_learning/inspect_dp_zarr.py \
  --zarr_path ./data/chicken_rgb_state/replay_buffer.zarr
```

Generate low-dimensional plots and camera contact sheets:

```bash
python scripts/imitation_learning/visualize_chicken_zarr.py \
  --zarr_path ./data/chicken_rgb_state/replay_buffer.zarr \
  --episode 0
```

This creates:

```text
data/chicken_rgb_state/plots/
  dataset_overview.png
  episode_000000_lowdim.png
  episode_000000_camera_sheet.png
```

## Train Diffusion Policy

Training is run from the sibling repo, but this wrapper is provided here for convenience:

```bash
cd /home/wanglab22/3_chicken-isaaclab
conda activate robodiff
unset PYTHONPATH

python scripts/imitation_learning/02_train_chicken_rgb_state_policy.py \
  --zarr_path /home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
  --num_epochs 450 \
  --batch_size 32 \
  --num_workers 4 \
  --logging_mode offline
```

For full training details, see:

```text
/home/wanglab22/ChicGrasp-IsaacChicken/README.md
```

## Evaluate In Isaac Simulation

After training, evaluate a checkpoint:

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/03_eval_chicken_rgb_state_policy.py \
  --checkpoint /home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/latest.ckpt \
  --num_episodes 3 \
  --episode_steps 300
```

Low-dimensional checkpoints can be evaluated with:

```bash
./isaaclab.sh -p scripts/imitation_learning/eval_chicken_diffusion_policy.py \
  --checkpoint /home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/latest.ckpt \
  --num_episodes 3 \
  --episode_steps 300
```

## Main Files

```text
scripts/imitation_learning/01_collect_chicken_rgb_state_demos.py
scripts/imitation_learning/02_train_chicken_rgb_state_policy.py
scripts/imitation_learning/03_eval_chicken_rgb_state_policy.py
scripts/imitation_learning/inspect_dp_zarr.py
scripts/imitation_learning/visualize_chicken_zarr.py

source/isaaclab_assets/isaaclab_assets/robots/chicken.py
source/isaaclab_assets/isaaclab_assets/robots/universal_robots.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/chicken_lift/
```

## Git Hygiene

Do not commit:

```text
data/
*.zarr/
*.ckpt
*.pt
*.pth
videos/
plots/
wandb/
outputs/
```
