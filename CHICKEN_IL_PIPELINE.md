# Chicken Imitation Learning Pipeline

This project uses two sibling repositories:

- `/home/wanglab22/3_chicken-isaaclab`: Isaac Lab simulation, GELLO data collection, zarr inspection, and Isaac policy evaluation.
- `/home/wanglab22/ChicGrasp-IsaacChicken`: separate Diffusion Policy training repo for the Isaac chicken task.

Keep large datasets, checkpoints, videos, and zarr stores out of Git. Use Git LFS only for large assets that must be versioned.

## Quick Commands

### 1. Collect RGB + Low-Dim Demonstrations

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/01_collect_chicken_rgb_state_demos.py \
  --out_dir ./data/chicken_rgb_state \
  --num_demos 50 \
  --episode_steps 300 \
  --preview_stride 5
```

Keyboard controls during collection:

```text
C          start recording
S          save current episode
Backspace  discard current episode
Q          quit
```

The live camera preview is shown in an OpenCV window. For smoother teleoperation,
do not encode MP4 videos during collection. Export review videos afterward:

```bash
python scripts/imitation_learning/export_chicken_zarr_videos.py \
  --zarr_path ./data/chicken_rgb_state/replay_buffer.zarr
```

### 2. Train RGB + Low-Dim Diffusion Policy

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

### 3. Evaluate RGB + Low-Dim Policy In Isaac

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/03_eval_chicken_rgb_state_policy.py \
  --checkpoint /home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/latest.ckpt \
  --num_episodes 3 \
  --episode_steps 300
```

## 1. Collect Demonstrations

Run from the Isaac Lab repo:

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/01_collect_chicken_rgb_state_demos.py \
  --out_dir ./data/chicken_rgb_state \
  --num_demos 50 \
  --episode_steps 300 \
  --preview_stride 5
```

The default dataset layout is:

```text
data/chicken_rgb_state/
  replay_buffer.zarr/
    .zgroup
    data/
      action        # (T, 8)
      state         # (T, 20)
      camera_rgb    # (T, 480, 640, 3)
      ...
    meta/
      episode_ends
```

The collector stores RGB frames in zarr for training. Export MP4 review videos
after collection to avoid blocking the real-time GELLO loop:

```bash
python scripts/imitation_learning/export_chicken_zarr_videos.py \
  --zarr_path ./data/chicken_rgb_state/replay_buffer.zarr
```

Inspect a dataset:

```bash
python scripts/imitation_learning/inspect_dp_zarr.py \
  --zarr_path ./data/chicken_rgb_state/replay_buffer.zarr
```

## 2. ChicGrasp Environment

Install Miniforge if needed, then create the training environment:

```bash
source /home/wanglab22/miniforge3/etc/profile.d/conda.sh
cd /home/wanglab22/ChicGrasp-IsaacChicken
unset PYTHONPATH

export MUJOCO_PATH=$HOME/.mujoco/mujoco-2.3.1
export MUJOCO_PLUGIN_PATH=$MUJOCO_PATH/plugin
export LD_LIBRARY_PATH=$MUJOCO_PATH/lib:$LD_LIBRARY_PATH

conda env create -f conda_environment.yaml -n robodiff
conda activate robodiff
pip install -e .
pip install "huggingface_hub==0.10.1"
```

Check the environment:

```bash
python -c "import mujoco, dm_control, hydra, dill, torch, zarr, diffusers, robomimic; print('ok')"
```

## 3. Train RGB + Low-Dim Diffusion Policy

This is the preferred training path when the camera should be part of the policy input.

Smoke test:

```bash
cd /home/wanglab22/3_chicken-isaaclab
conda activate robodiff
unset PYTHONPATH

python scripts/imitation_learning/02_train_chicken_rgb_state_policy.py \
  --zarr_path /home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
  --num_epochs 5 \
  --max_train_steps 20 \
  --max_val_steps 5 \
  --batch_size 8 \
  --num_workers 0 \
  --logging_mode offline
```

Longer training:

```bash
python scripts/imitation_learning/02_train_chicken_rgb_state_policy.py \
  --zarr_path /home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
  --num_epochs 450 \
  --batch_size 32 \
  --num_workers 4 \
  --logging_mode offline
```

The script installs the Isaac chicken RGB+state dataset/config bridge into:

```text
/home/wanglab22/ChicGrasp-IsaacChicken/diffusion_policy/
```

Checkpoints are written under:

```text
/home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/
```

## 4. Train Low-Dim Diffusion Policy

Smoke test:

```bash
cd /home/wanglab22/ChicGrasp-IsaacChicken
conda activate robodiff
unset PYTHONPATH

python train.py \
  --config-name=train_diffusion_unet_lowdim_isaac_chicken_workspace \
  task.obs_dim=20 \
  task.dataset.zarr_path=/home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
  +task.dataset.obs_key=state \
  task.dataset.val_ratio=0.25 \
  dataloader.batch_size=32 \
  val_dataloader.batch_size=32 \
  dataloader.num_workers=0 \
  val_dataloader.num_workers=0 \
  training.num_epochs=5 \
  training.max_train_steps=20 \
  training.max_val_steps=5 \
  logging.mode=offline
```

Longer training:

```bash
python train.py \
  --config-name=train_diffusion_unet_lowdim_isaac_chicken_workspace \
  task.obs_dim=20 \
  task.dataset.zarr_path=/home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
  +task.dataset.obs_key=state \
  task.dataset.val_ratio=0.25 \
  dataloader.batch_size=64 \
  val_dataloader.batch_size=64 \
  training.num_epochs=1000 \
  logging.mode=offline
```

Checkpoints are written under:

```text
/home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/
```

## 5. Prepare Isaac Eval Dependencies

The Isaac Lab environment must keep Isaac-compatible package versions. Do not install the old ChicGrasp pins for `huggingface_hub` or `protobuf` into Isaac.

Repair/confirm compatible packages:

```bash
/home/wanglab22/3_chicken-isaaclab/env_isaacsim/bin/python3 -m pip install \
  click==8.1.7 \
  huggingface_hub==0.36.2 \
  protobuf==4.25.8 \
  dill==0.3.5.1 \
  einops==0.4.1 \
  diffusers
```

Check:

```bash
/home/wanglab22/3_chicken-isaaclab/env_isaacsim/bin/python3 -c \
  "import dill, einops, transformers, huggingface_hub; print('ok')"
```

## 6. Evaluate RGB + Low-Dim Policy In Isaac

Run from the Isaac Lab repo:

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/03_eval_chicken_rgb_state_policy.py \
  --checkpoint /home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/latest.ckpt \
  --num_episodes 3 \
  --episode_steps 300
```

The evaluator reconstructs the same 20D state vector, reads the live Isaac camera, and feeds both `state` and `camera_rgb` to the policy.

## 7. Evaluate Low-Dim Policy In Isaac

Run from the Isaac Lab repo:

```bash
cd /home/wanglab22/3_chicken-isaaclab

./isaaclab.sh -p scripts/imitation_learning/eval_chicken_diffusion_policy.py \
  --checkpoint /home/wanglab22/ChicGrasp-IsaacChicken/data/outputs/<date>/<run_name>/checkpoints/latest.ckpt \
  --num_episodes 3 \
  --episode_steps 300
```

The evaluator loads the low-dimensional ChicGrasp checkpoint, reconstructs the same 20D state vector in Isaac, and executes the predicted 8D action chunks.

## 8. Git Hygiene

Commit source/config/docs only. Do not commit:

```text
data/
*.zarr/
*.ckpt
*.pt
*.pth
*.zip
videos/
wandb/
outputs/
```

Use separate commits in the two repositories:

- `chicken-isaaclab`: simulation, collection, zarr inspection, Isaac eval.
- `ChicGrasp-IsaacChicken`: training dataset/config/env changes.
