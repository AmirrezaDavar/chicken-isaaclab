# Class Humanoid Training and Playback

Run these commands from the repository root.

## 1. Create and Activate the uv Environment

Use a named uv project environment for Isaac Sim:

```bash
UV_PROJECT_ENVIRONMENT=env_isaacsim uv sync --locked
```

Activate it:

```bash
source env_isaacsim/bin/activate
```

Optional quick check:

```bash
python --version
which python
```

## 2. Start a Training Run

This trains the final rough Class Humanoid task with RSL-RL. The task id
`Isaac-Velocity-Rough-ClassHumanoid-v0` is registered to:

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/rough_env_cfg.py
```

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless \
  --num_envs 1024 \
  --max_iterations 2000 \
  --seed 42 \
  --experiment_name class_humanoid_rough \
  --run_name final_rough \
  --task Isaac-Velocity-Rough-ClassHumanoid-v0
```

## 3. Pick a Checkpoint

List recent runs:

```bash
ls -1dt logs/rsl_rl/class_humanoid_rough/* | head
```

Pick the latest checkpoint from a run:

```bash
RUN_DIR=logs/rsl_rl/class_humanoid_rough/<run_dir>
CHECKPOINT=$(ls -1v "$RUN_DIR"/model_*.pt | tail -1)
echo "$CHECKPOINT"
```

Replace `<run_dir>` with the run directory you want to play, for example:

```bash
RUN_DIR=logs/rsl_rl/class_humanoid_rough/2026-04-22_13-30-00_final_rough
CHECKPOINT=$(ls -1v "$RUN_DIR"/model_*.pt | tail -1)
echo "$CHECKPOINT"
```

## 4. Play the Trained Policy

Use the same task that was used during training:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ClassHumanoid-v0 \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT"
```

For a lighter playback environment, use the play task:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ClassHumanoid-Play-v0 \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT"
```

## 5. Record a Playback Video

This records headless video with a follow camera attached to `base_link`:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --headless \
  --task Isaac-Velocity-Rough-ClassHumanoid-v0 \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT" \
  --video \
  --video_length 800 \
  env.viewer.origin_type=asset_body \
  env.viewer.asset_name=robot \
  env.viewer.body_name=base_link \
  env.viewer.env_index=0 \
  env.viewer.eye='[3.4,-1.6,2.0]' \
  env.viewer.lookat='[0.0,0.0,0.9]' \
  env.viewer.resolution='[1920,1080]'
```

Videos are written under:

```text
logs/rsl_rl/<experiment>/<run>/videos/play
```

To widen the camera view, increase the camera offset:

```bash
env.viewer.eye='[4.2,-2.1,2.4]'
```
