# Humanoid Base Frame Fix (`base +X` alignment)

## 1. Create fixed USD

```bash
./isaaclab.sh -p my_scripts/fix_humanoid_base_frame.py --headless \
  --input-usd my_assets/humanoid_moveable_default_params.usd \
  --output-usd my_assets/humanoid_moveable_default_params_base_x.usd \
  --robot-root /humanoid --base-link base_link --yaw-deg -90
```

Notes:
- `yaw-deg -90` is the typical case when current visual front is `-Y`.
- If direction is opposite, try `yaw-deg 90`.

## 2. Verify heading axis

```bash
./isaaclab.sh -p my_scripts/debug_heading_axis.py --num_envs 1 --max_steps 600 --print_every 20
```

Check:
- red arrow on robot (`base +X`) should point to the robot visual front.

## 3. Switch robot config to fixed USD

Edit:
- `source/isaaclab_assets/isaaclab_assets/robots/class_humanoid.py`

Set:

```python
usd_path="my_assets/humanoid_moveable_default_params_base_x.usd"
```

## 4. Re-train

Use your existing train command after step 3.
