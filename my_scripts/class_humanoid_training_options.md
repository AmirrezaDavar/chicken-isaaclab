# Class Humanoid オプション実行メモ

## Task ID 一覧

- 元設定（ベースライン）  
  `Isaac-Velocity-Rough-ClassHumanoid-v0`
- Option 1: 非足部接触ペナルティ  
  `Isaac-Velocity-Rough-ClassHumanoid-ContactPenalty-v0`
- Option 2: 姿勢崩れで終了（`bad_orientation`）  
  `Isaac-Velocity-Rough-ClassHumanoid-BadOrientation-v0`
- Option 3: `base_contact` の対象部位を拡張  
  `Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0`
- Option 4: 足上げ促進チューニング（Option 3ベース）  
  `Isaac-Velocity-Rough-ClassHumanoid-FootLift-v0`

定義ファイル:
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/__init__.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/class_robot/rough_env_cfg.py`

## Joint名 / Link名で知っておくこと

- `joint` と `body(link)` は別物。
- `joint名` を使う場所:
  - `CLASS_HUMANOID_CFG.init_state.joint_pos` のキー
  - `ImplicitActuatorCfg.joint_names_expr`
- `link名(body名)` を使う場所:
  - `SceneEntityCfg(..., body_names=...)`
  - 例: `base_contact`, `undesired_contacts`, `feet_air_time` の接触判定対象

### 現在の制御対象 joint名（20）

- `Left_Hip_Pitch_RS04`, `Left_Hip_Roll_RS03`, `Left_Hip_Yaw_RS03`, `Left_Knee_RS04`, `Left_Ankle_RS00`
- `Right_Hip_Pitch_RS04`, `Right_Hip_Roll_RS03`, `Right_Hip_Yaw_RS03`, `Right_Knee_RS04`, `Right_Ankle_RS00`
- `Left_Shoulder_Pitch_RS03`, `Left_Shoulder_Roll_RS03`, `Left_Shoulder_Yaw_RS02`, `Left_Elbow_RS02`, `Left_Wrist_RS00`
- `Right_Shoulder_Pitch_RS03`, `Right_Shoulder_Roll_RS03`, `Right_Shoulder_Yaw_RS02`, `Right_Elbow_RS02`, `Right_Wrist_RS00`

### 現在の主要 link名（接触判定で使う）

- `base_link`, `Head_1`, `Hip_1`
- `HipYoke_Left_1`, `HipYoke_Right_1`
- `UpperThigh_Left_1`, `UpperThigh_Right_1`
- `LowerThigh_Left_1`, `LowerThigh_Right_1`
- `Shin_Left_1`, `Shin_Right_1`
- `Foot_Left_1`, `Foot_Right_1`
- `Shoulder_Left_1`, `Shoulder_Right_1`
- `UpBicep_Left_1`, `UpBicep_Right_1`
- `LowBicep_Left_1`, `LowBicep_Right_1`
- `Forearm_Left_1`, `Forearm_Right_1`
- `Wrist_Left_1`, `Wrist_Right_1`

### 正規表現の注意（重要）

- `body_names` / `joint_names_expr` は「一致しない式」があると `ValueError` で停止する。
- `Neck_.*`, `Torso_.*`, `Chest_.*` のように、USD内に無い名前は使えない。
- 迷ったらまず `.*` を使って左右をまとめる:
  - 例: `Shoulder_.*`, `UpperThigh_.*`, `Wrist_.*`

### 名前の確認手順

1. 学習前にロボットが stage に存在する状態を作る。
2. `my_scripts/get_joint_params.py` で CSV ダンプする:

```bash
source .venv/bin/activate
./isaaclab.sh -p my_scripts/get_joint_params.py \
  --robot-root /World/envs/env_0/Robot \
  --out-dir /tmp/class_humanoid_params \
  --asset-cfg isaaclab_assets:CLASS_HUMANOID_CFG
```

`--robot-root` は stage 内の実際の prim path に合わせる（例: `/World/robot` や `/World/envs/env_0/Robot`）。

3. 確認ファイル:
   - `/tmp/class_humanoid_params/joints.csv`
   - `/tmp/class_humanoid_params/rigid_bodies.csv`
   - `/tmp/class_humanoid_params/actuator_cfg.csv`

## 学習コマンド（RSL-RL）

```bash
source .venv/bin/activate
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs 1024 --max_iterations 2000 --seed 42 \
  --experiment_name class_humanoid_opts --run_name opt1_contact \
  --task Isaac-Velocity-Rough-ClassHumanoid-ContactPenalty-v0
```

`--task` を差し替えて他オプションを実行:
- `Isaac-Velocity-Rough-ClassHumanoid-v0`
- `Isaac-Velocity-Rough-ClassHumanoid-BadOrientation-v0`
- `Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0`
- `Isaac-Velocity-Rough-ClassHumanoid-FootLift-v0`

## 再生コマンド（学習済みモデル）

### 1. 使う run と checkpoint を決める

```bash
source .venv/bin/activate
ls -1dt logs/rsl_rl/class_humanoid_rough/* | head
ls -1v logs/rsl_rl/class_humanoid_rough/<run_dir>/model_*.pt | tail -1
```

`<run_dir>` には例として `2026-03-09_19-33-24_opt3_contact` などを入れる。

### 2. 学習した task と同じ task で再生（推奨）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/class_humanoid_rough/2026-03-09_19-33-24_opt3_contact/model_300.pt
```

### 3. 軽量な Play 用 task で再生

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ClassHumanoid-Play-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/class_humanoid_rough/2026-03-09_19-33-24_opt3_contact/model_300.pt
```

### 4. 動画保存（必要なら）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/class_humanoid_rough/2026-03-09_19-33-24_opt3_contact/model_300.pt \
  --video --video_length 500
```

保存先は `logs/rsl_rl/<experiment>/<run>/videos/play`。

### 5. 追従カメラ + 高解像度で録画（推奨）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ClassHumanoid-ExtendedBaseContact-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/class_humanoid_rough/2026-03-09_19-33-24_opt3_contact/model_300.pt \
  --video --video_length 800 \
  env.viewer.origin_type=asset_body \
  env.viewer.asset_name=robot \
  env.viewer.body_name=base_link \
  env.viewer.env_index=0 \
  env.viewer.eye='[3.4,-1.6,2.0]' \
  env.viewer.lookat='[0.0,0.0,0.9]' \
  env.viewer.resolution='[1920,1080]'
```

ポイント:
- `origin_type=asset_body` でロボット（`base_link`）追従になる。
- `eye/lookat` は「追従原点からの相対位置」。
- `resolution` を上げると動画品質が改善する。
- `Expected: NoneType, Received: str` が出る場合は、`rough_env_cfg.py` 側で `viewer.asset_name/body_name` の既定値が未設定。最新版では修正済み。

画角をさらに広げたい場合:

```bash
env.viewer.eye='[4.2,-2.1,2.4]'
```

---

### 最小コマンド（汎用）

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Rough-ClassHumanoid-ContactPenalty-v0 \
  --checkpoint /absolute/path/to/model_XXXX.pt
```

## 比較時の注意

- 比較実験では `--seed`, `--num_envs`, `--max_iterations` を揃える。
- 変えるのは `--task` と `--run_name` のみにすると差分を見やすい。
