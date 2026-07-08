#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Run a trained ChicGrasp low-dimensional diffusion policy in Isaac Lab.

This evaluates checkpoints trained from collect_isaac_demos.py zarr data.
The policy observation is the same 20D state vector used for training:
  robot_joint(6), robot_joint_vel(6), robot_eef_pose(6), left_jaw(1), right_jaw(1)

Example:
  python scripts/imitation_learning/eval_chicken_diffusion_policy.py \
      --checkpoint /home/wanglab22/ChicGrasp/data/outputs/2026.07.08/14.13.51_train_diffusion_unet_lowdim_isaac_chicken_isaac_chicken_lowdim/checkpoints/latest.ckpt
"""

import argparse
import collections
import pathlib
import sys

from isaaclab.app import AppLauncher


CHICGRASP_ROOT = pathlib.Path("/home/wanglab22/ChicGrasp")
DEFAULT_CHECKPOINT = (
    CHICGRASP_ROOT
    / "data/outputs/2026.07.08/14.13.51_train_diffusion_unet_lowdim_isaac_chicken_isaac_chicken_lowdim"
    / "checkpoints/latest.ckpt"
)


parser = argparse.ArgumentParser(description="Evaluate ChicGrasp DP checkpoint in Isaac chicken-lift env.")
parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to ChicGrasp .ckpt file.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Chicken-UR10e-CustomGripper-GELLO-v0")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--episode_steps", type=int, default=300)
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--device_policy", type=str, default="cuda:0")
parser.add_argument("--chicgrasp_root", type=str, default=str(CHICGRASP_ROOT))
parser.add_argument("--action_stride", type=int, default=0, help="Actions to execute per policy call. 0 uses ckpt n_action_steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import dill  # noqa: E402
import gymnasium as gym  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


chicgrasp_root = pathlib.Path(args_cli.chicgrasp_root)
if str(chicgrasp_root) not in sys.path:
    sys.path.insert(0, str(chicgrasp_root))

OmegaConf.register_new_resolver("eval", eval, replace=True)


ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    r, p, y = euler_xyz_from_quat(t)
    return np.array([r.item(), p.item(), y.item()], dtype=np.float32)


def extract_state(env_uw) -> np.ndarray:
    scene = env_uw.scene
    robot = scene["robot"]
    rb_pos = robot.data.root_pos_w[0].cpu().numpy()

    ee_pos_w = scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    ee_quat_w = scene["ee_frame"].data.target_quat_w[0, 0].cpu().numpy()
    ee_pos_r = (ee_pos_w - rb_pos).astype(np.float32)
    ee_euler = quat_to_euler(ee_quat_w)

    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    robot_joint = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
    robot_joint_vel = robot.data.joint_vel[0, arm_ids].cpu().numpy().astype(np.float32)

    lj_ids, _ = robot.find_joints(["PrismaticJoint1", "PrismaticJoint2"])
    rj_ids, _ = robot.find_joints(["PrismaticJoint3", "PrismaticJoint4"])
    left_jaw = float(np.clip(robot.data.joint_pos[0, lj_ids].cpu().numpy() / -0.0093, 0.0, 1.0).mean())
    right_jaw = float(np.clip(robot.data.joint_pos[0, rj_ids].cpu().numpy() / -0.0093, 0.0, 1.0).mean())

    return np.concatenate(
        [
            robot_joint,
            robot_joint_vel,
            np.concatenate([ee_pos_r, ee_euler]).astype(np.float32),
            np.array([left_jaw], dtype=np.float32),
            np.array([right_jaw], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def load_policy(checkpoint_path: str, device: str):
    # ChicGrasp was trained with diffusers 0.11.x, whose top-level import asks
    # for huggingface_hub.cached_download. Modern Isaac Lab environments often
    # carry newer huggingface_hub where that symbol was removed. The low-dim
    # policy only needs schedulers, so this compatibility alias is enough.
    try:
        import huggingface_hub

        if not hasattr(huggingface_hub, "cached_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    except Exception:
        pass

    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if cfg.training.use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    policy.to(torch.device(device))
    policy.eval()
    print(f"[policy] Loaded {checkpoint_path}")
    print(f"[policy] Using checkpoint state_dicts/{state_key}")
    print(f"[policy] n_obs_steps={policy.n_obs_steps}, n_action_steps={policy.n_action_steps}")
    return policy


def make_idle_action(env_uw) -> np.ndarray:
    robot = env_uw.scene["robot"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    arm_joints = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
    return np.array([*arm_joints, 1.0, 1.0], dtype=np.float32)


def step_env(env, env_uw, action: np.ndarray):
    action_t = torch.tensor(action, dtype=torch.float32, device=env_uw.device).unsqueeze(0)
    return env.step(action_t)


def main():
    checkpoint_path = pathlib.Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = 10000.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_uw = env.unwrapped

    policy = load_policy(str(checkpoint_path), args_cli.device_policy)
    n_obs_steps = int(policy.n_obs_steps)
    stride = int(args_cli.action_stride) if args_cli.action_stride > 0 else int(policy.n_action_steps)

    for ep in range(args_cli.num_episodes):
        env.reset()
        idle = make_idle_action(env_uw)
        for _ in range(args_cli.settle_steps):
            step_env(env, env_uw, idle)

        state_hist = collections.deque(maxlen=n_obs_steps)
        state = extract_state(env_uw)
        for _ in range(n_obs_steps):
            state_hist.append(state)

        action_queue: list[np.ndarray] = []
        print(f"[eval] Episode {ep} start")

        for step in range(args_cli.episode_steps):
            if len(action_queue) == 0:
                obs_np = np.stack(list(state_hist), axis=0)[None]  # (1, To, 20)
                obs_t = torch.from_numpy(obs_np).to(args_cli.device_policy)
                with torch.no_grad():
                    result = policy.predict_action({"obs": obs_t})
                actions = result["action"][0].detach().cpu().numpy().astype(np.float32)
                action_queue.extend([actions[i] for i in range(min(stride, len(actions)))])

            action = action_queue.pop(0)
            _, _, terminated, truncated, _ = step_env(env, env_uw, action)
            state_hist.append(extract_state(env_uw))

            if step % 25 == 0:
                print(f"[eval] ep={ep} step={step} action={np.round(action, 3)}")

            if terminated or truncated:
                print(f"[eval] Episode {ep} terminated at step {step}")
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
