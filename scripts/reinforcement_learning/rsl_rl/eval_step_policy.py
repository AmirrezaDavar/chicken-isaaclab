#!/usr/bin/env python3
"""Evaluate Class Humanoid stepping checkpoints with task-specific metrics."""

import argparse
import json
import os
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint on a Class Humanoid stepping task.")
parser.add_argument("--task", type=str, default="Isaac-Step-ClassHumanoid-Shaping-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=2000, help="Number of policy steps to evaluate.")
parser.add_argument("--video", action="store_true", help="Record a rollout video.")
parser.add_argument("--video_length", type=int, default=1200, help="Video length in environment steps.")
parser.add_argument("--metrics_path", type=str, default=None, help="Optional JSON output path for metrics.")
parser.add_argument("--contact_threshold", type=float, default=1.0, help="Foot contact force threshold in N.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


FOOT_BODY_NAMES = ["Foot_Left_1", "Foot_Right_1"]


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _quantile_or_nan(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    tensor = torch.tensor(values, dtype=torch.float32)
    return float(torch.quantile(tensor, q).item())


def _get_cached_foot_body_ids(env) -> tuple[list[int], list[int]]:
    unwrapped = env.unwrapped
    cache = getattr(unwrapped, "_eval_step_policy_foot_id_cache", None)
    if cache is None:
        robot = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene["contact_forces"]
        foot_body_ids = [robot.data.body_names.index(name) for name in FOOT_BODY_NAMES]
        sensor_foot_ids, _ = contact_sensor.find_bodies(FOOT_BODY_NAMES)
        if len(sensor_foot_ids) != 2:
            raise RuntimeError(f"Expected two foot bodies in contact sensor, got {sensor_foot_ids}.")
        cache = (foot_body_ids, sensor_foot_ids)
        setattr(unwrapped, "_eval_step_policy_foot_id_cache", cache)
    return cache


def _foot_metrics(env, contact_threshold: float) -> dict[str, torch.Tensor]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    contact_sensor = unwrapped.scene["contact_forces"]
    foot_body_ids, sensor_foot_ids = _get_cached_foot_body_ids(env)

    swing_command = unwrapped.command_manager.get_command("swing_foot")[:, 0]
    target_xy_b = unwrapped.command_manager.get_command("target_foot_pos_xy")[:, :2]
    selected_idx = torch.where(
        swing_command > 0.0,
        torch.ones(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device),
        torch.zeros(unwrapped.num_envs, dtype=torch.long, device=unwrapped.device),
    )
    support_idx = 1 - selected_idx
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)

    foot_pos_w = robot.data.body_pos_w[:, foot_body_ids, :3]
    selected_foot_w = foot_pos_w[env_ids, selected_idx]
    support_foot_w = foot_pos_w[env_ids, support_idx]

    target_pos_b = torch.cat([target_xy_b, torch.zeros((unwrapped.num_envs, 1), device=unwrapped.device)], dim=1)
    target_pos_w, _ = math_utils.combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_pos_b)
    placement_error = torch.norm(selected_foot_w[:, :2] - target_pos_w[:, :2], dim=1)

    contact_force = torch.max(
        torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_foot_ids], dim=-1),
        dim=1,
    )[0]
    contacts = contact_force > contact_threshold
    swing_contact = contacts[env_ids, selected_idx]
    support_contact = contacts[env_ids, support_idx]
    active_swing = torch.logical_and(~swing_contact, support_contact)

    foot_clearance = selected_foot_w[:, 2] - support_foot_w[:, 2]
    both_air = torch.logical_and(~contacts[:, 0], ~contacts[:, 1])
    both_contact = torch.logical_and(contacts[:, 0], contacts[:, 1])

    return {
        "placement_error": placement_error,
        "active_placement_error": placement_error[active_swing],
        "active_swing": active_swing.float(),
        "swing_contact": swing_contact.float(),
        "support_contact": support_contact.float(),
        "both_air": both_air.float(),
        "both_contact": both_contact.float(),
        "foot_clearance": foot_clearance,
        "active_foot_clearance": foot_clearance[active_swing],
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_folder = None
    if args_cli.video:
        video_folder = os.path.join(log_dir, "videos", "eval")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    totals: dict[str, float] = defaultdict(float)
    placement_errors: list[float] = []
    active_placement_errors: list[float] = []
    active_clearances: list[float] = []
    done_count = 0
    timeout_count = 0
    terminated_count = 0

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    num_steps = min(args_cli.steps, args_cli.video_length) if args_cli.video else args_cli.steps
    for _ in range(num_steps):
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            policy_nn.reset(dones)

            metrics = _foot_metrics(env, args_cli.contact_threshold)
            for key in (
                "active_swing",
                "swing_contact",
                "support_contact",
                "both_air",
                "both_contact",
                "foot_clearance",
            ):
                totals[key] += float(metrics[key].mean().item())
            placement_errors.extend(metrics["placement_error"].detach().cpu().tolist())
            active_placement_errors.extend(metrics["active_placement_error"].detach().cpu().tolist())
            active_clearances.extend(metrics["active_foot_clearance"].detach().cpu().tolist())

            done_bool = dones.bool()
            timeouts = extras.get("time_outs", torch.zeros_like(done_bool, dtype=torch.bool)).bool()
            done_count += int(done_bool.sum().item())
            timeout_count += int(torch.logical_and(done_bool, timeouts).sum().item())
            terminated_count += int(torch.logical_and(done_bool, ~timeouts).sum().item())

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    denom_steps = max(num_steps, 1)
    denom_env_steps = max(num_steps * env.num_envs, 1)
    metrics_out = {
        "task": args_cli.task,
        "checkpoint_path": resume_path,
        "video_folder": video_folder,
        "num_envs": env.num_envs,
        "steps": num_steps,
        "foot_placement_error_mean_m": _mean_or_nan(placement_errors),
        "foot_placement_error_p50_m": _quantile_or_nan(placement_errors, 0.50),
        "foot_placement_error_p90_m": _quantile_or_nan(placement_errors, 0.90),
        "active_foot_placement_error_mean_m": _mean_or_nan(active_placement_errors),
        "active_foot_placement_error_p90_m": _quantile_or_nan(active_placement_errors, 0.90),
        "active_swing_ratio": totals["active_swing"] / denom_steps,
        "swing_contact_ratio": totals["swing_contact"] / denom_steps,
        "support_contact_ratio": totals["support_contact"] / denom_steps,
        "both_air_ratio": totals["both_air"] / denom_steps,
        "both_contact_ratio": totals["both_contact"] / denom_steps,
        "mean_foot_clearance_m": totals["foot_clearance"] / denom_steps,
        "active_mean_foot_clearance_m": _mean_or_nan(active_clearances),
        "done_rate_per_env_step": done_count / denom_env_steps,
        "terminated_rate_per_env_step": terminated_count / denom_env_steps,
        "timeout_rate_per_env_step": timeout_count / denom_env_steps,
        "done_count": done_count,
        "terminated_count": terminated_count,
        "timeout_count": timeout_count,
    }

    print("[EVAL_METRICS]")
    print(json.dumps(metrics_out, indent=2, sort_keys=True))
    if args_cli.metrics_path:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics_path)), exist_ok=True)
        with open(args_cli.metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, indent=2, sort_keys=True)
            f.write("\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
