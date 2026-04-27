#!/usr/bin/env python3
"""Debug utility for base-link contact and termination behavior."""

import argparse
import sys

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


parser = argparse.ArgumentParser(description="Debug base_link contact/termination for locomotion tasks.")
parser.add_argument("--task", type=str, default="Isaac-Step-ClassHumanoid-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps.")
parser.add_argument("--print_every", type=int, default=20, help="Print period in steps.")
parser.add_argument("--contact_debug_vis", action="store_true", help="Enable contact sensor debug visualization.")
parser.add_argument("--marker", action="store_true", help="Show a red frame marker at base_link.")
parser.add_argument("--random_actions", action="store_true", help="Use random actions instead of zero actions.")
parser.add_argument("--stop_on_done", action="store_true", help="Stop when any environment terminates.")
parser.add_argument("--topk_bodies", type=int, default=0, help="If >0, print top-k contact-force bodies for env_0.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import euler_xyz_from_quat

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _make_env(task: str, num_envs: int, device: str, contact_debug_vis: bool):
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    if getattr(env_cfg.scene, "contact_forces", None) is not None:
        env_cfg.scene.contact_forces.debug_vis = contact_debug_vis
    return gym.make(task, cfg=env_cfg)


def main():
    env = _make_env(
        task=args_cli.task,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        contact_debug_vis=args_cli.contact_debug_vis,
    )
    env_unwrapped = env.unwrapped
    robot = env_unwrapped.scene["robot"]
    contact_sensor = env_unwrapped.scene["contact_forces"]

    if "base_link" not in robot.data.body_names:
        raise RuntimeError("base_link not found in robot body names.")
    robot_base_id = robot.data.body_names.index("base_link")

    sensor_base_ids, sensor_base_names = contact_sensor.find_bodies("base_link")
    if len(sensor_base_ids) == 0:
        raise RuntimeError("base_link not found in contact sensor body names.")
    sensor_base_id = sensor_base_ids[0]

    print("[DEBUG] robot base id:", robot_base_id, flush=True)
    print("[DEBUG] sensor base id:", sensor_base_id, "name:", sensor_base_names[0], flush=True)
    print("[DEBUG] first 20 sensor bodies:", contact_sensor.body_names[:20], flush=True)

    marker = None
    if args_cli.marker:
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/base_link_debug"
        marker_cfg.markers["frame"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
        marker = VisualizationMarkers(marker_cfg)

    _ = env.reset()
    action_dim = env_unwrapped.action_manager.total_action_dim

    for step in range(args_cli.steps):
        if args_cli.random_actions:
            actions = 2.0 * torch.rand((env_unwrapped.num_envs, action_dim), device=env_unwrapped.device) - 1.0
        else:
            actions = torch.zeros((env_unwrapped.num_envs, action_dim), device=env_unwrapped.device)

        _, _, terminated, truncated, _ = env.step(actions)
        done = torch.as_tensor(terminated, device=env_unwrapped.device) | torch.as_tensor(
            truncated, device=env_unwrapped.device
        )

        base_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_base_id, :], dim=-1)
        body_force = torch.norm(contact_sensor.data.net_forces_w, dim=-1)
        root_h = robot.data.root_pos_w[:, 2]
        roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)

        if marker is not None:
            marker.visualize(
                translations=robot.data.body_pos_w[:, robot_base_id, :],
                orientations=robot.data.body_quat_w[:, robot_base_id, :],
            )

        if step % args_cli.print_every == 0 or bool(done.any()):
            print(
                f"[STEP {step:04d}] "
                f"max|F_base|={float(base_force.max()):8.3f} "
                f"mean_h={float(root_h.mean()):6.3f} "
                f"mean|roll|={float(torch.abs(roll).mean()):6.3f} "
                f"mean|pitch|={float(torch.abs(pitch).mean()):6.3f} "
                f"done={int(done.sum())}/{done.numel()}"
            , flush=True)
            if args_cli.topk_bodies > 0:
                k = min(args_cli.topk_bodies, body_force.shape[1])
                vals, idx = torch.topk(body_force[0], k=k, largest=True, sorted=True)
                summary = ", ".join(
                    f"{contact_sensor.body_names[int(i)]}:{float(v):.2f}" for v, i in zip(vals.tolist(), idx.tolist())
                )
                print(f"           top{k}_forces(env0): {summary}", flush=True)

        if args_cli.stop_on_done and bool(done.any()):
            print("[DEBUG] stop_on_done triggered.", flush=True)
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
