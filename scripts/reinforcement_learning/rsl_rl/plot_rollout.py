"""Plot per-step observation trajectories from a trained chicken-lift policy.

Must be launched via Isaac Sim:

  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/plot_rollout.py \\
      --task Isaac-Lift-Chicken-UR10e-v0 \\
      --num_envs 1 \\
      --num_steps 500

  # or point to a specific checkpoint:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/plot_rollout.py \\
      --task Isaac-Lift-Chicken-UR10e-v0 \\
      --checkpoint logs/rsl_rl/ur10e_chicken_seq_grasp/2026-06-04_10-19-16/model_0.pt \\
      --num_envs 1 --num_steps 500

Outputs: rollout_observations.png  (next to the checkpoint, or in the current dir)
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ---------------------------------------------------------------------------
# CLI arguments (must be parsed BEFORE AppLauncher consumes sys.argv)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task",       type=str,   required=True,  help="Gym task ID to evaluate.")
parser.add_argument("--num_envs",   type=int,   default=1,      help="Number of parallel environments (1 recommended for clean plots).")
parser.add_argument("--num_steps",  type=int,   default=500,    help="Number of steps to record per run.")
parser.add_argument("--num_episodes", type=int, default=3,      help="Number of full episodes to record (ignored if num_steps reached first).")
parser.add_argument("--out",        type=str,   default=None,   help="Output PNG path.")
parser.add_argument("--headless",   action="store_true",        help="Run without GUI.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True   # always headless for plotting
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Post-launch imports
# ---------------------------------------------------------------------------

import os
import math

import gymnasium as gym
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

_PRISM_TRAVEL = 0.0093   # metres, full prismatic travel open→closed

_ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
]
_LEFT_JAW  = ["PrismaticJoint1", "PrismaticJoint2"]
_RIGHT_JAW = ["PrismaticJoint3", "PrismaticJoint4"]

_JID_CACHE: dict = {}

def _jids(robot, names):
    key = tuple(names)
    if key not in _JID_CACHE:
        ids, _ = robot.find_joints(list(names))
        _JID_CACHE[key] = ids
    return _JID_CACHE[key]


def collect_step(env_unwrapped, buf: dict):
    """Append one step of observations to buf (always env index 0)."""
    robot   = env_unwrapped.scene["robot"]
    chicken = env_unwrapped.scene["chicken"]
    ee      = env_unwrapped.scene["ee_frame"]

    # ---- Chicken ----
    buf["chk_x"].append(chicken.data.root_pos_w[0, 0].item())
    buf["chk_y"].append(chicken.data.root_pos_w[0, 1].item())
    buf["chk_z"].append(chicken.data.root_pos_w[0, 2].item())
    vel = chicken.data.root_lin_vel_w[0]
    buf["chk_vel"].append(vel.norm().item())
    buf["chk_vel_x"].append(vel[0].item())
    buf["chk_vel_y"].append(vel[1].item())
    buf["chk_vel_z"].append(vel[2].item())

    # left/right leg positions
    lids = _jids(robot, _LEFT_JAW)   # reuse cache for joints; find bodies directly
    left_leg_idx  = chicken.data.body_names.index("left_leg")
    right_leg_idx = chicken.data.body_names.index("right_leg")
    buf["left_leg_z"].append(chicken.data.body_pos_w[0, left_leg_idx, 2].item())
    buf["right_leg_z"].append(chicken.data.body_pos_w[0, right_leg_idx, 2].item())

    # ---- Robot arm ----
    arm_ids = _jids(robot, tuple(_ARM_JOINTS))
    arm_pos = robot.data.joint_pos[0, arm_ids].cpu().numpy()   # (6,)
    buf["arm_joints"].append(arm_pos.copy())

    # ---- Gripper jaws ----
    left_ids  = _jids(robot, tuple(_LEFT_JAW))
    right_ids = _jids(robot, tuple(_RIGHT_JAW))
    left_pos  = robot.data.joint_pos[0, left_ids].mean().item()
    right_pos = robot.data.joint_pos[0, right_ids].mean().item()
    left_cl   = min(1.0, max(0.0, -left_pos  / _PRISM_TRAVEL))
    right_cl  = min(1.0, max(0.0, -right_pos / _PRISM_TRAVEL))
    buf["left_jaw"].append(left_cl * 100.0)    # percent closed
    buf["right_jaw"].append(right_cl * 100.0)

    # ---- EE height ----
    buf["ee_z"].append(ee.data.target_pos_w[0, 0, 2].item())

    # ---- distance EE→chicken torso ----
    ee_pos  = ee.data.target_pos_w[0, 0, :3]
    chk_pos = chicken.data.root_pos_w[0, :3]
    buf["ee_chk_dist"].append((ee_pos - chk_pos).norm().item())


def empty_buf() -> dict:
    return {
        "chk_x": [], "chk_y": [], "chk_z": [],
        "chk_vel": [], "chk_vel_x": [], "chk_vel_y": [], "chk_vel_z": [],
        "left_leg_z": [], "right_leg_z": [],
        "arm_joints": [],
        "left_jaw": [], "right_jaw": [],
        "ee_z": [], "ee_chk_dist": [],
        "episode_ends": [],   # step indices where episode resets
        "reward": [],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_DARK_BG  = "#0f0f23"
_PANEL_BG = "#1a1a2e"
_GRID_CLR = "#2a2a4a"

PALETTE = {
    "cyan":   "#00d4ff",
    "green":  "#00ff88",
    "orange": "#ff9f43",
    "red":    "#ff6b6b",
    "purple": "#c97bff",
    "yellow": "#ffe66d",
    "teal":   "#81ecec",
    "pink":   "#fd79a8",
}


def _style_ax(ax, title, xlabel="Step", ylabel=""):
    ax.set_facecolor(_DARK_BG)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=7)
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(True, color=_GRID_CLR, linewidth=0.5, alpha=0.8)


def _add_episodes(ax, ends, color="#ffffff", alpha=0.25):
    for e in ends:
        ax.axvline(e, color=color, linewidth=0.7, linestyle="--", alpha=alpha)


def _lift_threshold_line(ax, z_thresh=0.06, label=True):
    ax.axhline(z_thresh, color="#ffee00", linewidth=0.8, linestyle=":", alpha=0.8)
    if label:
        ax.text(2, z_thresh + 0.002, "lift threshold", color="#ffee00", fontsize=6.5)


def make_figure(buf: dict, title_suffix: str = "") -> plt.Figure:
    steps = np.arange(len(buf["chk_z"]))
    ends  = buf["episode_ends"]
    arm_arr = np.array(buf["arm_joints"])  # (T, 6)

    fig = plt.figure(figsize=(20, 14), facecolor=_PANEL_BG)
    fig.suptitle(
        f"Rollout Observations{title_suffix}",
        color="white", fontsize=14, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.32,
                           top=0.95, bottom=0.06, left=0.06, right=0.97)

    # =========================================================
    # ROW 0: Chicken position
    # =========================================================
    ax_chk_z = fig.add_subplot(gs[0, 0])
    ax_chk_z.plot(steps, buf["chk_z"], color=PALETTE["cyan"], linewidth=1.4, label="torso")
    ax_chk_z.plot(steps, buf["left_leg_z"],  color=PALETTE["green"],  linewidth=1.0, alpha=0.8, label="left leg")
    ax_chk_z.plot(steps, buf["right_leg_z"], color=PALETTE["orange"], linewidth=1.0, alpha=0.8, label="right leg")
    _lift_threshold_line(ax_chk_z)
    _add_episodes(ax_chk_z, ends)
    _style_ax(ax_chk_z, "Chicken Height (z)", ylabel="m")
    ax_chk_z.legend(fontsize=6.5, facecolor=_DARK_BG, labelcolor="white", loc="upper right")

    ax_chk_xy = fig.add_subplot(gs[0, 1])
    ax_chk_xy.plot(steps, buf["chk_x"], color=PALETTE["cyan"],   linewidth=1.2, label="x")
    ax_chk_xy.plot(steps, buf["chk_y"], color=PALETTE["purple"], linewidth=1.2, label="y")
    _add_episodes(ax_chk_xy, ends)
    _style_ax(ax_chk_xy, "Chicken XY Position", ylabel="m")
    ax_chk_xy.legend(fontsize=6.5, facecolor=_DARK_BG, labelcolor="white")

    ax_chk_vel = fig.add_subplot(gs[0, 2])
    ax_chk_vel.plot(steps, buf["chk_vel"], color=PALETTE["yellow"], linewidth=1.2, alpha=0.9)
    ax_chk_vel.fill_between(steps, buf["chk_vel"], alpha=0.15, color=PALETTE["yellow"])
    _add_episodes(ax_chk_vel, ends)
    _style_ax(ax_chk_vel, "Chicken Speed |v|", ylabel="m/s")
    # annotate mean speed during grasp (when jaw >50% closed)
    left_cl = np.array(buf["left_jaw"]) / 100.0
    grasping_mask = left_cl > 0.5
    if grasping_mask.any():
        mean_gsp_vel = np.array(buf["chk_vel"])[grasping_mask].mean()
        ax_chk_vel.axhline(mean_gsp_vel, color=PALETTE["red"], linewidth=0.8, linestyle=":")
        ax_chk_vel.text(2, mean_gsp_vel + 0.005, f"mean while grasping: {mean_gsp_vel:.3f}",
                        color=PALETTE["red"], fontsize=6.5)

    # =========================================================
    # ROW 1: EE tracking
    # =========================================================
    ax_ee = fig.add_subplot(gs[1, 0])
    ax_ee.plot(steps, buf["ee_z"],       color=PALETTE["teal"],   linewidth=1.4, label="EE z")
    ax_ee.plot(steps, buf["chk_z"],      color=PALETTE["cyan"],   linewidth=1.0, alpha=0.7, label="chicken z")
    _lift_threshold_line(ax_ee, label=False)
    _add_episodes(ax_ee, ends)
    _style_ax(ax_ee, "EE vs Chicken Height", ylabel="m")
    ax_ee.legend(fontsize=6.5, facecolor=_DARK_BG, labelcolor="white")

    ax_dist = fig.add_subplot(gs[1, 1])
    ax_dist.plot(steps, buf["ee_chk_dist"], color=PALETTE["pink"], linewidth=1.4)
    ax_dist.fill_between(steps, buf["ee_chk_dist"], alpha=0.12, color=PALETTE["pink"])
    _add_episodes(ax_dist, ends)
    _style_ax(ax_dist, "EE → Chicken Distance", ylabel="m")
    ax_dist.axhline(0.08, color=PALETTE["yellow"], linewidth=0.8, linestyle=":")
    ax_dist.text(2, 0.082, "grasp range", color=PALETTE["yellow"], fontsize=6.5)

    ax_rew = fig.add_subplot(gs[1, 2])
    if buf["reward"]:
        rew = np.array(buf["reward"])
        ax_rew.plot(steps[:len(rew)], rew, color=PALETTE["green"], linewidth=1.2, alpha=0.8)
        ax_rew.fill_between(steps[:len(rew)], rew, alpha=0.12, color=PALETTE["green"])
        # cumulative reward
        ax2 = ax_rew.twinx()
        ax2.plot(steps[:len(rew)], np.cumsum(rew), color=PALETTE["orange"],
                 linewidth=1.0, linestyle="--", alpha=0.7)
        ax2.set_ylabel("Cumulative", color=PALETTE["orange"], fontsize=7)
        ax2.tick_params(colors=PALETTE["orange"], labelsize=7)
        ax2.set_facecolor(_DARK_BG)
    _add_episodes(ax_rew, ends)
    _style_ax(ax_rew, "Step Reward", ylabel="reward")

    # =========================================================
    # ROW 2: Arm joints
    # =========================================================
    ax_arm = fig.add_subplot(gs[2, :2])
    arm_colors = [PALETTE["cyan"], PALETTE["green"], PALETTE["orange"],
                  PALETTE["red"],  PALETTE["purple"], PALETTE["yellow"]]
    for i, (jname, col) in enumerate(zip(_ARM_JOINTS, arm_colors)):
        ax_arm.plot(steps, arm_arr[:, i], color=col, linewidth=1.1,
                    label=jname.replace("_joint", ""))
    _add_episodes(ax_arm, ends)
    _style_ax(ax_arm, "Arm Joint Angles (rad)", ylabel="rad")
    ax_arm.legend(fontsize=6.5, facecolor=_DARK_BG, labelcolor="white",
                  ncol=3, loc="upper right")

    # joint velocity (approximate from diff)
    if len(steps) > 1:
        ax_jv = fig.add_subplot(gs[2, 2])
        arm_vel = np.diff(arm_arr, axis=0)
        ax_jv.plot(steps[1:], np.abs(arm_vel).sum(axis=1),
                   color=PALETTE["teal"], linewidth=1.2, alpha=0.9)
        ax_jv.fill_between(steps[1:], np.abs(arm_vel).sum(axis=1),
                           alpha=0.12, color=PALETTE["teal"])
        _add_episodes(ax_jv, ends)
        _style_ax(ax_jv, "Arm Activity (Σ|Δθ|)", ylabel="rad/step")

    # =========================================================
    # ROW 3: Gripper jaws
    # =========================================================
    ax_lj = fig.add_subplot(gs[3, 0])
    ax_lj.plot(steps, buf["left_jaw"],  color=PALETTE["green"],  linewidth=1.6)
    ax_lj.fill_between(steps, buf["left_jaw"], alpha=0.15, color=PALETTE["green"])
    ax_lj.axhline(50, color="white", linewidth=0.6, linestyle=":", alpha=0.5)
    ax_lj.set_ylim(-5, 105)
    _add_episodes(ax_lj, ends)
    _style_ax(ax_lj, "Left Jaw Closure %", ylabel="% closed")

    ax_rj = fig.add_subplot(gs[3, 1])
    ax_rj.plot(steps, buf["right_jaw"], color=PALETTE["orange"], linewidth=1.6)
    ax_rj.fill_between(steps, buf["right_jaw"], alpha=0.15, color=PALETTE["orange"])
    ax_rj.axhline(50, color="white", linewidth=0.6, linestyle=":", alpha=0.5)
    ax_rj.set_ylim(-5, 105)
    _add_episodes(ax_rj, ends)
    _style_ax(ax_rj, "Right Jaw Closure %", ylabel="% closed")

    # ---- grasp phase timeline ----
    ax_phase = fig.add_subplot(gs[3, 2])
    left_cl  = np.array(buf["left_jaw"])  / 100.0
    right_cl = np.array(buf["right_jaw"]) / 100.0
    chk_z    = np.array(buf["chk_z"])

    phase_left  = (left_cl > 0.3).astype(float)
    phase_right = (right_cl > 0.3).astype(float)
    phase_lift  = (chk_z > 0.06).astype(float)

    ax_phase.stackplot(
        steps,
        phase_left, phase_right, phase_lift,
        colors=[PALETTE["green"], PALETTE["orange"], PALETTE["cyan"]],
        alpha=0.6,
    )
    _add_episodes(ax_phase, ends)
    _style_ax(ax_phase, "Grasp Phase Timeline", ylabel="active")
    legend_items = [
        Line2D([0], [0], color=PALETTE["green"],  linewidth=4, label="L-jaw ≥30%"),
        Line2D([0], [0], color=PALETTE["orange"], linewidth=4, label="R-jaw ≥30%"),
        Line2D([0], [0], color=PALETTE["cyan"],   linewidth=4, label="Chicken lifted"),
    ]
    ax_phase.legend(handles=legend_items, fontsize=6.5, facecolor=_DARK_BG,
                    labelcolor="white", loc="upper left")

    # ---- episode-boundary legend ----
    if ends:
        fig.text(
            0.01, 0.01,
            "  ╌╌╌  episode reset",
            color="white", alpha=0.5, fontsize=7,
        )

    return fig


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1    # single env → clean individual trajectory
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- resolve checkpoint ----
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    # ---- env + policy ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- rollout ----
    buf    = empty_buf()
    obs, _ = env.get_observations()
    env_u  = env.unwrapped
    episodes_done = 0
    step = 0

    while step < args_cli.num_steps and episodes_done < args_cli.num_episodes:
        with torch.no_grad():
            actions = policy(obs)
        obs, rew, terminated, truncated, _ = env.step(actions)

        collect_step(env_u, buf)
        buf["reward"].append(rew[0].item())

        done = (terminated[0] | truncated[0]).item()
        if done:
            buf["episode_ends"].append(step)
            episodes_done += 1

        step += 1

    env.close()
    print(f"[INFO] Collected {step} steps across {episodes_done} episode resets.")

    # ---- plot ----
    ckpt_name = Path(resume_path).stem
    fig = make_figure(buf, title_suffix=f"  [{ckpt_name}]")

    out_path = args_cli.out if args_cli.out else os.path.join(log_dir, "rollout_observations.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[INFO] Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    simulation_app.close()
