#!/usr/bin/env python3
"""
Visualise low-dim data from replay_buffer.zarr collected by collect_isaac_demos.py

Usage:
  # Overview of all episodes:
  python scripts/imitation_learning/plot_demos.py --zarr_path ./data/isaac_chicken/replay_buffer.zarr

  # Single episode detail (0-indexed):
  python scripts/imitation_learning/plot_demos.py --zarr_path ./data/isaac_chicken/replay_buffer.zarr --episode 0
"""

import argparse
import sys
import numpy as np
import zarr
import matplotlib
matplotlib.use("TkAgg")          # change to "Agg" if no display → saves PNG instead
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--zarr_path", type=str, default="./data/isaac_chicken/replay_buffer.zarr")
parser.add_argument("--episode",   type=int, default=None,
                    help="Plot a single episode in detail (0-indexed). "
                         "Omit to show dataset overview.")
parser.add_argument("--save",      type=str, default=None,
                    help="Save figure to this PNG path instead of showing interactively.")
args = parser.parse_args()

# ── Load zarr ─────────────────────────────────────────────────────────────────
store = zarr.DirectoryStore(args.zarr_path)
root  = zarr.open_group(store, mode="r")
data  = root["data"]
meta  = root["meta"]

episode_ends = meta["episode_ends"][:]        # (N,) cumulative
N_ep   = len(episode_ends)
T_total = int(episode_ends[-1]) if N_ep > 0 else 0

starts = np.concatenate([[0], episode_ends[:-1]])
ends   = episode_ends

print(f"Dataset: {N_ep} episodes, {T_total} total steps")
print(f"Arrays : {sorted(data.keys())}")

# ── Helpers ───────────────────────────────────────────────────────────────────
ARM_JOINT_LABELS  = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]
EEF_POSE_LABELS   = ["ee_x", "ee_y", "ee_z", "roll", "pitch", "yaw"]
EEF_VEL_LABELS    = ["vx",   "vy",   "vz",   "wx",   "wy",    "wz"]
ACTION_LABELS     = ["a_pan","a_lift","a_elbow","a_w1","a_w2","a_w3","L_jaw","R_jaw"]

COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]

def ep_slice(ep_idx):
    return slice(int(starts[ep_idx]), int(ends[ep_idx]))

def ep_time(ep_idx):
    T = int(ends[ep_idx]) - int(starts[ep_idx])
    ts = data["timestamp"][ep_slice(ep_idx), 0]
    return ts if ts[-1] > 0 else np.arange(T) * 0.02

def lifted(ep_idx):
    """True if stage==1 at any point in the episode."""
    return bool(data["stage"][ep_slice(ep_idx), 0].max() > 0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  MODE A — Single episode detail
# ══════════════════════════════════════════════════════════════════════════════
if args.episode is not None:
    ep = args.episode
    if ep >= N_ep:
        sys.exit(f"Episode {ep} does not exist (dataset has {N_ep} episodes).")

    sl   = ep_slice(ep)
    t    = ep_time(ep)
    suc  = lifted(ep)
    T    = len(t)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Episode {ep}  |  {T} steps  |  {'✓ lifted' if suc else '✗ not lifted'}",
                 fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Arm joints ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    jpos = data["robot_joint"][sl]           # (T, 6) rad
    for i, lbl in enumerate(ARM_JOINT_LABELS):
        ax1.plot(t, np.rad2deg(jpos[:, i]), label=lbl)
    ax1.set_title("Arm joint positions (°)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("degrees")
    ax1.legend(ncol=3, fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Arm joint velocities ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    jvel = data["robot_joint_vel"][sl]
    for i, lbl in enumerate(ARM_JOINT_LABELS):
        ax2.plot(t, np.rad2deg(jvel[:, i]), label=lbl)
    ax2.set_title("Arm joint velocities (°/s)")
    ax2.set_xlabel("time (s)")
    ax2.legend(ncol=2, fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── 3. EE position ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    eef = data["robot_eef_pose"][sl]         # (T, 6) [pos(3), euler(3)]
    for i, lbl in enumerate(["ee_x", "ee_y", "ee_z"]):
        ax3.plot(t, eef[:, i], label=lbl)
    ax3.set_title("EE position (m, robot frame)")
    ax3.set_xlabel("time (s)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. EE orientation ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for i, lbl in enumerate(["roll", "pitch", "yaw"]):
        ax4.plot(t, np.rad2deg(eef[:, 3+i]), label=lbl)
    ax4.set_title("EE orientation (°)")
    ax4.set_xlabel("time (s)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 5. Gripper jaws + stage ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    lj = data["left_jaw"][sl,  0]
    rj = data["right_jaw"][sl, 0]
    st = data["stage"][sl, 0]
    ax5.plot(t, lj, label="left jaw",  color="tab:blue")
    ax5.plot(t, rj, label="right jaw", color="tab:orange")
    ax5.fill_between(t, 0, st, alpha=0.15, color="green", label="stage=1 (lifted)")
    ax5.set_title("Gripper closure & stage")
    ax5.set_xlabel("time (s)")
    ax5.set_ylim(-0.05, 1.1)
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── 6. Actions (arm) ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    act = data["action"][sl]                 # (T, 8)
    for i, lbl in enumerate(ARM_JOINT_LABELS):
        ax6.plot(t, np.rad2deg(act[:, i]), label=lbl)
    ax6.set_title("Action — arm joint targets (°)")
    ax6.set_xlabel("time (s)")
    ax6.set_ylabel("degrees")
    ax6.legend(ncol=3, fontsize=8)
    ax6.grid(True, alpha=0.3)

    # ── 7. Actions (gripper) ──────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.step(t, act[:, 6], label="left jaw cmd",  where="post", color="tab:blue")
    ax7.step(t, act[:, 7], label="right jaw cmd", where="post", color="tab:orange")
    ax7.set_title("Action — jaw commands (+1=open, -1=close)")
    ax7.set_xlabel("time (s)")
    ax7.set_ylim(-1.3, 1.3)
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])


# ══════════════════════════════════════════════════════════════════════════════
#  MODE B — Dataset overview (all episodes)
# ══════════════════════════════════════════════════════════════════════════════
else:
    ep_lengths = (ends - starts).astype(int)
    ep_success = np.array([lifted(i) for i in range(N_ep)])

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Dataset overview — {N_ep} episodes, {T_total} steps total",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.35)

    ep_ids = np.arange(N_ep)
    colors = ["tab:green" if s else "tab:red" for s in ep_success]

    # ── 1. Episode lengths ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(ep_ids, ep_lengths, color=colors, edgecolor="white", linewidth=0.3)
    ax1.set_title(f"Episode lengths  |  success (green): {ep_success.sum()}/{N_ep}")
    ax1.set_xlabel("episode index")
    ax1.set_ylabel("steps")
    ax1.grid(True, alpha=0.3, axis="y")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color="tab:green", label="lifted"),
                        Patch(color="tab:red",   label="not lifted")], fontsize=9)

    # ── 2. Arm joints — all episodes overlaid ────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :2])
    for ep in range(N_ep):
        sl = ep_slice(ep)
        t  = ep_time(ep)
        jpos = data["robot_joint"][sl]
        alpha = 0.6 / max(N_ep, 1) + 0.1
        for i in range(6):
            ax2.plot(t, np.rad2deg(jpos[:, i]),
                     color=COLOR_CYCLE[i % len(COLOR_CYCLE)], alpha=alpha, linewidth=0.7)
    # Legend via dummy lines
    for i, lbl in enumerate(ARM_JOINT_LABELS):
        ax2.plot([], [], color=COLOR_CYCLE[i % len(COLOR_CYCLE)], label=lbl)
    ax2.set_title("Arm joint positions — all episodes (°)")
    ax2.set_xlabel("time (s)")
    ax2.legend(ncol=3, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── 3. Gripper jaw closure — all episodes ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    for ep in range(N_ep):
        sl = ep_slice(ep)
        t  = ep_time(ep)
        alpha = 0.5 / max(N_ep, 1) + 0.1
        ax3.plot(t, data["left_jaw"][sl, 0],  color="tab:blue",   alpha=alpha, linewidth=0.8)
        ax3.plot(t, data["right_jaw"][sl, 0], color="tab:orange", alpha=alpha, linewidth=0.8)
    ax3.plot([], [], color="tab:blue",   label="left jaw")
    ax3.plot([], [], color="tab:orange", label="right jaw")
    ax3.set_title("Gripper closure [0=open, 1=closed]")
    ax3.set_xlabel("time (s)")
    ax3.set_ylim(-0.05, 1.1)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. EE XYZ — all episodes ─────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    eef_labels = ["ee_x", "ee_y", "ee_z"]
    for ep in range(N_ep):
        sl = ep_slice(ep)
        t  = ep_time(ep)
        eef = data["robot_eef_pose"][sl]
        alpha = 0.5 / max(N_ep, 1) + 0.1
        for i in range(3):
            ax4.plot(t, eef[:, i], color=COLOR_CYCLE[i], alpha=alpha, linewidth=0.8)
    for i, lbl in enumerate(eef_labels):
        ax4.plot([], [], color=COLOR_CYCLE[i], label=lbl)
    ax4.set_title("EE position (m, robot frame)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ── 5. EE euler — all episodes ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    for ep in range(N_ep):
        sl = ep_slice(ep)
        t  = ep_time(ep)
        eef = data["robot_eef_pose"][sl]
        alpha = 0.5 / max(N_ep, 1) + 0.1
        for i in range(3):
            ax5.plot(t, np.rad2deg(eef[:, 3+i]), color=COLOR_CYCLE[3+i], alpha=alpha, linewidth=0.8)
    for i, lbl in enumerate(["roll", "pitch", "yaw"]):
        ax5.plot([], [], color=COLOR_CYCLE[3+i], label=lbl)
    ax5.set_title("EE orientation (°)")
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # ── 6. Stage (lifted fraction per episode) ────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    lifted_fracs = []
    for ep in range(N_ep):
        sl = ep_slice(ep)
        st = data["stage"][sl, 0]
        lifted_fracs.append(float(st.mean()))
    ax6.bar(ep_ids, lifted_fracs, color=colors, edgecolor="white", linewidth=0.3)
    ax6.set_title("Fraction of steps with chicken lifted")
    ax6.set_xlabel("episode index")
    ax6.set_ylabel("fraction")
    ax6.set_ylim(0, 1.05)
    ax6.grid(True, alpha=0.3, axis="y")

    # ── 7. Joint position histograms ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, :])
    all_joints = data["robot_joint"][:]   # (T_total, 6)
    for i, lbl in enumerate(ARM_JOINT_LABELS):
        ax7.hist(np.rad2deg(all_joints[:, i]), bins=60, alpha=0.55, label=lbl,
                 color=COLOR_CYCLE[i % len(COLOR_CYCLE)])
    ax7.set_title("Arm joint position distribution across all episodes (°)")
    ax7.set_xlabel("degrees")
    ax7.set_ylabel("count")
    ax7.legend(ncol=6, fontsize=8)
    ax7.grid(True, alpha=0.3, axis="y")

    plt.tight_layout(rect=[0, 0, 1, 0.96])


# ── Show / Save ───────────────────────────────────────────────────────────────
if args.save:
    plt.savefig(args.save, dpi=150, bbox_inches="tight")
    print(f"Saved → {args.save}")
else:
    plt.show()
