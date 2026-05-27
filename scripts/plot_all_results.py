"""Comprehensive training plots for CSCE50103 Isaac Lab project.

Covers:
  - Reach task (v7, v10, v11): GT-position reaching, right arm
  - Reach Left Fixed (v1): left arm, fixed base
  - Reach Depth Camera (v1): real depth camera perception, fixed base
  - Shoot Ball (v1-broken, v3-buggy, v4-working, v7-finetuned)

Run:  python scripts/plot_all_results.py
Output: logs/presentation_plots/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing import event_accumulator

OUT_DIR = "logs/presentation_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
C = {
    "v7":        "#2196F3",
    "v10":       "#FF9800",
    "v11":       "#4CAF50",
    "left_fix":  "#9C27B0",
    "depth_cam": "#00BCD4",
    "sb_v1":     "#F44336",
    "sb_v3":     "#FF5722",
    "sb_v4":     "#4CAF50",
    "sb_v7":     "#2196F3",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ── Helpers ─────────────────────────────────────────────────────────────────

def load(run_dir, tag):
    if not os.path.isdir(run_dir):
        return np.array([]), np.array([])
    evts = [f for f in os.listdir(run_dir) if f.startswith("events.out")]
    if not evts:
        return np.array([]), np.array([])
    ea = event_accumulator.EventAccumulator(
        os.path.join(run_dir, evts[0]), size_guidance={"scalars": 0}
    )
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    data = ea.Scalars(tag)
    return np.array([d.step for d in data]), np.array([d.value for d in data])


def smooth(y, w=15):
    if len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="same")


def plot_line(ax, steps, vals, label, color, w=15, alpha_fill=0.12):
    if len(steps) == 0:
        return
    s = smooth(vals, w)
    ax.plot(steps, s, color=color, label=label, linewidth=2)
    ax.fill_between(steps, s * 0.95, s * 1.05, color=color, alpha=alpha_fill)


# ── Run directories ─────────────────────────────────────────────────────────
REACH = "logs/rsl_rl/class_humanoid_reach_depth"
LEFT  = "logs/rsl_rl/class_humanoid_reach_left_fixed"
DCAM  = "logs/rsl_rl/class_humanoid_reach_depth_camera"
SHOOT = "logs/rsl_rl/class_humanoid_shoot_ball"

REACH_RUNS = {
    "reach_v7":  os.path.join(REACH, "2026-05-04_18-38-54_reach_v7"),
    "reach_v10": os.path.join(REACH, "2026-05-04_22-16-00_reach_v10"),
    "reach_v11": os.path.join(REACH, "2026-05-05_11-55-55_reach_v11"),
}
LEFT_RUN  = os.path.join(LEFT, "2026-05-06_00-55-31_reach_left_fixed_v1")
DCAM_RUN  = os.path.join(DCAM, "2026-05-06_12-19-38")

SHOOT_RUNS = {
    "v1 (broken-legs)": (os.path.join(SHOOT, "2026-05-05_18-43-22_shoot_ball_v1"), C["sb_v1"]),
    "v3 (goal-bug)":    (os.path.join(SHOOT, "2026-05-06_00-24-11_shoot_ball_v3"),  C["sb_v3"]),
    "v4 (working)":     (os.path.join(SHOOT, "2026-05-06_10-27-23"),                C["sb_v4"]),
    "v7 (finetuned)":   (os.path.join(SHOOT, "2026-05-06_12-06-11"),                C["sb_v7"]),
}

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — REACH TASK
# ════════════════════════════════════════════════════════════════════════════

# Plot 1-A: Reward comparison reach v7 / v10 / v11
fig, ax = plt.subplots(figsize=(11, 5))
for name, run in REACH_RUNS.items():
    key = name.replace("reach_", "")
    steps, vals = load(run, "Train/mean_reward")
    plot_line(ax, steps, vals, name, C[key])
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("Task 1: Right-Arm Reaching — Training Reward (GT observation)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/01_reach_reward_comparison.png")
plt.close(fig)
print("Saved 01_reach_reward_comparison.png")

# Plot 1-B: Episode length
fig, ax = plt.subplots(figsize=(11, 5))
for name, run in REACH_RUNS.items():
    key = name.replace("reach_", "")
    steps, vals = load(run, "Train/mean_episode_length")
    plot_line(ax, steps, vals, name, C[key])
ax.axhline(160, color="gray", linestyle="--", linewidth=1, label="max (8s)")
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Episode Length (steps)")
ax.set_title("Task 1: Reaching — Episode Length (longer = more stable)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_reach_episode_length.png")
plt.close(fig)
print("Saved 02_reach_episode_length.png")

# Plot 1-C: Reward breakdown for reach_v11 (best)
reward_tags_reach = {
    "reach_target":        "Episode_Reward/reach_target",
    "reach_distance":      "Episode_Reward/reach_distance",
    "arm_joint_deviation": "Episode_Reward/arm_joint_deviation",
    "undesired_contacts":  "Episode_Reward/undesired_contacts",
    "torso_tilt":          "Episode_Reward/torso_tilt_l2",
    "action_rate":         "Episode_Reward/action_rate_l2",
}
tag_colors = ["#1565C0", "#E53935", "#7B1FA2", "#F57F17", "#2E7D32", "#795548"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, (label, tag), color in zip(axes, reward_tags_reach.items(), tag_colors):
    steps, vals = load(REACH_RUNS["reach_v11"], tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 15), color=color, linewidth=2)
        ax.fill_between(steps, smooth(vals, 15) * 0.95, smooth(vals, 15) * 1.05,
                        color=color, alpha=0.15)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("Reward", fontsize=8)
fig.suptitle("reach_v11 — Individual Reward Components", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/03_reach_v11_reward_breakdown.png")
plt.close(fig)
print("Saved 03_reach_v11_reward_breakdown.png")

# Plot 1-D: Termination breakdown v11
fig, ax = plt.subplots(figsize=(11, 5))
term_map = {
    "Timeout (success)":   ("Episode_Termination/time_out",       "#4CAF50"),
    "Bad orientation":     ("Episode_Termination/bad_orientation", "#F44336"),
    "Root too low (fall)": ("Episode_Termination/root_too_low",   "#FF9800"),
}
for label, (tag, color) in term_map.items():
    steps, vals = load(REACH_RUNS["reach_v11"], tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 15), color=color, label=label, linewidth=2)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Fraction of Episodes")
ax.set_title("reach_v11 — Termination Reasons")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_reach_v11_terminations.png")
plt.close(fig)
print("Saved 04_reach_v11_terminations.png")

# Plot 1-E: Exploration decay
fig, ax = plt.subplots(figsize=(11, 4))
for name, run in REACH_RUNS.items():
    key = name.replace("reach_", "")
    steps, vals = load(run, "Policy/mean_noise_std")
    plot_line(ax, steps, vals, name, C[key], w=10)
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Action Noise Std")
ax.set_title("Reach — Policy Exploration Decay")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/05_reach_exploration_decay.png")
plt.close(fig)
print("Saved 05_reach_exploration_decay.png")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REACH LEFT FIXED
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for tag, ylabel, ax in [
    ("Train/mean_reward",         "Mean Episode Reward",       axes[0]),
    ("Train/mean_episode_length", "Mean Episode Length (steps)", axes[1]),
]:
    steps, vals = load(LEFT_RUN, tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 15), color=C["left_fix"], linewidth=2)
        ax.fill_between(steps, smooth(vals, 15) * 0.95, smooth(vals, 15) * 1.05,
                        color=C["left_fix"], alpha=0.15)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(ylabel)
axes[0].set_title("Task 2: Left-Arm Reach (Fixed Base) — Reward")
axes[1].set_title("Task 2: Left-Arm Reach (Fixed Base) — Episode Length")
fig.suptitle("Left-Arm Fixed-Base Reaching (3999 iterations, 512 envs)", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/06_reach_left_fixed.png")
plt.close(fig)
print("Saved 06_reach_left_fixed.png")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — REACH DEPTH CAMERA
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for tag, ylabel, ax in [
    ("Train/mean_reward",         "Mean Episode Reward",         axes[0]),
    ("Train/mean_episode_length", "Mean Episode Length (steps)", axes[1]),
]:
    steps, vals = load(DCAM_RUN, tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 5), color=C["depth_cam"], linewidth=2)
        ax.fill_between(steps, smooth(vals, 5) * 0.95, smooth(vals, 5) * 1.05,
                        color=C["depth_cam"], alpha=0.15)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel(ylabel)
axes[0].set_title("Task 3: Reach w/ Depth Camera — Reward (in progress)")
axes[1].set_title("Task 3: Reach w/ Depth Camera — Episode Length")
fig.suptitle("Depth Camera Reaching — Policy observes depth estimate, NOT ground truth",
             fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/07_reach_depth_camera.png")
plt.close(fig)
print("Saved 07_reach_depth_camera.png")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SHOOT BALL
# ════════════════════════════════════════════════════════════════════════════

# Plot 4-A: Reward comparison all versions
fig, ax = plt.subplots(figsize=(13, 6))
for name, (run, color) in SHOOT_RUNS.items():
    steps, vals = load(run, "Train/mean_reward")
    plot_line(ax, steps, vals, name, color)
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("Task 4: Shoot Ball to Target — All Versions")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/08_shoot_reward_comparison.png")
plt.close(fig)
print("Saved 08_shoot_reward_comparison.png")

# Plot 4-B: Episode length comparison
fig, ax = plt.subplots(figsize=(13, 5))
for name, (run, color) in SHOOT_RUNS.items():
    steps, vals = load(run, "Train/mean_episode_length")
    plot_line(ax, steps, vals, name, color)
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Episode Length (steps)")
ax.set_title("Shoot Ball — Episode Length (longer = more stable)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/09_shoot_episode_length.png")
plt.close(fig)
print("Saved 09_shoot_episode_length.png")

# Plot 4-C: Termination breakdown for v1 (broken) vs v4 (working)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (run_name, run_dir, color_set) in zip(axes, [
    ("v1 — broken (leg joints, robot falls)",
     os.path.join(SHOOT, "2026-05-05_18-43-22_shoot_ball_v1"),
     {"Timeout": "#4CAF50", "Bad orientation": "#F44336", "Root too low": "#FF9800"}),
    ("v4 — working (arm, fixed reward)",
     os.path.join(SHOOT, "2026-05-06_10-27-23"),
     {"Timeout": "#4CAF50", "Bad orientation": "#F44336", "Root too low": "#FF9800"}),
]):
    for label, color in color_set.items():
        tag_map = {
            "Timeout":        "Episode_Termination/time_out",
            "Bad orientation":"Episode_Termination/bad_orientation",
            "Root too low":   "Episode_Termination/root_too_low",
        }
        steps, vals = load(run_dir, tag_map[label])
        if len(steps):
            ax.plot(steps, smooth(vals, 15), color=color, label=label, linewidth=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(run_name, fontsize=9)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of Episodes")
    ax.legend(fontsize=8)
fig.suptitle("Shoot Ball — Termination Reasons: Broken vs Working", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/10_shoot_terminations_comparison.png")
plt.close(fig)
print("Saved 10_shoot_terminations_comparison.png")

# Plot 4-D: Reward breakdown for v4 (working)
reward_tags_shoot = {
    "arm_to_ball":               "Episode_Reward/arm_to_ball",
    "ball_to_goal":              "Episode_Reward/ball_to_goal",
    "ball_velocity_toward_goal": "Episode_Reward/ball_velocity_toward_goal",
    "ball_at_goal":              "Episode_Reward/ball_at_goal",
    "termination_penalty":       "Episode_Reward/termination_penalty",
    "torso_tilt":                "Episode_Reward/torso_tilt_l2",
}
tag_colors_shoot = ["#1565C0", "#E53935", "#2E7D32", "#F57F17", "#7B1FA2", "#795548"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()
for ax, (label, tag), color in zip(axes, reward_tags_shoot.items(), tag_colors_shoot):
    steps, vals = load(os.path.join(SHOOT, "2026-05-06_10-27-23"), tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 10), color=color, linewidth=2)
        ax.fill_between(steps, smooth(vals, 10) * 0.95, smooth(vals, 10) * 1.05,
                        color=color, alpha=0.15)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Iteration", fontsize=8)
    ax.set_ylabel("Reward", fontsize=8)
fig.suptitle("shoot_ball_v4 (working) — Individual Reward Components", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/11_shoot_v4_reward_breakdown.png")
plt.close(fig)
print("Saved 11_shoot_v4_reward_breakdown.png")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MASTER DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.35)

panels = [
    # Row 0: Reach tasks
    (gs[0, 0], REACH_RUNS["reach_v11"],     "Train/mean_reward",              "Reach v11 — Reward",          C["v11"]),
    (gs[0, 1], REACH_RUNS["reach_v11"],     "Train/mean_episode_length",      "Reach v11 — Episode Length",  C["v11"]),
    (gs[0, 2], LEFT_RUN,                    "Train/mean_reward",              "Left Fixed — Reward",         C["left_fix"]),
    (gs[0, 3], DCAM_RUN,                    "Train/mean_reward",              "Depth Cam — Reward",          C["depth_cam"]),
    # Row 1: Shoot ball
    (gs[1, 0], os.path.join(SHOOT,"2026-05-06_10-27-23"), "Train/mean_reward",         "Shoot v4 — Reward",           C["sb_v4"]),
    (gs[1, 1], os.path.join(SHOOT,"2026-05-06_10-27-23"), "Train/mean_episode_length", "Shoot v4 — Episode Length",   C["sb_v4"]),
    (gs[1, 2], os.path.join(SHOOT,"2026-05-06_10-27-23"), "Episode_Reward/arm_to_ball","Shoot v4 — Arm→Ball Reward",  "#1565C0"),
    (gs[1, 3], os.path.join(SHOOT,"2026-05-06_10-27-23"), "Episode_Reward/ball_at_goal","Shoot v4 — Ball@Goal Reward", "#F57F17"),
    # Row 2: Comparisons
    (gs[2, 0], REACH_RUNS["reach_v7"],  "Train/mean_reward", "Reach v7 — Reward",  C["v7"]),
    (gs[2, 1], REACH_RUNS["reach_v10"], "Train/mean_reward", "Reach v10 — Reward", C["v10"]),
    (gs[2, 2], os.path.join(SHOOT,"2026-05-05_18-43-22_shoot_ball_v1"), "Train/mean_reward", "Shoot v1 — BROKEN", C["sb_v1"]),
    (gs[2, 3], os.path.join(SHOOT,"2026-05-06_00-24-11_shoot_ball_v3"), "Train/mean_reward", "Shoot v3 — Goal Bug", C["sb_v3"]),
]
for spec, run_dir, tag, title, color in panels:
    ax = fig.add_subplot(spec)
    steps, vals = load(run_dir, tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 10), color=color, linewidth=1.8)
        ax.fill_between(steps, smooth(vals, 10) * 0.95, smooth(vals, 10) * 1.05,
                        color=color, alpha=0.15)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("Iteration", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("CSCE50103 — Humanoid RL Training Dashboard\n"
             "Tasks: Right-Arm Reach · Left-Arm Fixed · Depth Camera Reach · Shoot Ball to Target",
             fontsize=14, fontweight="bold")
fig.savefig(f"{OUT_DIR}/00_master_dashboard.png", bbox_inches="tight")
plt.close(fig)
print("Saved 00_master_dashboard.png")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — BUG ANALYSIS PLOT (great for presentation)
# ════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: GOAL_XY bug — ball_at_goal near zero in v3
ax = axes[0]
for name, run, color in [
    ("v3 (GOAL_XY bug)",  os.path.join(SHOOT,"2026-05-06_00-24-11_shoot_ball_v3"), C["sb_v3"]),
    ("v4 (bug fixed)",    os.path.join(SHOOT,"2026-05-06_10-27-23"),               C["sb_v4"]),
]:
    steps, vals = load(run, "Episode_Reward/ball_at_goal")
    if len(steps):
        ax.plot(steps, smooth(vals, 10), color=color, label=name, linewidth=2)
ax.set_title("Bug Fix: Hardcoded GOAL_XY\n→ Wrong gradient for 1535/1536 envs", fontsize=9)
ax.set_xlabel("Iteration")
ax.set_ylabel("ball_at_goal reward")
ax.legend(fontsize=8)

# Panel B: v6 degenerate — ball_at_goal maxed from spawn overlap
ax = axes[1]
for name, run, color in [
    ("v6 (ball in goal at spawn)", os.path.join(SHOOT,"2026-05-06_11-51-23"), "#E91E63"),
    ("v4 (correct)",               os.path.join(SHOOT,"2026-05-06_10-27-23"), C["sb_v4"]),
]:
    steps, vals = load(run, "Episode_Reward/ball_at_goal")
    if len(steps):
        ax.plot(steps, smooth(vals, 10), color=color, label=name, linewidth=2)
ax.set_title("Bug: Ball spawned inside goal threshold\n→ Free reward, robot learns to do nothing", fontsize=9)
ax.set_xlabel("Iteration")
ax.set_ylabel("ball_at_goal reward")
ax.legend(fontsize=8)

# Panel C: v1 broken — robot always falls
ax = axes[2]
for name, run, color in [
    ("v1 (leg joints, falls)", os.path.join(SHOOT,"2026-05-05_18-43-22_shoot_ball_v1"), C["sb_v1"]),
    ("v4 (arm joints, stable)", os.path.join(SHOOT,"2026-05-06_10-27-23"),              C["sb_v4"]),
]:
    steps, vals = load(run, "Episode_Termination/root_too_low")
    if len(steps):
        ax.plot(steps, smooth(vals, 10), color=color, label=name, linewidth=2)
ax.set_title("Bug: Leg joints made robot fall every episode\n→ Switched to arm joints + fine-tune from reach", fontsize=9)
ax.set_xlabel("Iteration")
ax.set_ylabel("root_too_low (fraction)")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)

fig.suptitle("Shoot Ball — Bug Analysis & Fixes", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/12_shoot_bug_analysis.png")
plt.close(fig)
print("Saved 12_shoot_bug_analysis.png")

print(f"\nAll plots saved to {OUT_DIR}/")
print("\nFiles:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
