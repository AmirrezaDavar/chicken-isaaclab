"""Generate training plots for the reach task (reach_v7, v10, v11)."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing import event_accumulator

LOG_ROOT = "logs/rsl_rl/class_humanoid_reach_depth"
OUT_DIR = "logs/reach_plots"
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = {
    "reach_v7":  "2026-05-04_18-38-54_reach_v7",
    "reach_v10": "2026-05-04_22-16-00_reach_v10",
    "reach_v11": "2026-05-05_11-55-55_reach_v11",
}
COLORS = {"reach_v7": "#2196F3", "reach_v10": "#FF9800", "reach_v11": "#4CAF50"}


def load(run_dir, tag):
    path = os.path.join(LOG_ROOT, run_dir)
    evts = [f for f in os.listdir(path) if f.startswith("events.out")]
    if not evts:
        return np.array([]), np.array([])
    ea = event_accumulator.EventAccumulator(os.path.join(path, evts[0]),
                                            size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return np.array([]), np.array([])
    data = ea.Scalars(tag)
    steps = np.array([d.step for d in data])
    vals  = np.array([d.value for d in data])
    return steps, vals


def smooth(y, w=10):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")


plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ── Plot 1: Training reward comparison across all runs ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for name, run_dir in RUNS.items():
    steps, vals = load(run_dir, "Train/mean_reward")
    if len(steps):
        ax.plot(steps, smooth(vals, 15), color=COLORS[name], label=name, linewidth=2)
        ax.fill_between(steps, smooth(vals, 15) - 2, smooth(vals, 15) + 2,
                        color=COLORS[name], alpha=0.12)
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Episode Reward")
ax.set_title("Reaching Task — Training Reward Across Runs")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/01_reward_comparison.png")
plt.close(fig)
print("Saved 01_reward_comparison.png")

# ── Plot 2: Episode length comparison ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for name, run_dir in RUNS.items():
    steps, vals = load(run_dir, "Train/mean_episode_length")
    if len(steps):
        ax.plot(steps, smooth(vals, 15), color=COLORS[name], label=name, linewidth=2)
ax.axhline(160, color="gray", linestyle="--", linewidth=1, label="max (8 s × 20 Hz)")
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Mean Episode Length (steps)")
ax.set_title("Reaching Task — Episode Length (longer = more stable)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/02_episode_length.png")
plt.close(fig)
print("Saved 02_episode_length.png")

# ── Plot 3: Reward breakdown for reach_v7 ────────────────────────────────────
reward_tags = {
    "reach_target":       "Episode_Reward/reach_target",
    "reach_distance":     "Episode_Reward/reach_distance",
    "arm_joint_deviation":"Episode_Reward/arm_joint_deviation",
    "undesired_contacts": "Episode_Reward/undesired_contacts",
}
tag_colors = ["#1565C0", "#E53935", "#7B1FA2", "#F57F17"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, (label, tag), color in zip(axes, reward_tags.items(), tag_colors):
    steps, vals = load(RUNS["reach_v7"], tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 15), color=color, linewidth=2)
        ax.fill_between(steps, smooth(vals, 15) - 0.05,
                        smooth(vals, 15) + 0.05, color=color, alpha=0.15)
    ax.set_title(label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
fig.suptitle("reach_v7 — Individual Reward Components", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/03_reward_breakdown_v7.png")
plt.close(fig)
print("Saved 03_reward_breakdown_v7.png")

# ── Plot 4: Policy noise std (exploration decay) ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
for name, run_dir in RUNS.items():
    steps, vals = load(run_dir, "Policy/mean_noise_std")
    if len(steps):
        ax.plot(steps, smooth(vals, 10), color=COLORS[name], label=name, linewidth=2)
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Action Noise Std")
ax.set_title("Policy Exploration Decay (lower = more deterministic)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/04_exploration_decay.png")
plt.close(fig)
print("Saved 04_exploration_decay.png")

# ── Plot 5: Loss curves for reach_v7 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
loss_tags = [("Value Function Loss", "Loss/value_function", "#1565C0"),
             ("Surrogate Loss",      "Loss/surrogate",      "#E53935"),
             ("Entropy Loss",        "Loss/entropy",        "#2E7D32")]
for ax, (title, tag, color) in zip(axes, loss_tags):
    steps, vals = load(RUNS["reach_v7"], tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 10), color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
fig.suptitle("reach_v7 — Loss Curves", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/05_loss_curves_v7.png")
plt.close(fig)
print("Saved 05_loss_curves_v7.png")

# ── Plot 6: Termination breakdown for reach_v7 ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
term_tags = {
    "Timeout (success)":   ("Episode_Termination/time_out",       "#4CAF50"),
    "Bad orientation":     ("Episode_Termination/bad_orientation", "#F44336"),
    "Root too low (fall)": ("Episode_Termination/root_too_low",   "#FF9800"),
}
for label, (tag, color) in term_tags.items():
    steps, vals = load(RUNS["reach_v7"], tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 15), color=color, label=label, linewidth=2)
ax.set_xlabel("Training Iteration")
ax.set_ylabel("Fraction of Episodes")
ax.set_title("reach_v7 — Termination Reasons Over Training")
ax.set_ylim(-0.05, 1.05)
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/06_termination_breakdown.png")
plt.close(fig)
print("Saved 06_termination_breakdown.png")

# ── Plot 7: Big summary dashboard for reach_v7 ───────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

panels = [
    (gs[0, 0], "Train/mean_reward",              "Mean Reward",         "#2196F3"),
    (gs[0, 1], "Train/mean_episode_length",      "Episode Length",      "#4CAF50"),
    (gs[0, 2], "Episode_Reward/reach_target",    "Reach Target Reward", "#1565C0"),
    (gs[1, 0], "Episode_Reward/reach_distance",  "Reach Distance Reward","#E53935"),
    (gs[1, 1], "Policy/mean_noise_std",          "Action Noise Std",    "#7B1FA2"),
    (gs[1, 2], "Loss/value_function",            "Value Function Loss",  "#F57F17"),
]
for spec, tag, title, color in panels:
    ax = fig.add_subplot(spec)
    steps, vals = load(RUNS["reach_v7"], tag)
    if len(steps):
        ax.plot(steps, smooth(vals, 15), color=color, linewidth=2)
        ax.fill_between(steps, smooth(vals, 15) * 0.95,
                        smooth(vals, 15) * 1.05, color=color, alpha=0.15)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Iteration", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("reach_v7 — Training Dashboard (2600 iterations, 1536 envs)",
             fontsize=14, fontweight="bold")
fig.savefig(f"{OUT_DIR}/07_dashboard_v7.png", bbox_inches="tight")
plt.close(fig)
print("Saved 07_dashboard_v7.png")

print(f"\nAll plots saved to: {OUT_DIR}/")
