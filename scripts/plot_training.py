#!/usr/bin/env python3
"""Plot training progress from RSL-RL tensorboard logs.

Does NOT require Isaac Sim — run with plain Python 3:

  python3 scripts/plot_training.py
  python3 scripts/plot_training.py --experiment ur10e_chicken_seq_grasp
  python3 scripts/plot_training.py --log_dir logs/rsl_rl/ur10e_chicken_seq_grasp/2026-06-04_10-19-16

Outputs: training_progress.png  (in the run directory, or current dir if --log_dir not given)
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ---------------------------------------------------------------------------
# Tensorboard reader
# ---------------------------------------------------------------------------

def read_tb_scalars(log_dir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read all scalar summaries from an RSL-RL run directory.

    Returns {tag: (steps_array, values_array)}.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise SystemExit(
            "[ERROR] tensorboard not found.\n"
            "Install it: pip install tensorboard\n"
            "Or activate the isaaclab conda env first."
        )

    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        vals = np.array([e.value for e in events])
        data[tag] = (steps, vals)
    return data


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple box-car smoothing, preserving array length."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def find_latest_run(experiment: str, log_root: str = "logs/rsl_rl") -> str:
    exp_dir = Path(log_root) / experiment
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    runs = sorted(exp_dir.iterdir())
    if not runs:
        raise FileNotFoundError(f"No runs found in {exp_dir}")
    return str(runs[-1])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_training(log_dir: str, out_path: str | None = None):
    print(f"[INFO] Reading logs from: {log_dir}")
    data = read_tb_scalars(log_dir)

    if not data:
        raise SystemExit("[ERROR] No scalar data found. Check the log directory path.")

    # Tag aliases: RSL-RL may use slightly different names across versions
    def get(preferred, *fallbacks):
        for name in (preferred, *fallbacks):
            if name in data:
                return data[name]
        return None

    reward        = get("Train/mean_reward")
    ep_len        = get("Train/mean_episode_length")
    policy_loss   = get("Loss/surrogate", "Loss/policy_loss")
    value_loss    = get("Loss/value_function", "Loss/value_loss")
    entropy       = get("Loss/entropy")
    lr            = get("Loss/learning_rate", "Train/learning_rate")
    fps           = get("Perf/total_fps", "Train/fps")

    available = [x for x in [reward, ep_len, policy_loss, value_loss, entropy, lr, fps] if x is not None]
    if not available:
        available_tags = "\n  ".join(sorted(data.keys()))
        raise SystemExit(f"[ERROR] No recognised metrics found.\nAvailable tags:\n  {available_tags}")

    # ---- layout ----
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    fig.suptitle(
        f"Training Progress — {Path(log_dir).parent.name}",
        color="white", fontsize=15, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    ax_map = {}
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)]
    entries = [
        ("Mean Reward",     reward,      "#00d4ff", True),
        ("Episode Length",  ep_len,      "#7fff7f", False),
        ("Policy Loss",     policy_loss, "#ff9f43", False),
        ("Value Loss",      value_loss,  "#ff6b6b", False),
        ("Entropy",         entropy,     "#c97bff", False),
        ("Learning Rate",   lr,          "#ffe66d", False),
        ("Throughput (FPS)",fps,         "#81ecec", False),
    ]

    for (row, col), (title, series, color, is_reward) in zip(positions, entries):
        if series is None:
            continue
        ax = fig.add_subplot(gs[row, col])
        steps, vals = series

        # raw (dim)
        ax.plot(steps, vals, color=color, alpha=0.25, linewidth=0.8)
        # smoothed (bright)
        sm = smooth(vals, window=min(30, max(1, len(vals) // 20)))
        ax.plot(steps, sm, color=color, linewidth=1.8, label="smoothed")

        if is_reward:
            # shade below the smoothed reward curve
            ax.fill_between(steps, sm, alpha=0.15, color=color)
            # mark best reward
            best_idx = np.argmax(sm)
            ax.axvline(steps[best_idx], color=color, linestyle="--", alpha=0.5, linewidth=0.8)
            ax.text(
                steps[best_idx], sm[best_idx],
                f"  best: {sm[best_idx]:.2f}",
                color=color, fontsize=7, va="bottom",
            )

        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.set_xlabel("Iteration", color="#aaaaaa", fontsize=7)
        ax.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")
        ax.set_facecolor("#0f0f23")
        ax.grid(True, color="#333355", linewidth=0.5, alpha=0.7)

        # current / final value annotation
        ax.text(
            0.97, 0.05, f"last: {sm[-1]:.4g}",
            transform=ax.transAxes, ha="right", va="bottom",
            color=color, fontsize=7, alpha=0.9,
        )

    # ---- extra: reward histogram in the last cell ----
    if reward is not None:
        ax_hist = fig.add_subplot(gs[2, 1:])
        _, vals = reward
        ax_hist.hist(vals, bins=60, color="#00d4ff", alpha=0.7, edgecolor="#003344")
        ax_hist.set_title("Reward Distribution (all iterations)", color="white", fontsize=9, pad=4)
        ax_hist.set_xlabel("Reward", color="#aaaaaa", fontsize=7)
        ax_hist.set_ylabel("Count", color="#aaaaaa", fontsize=7)
        ax_hist.tick_params(colors="#aaaaaa", labelsize=7)
        for spine in ax_hist.spines.values():
            spine.set_edgecolor("#444466")
        ax_hist.set_facecolor("#0f0f23")
        ax_hist.grid(True, color="#333355", linewidth=0.5, alpha=0.7)

    # ---- save ----
    if out_path is None:
        out_path = os.path.join(log_dir, "training_progress.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[INFO] Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Explicit path to an RSL-RL run directory (contains events.out.tfevents.*).",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="ur10e_chicken_seq_grasp",
        help="Experiment name; auto-selects the latest run under logs/rsl_rl/<experiment>/.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="logs/rsl_rl",
        help="Root directory for RSL-RL logs (default: logs/rsl_rl).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: <log_dir>/training_progress.png).",
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    if log_dir is None:
        log_dir = find_latest_run(args.experiment, args.log_root)
        print(f"[INFO] Auto-selected run: {log_dir}")

    plot_training(log_dir, out_path=args.out)
