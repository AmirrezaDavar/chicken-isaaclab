#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Collect teleoperated demonstrations in Isaac Sim using a GELLO device.

Saves episodes in diffusion-policy zarr format:
  <out_dir>/
    replay_buffer.zarr/
      data/
        action/             (T_total, 8)  [arm_joints(6), left_jaw_bin(1), right_jaw_bin(1)]
        left_jaw/           (T_total, 1)  left-jaw closure fraction [0=open, 1=closed]
        right_jaw/          (T_total, 1)  right-jaw closure fraction [0=open, 1=closed]
        robot_eef_pose/     (T_total, 6)  [ee_pos(3), ee_euler(3)] in robot-root frame
        robot_eef_pose_vel/ (T_total, 6)  [ee_lin_vel(3), ee_ang_vel(3)] in world frame
        robot_joint/        (T_total, 6)  arm joint positions (rad)
        robot_joint_vel/    (T_total, 6)  arm joint velocities (rad/s)
        stage/              (T_total, 1)  0=reaching/grasping, 1=lifted
        timestamp/          (T_total, 1)  per-step timestamp (s, 0-based per episode)
      meta/
        episode_ends/       (N_episodes,) cumulative step count at each episode end
    videos/
      episode_000000.mp4    per-episode wrist-camera recording for human review

Controls:
  GELLO handle   → arm joint positions
  GELLO trigger  → all 4 gripper jaws
  C              → START recording
  S              → STOP + SAVE (writes zarr arrays + mp4)
  Backspace      → discard current episode
  Q              → quit

Usage:
  cd /home/wanglab22/3_chicken-isaaclab
  python scripts/imitation_learning/collect_isaac_demos.py \\
      --out_dir ./data/isaac_chicken --num_demos 50
"""

import argparse
import pathlib
import sys
import os

GELLO_SOFTWARE_DIR = (
    "/home/wanglab22/1_gello_software"
    "(pressure+gelsight+tele speed alighnment))/gello_software"
)
if GELLO_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, GELLO_SOFTWARE_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Isaac Sim chicken-lift demos via GELLO.")
parser.add_argument("--out_dir",       type=str,  default="./data/isaac_chicken")
parser.add_argument("--num_demos",     type=int,  default=0,
                    help="Number of demos (0 = infinite).")
parser.add_argument("--episode_steps", type=int,  default=300,
                    help="Max steps per episode before auto-save.")
parser.add_argument("--gello_port",    type=str,  default=None)
parser.add_argument("--calib_path",    type=str,
                    default=os.path.join(GELLO_SOFTWARE_DIR, "gello_calibration.json"))
parser.add_argument("--diagnose",      action="store_true",
                    help="Print GELLO vs sim joint table for calibration.")
parser.add_argument("--video_fps",     type=int,  default=30,
                    help="FPS for saved MP4 videos (default 30).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after sim is live ─────────────────────────────────────────────────
import json
import numpy as np
import torch
import zarr
import gymnasium as gym

import carb.input
import omni.appwindow

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("[WARN] opencv-python not found — camera view + MP4 disabled.")

# ─────────────────────────────────────────────────────────────────────────────
TASK_ID     = "Isaac-Lift-Chicken-UR10e-CustomGripper-GELLO-v0"
SIM_STEP_DT = 0.02     # decimation=2, dt=0.01
GRIPPER_THRESH = 0.5   # GELLO gripper fraction below this → open command
LIFT_HEIGHT_M  = 0.06  # chicken Z above robot base → "lifted" (stage=1)

GELLO_SIGNS   = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
GELLO_OFFSETS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# ─────────────────────────────────────────────────────────────────────────────
# GELLO reader
# ─────────────────────────────────────────────────────────────────────────────

def _find_gello_port() -> str:
    from glob import glob
    ports = glob("/dev/serial/by-id/*FTDI*")
    if not ports:
        raise RuntimeError("No FTDI serial device found.")
    if len(ports) > 1:
        print(f"[GELLO] Multiple ports: {ports}. Using {ports[0]}")
    return ports[0]


class GelloReader:
    def __init__(self, port: str, calib_path: str):
        from gello.robots.dynamixel import DynamixelRobot
        with open(calib_path) as f:
            calib = json.load(f)
        g_open  = calib["gripper_offset_deg"]
        g_close = g_open - 41.8
        print(f"[GELLO] gripper open={g_open:.2f}°  close={g_close:.2f}°")
        print(f"[GELLO] Connecting → {port}")
        self._robot = DynamixelRobot(
            joint_ids=(1, 2, 3, 4, 5, 6),
            joint_offsets=calib["offsets"],
            joint_signs=calib["signs"],
            real=True, port=port,
            gripper_config=(7, g_open, g_close),
        )
        print("[GELLO] Connected.")

    def get_joints(self) -> np.ndarray:
        return self._robot.get_joint_state()   # (7,)

    def get_arm_joints(self) -> np.ndarray:
        return self.get_joints()[:6]

    def get_gripper_frac(self) -> float:
        return float(self.get_joints()[6])


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard
# ─────────────────────────────────────────────────────────────────────────────

class SimpleKeyboard:
    def __init__(self):
        self._input     = carb.input.acquire_input_interface()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._keyboard  = self._appwindow.get_keyboard()
        self._callbacks = {}
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_event)

    def add_callback(self, key: str, func):
        self._callbacks[key.upper()] = func

    def _on_event(self, event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            fn = self._callbacks.get(event.input.name)
            if fn:
                fn()
        return True

    def close(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)


# ─────────────────────────────────────────────────────────────────────────────
# Observation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_euler(q: np.ndarray) -> np.ndarray:
    t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    r, p, y = euler_xyz_from_quat(t)
    return np.array([r.item(), p.item(), y.item()], dtype=np.float32)


def extract_obs_dict(env_uw) -> dict:
    """Extract per-feature observation dict matching the zarr array layout."""
    scene = env_uw.scene
    robot = scene["robot"]
    rb_pos = robot.data.root_pos_w[0].cpu().numpy()

    # EE pose in robot-root frame
    ee_pos_w  = scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    ee_quat_w = scene["ee_frame"].data.target_quat_w[0, 0].cpu().numpy()
    ee_pos_r  = (ee_pos_w - rb_pos).astype(np.float32)
    ee_euler  = _quat_to_euler(ee_quat_w)

    # EE velocity via wrist_3_link body velocity: (6,) [lin_vel(3), ang_vel(3)]
    body_ids, _ = robot.find_bodies(["wrist_3_link"])
    ee_lin_vel = robot.data.body_lin_vel_w[0, body_ids[0]].cpu().numpy().astype(np.float32)
    ee_ang_vel = robot.data.body_ang_vel_w[0, body_ids[0]].cpu().numpy().astype(np.float32)
    ee_vel_w = np.concatenate([ee_lin_vel, ee_ang_vel])

    # Arm joint positions and velocities
    arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    robot_joint     = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
    robot_joint_vel = robot.data.joint_vel[0, arm_ids].cpu().numpy().astype(np.float32)

    # Gripper jaw closure fractions (0=open, 1=closed) — left and right pairs
    lj_ids, _ = robot.find_joints(["PrismaticJoint1", "PrismaticJoint2"])
    rj_ids, _ = robot.find_joints(["PrismaticJoint3", "PrismaticJoint4"])
    lj_frac = float(np.clip(robot.data.joint_pos[0, lj_ids].cpu().numpy() / -0.0093, 0.0, 1.0).mean())
    rj_frac = float(np.clip(robot.data.joint_pos[0, rj_ids].cpu().numpy() / -0.0093, 0.0, 1.0).mean())

    # Chicken height relative to robot base (for stage/reward)
    ck_z_r = float(scene["chicken"].data.root_pos_w[0, 2].item() - rb_pos[2])

    return {
        "robot_joint":        robot_joint,
        "robot_joint_vel":    robot_joint_vel,
        "robot_eef_pose":     np.concatenate([ee_pos_r, ee_euler]),
        "robot_eef_pose_vel": ee_vel_w,
        "left_jaw":           np.array([lj_frac], dtype=np.float32),
        "right_jaw":          np.array([rj_frac], dtype=np.float32),
        "ck_z_r":             ck_z_r,
    }


def compute_reward_stage(obs_dict: dict) -> tuple[float, int]:
    lifted = obs_dict["ck_z_r"] > LIFT_HEIGHT_M
    return (1.0 if lifted else 0.0), (1 if lifted else 0)


def get_camera_frame(env_uw) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB."""
    rgb_t  = env_uw.scene["camera"].data.output["rgb"][0, :, :, :3]
    rgb_np = rgb_t.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
    return rgb_np


def get_sim_arm_joints(env_uw) -> np.ndarray:
    robot = env_uw.scene["robot"]
    ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    return robot.data.joint_pos[0, ids].cpu().numpy()


def snap_to_gello(env_uw, gello: GelloReader) -> np.ndarray:
    gello_pos = gello.get_arm_joints() * GELLO_SIGNS + GELLO_OFFSETS
    robot = env_uw.scene["robot"]
    ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    pos_t = torch.tensor(gello_pos, device=env_uw.device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_position_to_sim(pos_t, joint_ids=ids)
    return gello_pos


# ─────────────────────────────────────────────────────────────────────────────
# Diagnose
# ─────────────────────────────────────────────────────────────────────────────

_JOINT_NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]


def print_joint_table(gello_raw, sim_joints):
    corrected = gello_raw * GELLO_SIGNS + GELLO_OFFSETS
    print("\033[2J\033[H", end="")
    print("─" * 66)
    print(f"  {'Joint':<10}  {'GELLO raw(°)':>12}  {'Corrected(°)':>12}  {'SIM(°)':>10}  {'Δ(°)':>8}")
    print("─" * 66)
    for i, name in enumerate(_JOINT_NAMES):
        g, c, s = np.rad2deg(gello_raw[i]), np.rad2deg(corrected[i]), np.rad2deg(sim_joints[i])
        flag = "  ← fix" if abs(c - s) > 10 else ""
        print(f"  {name:<10}  {g:>+12.2f}  {c:>+12.2f}  {s:>+10.2f}  {(c-s):>+8.2f}{flag}")
    print("─" * 66)


def run_diagnose_loop(env, env_uw, gello):
    kb = SimpleKeyboard()
    quit_flag = {"v": False}
    kb.add_callback("Q", lambda: quit_flag.update({"v": True}))
    env.reset()
    step = 0
    print("\n[DIAGNOSE] Move GELLO. Press Q to quit.\n")
    while simulation_app.is_running() and not quit_flag["v"]:
        raw = gello.get_arm_joints()
        corrected = raw * GELLO_SIGNS + GELLO_OFFSETS
        gf = float(gello.get_gripper_frac())
        gb = 1.0 if gf < GRIPPER_THRESH else -1.0
        env.step(torch.tensor(np.array([*corrected, gb, gb], dtype=np.float32),
                              device=env_uw.device).unsqueeze(0))
        if step % 30 == 0:
            print_joint_table(raw, get_sim_arm_joints(env_uw))
        step += 1
    kb.close()


# ─────────────────────────────────────────────────────────────────────────────
# Zarr demo writer — diffusion-policy format
# ─────────────────────────────────────────────────────────────────────────────

class ZarrDemoWriter:
    """Saves episodes in diffusion-policy zarr format with per-feature array folders."""

    # zarr array name → feature dimension
    FEATURES = {
        "action":             8,
        "left_jaw":           1,
        "right_jaw":          1,
        "robot_eef_pose":     6,
        "robot_eef_pose_vel": 6,
        "robot_joint":        6,
        "robot_joint_vel":    6,
        "stage":              1,
        "timestamp":          1,
    }

    def __init__(self, out_dir: str, video_fps: int = 30):
        root      = pathlib.Path(out_dir)
        zarr_path = root / "replay_buffer.zarr"
        vid_dir   = root / "videos"
        vid_dir.mkdir(parents=True, exist_ok=True)

        self.vid_dir   = vid_dir
        self.video_fps = video_fps

        store      = zarr.DirectoryStore(str(zarr_path))
        self._root = zarr.open_group(store, mode="a")
        self._root.require_group("data")
        self._root.require_group("meta")

        if "episode_ends" in self._root["meta"]:
            ep_ends = self._root["meta"]["episode_ends"][:]
            self._n_episodes  = len(ep_ends)
            self._total_steps = int(ep_ends[-1]) if len(ep_ends) > 0 else 0
        else:
            self._n_episodes  = 0
            self._total_steps = 0

        print(f"[writer] zarr → {zarr_path}")
        print(f"[writer] Resuming: {self._n_episodes} episodes, {self._total_steps} steps")

        self._bufs: dict[str, list] = {k: [] for k in self.FEATURES}
        self._frames: list[np.ndarray] = []

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def ep_len(self) -> int:
        return len(self._bufs["timestamp"])

    def add_step(self, obs_dict: dict, action: np.ndarray,
                 camera: np.ndarray | None, timestamp: float, stage: int = 0):
        self._bufs["action"].append(action.astype(np.float32))
        self._bufs["left_jaw"].append(obs_dict["left_jaw"])
        self._bufs["right_jaw"].append(obs_dict["right_jaw"])
        self._bufs["robot_eef_pose"].append(obs_dict["robot_eef_pose"])
        self._bufs["robot_eef_pose_vel"].append(obs_dict["robot_eef_pose_vel"])
        self._bufs["robot_joint"].append(obs_dict["robot_joint"])
        self._bufs["robot_joint_vel"].append(obs_dict["robot_joint_vel"])
        self._bufs["stage"].append(np.array([stage], dtype=np.float32))
        self._bufs["timestamp"].append(np.array([timestamp], dtype=np.float32))

        if camera is not None and _HAS_CV2:
            bgr = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
            self._frames.append(bgr)

    def save_episode(self) -> bool:
        T = self.ep_len
        if T == 0:
            print("[writer] nothing to save (episode empty)")
            return False

        data_grp = self._root["data"]
        meta_grp = self._root["meta"]

        # Append each feature array to zarr
        for key, dim in self.FEATURES.items():
            arr = np.stack(self._bufs[key]).astype(np.float32)  # (T, dim)
            if key not in data_grp:
                data_grp.create_dataset(key, data=arr,
                                        chunks=(100, dim), dtype="float32")
            else:
                data_grp[key].append(arr)

        # Update episode_ends in meta
        new_end = np.array([self._total_steps + T], dtype=np.int64)
        if "episode_ends" not in meta_grp:
            meta_grp.create_dataset("episode_ends", data=new_end,
                                    chunks=(100,), dtype="int64")
        else:
            meta_grp["episode_ends"].append(new_end)

        self._total_steps += T
        print(f"[writer] zarr ← episode {self._n_episodes}  ({T} steps, total {self._total_steps})")

        # Save MP4 for human review
        ep_name = f"episode_{self._n_episodes:06d}"
        if self._frames and _HAS_CV2:
            vpath  = self.vid_dir / f"{ep_name}.mp4"
            h, w   = self._frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out    = cv2.VideoWriter(str(vpath), fourcc, self.video_fps, (w, h))
            for frame in self._frames:
                out.write(frame)
            out.release()
            print(f"[writer] mp4  → {vpath}  ({len(self._frames)} frames @ {self.video_fps} fps)")

        self._n_episodes += 1
        self._reset_buffers()
        return True

    def discard_episode(self):
        n = self.ep_len
        self._reset_buffers()
        print(f"[writer] discarded {n} steps")

    def _reset_buffers(self):
        for k in self._bufs:
            self._bufs[k].clear()
        self._frames.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = 10000.0
    env    = gym.make(TASK_ID, cfg=env_cfg)
    env_uw = env.unwrapped

    gello_port = args_cli.gello_port or _find_gello_port()
    gello = GelloReader(port=gello_port, calib_path=args_cli.calib_path)

    if args_cli.diagnose:
        run_diagnose_loop(env, env_uw, gello)
        env.close()
        return

    # ── keyboard ──────────────────────────────────────────────────────────────
    kb    = SimpleKeyboard()
    flags = {"recording": False, "save": False, "discard": False, "quit": False}
    kb.add_callback("C",         lambda: (flags.update({"recording": True})
                                          or print("\n[REC] Recording — S to save, Backspace to discard")))
    kb.add_callback("S",         lambda: flags.update({"save": True}))
    kb.add_callback("BACKSPACE", lambda: flags.update({"discard": True}))
    kb.add_callback("Q",         lambda: flags.update({"quit": True}))

    # ── writer ────────────────────────────────────────────────────────────────
    writer = ZarrDemoWriter(args_cli.out_dir, video_fps=args_cli.video_fps)
    print(f"\n[INFO] Output: {args_cli.out_dir}")
    print(f"[INFO] Resuming from {writer.n_episodes} episodes\n")
    print("Controls: C=record  S=save  Backspace=discard  Q=quit\n")

    # ── startup: snap robot to GELLO's current pose ───────────────────────────
    env.reset()
    gello_start = snap_to_gello(env_uw, gello)
    for _ in range(10):
        env.step(torch.tensor(np.array([*gello_start, 1.0, 1.0], dtype=np.float32),
                              device=env_uw.device).unsqueeze(0))
    print("[GELLO] Ready. Press C to start recording.")

    # ── main loop ─────────────────────────────────────────────────────────────
    demos_saved  = writer.n_episodes
    rec_steps    = 0
    ep_timestamp = 0.0

    while simulation_app.is_running() and not flags["quit"]:
        if args_cli.num_demos > 0 and demos_saved >= args_cli.num_demos:
            print(f"\nTarget of {args_cli.num_demos} demos reached. Exiting.")
            break

        # ── GELLO read ────────────────────────────────────────────────────────
        gello_state  = gello.get_joints()
        arm_joints   = gello_state[:6].astype(np.float32) * GELLO_SIGNS + GELLO_OFFSETS
        gripper_frac = float(gello_state[6])
        gripper_bin  = 1.0 if gripper_frac < GRIPPER_THRESH else -1.0
        action_np    = np.array([*arm_joints, gripper_bin, gripper_bin], dtype=np.float32)

        # ── observe + camera (before step) ───────────────────────────────────
        obs_dict  = extract_obs_dict(env_uw)
        cam_frame = get_camera_frame(env_uw)
        reward, stage = compute_reward_stage(obs_dict)

        # ── live camera popup ─────────────────────────────────────────────────
        if _HAS_CV2:
            cv2.imshow("RealSense Camera", cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # ── step sim ──────────────────────────────────────────────────────────
        _, _, terminated, truncated, _ = env.step(
            torch.tensor(action_np, device=env_uw.device).unsqueeze(0))

        # ── record ────────────────────────────────────────────────────────────
        if flags["recording"]:
            writer.add_step(obs_dict, action_np, cam_frame,
                            timestamp=ep_timestamp, stage=stage)
            rec_steps    += 1
            ep_timestamp += SIM_STEP_DT

            if rec_steps % 50 == 0:
                print(f"  [REC] {rec_steps} steps  (saved: {demos_saved})  reward={reward:.1f}")

            if rec_steps >= args_cli.episode_steps:
                print("\n[INFO] Max episode length — auto-saving")
                flags["save"] = True

        # ── save ──────────────────────────────────────────────────────────────
        if flags["save"]:
            if writer.save_episode():
                demos_saved += 1
            flags.update({"recording": False, "save": False})
            rec_steps    = 0
            ep_timestamp = 0.0
            _reset_env(env, env_uw, gello)
            print(f"  → {demos_saved} episodes saved. Press C for next.\n")

        # ── discard ───────────────────────────────────────────────────────────
        if flags["discard"]:
            writer.discard_episode()
            flags.update({"recording": False, "discard": False})
            rec_steps    = 0
            ep_timestamp = 0.0
            _reset_env(env, env_uw, gello)
            print("  → Discarded. Press C for a new episode.\n")

        # ── auto-reset on env termination ─────────────────────────────────────
        if terminated or truncated:
            if flags["recording"] and writer.ep_len > 0:
                print("\n[WARN] Episode terminated early — auto-saving")
                if writer.save_episode():
                    demos_saved += 1
                flags["recording"] = False
                rec_steps    = 0
                ep_timestamp = 0.0
            _reset_env(env, env_uw, gello)

    # ── cleanup ───────────────────────────────────────────────────────────────
    kb.close()
    if _HAS_CV2:
        cv2.destroyAllWindows()
    env.close()
    print(f"\nDone. {demos_saved} episodes in {args_cli.out_dir}/replay_buffer.zarr/")


def _reset_env(env, env_uw, gello):
    env.reset()
    g = snap_to_gello(env_uw, gello)
    for _ in range(5):
        env.step(torch.tensor(np.array([*g, 1.0, 1.0], dtype=np.float32),
                              device=env_uw.device).unsqueeze(0))


if __name__ == "__main__":
    main()
    simulation_app.close()
