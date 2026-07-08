#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Collect teleoperated demonstrations in Isaac Sim using a GELLO device.

Saves episodes in diffusion-policy zarr format:
  <out_dir>/
    replay_buffer.zarr/
      data/
        action/             (T_total, 8)  [arm_joints(6), left_jaw_bin(1), right_jaw_bin(1)]
        camera_rgb/         (T_total, H, W, 3) uint8 wrist-camera RGB frames
        state/              (T_total, 20) low-dim DP state vector
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
    Optional review videos are only written when --save_videos is passed.

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
      --out_dir ./data --num_demos 50
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
parser.add_argument("--out_dir",       type=str,  default="./data")
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
parser.add_argument("--save_videos", action="store_true",
                    help="Also save per-episode MP4 review videos under <out_dir>/videos.")
parser.add_argument("--image_key",     type=str,  default="camera_rgb",
                    help="Zarr key for high-dimensional RGB observations.")
parser.add_argument("--no_zarr_images", action="store_true",
                    help="Skip storing camera frames inside replay_buffer.zarr.")
parser.add_argument("--no_live_camera", action="store_true",
                    help="Disable the OpenCV camera popup to reduce teleop lag.")
parser.add_argument("--live_camera_recording_only", action="store_true",
                    help="Show the OpenCV camera popup only while recording.")
parser.add_argument("--preview_stride", type=int, default=1,
                    help="Show one live preview frame every N sim steps.")
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
from isaaclab_tasks.manager_based.manipulation.chicken_lift.chicken_lift_env_cfg import (
    CHICKEN_LIFT_MIN_HEIGHT,
    CHICKEN_SPAWN_Z,
)
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
ARM_IDLE_DEADBAND_RAD = 0.01  # Ignore tiny idle encoder changes when GELLO is not being moved.
MAX_ARM_TARGET_STEP_RAD = 0.025  # Rate-limit absolute GELLO joint targets so PhysX contacts can resolve.
ROBOT_BASE_Z = 0.63
LIFT_HEIGHT_M = CHICKEN_LIFT_MIN_HEIGHT - ROBOT_BASE_Z

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

    # The final chicken USD is spawned as a plain scene asset for GELLO teleop,
    # so it is not queried through an articulation/rigid-object data handle.
    ck_z_r = float(CHICKEN_SPAWN_Z - rb_pos[2])

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


def make_low_dim_state(obs_dict: dict) -> np.ndarray:
    """Return the low-dimensional state vector used by diffusion-policy configs."""

    state = np.concatenate(
        [
            obs_dict["robot_joint"],
            obs_dict["robot_joint_vel"],
            obs_dict["robot_eef_pose"],
            obs_dict["left_jaw"],
            obs_dict["right_jaw"],
        ]
    )
    return state.astype(np.float32, copy=False)


def get_sim_arm_joints(env_uw) -> np.ndarray:
    robot = env_uw.scene["robot"]
    ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    return robot.data.joint_pos[0, ids].cpu().numpy()


def snap_to_gello(env_uw, gello: GelloReader) -> np.ndarray:
    gello_pos = gello.get_arm_joints() * GELLO_SIGNS + GELLO_OFFSETS
    return gello_pos


def rate_limit_arm_targets(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.clip(target - prev, -max_step, max_step)
    return (prev + delta).astype(np.float32)


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
        "state":              20,
        "left_jaw":           1,
        "right_jaw":          1,
        "robot_eef_pose":     6,
        "robot_eef_pose_vel": 6,
        "robot_joint":        6,
        "robot_joint_vel":    6,
        "stage":              1,
        "timestamp":          1,
    }

    def __init__(
        self,
        out_dir: str,
        video_fps: int = 30,
        image_key: str = "camera_rgb",
        store_images: bool = True,
        save_videos: bool = False,
    ):
        root      = pathlib.Path(out_dir)
        zarr_path = root / "replay_buffer.zarr"
        vid_dir   = root / "videos"

        self.vid_dir   = vid_dir
        self.video_fps = video_fps
        self.image_key = image_key
        self.store_images = store_images
        self.save_videos = save_videos
        if self.save_videos:
            vid_dir.mkdir(parents=True, exist_ok=True)

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
        self._migrate_state_if_needed()
        if self.store_images:
            self._validate_or_explain_resume_state()

        self._bufs: dict[str, list] = {k: [] for k in self.FEATURES}
        self._frames: list[np.ndarray] = []
        self._zarr_images: list[np.ndarray] = []

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def ep_len(self) -> int:
        return len(self._bufs["timestamp"])

    def add_step(self, obs_dict: dict, action: np.ndarray,
                 camera: np.ndarray | None, timestamp: float, stage: int = 0):
        self._bufs["action"].append(action.astype(np.float32))
        self._bufs["state"].append(make_low_dim_state(obs_dict))
        self._bufs["left_jaw"].append(obs_dict["left_jaw"])
        self._bufs["right_jaw"].append(obs_dict["right_jaw"])
        self._bufs["robot_eef_pose"].append(obs_dict["robot_eef_pose"])
        self._bufs["robot_eef_pose_vel"].append(obs_dict["robot_eef_pose_vel"])
        self._bufs["robot_joint"].append(obs_dict["robot_joint"])
        self._bufs["robot_joint_vel"].append(obs_dict["robot_joint_vel"])
        self._bufs["stage"].append(np.array([stage], dtype=np.float32))
        self._bufs["timestamp"].append(np.array([timestamp], dtype=np.float32))

        if camera is not None and _HAS_CV2 and self.save_videos:
            bgr = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
            self._frames.append(bgr)
        if self.store_images:
            if camera is None:
                raise RuntimeError("store_images=True but no camera frame was provided for this step.")
            self._zarr_images.append(camera.astype(np.uint8, copy=False))

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
                self._validate_append_shape(data_grp[key], arr, key)
                data_grp[key].append(arr)

        if self.store_images:
            img_arr = np.stack(self._zarr_images).astype(np.uint8)  # (T, H, W, 3)
            if self.image_key not in data_grp:
                if self._total_steps != 0:
                    raise RuntimeError(
                        f"Cannot create image dataset '{self.image_key}' in a non-empty replay buffer. "
                        "Start a fresh --out_dir for high-dimensional collection, or use --no_zarr_images."
                    )
                data_grp.create_dataset(
                    self.image_key,
                    data=img_arr,
                    chunks=(min(32, T), *img_arr.shape[1:]),
                    dtype="uint8",
                )
            else:
                self._validate_append_shape(data_grp[self.image_key], img_arr, self.image_key)
                data_grp[self.image_key].append(img_arr)

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
        self._zarr_images.clear()

    def _validate_or_explain_resume_state(self):
        data_grp = self._root["data"]
        if self._total_steps > 0 and self.image_key not in data_grp:
            raise RuntimeError(
                f"Existing dataset has {self._total_steps} low-dimensional steps but no "
                f"data/{self.image_key} image array. Use a fresh --out_dir for image+state collection "
                "or pass --no_zarr_images to continue low-dimensional-only collection."
            )

    def _migrate_state_if_needed(self):
        data_grp = self._root["data"]
        if self._total_steps == 0 or "state" in data_grp:
            return
        required_keys = ("robot_joint", "robot_joint_vel", "robot_eef_pose", "left_jaw", "right_jaw")
        if not all(key in data_grp for key in required_keys):
            return
        lengths = {key: data_grp[key].shape[0] for key in required_keys}
        if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != self._total_steps:
            raise RuntimeError(f"Cannot backfill data/state because existing low-dimensional lengths differ: {lengths}")

        state = np.concatenate([data_grp[key][:] for key in required_keys], axis=1).astype(np.float32, copy=False)
        data_grp.create_dataset("state", data=state, chunks=(100, state.shape[1]), dtype="float32")
        print(f"[writer] Backfilled data/state from existing low-dimensional arrays: {state.shape}")

    @staticmethod
    def _validate_append_shape(existing, new_data: np.ndarray, key: str):
        if len(existing.shape) != len(new_data.shape) or tuple(existing.shape[1:]) != tuple(new_data.shape[1:]):
            raise RuntimeError(
                f"Cannot append data/{key}: existing shape per step is {existing.shape[1:]}, "
                f"new shape per step is {new_data.shape[1:]}."
            )


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
    writer = ZarrDemoWriter(
        args_cli.out_dir,
        video_fps=args_cli.video_fps,
        image_key=args_cli.image_key,
        store_images=not args_cli.no_zarr_images,
        save_videos=args_cli.save_videos,
    )
    print(f"\n[INFO] Output: {args_cli.out_dir}")
    print(f"[INFO] Resuming from {writer.n_episodes} episodes\n")
    print("Controls: C=record  S=save  Backspace=discard  Q=quit\n")

    # ── startup: drive robot toward GELLO's current pose through articulation drives ──
    env.reset()
    rb_pos_z = env_uw.scene["robot"].data.root_pos_w[0, 2].item()  # type: ignore[union-attr]
    print(f"[DEBUG] Robot base Z after reset = {rb_pos_z:.4f}  (expected 0.63)")
    robot = env_uw.scene["robot"]
    arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    last_arm_joints = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
    gello_start = snap_to_gello(env_uw, gello).astype(np.float32)
    for _ in range(120):
        last_arm_joints = rate_limit_arm_targets(last_arm_joints, gello_start, MAX_ARM_TARGET_STEP_RAD)
        env.step(torch.tensor(np.array([*last_arm_joints, 1.0, 1.0], dtype=np.float32),
                              device=env_uw.device).unsqueeze(0))
    rb_pos_z = env_uw.scene["robot"].data.root_pos_w[0, 2].item()  # type: ignore[union-attr]
    print(f"[DEBUG] Robot base Z after snap  = {rb_pos_z:.4f}  (expected 0.63)")
    print("[GELLO] Ready. Press C to start recording.")

    # ── main loop ─────────────────────────────────────────────────────────────
    demos_saved  = writer.n_episodes
    rec_steps    = 0
    ep_timestamp = 0.0
    loop_step    = 0

    while simulation_app.is_running() and not flags["quit"]:
        if args_cli.num_demos > 0 and demos_saved >= args_cli.num_demos:
            print(f"\nTarget of {args_cli.num_demos} demos reached. Exiting.")
            break

        # ── GELLO read ────────────────────────────────────────────────────────
        gello_state  = gello.get_joints()
        arm_joints   = gello_state[:6].astype(np.float32) * GELLO_SIGNS + GELLO_OFFSETS
        if np.max(np.abs(arm_joints - last_arm_joints)) < ARM_IDLE_DEADBAND_RAD:
            arm_joints = last_arm_joints.copy()
        else:
            arm_joints = rate_limit_arm_targets(last_arm_joints, arm_joints, MAX_ARM_TARGET_STEP_RAD)
            last_arm_joints = arm_joints.copy()
        gripper_frac = float(gello_state[6])
        gripper_bin  = 1.0 if gripper_frac < GRIPPER_THRESH else -1.0
        action_np    = np.array([*arm_joints, gripper_bin, gripper_bin], dtype=np.float32)

        # ── observe + optional camera (before step) ──────────────────────────
        obs_dict  = extract_obs_dict(env_uw)
        reward, stage = compute_reward_stage(obs_dict)

        preview_stride = max(args_cli.preview_stride, 1)
        preview_allowed = (
            _HAS_CV2
            and not args_cli.no_live_camera
            and (not args_cli.live_camera_recording_only or flags["recording"])
        )
        need_preview = preview_allowed and loop_step % preview_stride == 0
        need_record_frame = flags["recording"]
        cam_frame = get_camera_frame(env_uw) if (need_preview or need_record_frame) else None

        # ── live camera popup ────────────────────────────────────────────────
        if need_preview and cam_frame is not None:
            cv2.imshow("RealSense Camera", cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        elif preview_allowed:
            cv2.waitKey(1)

        # ── step sim ──────────────────────────────────────────────────────────
        _, _, terminated, truncated, _ = env.step(
            torch.tensor(action_np, device=env_uw.device).unsqueeze(0))

        # ── record ────────────────────────────────────────────────────────────
        if flags["recording"]:
            if cam_frame is None:
                cam_frame = get_camera_frame(env_uw)
            writer.add_step(obs_dict, action_np, cam_frame,
                            timestamp=ep_timestamp, stage=stage)
            rec_steps    += 1
            ep_timestamp += SIM_STEP_DT

            if rec_steps % 50 == 0:
                print(
                    f"  [REC] {rec_steps} steps  (saved: {demos_saved})  "
                    f"reward={reward:.1f}"
                )

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
            last_arm_joints = _reset_env(env, env_uw, gello).astype(np.float32).copy()
            print(f"  → {demos_saved} episodes saved. Press C for next.\n")

        # ── discard ───────────────────────────────────────────────────────────
        if flags["discard"]:
            writer.discard_episode()
            flags.update({"recording": False, "discard": False})
            rec_steps    = 0
            ep_timestamp = 0.0
            last_arm_joints = _reset_env(env, env_uw, gello).astype(np.float32).copy()
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
            last_arm_joints = _reset_env(env, env_uw, gello).astype(np.float32).copy()

        loop_step += 1

    # ── cleanup ───────────────────────────────────────────────────────────────
    kb.close()
    if _HAS_CV2:
        cv2.destroyAllWindows()
    env.close()
    print(f"\nDone. {demos_saved} episodes in {args_cli.out_dir}/replay_buffer.zarr/")


def _reset_env(env, env_uw, gello):
    env.reset()
    robot = env_uw.scene["robot"]
    ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    current = robot.data.joint_pos[0, ids].cpu().numpy().astype(np.float32)
    g = snap_to_gello(env_uw, gello).astype(np.float32)
    for _ in range(120):
        current = rate_limit_arm_targets(current, g, MAX_ARM_TARGET_STEP_RAD)
        env.step(torch.tensor(np.array([*current, 1.0, 1.0], dtype=np.float32),
                              device=env_uw.device).unsqueeze(0))
    return current


if __name__ == "__main__":
    main()
    simulation_app.close()
