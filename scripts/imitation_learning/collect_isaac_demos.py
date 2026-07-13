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
import queue
import threading
import traceback

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
parser.add_argument("--episode_steps", type=int,  default=0,
                    help="Max steps per episode before auto-save. 0 disables auto-save.")
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
                    help="Zarr key for the gripper/wrist RGB observation.")
parser.add_argument("--left_image_key", type=str, default="camera_left_rgb",
                    help="Zarr key for the left table-side RGB observation.")
parser.add_argument("--right_image_key", type=str, default="camera_right_rgb",
                    help="Zarr key for the right table-side RGB observation.")
parser.add_argument("--no_zarr_images", action="store_true",
                    help="Skip storing camera frames inside replay_buffer.zarr.")
parser.add_argument("--no_live_camera", action="store_true",
                    help="Disable the OpenCV camera popup to reduce teleop lag.")
parser.add_argument("--live_camera_recording_only", action="store_true",
                    help="Show the OpenCV camera popup only while recording.")
parser.add_argument("--preview_stride", type=int, default=1,
                    help="Show one live preview frame every N sim steps.")
parser.add_argument("--preview_width", type=int, default=1280,
                    help="Initial live camera popup width in pixels.")
parser.add_argument("--preview_height", type=int, default=720,
                    help="Initial live camera popup height in pixels.")
parser.add_argument("--chicken_xy_range", type=float, nargs=2, default=(0.08, 0.12),
                    metavar=("X_RANGE", "Y_RANGE"),
                    help="Random chicken XY half-ranges in meters after each episode.")
parser.add_argument("--chicken_seed", type=int, default=None,
                    help="Random seed for chicken XY placement.")
parser.add_argument("--disable_chicken_drop_reset", action="store_true",
                    help="Do not automatically bring the chicken back if it drops below the table.")
parser.add_argument("--max_gripper_height_above_table", type=float, default=0.10,
                    help="Max gripper-tip height above the table in meters. <=0 disables the clamp.")
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
from pxr import Usd, UsdGeom

import carb.input
import omni.appwindow

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.chicken_lift.chicken_lift_env_cfg import (
    CHICKEN_DROP_MIN_HEIGHT,
    CHICKEN_LIFT_MIN_HEIGHT,
    CHICKEN_SPAWN_Z,
    TABLE_CENTER_X,
    TABLE_CENTER_Y,
    TABLE_TOP_Z,
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
CHICKEN_SPAWN_ROT = np.array([0.0, 0.7071068, 0.7071068, 0.0], dtype=np.float32)
CHICKEN_DROP_CHECK_INTERVAL = 15
PREVIEW_WINDOW_NAME = "RealSense Camera"
PREVIEW_TILE_W = 640
PREVIEW_TILE_H = 360
PREVIEW_LABEL_H = 34
GRIPPER_Z_LIMIT_WARN_INTERVAL = 50

GELLO_SIGNS   = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
GELLO_OFFSETS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
_SHOULDER_LIFT_ID = 1
_ELBOW_ID = 2
_WRIST_PITCH_ID = 3
_WRIST_ROLL_ID = 4

# Change this to bias the frozen tool pitch after reset.
# Positive values pitch one way, negative values pitch the other way.
TOOL_PITCH_OFFSET_DEG = 9.5

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


def get_camera_frame(env_uw, camera_name: str = "camera") -> np.ndarray:
    """Return (H, W, 3) uint8 RGB."""
    rgb_t  = env_uw.scene[camera_name].data.output["rgb"][0, :, :, :3]
    rgb_np = rgb_t.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
    return rgb_np


def get_camera_frames(env_uw) -> dict[str, np.ndarray]:
    return {
        args_cli.image_key: get_camera_frame(env_uw, "camera"),
        args_cli.left_image_key: get_camera_frame(env_uw, "left_camera"),
        args_cli.right_image_key: get_camera_frame(env_uw, "right_camera"),
    }


def _find_chicken_prim() -> Usd.Prim:
    stage = sim_utils.get_current_stage()
    preferred = stage.GetPrimAtPath("/World/envs/env_0/Chicken")
    if preferred.IsValid():
        return preferred
    for prim in stage.Traverse():
        if prim.GetPath().pathString.endswith("/Chicken"):
            return prim
    raise RuntimeError("Could not find a USD prim ending in /Chicken.")


def get_chicken_root_pos_w(env_uw) -> np.ndarray:
    """Return chicken root position in world frame."""
    try:
        chicken = env_uw.scene["chicken"]
        if hasattr(chicken, "data") and hasattr(chicken.data, "root_pos_w"):
            return chicken.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
    except Exception:
        pass

    prim = _find_chicken_prim()
    xform = UsdGeom.Xformable(prim)
    world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(world_tf.ExtractTranslation(), dtype=np.float32)


def random_chicken_xy(rng: np.random.Generator) -> tuple[float, float]:
    x_half_range, y_half_range = args_cli.chicken_xy_range
    x = TABLE_CENTER_X + float(rng.uniform(-abs(x_half_range), abs(x_half_range)))
    y = TABLE_CENTER_Y + float(rng.uniform(-abs(y_half_range), abs(y_half_range)))
    return x, y


def place_chicken_on_table(env_uw, rng: np.random.Generator, reason: str) -> np.ndarray:
    """Place the chicken on the table with randomized XY and zero root velocity when possible."""
    pos = np.array([*random_chicken_xy(rng), CHICKEN_SPAWN_Z], dtype=np.float32)
    quat = CHICKEN_SPAWN_ROT.copy()

    moved_with_asset_api = False
    try:
        chicken = env_uw.scene["chicken"]
        root_pose = torch.tensor([*pos, *quat], dtype=torch.float32, device=env_uw.device).unsqueeze(0)
        if hasattr(chicken, "write_root_pose_to_sim"):
            chicken.write_root_pose_to_sim(root_pose)
            moved_with_asset_api = True
        if hasattr(chicken, "write_root_velocity_to_sim"):
            root_vel = torch.zeros((1, 6), dtype=torch.float32, device=env_uw.device)
            chicken.write_root_velocity_to_sim(root_vel)
        if hasattr(chicken, "reset"):
            chicken.reset()
    except Exception:
        moved_with_asset_api = False

    if not moved_with_asset_api:
        prim = _find_chicken_prim()
        sim_utils.standardize_xform_ops(
            prim,
            translation=tuple(float(v) for v in pos),
            orientation=tuple(float(v) for v in quat),
        )

    print(f"[chicken] {reason}: placed at x={pos[0]:+.3f}, y={pos[1]:+.3f}, z={pos[2]:+.3f}")
    return pos


def recover_chicken_if_dropped(env_uw, rng: np.random.Generator) -> bool:
    if args_cli.disable_chicken_drop_reset:
        return False
    pos = get_chicken_root_pos_w(env_uw)
    if float(pos[2]) < CHICKEN_DROP_MIN_HEIGHT:
        place_chicken_on_table(env_uw, rng, reason="drop reset")
        return True
    return False


def draw_recording_overlay(rgb_frame: np.ndarray, recording: bool, elapsed_s: float) -> np.ndarray:
    """Return BGR preview frame with recording status overlay."""
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    if not recording:
        cv2.putText(
            bgr,
            "READY",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        return bgr

    mins = int(elapsed_s // 60)
    secs = int(elapsed_s % 60)
    label = f"REC {mins:02d}:{secs:02d}"
    cv2.circle(bgr, (30, 30), 10, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(
        bgr,
        label,
        (50, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return bgr


def _draw_preview_tile(frame: np.ndarray | None, label: str) -> np.ndarray:
    if frame is None:
        tile = np.zeros((PREVIEW_TILE_H, PREVIEW_TILE_W, 3), dtype=np.uint8)
        cv2.putText(tile, label, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA)
        cv2.putText(tile, "NO CAMERA", (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (120, 120, 120), 2, cv2.LINE_AA)
        return tile

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tile = cv2.resize(bgr, (PREVIEW_TILE_W, PREVIEW_TILE_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (PREVIEW_TILE_W, PREVIEW_LABEL_H), (12, 12, 12), -1)
    cv2.putText(tile, label, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    return tile


def draw_multi_camera_preview(frames: dict[str, np.ndarray], recording: bool, elapsed_s: float) -> np.ndarray:
    """Return a BGR 2x2 preview grid with wrist, left, right, and a black empty tile."""

    labels = [
        (args_cli.image_key, "WRIST"),
        (args_cli.left_image_key, "LEFT TABLE"),
        (args_cli.right_image_key, "RIGHT TABLE"),
        ("__empty__", ""),
    ]
    tiles = [_draw_preview_tile(frames.get(key), label) for key, label in labels]
    preview = np.vstack((np.hstack((tiles[0], tiles[1])), np.hstack((tiles[2], tiles[3]))))
    mins = int(elapsed_s // 60)
    secs = int(elapsed_s % 60)
    if recording:
        cv2.circle(preview, (24, 52), 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            preview,
            f"REC {mins:02d}:{secs:02d}",
            (42, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(preview, "READY", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    return preview


def create_resizable_preview_window(width: int, height: int) -> list[int]:
    flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
    cv2.namedWindow(PREVIEW_WINDOW_NAME, flags)
    width = max(int(width), 640)
    height = max(int(height), 360)
    cv2.resizeWindow(PREVIEW_WINDOW_NAME, width, height)
    return [width, height]


def handle_preview_window_key(key: int, window_size: list[int]) -> None:
    if key in (ord("+"), ord("=")):
        scale = 1.15
    elif key in (ord("-"), ord("_")):
        scale = 1.0 / 1.15
    else:
        return

    window_size[0] = max(640, int(window_size[0] * scale))
    window_size[1] = max(360, int(window_size[1] * scale))
    cv2.resizeWindow(PREVIEW_WINDOW_NAME, window_size[0], window_size[1])


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


def get_gripper_tip_z_w(env_uw) -> float:
    return float(env_uw.scene["ee_frame"].data.target_pos_w[0, 0, 2].item())


def clamp_arm_target_to_gripper_z_limit(
    env_uw,
    current: np.ndarray,
    target: np.ndarray,
    z_limit_w: float | None,
) -> tuple[np.ndarray, bool]:
    """Remove only the upward joint-space component when the gripper tip is at the Z limit."""

    if z_limit_w is None or get_gripper_tip_z_w(env_uw) < z_limit_w:
        return target, False

    robot = env_uw.scene["robot"]
    arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    body_ids, _ = robot.find_bodies(["wrist_3_link"])
    jacobian_body_id = body_ids[0] - 1 if robot.root_physx_view.shared_metatype.fixed_base else body_ids[0]

    jacobian = robot.root_physx_view.get_jacobians()[0, jacobian_body_id, 2, arm_ids].cpu().numpy().astype(np.float32)
    delta = (target - current).astype(np.float32, copy=True)
    predicted_dz = float(jacobian @ delta)
    if predicted_dz <= 0.0:
        return target, False

    denom = float(jacobian @ jacobian)
    if denom < 1e-8:
        return current.astype(np.float32, copy=True), True

    clamped = current + delta - (predicted_dz / denom) * jacobian
    clamped[_WRIST_ROLL_ID] = target[_WRIST_ROLL_ID]
    return clamped.astype(np.float32), True


def snap_to_gello(env_uw, gello: GelloReader) -> np.ndarray:
    gello_pos = gello.get_arm_joints() * GELLO_SIGNS + GELLO_OFFSETS
    return gello_pos


def rate_limit_arm_targets(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.clip(target - prev, -max_step, max_step)
    return (prev + delta).astype(np.float32)


def keep_tool_perpendicular_targets(target: np.ndarray, reference_joints: np.ndarray) -> np.ndarray:
    """Keep the tool pitch/roll close to the reset pose while leaving wrist yaw active."""

    target = target.astype(np.float32, copy=True)
    reference_pitch_sum = (
        reference_joints[_SHOULDER_LIFT_ID]
        + reference_joints[_ELBOW_ID]
        + reference_joints[_WRIST_PITCH_ID]
        + np.deg2rad(TOOL_PITCH_OFFSET_DEG)
    )
    target[_WRIST_PITCH_ID] = reference_pitch_sum - target[_SHOULDER_LIFT_ID] - target[_ELBOW_ID]
    target[_WRIST_ROLL_ID] = reference_joints[_WRIST_ROLL_ID]
    return target


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
        extra_image_keys: list[str] | None = None,
        store_images: bool = True,
        save_videos: bool = False,
    ):
        root      = pathlib.Path(out_dir)
        zarr_path = root / "replay_buffer.zarr"
        vid_dir   = root / "videos"

        self.vid_dir   = vid_dir
        self.video_fps = video_fps
        self.image_key = image_key
        self.image_keys = [image_key]
        for key in extra_image_keys or []:
            if key not in self.image_keys:
                self.image_keys.append(key)
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
        self._frames: dict[str, list[np.ndarray]] = {key: [] for key in self.image_keys}
        self._zarr_images: dict[str, list[np.ndarray]] = {key: [] for key in self.image_keys}
        self._write_queue: queue.Queue[dict | None] = queue.Queue()
        self._worker_error: BaseException | None = None
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="chicken-demo-writer",
            daemon=False,
        )
        self._writer_thread.start()

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def ep_len(self) -> int:
        return len(self._bufs["timestamp"])

    def add_step(self, obs_dict: dict, action: np.ndarray,
                 cameras: dict[str, np.ndarray] | np.ndarray | None, timestamp: float, stage: int = 0):
        self._raise_worker_error_if_needed()
        if isinstance(cameras, np.ndarray):
            cameras = {self.image_key: cameras}
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

        if cameras is not None and _HAS_CV2 and self.save_videos:
            for key, camera in cameras.items():
                if key in self._frames:
                    bgr = cv2.cvtColor(camera, cv2.COLOR_RGB2BGR)
                    self._frames[key].append(bgr)
        if self.store_images:
            if cameras is None:
                raise RuntimeError("store_images=True but no camera frames were provided for this step.")
            missing = [key for key in self.image_keys if key not in cameras]
            if missing:
                raise RuntimeError(f"Missing camera frame(s) for zarr image keys: {missing}")
            for key in self.image_keys:
                self._zarr_images[key].append(cameras[key].astype(np.uint8, copy=False))

    def save_episode(self) -> bool:
        self._raise_worker_error_if_needed()
        T = self.ep_len
        if T == 0:
            print("[writer] nothing to save (episode empty)")
            return False

        ep_idx = self._n_episodes
        new_total_steps = self._total_steps + T
        payload = {
            "episode_idx": ep_idx,
            "length": T,
            "total_steps": new_total_steps,
            "features": {},
            "images": None,
            "frames": {},
        }
        for key, dim in self.FEATURES.items():
            payload["features"][key] = np.stack(self._bufs[key]).astype(np.float32)

        if self.store_images:
            payload["images"] = {
                key: np.stack(self._zarr_images[key]).astype(np.uint8)
                for key in self.image_keys
            }
        if self.save_videos:
            payload["frames"] = {
                key: list(frames)
                for key, frames in self._frames.items()
                if frames
            }

        self._n_episodes += 1
        self._total_steps = new_total_steps
        self._write_queue.put(payload)
        print(
            f"[writer] queued episode {ep_idx}  "
            f"({T} steps, total {self._total_steps}, pending writes={self._write_queue.qsize()})"
        )
        self._reset_buffers()
        return True

    def discard_episode(self):
        self._raise_worker_error_if_needed()
        n = self.ep_len
        self._reset_buffers()
        print(f"[writer] discarded {n} steps")

    def close(self):
        self._write_queue.put(None)
        self._writer_thread.join()
        self._raise_worker_error_if_needed()

    def _reset_buffers(self):
        for k in self._bufs:
            self._bufs[k].clear()
        for key in self._frames:
            self._frames[key].clear()
        for key in self._zarr_images:
            self._zarr_images[key].clear()

    def _validate_or_explain_resume_state(self):
        data_grp = self._root["data"]
        missing = [key for key in self.image_keys if key not in data_grp]
        if self._total_steps > 0 and missing:
            raise RuntimeError(
                f"Existing dataset has {self._total_steps} low-dimensional steps but no "
                f"image array(s) {missing}. Use a fresh --out_dir for image+state collection "
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

    def _writer_loop(self):
        while True:
            payload = self._write_queue.get()
            try:
                if payload is None:
                    return
                self._write_episode_payload(payload)
            except BaseException as exc:
                self._worker_error = exc
                traceback.print_exc()
            finally:
                self._write_queue.task_done()

    def _write_episode_payload(self, payload: dict):
        data_grp = self._root["data"]
        meta_grp = self._root["meta"]
        ep_idx = payload["episode_idx"]
        T = payload["length"]

        for key, dim in self.FEATURES.items():
            arr = payload["features"][key]
            if key not in data_grp:
                data_grp.create_dataset(key, data=arr, chunks=(100, dim), dtype="float32")
            else:
                self._validate_append_shape(data_grp[key], arr, key)
                data_grp[key].append(arr)

        image_payload = payload["images"]
        if image_payload is not None:
            for key, img_arr in image_payload.items():
                if key not in data_grp:
                    if payload["total_steps"] != T:
                        raise RuntimeError(
                            f"Cannot create image dataset '{key}' after earlier non-image steps. "
                            "Start a fresh --out_dir for high-dimensional collection."
                        )
                    data_grp.create_dataset(
                        key,
                        data=img_arr,
                        chunks=(min(32, T), *img_arr.shape[1:]),
                        dtype="uint8",
                    )
                else:
                    self._validate_append_shape(data_grp[key], img_arr, key)
                    data_grp[key].append(img_arr)

        new_end = np.array([payload["total_steps"]], dtype=np.int64)
        if "episode_ends" not in meta_grp:
            meta_grp.create_dataset("episode_ends", data=new_end, chunks=(100,), dtype="int64")
        else:
            meta_grp["episode_ends"].append(new_end)
        print(f"[writer] zarr ← episode {ep_idx}  ({T} steps, total {payload['total_steps']})")

        frame_payload = payload["frames"]
        if frame_payload and _HAS_CV2:
            ep_name = f"episode_{ep_idx:06d}"
            for key, frames in frame_payload.items():
                if key == self.image_key:
                    vpath = self.vid_dir / f"{ep_name}.mp4"
                else:
                    camera_vid_dir = self.vid_dir / key
                    camera_vid_dir.mkdir(parents=True, exist_ok=True)
                    vpath = camera_vid_dir / f"{ep_name}.mp4"
                h, w = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(vpath), fourcc, self.video_fps, (w, h))
                if not out.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {vpath}")
                for frame in frames:
                    out.write(frame)
                out.release()
                print(f"[writer] mp4  → {vpath}  ({len(frames)} frames @ {self.video_fps} fps)")

    def _raise_worker_error_if_needed(self):
        if self._worker_error is not None:
            raise RuntimeError("Background demo writer failed.") from self._worker_error


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = 10000.0
    env    = gym.make(TASK_ID, cfg=env_cfg)
    env_uw = env.unwrapped
    rng = np.random.default_rng(args_cli.chicken_seed)

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
        extra_image_keys=[args_cli.left_image_key, args_cli.right_image_key],
        store_images=not args_cli.no_zarr_images,
        save_videos=args_cli.save_videos,
    )
    print(f"\n[INFO] Output: {args_cli.out_dir}")
    print(f"[INFO] Resuming from {writer.n_episodes} episodes\n")
    print("Controls: C=record  S=save  Backspace=discard  Q=quit\n")

    # ── startup: drive robot toward GELLO's current pose through articulation drives ──
    env.reset()
    place_chicken_on_table(env_uw, rng, reason="initial placement")
    rb_pos_z = env_uw.scene["robot"].data.root_pos_w[0, 2].item()  # type: ignore[union-attr]
    print(f"[DEBUG] Robot base Z after reset = {rb_pos_z:.4f}  (expected 0.63)")
    robot = env_uw.scene["robot"]
    arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    last_arm_joints = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
    perpendicular_reference_joints = last_arm_joints.copy()
    gripper_z_limit_w = None
    if args_cli.max_gripper_height_above_table > 0.0:
        gripper_z_limit_w = TABLE_TOP_Z + float(args_cli.max_gripper_height_above_table)
        print(
            "[GELLO] Gripper-tip Z clamp enabled: "
            f"table_z={TABLE_TOP_Z:.4f} m, "
            f"max_above_table={args_cli.max_gripper_height_above_table:.3f} m, "
            f"limit_z={gripper_z_limit_w:.4f} m."
        )
    print(
        "[GELLO] Keeping gripper pitch/roll near reset pose: "
        f"wrist_2_joint={last_arm_joints[_WRIST_ROLL_ID]:+.3f} rad fixed, "
        "wrist_1_joint compensates shoulder/elbow, wrist_3_joint remains active for yaw."
    )
    gello_start = keep_tool_perpendicular_targets(snap_to_gello(env_uw, gello), perpendicular_reference_joints)
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
    preview_window_size = None
    z_limit_warn_step = -GRIPPER_Z_LIMIT_WARN_INTERVAL

    while simulation_app.is_running() and not flags["quit"]:
        if args_cli.num_demos > 0 and demos_saved >= args_cli.num_demos:
            print(f"\nTarget of {args_cli.num_demos} demos reached. Exiting.")
            break

        # ── GELLO read ────────────────────────────────────────────────────────
        gello_state  = gello.get_joints()
        arm_joints   = gello_state[:6].astype(np.float32) * GELLO_SIGNS + GELLO_OFFSETS
        arm_joints   = keep_tool_perpendicular_targets(arm_joints, perpendicular_reference_joints)
        if gripper_z_limit_w is not None:
            gripper_z_w = get_gripper_tip_z_w(env_uw)
            arm_joints, z_limited = clamp_arm_target_to_gripper_z_limit(
                env_uw, last_arm_joints, arm_joints, gripper_z_limit_w
            )
            if z_limited:
                if loop_step - z_limit_warn_step >= GRIPPER_Z_LIMIT_WARN_INTERVAL:
                    print(
                        "[LIMIT] Gripper tip reached Z clamp "
                        f"({gripper_z_w:.3f} m > {gripper_z_limit_w:.3f} m). "
                        "Removing upward motion; downward/sideways/yaw commands remain active."
                    )
                    z_limit_warn_step = loop_step
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
        cam_frames = get_camera_frames(env_uw) if (need_preview or need_record_frame) else None
        cam_frame = cam_frames[args_cli.image_key] if cam_frames is not None else None

        # ── live camera popup ────────────────────────────────────────────────
        if need_preview and cam_frame is not None:
            if preview_window_size is None:
                preview_window_size = create_resizable_preview_window(args_cli.preview_width, args_cli.preview_height)
            preview = draw_multi_camera_preview(cam_frames, flags["recording"], ep_timestamp)
            cv2.imshow(PREVIEW_WINDOW_NAME, preview)
            handle_preview_window_key(cv2.waitKey(1) & 0xFF, preview_window_size)
        elif preview_allowed:
            if preview_window_size is not None:
                handle_preview_window_key(cv2.waitKey(1) & 0xFF, preview_window_size)

        # ── step sim ──────────────────────────────────────────────────────────
        _, _, terminated, truncated, _ = env.step(
            torch.tensor(action_np, device=env_uw.device).unsqueeze(0))

        dropped = False
        if loop_step % CHICKEN_DROP_CHECK_INTERVAL == 0:
            dropped = recover_chicken_if_dropped(env_uw, rng)
            if dropped and flags["recording"]:
                print("[chicken] dropped during recording; reset on table and skipped this sample")

        # ── record ────────────────────────────────────────────────────────────
        if flags["recording"] and not dropped:
            if cam_frames is None:
                cam_frames = get_camera_frames(env_uw)
            writer.add_step(obs_dict, action_np, cam_frames,
                            timestamp=ep_timestamp, stage=stage)
            rec_steps    += 1
            ep_timestamp += SIM_STEP_DT

            if rec_steps % 50 == 0:
                print(
                    f"  [REC] {rec_steps} steps  (saved: {demos_saved})  "
                    f"reward={reward:.1f}"
                )

            if args_cli.episode_steps > 0 and rec_steps >= args_cli.episode_steps:
                print("\n[INFO] Max episode length — auto-saving")
                flags["save"] = True

        # ── save ──────────────────────────────────────────────────────────────
        if flags["save"]:
            if writer.save_episode():
                demos_saved += 1
            flags.update({"recording": False, "save": False})
            rec_steps    = 0
            ep_timestamp = 0.0
            last_arm_joints = _reset_env(env, env_uw, gello, rng, perpendicular_reference_joints).astype(np.float32).copy()
            print(f"  → {demos_saved} episodes saved. Press C for next.\n")

        # ── discard ───────────────────────────────────────────────────────────
        if flags["discard"]:
            writer.discard_episode()
            flags.update({"recording": False, "discard": False})
            rec_steps    = 0
            ep_timestamp = 0.0
            last_arm_joints = _reset_env(env, env_uw, gello, rng, perpendicular_reference_joints).astype(np.float32).copy()
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
            last_arm_joints = _reset_env(env, env_uw, gello, rng, perpendicular_reference_joints).astype(np.float32).copy()

        loop_step += 1

    # ── cleanup ───────────────────────────────────────────────────────────────
    kb.close()
    print("[writer] Waiting for queued zarr/video writes to finish...")
    writer.close()
    if _HAS_CV2:
        cv2.destroyAllWindows()
    env.close()
    print(f"\nDone. {demos_saved} episodes in {args_cli.out_dir}/replay_buffer.zarr/")


def _reset_env(env, env_uw, gello, rng: np.random.Generator, perpendicular_reference_joints: np.ndarray):
    env.reset()
    place_chicken_on_table(env_uw, rng, reason="episode reset")
    robot = env_uw.scene["robot"]
    ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    current = robot.data.joint_pos[0, ids].cpu().numpy().astype(np.float32)
    current = keep_tool_perpendicular_targets(current, perpendicular_reference_joints)
    g = keep_tool_perpendicular_targets(snap_to_gello(env_uw, gello), perpendicular_reference_joints)
    for _ in range(120):
        current = rate_limit_arm_targets(current, g, MAX_ARM_TARGET_STEP_RAD)
        env.step(torch.tensor(np.array([*current, 1.0, 1.0], dtype=np.float32),
                              device=env_uw.device).unsqueeze(0))
    return current


if __name__ == "__main__":
    main()
    simulation_app.close()
