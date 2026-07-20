#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Collect teleoperated chicken demonstrations in Isaac Sim using a GELLO device.

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
  python scripts/imitation_learning/collect_isaac_pile_demos.py \\
      --out_dir ./data/chicken_rgb_state --num_demos 150
"""

import argparse
import pathlib
import sys
import os
import queue
import threading
import traceback
import time

GELLO_SOFTWARE_DIR = (
    "/home/wanglab22/1_gello_software"
    "(pressure+gelsight+tele speed alighnment))/gello_software"
)
if GELLO_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, GELLO_SOFTWARE_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Isaac Sim chicken demos via GELLO.")
parser.add_argument("--out_dir",       type=str,  default="./data/chicken_rgb_state")
parser.add_argument("--num_demos",     type=int,  default=150,
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
parser.add_argument("--chicken_xy_range", type=float, nargs=2, default=(0.25, 0.30),
                    metavar=("X_RANGE", "Y_RANGE"),
                    help="Max chicken XY half-ranges in meters. Clipped to safe tabletop coverage.")
parser.add_argument("--chicken_yaw_range_deg", type=float, default=0.0,
                    help="Deprecated: chicken yaw is fixed straight for consistent GELLO demos.")
parser.add_argument("--chicken_seed", type=int, default=None,
                    help="Random seed for chicken XY placement.")
parser.add_argument("--placement_vertical_step", type=float, default=0.03,
                    help="Vertical step in meters for deterministic chicken tabletop scan.")
parser.add_argument("--placement_horizontal_step", type=float, default=0.02,
                    help="Horizontal step in meters for deterministic chicken tabletop scan.")
parser.add_argument("--placement_grid", type=int, nargs=2, default=(5, 5), help=argparse.SUPPRESS)
parser.add_argument("--placement_jitter_fraction", type=float, default=0.0, help=argparse.SUPPRESS)
parser.add_argument("--chicken_spawn_z_offset", type=float, default=-0.005,
                    help="Extra root-Z offset in meters when placing chicken. Default removes old drop clearance.")
parser.add_argument("--chicken_pin_steps", type=int, default=8,
                    help="Reset-only steps that hold the chicken at the final pose with zero velocity.")
parser.add_argument("--chicken_stable_speed", type=float, default=0.001,
                    help="Chicken root motion threshold in m/step used to finish reset settling early.")
parser.add_argument("--robot_start_noise_deg", type=float, nargs=6,
                    default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                    help="Per-episode GELLO-to-sim joint offset half-ranges in degrees.")
parser.add_argument("--auto_stop_on_lift", action=argparse.BooleanOptionalAction, default=True,
                    help="Automatically save when the chicken has been lifted and held above threshold.")
parser.add_argument("--success_lift_height", type=float, default=0.0800,
                    help="Chicken root height increase in meters required for a successful lift.")
parser.add_argument("--auto_stop_hold_steps", type=int, default=10,
                    help="Consecutive lifted recording steps required before auto-save.")
parser.add_argument("--no_contact_sheets", action="store_true",
                    help="Disable per-episode JPG review contact sheets.")
parser.add_argument("--contact_sheet_stride", type=int, default=15,
                    help="Keep one review frame every N recorded steps for contact sheets.")
parser.add_argument("--coverage_footprint_size", type=float, nargs=2, default=(0.30, 0.12),
                    metavar=("LENGTH", "WIDTH"),
                    help="Approx chicken footprint size in meters for top-down coverage preview.")
parser.add_argument("--coverage_map_half_range", type=float, default=0.18,
                    help="Top-down coverage map XY half-range around table center in meters.")
parser.add_argument("--table_safe_margin", type=float, default=0.025,
                    help="Extra tabletop edge margin in meters beyond the chicken footprint.")
parser.add_argument("--disable_table_bounds", action="store_true",
                    help="Allow chicken randomization to exceed safe tabletop footprint limits.")
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
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import carb.input
import omni.appwindow

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
from isaaclab.assets import RigidObjectCfg
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
CHICKEN_SETTLE_STEPS = 120
TABLE_TOP_HALF_X = 0.254
TABLE_TOP_HALF_Y = 0.3048

_PLACEMENT_SCAN_CURSOR = 0

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


def _quat_wxyz_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_multiply_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = [float(v) for v in a]
    bw, bx, by, bz = [float(v) for v in b]
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


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

    ck_z_r = float(get_chicken_root_pos_w(env_uw)[2] - rb_pos[2])

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


def _find_chicken_rigid_prim() -> Usd.Prim:
    root = _find_chicken_prim()
    if root.HasAPI(UsdPhysics.RigidBodyAPI):
        return root
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    return root


def _world_pose_to_parent_local(prim: Usd.Prim, pos_w: np.ndarray, quat_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    parent = prim.GetParent()
    if not parent.IsValid():
        return pos_w.astype(np.float32), quat_w.astype(np.float32)

    parent_tf = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    parent_inv = parent_tf.GetInverse()
    world_tf = Gf.Matrix4d()
    world_tf.SetRotate(Gf.Quatd(float(quat_w[0]), float(quat_w[1]), float(quat_w[2]), float(quat_w[3])))
    world_tf.SetTranslateOnly(Gf.Vec3d(float(pos_w[0]), float(pos_w[1]), float(pos_w[2])))
    local_tf = world_tf * parent_inv
    local_pos = np.array(local_tf.ExtractTranslation(), dtype=np.float32)
    local_quat = local_tf.ExtractRotationQuat()
    local_imag = local_quat.GetImaginary()
    local_rot = np.array([local_quat.GetReal(), local_imag[0], local_imag[1], local_imag[2]], dtype=np.float32)
    return local_pos, local_rot


def force_chicken_visuals_visible(root_prim: Usd.Prim) -> None:
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeVisible()


def get_chicken_root_pos_w(env_uw) -> np.ndarray:
    """Return chicken root position in world frame."""
    try:
        chicken = env_uw.scene["chicken"]
        if hasattr(chicken, "data") and hasattr(chicken.data, "root_pos_w"):
            return chicken.data.root_pos_w[0].detach().cpu().numpy().astype(np.float32)
    except Exception:
        pass

    prim = _find_chicken_rigid_prim()
    xform = UsdGeom.Xformable(prim)
    world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(world_tf.ExtractTranslation(), dtype=np.float32)


def get_chicken_root_quat_w(env_uw) -> np.ndarray:
    """Return chicken root orientation as (w, x, y, z)."""
    try:
        chicken = env_uw.scene["chicken"]
        if hasattr(chicken, "data") and hasattr(chicken.data, "root_quat_w"):
            return chicken.data.root_quat_w[0].detach().cpu().numpy().astype(np.float32)
    except Exception:
        pass

    prim = _find_chicken_rigid_prim()
    xform = UsdGeom.Xformable(prim)
    world_tf = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    quat = world_tf.ExtractRotationQuat()
    imag = quat.GetImaginary()
    return np.array([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)


def rotated_chicken_footprint_half_extents(yaw: float) -> tuple[float, float]:
    length, width = [float(v) for v in args_cli.coverage_footprint_size]
    contour = _chicken_contour_points_m(length, width)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    rotated = contour @ rot.T
    return float(np.max(np.abs(rotated[:, 0]))), float(np.max(np.abs(rotated[:, 1])))


def safe_chicken_xy_half_ranges(yaw: float) -> tuple[float, float]:
    x_half_range, y_half_range = args_cli.chicken_xy_range
    x_half_range = abs(float(x_half_range))
    y_half_range = abs(float(y_half_range))

    if args_cli.disable_table_bounds:
        return x_half_range, y_half_range

    footprint_half_x, footprint_half_y = rotated_chicken_footprint_half_extents(yaw)
    margin = max(float(args_cli.table_safe_margin), 0.0)
    safe_half_x = max(TABLE_TOP_HALF_X - footprint_half_x - margin, 0.0)
    safe_half_y = max(TABLE_TOP_HALF_Y - footprint_half_y - margin, 0.0)
    return min(x_half_range, safe_half_x), min(y_half_range, safe_half_y)


def set_placement_scan_cursor(cursor: int) -> None:
    global _PLACEMENT_SCAN_CURSOR
    _PLACEMENT_SCAN_CURSOR = max(int(cursor), 0)


def placement_scan_shape(yaw: float) -> tuple[int, int]:
    x_half_range, y_half_range = safe_chicken_xy_half_ranges(yaw)
    vertical_step = max(abs(float(args_cli.placement_vertical_step)), 1e-4)
    horizontal_step = max(abs(float(args_cli.placement_horizontal_step)), 1e-4)
    row_count = max(int(np.ceil((2.0 * y_half_range) / vertical_step)) + 1, 1)
    col_count = max(int(np.ceil((2.0 * x_half_range) / horizontal_step)) + 1, 1)
    return row_count, col_count


def scan_coverage_chicken_xy(yaw: float) -> tuple[float, float]:
    global _PLACEMENT_SCAN_CURSOR

    x_half_range, y_half_range = safe_chicken_xy_half_ranges(yaw)
    vertical_step = max(abs(float(args_cli.placement_vertical_step)), 1e-4)
    horizontal_step = max(abs(float(args_cli.placement_horizontal_step)), 1e-4)
    row_count, col_count = placement_scan_shape(yaw)

    scan_index = _PLACEMENT_SCAN_CURSOR % (row_count * col_count)
    col = scan_index // row_count
    row = scan_index % row_count
    _PLACEMENT_SCAN_CURSOR += 1

    x_local = x_half_range - col * horizontal_step
    if col % 2 == 0:
        y_local = -y_half_range + row * vertical_step
    else:
        y_local = y_half_range - row * vertical_step

    x_local = float(np.clip(x_local, -x_half_range, x_half_range))
    y_local = float(np.clip(y_local, -y_half_range, y_half_range))
    x = TABLE_CENTER_X + x_local
    y = TABLE_CENTER_Y + y_local
    return x, y


def chicken_placement_z() -> float:
    return float(CHICKEN_SPAWN_Z + args_cli.chicken_spawn_z_offset)


def random_chicken_pose(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    yaw = 0.0
    x, y = scan_coverage_chicken_xy(yaw)
    yaw_quat = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float32)
    pos = np.array(
        [
            x,
            y,
            chicken_placement_z(),
        ],
        dtype=np.float32,
    )
    return pos, _quat_multiply_wxyz(yaw_quat, CHICKEN_SPAWN_ROT)


def write_chicken_pose_to_sim(env_uw, pos: np.ndarray, quat: np.ndarray) -> bool:
    try:
        chicken = env_uw.scene["chicken"]
        root_pose = torch.tensor([*pos, *quat], dtype=torch.float32, device=env_uw.device).unsqueeze(0)
        root_vel = torch.zeros((1, 6), dtype=torch.float32, device=env_uw.device)
        if hasattr(chicken, "write_root_state_to_sim"):
            chicken.write_root_state_to_sim(torch.cat([root_pose, root_vel], dim=1))
        elif hasattr(chicken, "write_root_pose_to_sim"):
            chicken.write_root_pose_to_sim(root_pose)
            if hasattr(chicken, "write_root_velocity_to_sim"):
                chicken.write_root_velocity_to_sim(root_vel)
        else:
            return False
        return True
    except Exception as exc:
        print(f"[chicken] warning: could not move chicken through asset API: {exc}")
        return False


def write_chicken_pose_to_usd(pos: np.ndarray, quat: np.ndarray) -> None:
    root_prim = _find_chicken_prim()
    rigid_prim = _find_chicken_rigid_prim()
    local_pos, local_quat = _world_pose_to_parent_local(rigid_prim, pos, quat)
    sim_utils.standardize_xform_ops(
        rigid_prim,
        translation=tuple(float(v) for v in local_pos),
        orientation=tuple(float(v) for v in local_quat),
    )
    force_chicken_visuals_visible(root_prim)


def place_chicken_on_table(env_uw, rng: np.random.Generator, reason: str) -> tuple[np.ndarray, np.ndarray]:
    """Place the chicken on/above the table."""

    pos, quat = random_chicken_pose(rng)
    moved_with_asset_api = write_chicken_pose_to_sim(env_uw, pos, quat)
    if not moved_with_asset_api:
        write_chicken_pose_to_usd(pos, quat)

    actual_pos = get_chicken_root_pos_w(env_uw)
    actual_quat = get_chicken_root_quat_w(env_uw)
    print(
        f"[chicken] {reason}: "
        f"x={actual_pos[0]:+.3f}, y={actual_pos[1]:+.3f}, z={actual_pos[2]:+.3f} "
        f"yaw={np.rad2deg(_quat_wxyz_to_yaw(actual_quat)):+.1f} deg "
        f"target=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) "
        f"(asset api={'yes' if moved_with_asset_api else 'no'})"
    )
    return pos, quat


def hold_arm_step(env, env_uw, arm_joints: np.ndarray) -> None:
    env.step(
        torch.tensor(
            np.array([*arm_joints, 1.0, 1.0], dtype=np.float32),
            device=env_uw.device,
        ).unsqueeze(0)
    )


def settle_chicken_reset_pose(
    env,
    env_uw,
    arm_joints: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> np.ndarray:
    """Settle contacts, then restore deterministic XY while preserving the rested orientation."""

    prev_pos = get_chicken_root_pos_w(env_uw)
    stable_steps = 0
    for _ in range(max(CHICKEN_SETTLE_STEPS, 0)):
        hold_arm_step(env, env_uw, arm_joints)
        actual_pos = get_chicken_root_pos_w(env_uw)
        motion = float(np.linalg.norm(actual_pos - prev_pos))
        if motion <= max(float(args_cli.chicken_stable_speed), 0.0):
            stable_steps += 1
        else:
            stable_steps = 0
        prev_pos = actual_pos
        if stable_steps >= 8:
            break

    actual_pos = get_chicken_root_pos_w(env_uw)
    settled_quat = get_chicken_root_quat_w(env_uw)
    corrected_pos = target_pos.astype(np.float32, copy=True)
    corrected_pos[2] = actual_pos[2]
    write_chicken_pose_to_sim(env_uw, corrected_pos, settled_quat)
    for _ in range(max(int(args_cli.chicken_pin_steps), 0)):
        write_chicken_pose_to_sim(env_uw, corrected_pos, settled_quat)
        hold_arm_step(env, env_uw, arm_joints)
    write_chicken_pose_to_sim(env_uw, corrected_pos, settled_quat)
    print(
        "[chicken] settled reset pose: "
        f"target_xy=({target_pos[0]:+.3f}, {target_pos[1]:+.3f}), "
        f"final=({corrected_pos[0]:+.3f}, {corrected_pos[1]:+.3f}, {corrected_pos[2]:+.3f}), "
        f"settled_yaw={np.rad2deg(_quat_wxyz_to_yaw(settled_quat)):+.1f} deg"
    )
    return corrected_pos


def recover_chicken_if_dropped(env_uw, rng: np.random.Generator) -> bool:
    if args_cli.disable_chicken_drop_reset:
        return False
    pos = get_chicken_root_pos_w(env_uw)
    if float(pos[2]) < CHICKEN_DROP_MIN_HEIGHT:
        place_chicken_on_table(env_uw, rng, reason="drop reset")
        return True
    return False


def sample_episode_arm_offset(rng: np.random.Generator) -> np.ndarray:
    ranges = np.deg2rad(np.array(args_cli.robot_start_noise_deg, dtype=np.float32))
    if np.max(np.abs(ranges)) <= 0.0:
        return np.zeros(6, dtype=np.float32)
    return rng.uniform(-np.abs(ranges), np.abs(ranges)).astype(np.float32)


def make_episode_quality_tracker(env_uw, episode_idx: int, arm_offset: np.ndarray) -> dict:
    chicken_pos = get_chicken_root_pos_w(env_uw)
    chicken_quat = get_chicken_root_quat_w(env_uw)
    robot_joints = get_sim_arm_joints(env_uw).astype(np.float32)
    return {
        "episode_idx": int(episode_idx),
        "started_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chicken_start_pos": chicken_pos.astype(float).tolist(),
        "chicken_start_quat_wxyz": chicken_quat.astype(float).tolist(),
        "chicken_start_yaw_rad": _quat_wxyz_to_yaw(chicken_quat),
        "robot_start_joint": robot_joints.astype(float).tolist(),
        "episode_arm_offset": arm_offset.astype(float).tolist(),
        "first_gripper_close_step": None,
        "first_gripper_close_time_s": None,
        "max_chicken_height": float(chicken_pos[2]),
        "max_lift_above_start": 0.0,
        "success": False,
        "human_clean": None,
        "episode_length": 0,
        "ended_reason": None,
    }


def update_episode_quality_tracker(
    tracker: dict,
    env_uw,
    action: np.ndarray,
    step_idx: int,
    timestamp: float,
) -> None:
    chicken_z = float(get_chicken_root_pos_w(env_uw)[2])
    start_z = float(tracker["chicken_start_pos"][2])
    lift = chicken_z - start_z
    tracker["max_chicken_height"] = max(float(tracker["max_chicken_height"]), chicken_z)
    tracker["max_lift_above_start"] = max(float(tracker["max_lift_above_start"]), lift)
    tracker["episode_length"] = int(step_idx + 1)
    tracker["success"] = bool(float(tracker["max_lift_above_start"]) >= args_cli.success_lift_height)
    if tracker["first_gripper_close_step"] is None and float(action[6]) < 0.0:
        tracker["first_gripper_close_step"] = int(step_idx)
        tracker["first_gripper_close_time_s"] = float(timestamp)


def make_contact_sheet(frames: list[np.ndarray], max_cols: int = 4) -> np.ndarray | None:
    if not frames or not _HAS_CV2:
        return None
    thumbs = []
    for idx, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        thumb = cv2.resize(bgr, (240, 135), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, f"{idx}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)
    cols = min(max_cols, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    blank = np.zeros_like(thumbs[0])
    padded = thumbs + [blank] * (rows * cols - len(thumbs))
    return np.vstack([np.hstack(padded[row * cols:(row + 1) * cols]) for row in range(rows)])


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


def _coverage_world_to_pixel(x: float, y: float, half_range: float) -> tuple[int, int]:
    half_range = max(float(half_range), 0.05)
    px = int((x - (TABLE_CENTER_X - half_range)) / (2.0 * half_range) * (PREVIEW_TILE_W - 1))
    py = int((1.0 - (y - (TABLE_CENTER_Y - half_range)) / (2.0 * half_range)) * (PREVIEW_TILE_H - 1))
    return px, py


def _chicken_contour_points_m(length: float, width: float) -> np.ndarray:
    half_l = 0.5 * length
    half_w = 0.5 * width
    return np.array(
        [
            [-0.95 * half_l, -0.12 * half_w],
            [-0.75 * half_l, -0.72 * half_w],
            [-0.34 * half_l, -0.96 * half_w],
            [0.18 * half_l, -0.72 * half_w],
            [0.56 * half_l, -0.42 * half_w],
            [0.92 * half_l, -0.28 * half_w],
            [1.16 * half_l, 0.00 * half_w],
            [0.92 * half_l, 0.28 * half_w],
            [0.56 * half_l, 0.42 * half_w],
            [0.18 * half_l, 0.72 * half_w],
            [-0.34 * half_l, 0.96 * half_w],
            [-0.75 * half_l, 0.72 * half_w],
            [-0.95 * half_l, 0.12 * half_w],
        ],
        dtype=np.float32,
    )


def _transform_footprint_points(
    local_xy_m: np.ndarray,
    center_w: tuple[float, float],
    yaw: float,
    half_range: float,
) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    world_xy = local_xy_m @ rot.T + np.array(center_w, dtype=np.float32)
    return np.array([_coverage_world_to_pixel(float(x), float(y), half_range) for x, y in world_xy], dtype=np.int32)


def _draw_coverage_footprint(
    tile: np.ndarray,
    entry: dict,
    half_range: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    pos = entry.get("chicken_start_pos")
    if not pos or len(pos) < 2:
        return
    yaw = float(entry.get("chicken_start_yaw_rad", 0.0))
    length, width = [float(v) for v in args_cli.coverage_footprint_size]
    pixels_per_meter_x = PREVIEW_TILE_W / (2.0 * max(half_range, 0.05))
    center = _coverage_world_to_pixel(float(pos[0]), float(pos[1]), half_range)
    contour_m = _chicken_contour_points_m(length, width)
    # Convert through world coordinates so yaw and map aspect are handled consistently.
    contour_px = _transform_footprint_points(contour_m, (float(pos[0]), float(pos[1])), yaw, half_range)
    if thickness < 0:
        cv2.fillPoly(tile, [contour_px], color, cv2.LINE_AA)
    else:
        cv2.polylines(tile, [contour_px], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Add two darker leg/wing hints so the footprint reads as a chicken-like shape.
    leg_offset = 0.16 * length
    leg_span = 0.35 * width
    hints = np.array(
        [
            [-leg_offset, -0.5 * leg_span],
            [-leg_offset, 0.5 * leg_span],
            [0.12 * length, 0.0],
        ],
        dtype=np.float32,
    )
    hint_px = _transform_footprint_points(hints, (float(pos[0]), float(pos[1])), yaw, half_range)
    hint_color = tuple(max(0, int(v) - 45) for v in color)
    cv2.polylines(tile, [hint_px], isClosed=False, color=hint_color, thickness=max(1, thickness if thickness > 0 else 1), lineType=cv2.LINE_AA)
    cv2.circle(tile, center, max(2, int(0.012 * pixels_per_meter_x)), hint_color, -1, cv2.LINE_AA)


def draw_coverage_tile(saved_entries: list[dict], current_entry: dict | None = None) -> np.ndarray:
    half_range = max(
        float(args_cli.coverage_map_half_range),
        abs(float(args_cli.chicken_xy_range[0])) + 0.08,
        abs(float(args_cli.chicken_xy_range[1])) + 0.08,
    )
    tile = np.full((PREVIEW_TILE_H, PREVIEW_TILE_W, 3), 28, dtype=np.uint8)
    cv2.rectangle(tile, (0, 0), (PREVIEW_TILE_W, PREVIEW_LABEL_H), (12, 12, 12), -1)
    cv2.putText(tile, "COVERAGE", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)

    for frac in (0.25, 0.5, 0.75):
        x = int(frac * PREVIEW_TILE_W)
        y = int(frac * PREVIEW_TILE_H)
        cv2.line(tile, (x, PREVIEW_LABEL_H), (x, PREVIEW_TILE_H), (55, 55, 55), 1, cv2.LINE_AA)
        cv2.line(tile, (0, y), (PREVIEW_TILE_W, y), (55, 55, 55), 1, cv2.LINE_AA)

    center = _coverage_world_to_pixel(TABLE_CENTER_X, TABLE_CENTER_Y, half_range)
    cv2.drawMarker(tile, center, (230, 230, 230), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

    for entry in saved_entries:
        color = (130, 130, 130)
        if entry.get("success") and entry.get("human_clean") is not False:
            color = (165, 165, 165)
        _draw_coverage_footprint(tile, entry, half_range, color, -1)

    if current_entry is not None:
        _draw_coverage_footprint(tile, current_entry, half_range, (0, 220, 255), 2)

    label = f"{len(saved_entries)} saved | span +/-{half_range:.2f}m"
    cv2.putText(tile, label, (14, PREVIEW_TILE_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return tile


def save_coverage_map(out_dir: pathlib.Path, saved_entries: list[dict], current_entry: dict | None = None) -> None:
    if not _HAS_CV2:
        return
    quality_dir = out_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(quality_dir / "coverage_map_latest.jpg"), draw_coverage_tile(saved_entries, current_entry))


def load_saved_coverage_entries(out_dir: pathlib.Path) -> list[dict]:
    path = out_dir / "quality" / "episode_metadata.jsonl"
    entries = []
    if not path.exists():
        return entries
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "chicken_start_pos" in entry:
                entries.append(entry)
    return entries


def make_runtime_chicken_rigid_object_cfg(chicken_cfg) -> RigidObjectCfg:
    init_state = chicken_cfg.init_state
    return RigidObjectCfg(
        prim_path=chicken_cfg.prim_path,
        spawn=chicken_cfg.spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(float(init_state.pos[0]), float(init_state.pos[1]), chicken_placement_z()),
            rot=tuple(float(v) for v in init_state.rot),
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
        ),
    )


def draw_multi_camera_preview(
    frames: dict[str, np.ndarray],
    recording: bool,
    elapsed_s: float,
    quality: dict | None = None,
    coverage_entries: list[dict] | None = None,
    saved_episodes: int = 0,
    target_episodes: int = 0,
) -> np.ndarray:
    """Return a BGR 2x2 preview grid with wrist, left, right, and a black empty tile."""

    labels = [
        (args_cli.image_key, "WRIST"),
        (args_cli.left_image_key, "LEFT TABLE"),
        (args_cli.right_image_key, "RIGHT TABLE"),
        ("__empty__", ""),
    ]
    tiles = [_draw_preview_tile(frames.get(key), label) for key, label in labels]
    tiles[3] = draw_coverage_tile(coverage_entries or [], quality)
    preview = np.vstack((np.hstack((tiles[0], tiles[1])), np.hstack((tiles[2], tiles[3]))))
    mins = int(elapsed_s // 60)
    secs = int(elapsed_s % 60)
    episode_label = (
        f"EPISODES {saved_episodes}/{target_episodes}"
        if target_episodes > 0
        else f"EPISODES {saved_episodes}"
    )
    (label_w, label_h), _ = cv2.getTextSize(episode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    label_x = max(18, preview.shape[1] - label_w - 18)
    label_y = 60
    cv2.rectangle(
        preview,
        (label_x - 10, label_y - label_h - 10),
        (label_x + label_w + 10, label_y + 10),
        (12, 12, 12),
        -1,
    )
    cv2.putText(
        preview,
        episode_label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
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
        if quality is not None:
            clean = quality.get("human_clean")
            clean_label = "unset" if clean is None else ("yes" if clean else "no")
            stats = (
                f"steps {quality.get('episode_length', 0)} | "
                f"close {quality.get('first_gripper_close_step')} | "
                f"lift {float(quality.get('max_lift_above_start', 0.0)):.3f}m | "
                f"success {quality.get('success', False)} | clean {clean_label}"
            )
            cv2.putText(preview, stats, (18, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
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
        save_contact_sheets: bool = True,
        contact_sheet_stride: int = 15,
    ):
        root      = pathlib.Path(out_dir)
        zarr_path = root / "replay_buffer.zarr"
        vid_dir   = root / "videos"
        quality_dir = root / "quality"

        self.vid_dir   = vid_dir
        self.quality_dir = quality_dir
        self.metadata_jsonl = quality_dir / "episode_metadata.jsonl"
        self.video_fps = video_fps
        self.image_key = image_key
        self.image_keys = [image_key]
        for key in extra_image_keys or []:
            if key not in self.image_keys:
                self.image_keys.append(key)
        self.store_images = store_images
        self.save_videos = save_videos
        self.save_contact_sheets = save_contact_sheets and _HAS_CV2
        self.contact_sheet_stride = max(int(contact_sheet_stride), 1)
        if self.save_videos:
            vid_dir.mkdir(parents=True, exist_ok=True)
        quality_dir.mkdir(parents=True, exist_ok=True)

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
        self._review_frames: list[np.ndarray] = []
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
        if (
            self.save_contact_sheets
            and cameras is not None
            and self.image_key in cameras
            and len(self._bufs["timestamp"]) % self.contact_sheet_stride == 1
        ):
            self._review_frames.append(cameras[self.image_key].astype(np.uint8, copy=False))

    def save_episode(self, metadata: dict | None = None) -> bool:
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
            "review_frames": list(self._review_frames),
            "metadata": metadata or {},
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
        self._review_frames.clear()

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

        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault("episode_idx", int(ep_idx))
        metadata.setdefault("episode_length", int(T))
        metadata.setdefault("total_steps_after_episode", int(payload["total_steps"]))
        metadata_path = self.quality_dir / f"episode_{ep_idx:06d}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        with open(self.metadata_jsonl, "a") as f:
            f.write(json.dumps(metadata, sort_keys=True) + "\n")
        print(f"[writer] quality → {metadata_path}")

        review_frames = payload.get("review_frames") or []
        sheet = make_contact_sheet(review_frames)
        if sheet is not None:
            sheet_path = self.quality_dir / f"episode_{ep_idx:06d}_contact_sheet.jpg"
            cv2.imwrite(str(sheet_path), sheet)
            print(f"[writer] contact sheet → {sheet_path}")

    def _raise_worker_error_if_needed(self):
        if self._worker_error is not None:
            raise RuntimeError("Background demo writer failed.") from self._worker_error


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.scene.chicken = make_runtime_chicken_rigid_object_cfg(env_cfg.scene.chicken)
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
    flags = {"recording": False, "save": False, "discard": False, "quit": False, "human_clean": None}
    kb.add_callback("C",         lambda: (flags.update({"recording": True})
                                          or print("\n[REC] Recording — S to save, Backspace to discard")))
    kb.add_callback("S",         lambda: flags.update({"save": True}))
    kb.add_callback("BACKSPACE", lambda: flags.update({"discard": True}))
    kb.add_callback("Q",         lambda: flags.update({"quit": True}))
    kb.add_callback("Y",         lambda: (flags.update({"human_clean": True})
                                          or print("[quality] Marked current episode clean.")))
    kb.add_callback("N",         lambda: (flags.update({"human_clean": False})
                                          or print("[quality] Marked current episode not clean.")))

    # ── writer ────────────────────────────────────────────────────────────────
    writer = ZarrDemoWriter(
        args_cli.out_dir,
        video_fps=args_cli.video_fps,
        image_key=args_cli.image_key,
        extra_image_keys=[args_cli.left_image_key, args_cli.right_image_key],
        store_images=not args_cli.no_zarr_images,
        save_videos=args_cli.save_videos,
        save_contact_sheets=not args_cli.no_contact_sheets,
        contact_sheet_stride=args_cli.contact_sheet_stride,
    )
    out_root = pathlib.Path(args_cli.out_dir)
    coverage_entries = load_saved_coverage_entries(out_root)
    set_placement_scan_cursor(len(coverage_entries))
    if coverage_entries:
        save_coverage_map(out_root, coverage_entries)
    print(f"\n[INFO] Output: {args_cli.out_dir}")
    print(f"[INFO] Resuming from {writer.n_episodes} episodes\n")
    print(f"[INFO] Coverage map has {len(coverage_entries)} saved chicken placements\n")
    safe_x, safe_y = safe_chicken_xy_half_ranges(0.0)
    scan_rows, scan_cols = placement_scan_shape(0.0)
    print(
        "[INFO] Table-safe placement: "
        f"table={2.0 * TABLE_TOP_HALF_X:.3f}m x {2.0 * TABLE_TOP_HALF_Y:.3f}m, "
        f"safe_root_half_range=({safe_x:.3f}, {safe_y:.3f})m, "
        f"scan={scan_cols} columns x {scan_rows} rows, "
        f"step=({args_cli.placement_horizontal_step:.3f}m horizontal, "
        f"{args_cli.placement_vertical_step:.3f}m vertical), "
        f"z={chicken_placement_z():.4f}m, "
        "yaw=0.0deg"
    )
    print("Controls: C=record  S=save  Backspace=discard  Q=quit\n")

    # ── startup: drive robot toward GELLO's current pose through articulation drives ──
    env.reset()
    episode_arm_offset = sample_episode_arm_offset(rng)
    if np.max(np.abs(episode_arm_offset)) > 0.0:
        print(f"[reset] Episode arm offset deg: {np.rad2deg(episode_arm_offset)}")
    rb_pos_z = env_uw.scene["robot"].data.root_pos_w[0, 2].item()  # type: ignore[union-attr]
    print(f"[DEBUG] Robot base Z after reset = {rb_pos_z:.4f}  (expected 0.63)")
    robot = env_uw.scene["robot"]
    arm_ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    last_arm_joints = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
    perpendicular_reference_joints = last_arm_joints.copy()
    chicken_target_pos, chicken_target_quat = place_chicken_on_table(env_uw, rng, reason="initial placement")
    settle_chicken_reset_pose(env, env_uw, last_arm_joints, chicken_target_pos, chicken_target_quat)
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
    gello_start = keep_tool_perpendicular_targets(
        snap_to_gello(env_uw, gello) + episode_arm_offset,
        perpendicular_reference_joints,
    )
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
    episode_tracker = None
    lifted_hold_steps = 0

    while simulation_app.is_running() and not flags["quit"]:
        if args_cli.num_demos > 0 and demos_saved >= args_cli.num_demos:
            print(f"\nTarget of {args_cli.num_demos} demos reached. Exiting.")
            break

        # ── GELLO read ────────────────────────────────────────────────────────
        gello_state  = gello.get_joints()
        arm_joints   = gello_state[:6].astype(np.float32) * GELLO_SIGNS + GELLO_OFFSETS + episode_arm_offset
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
        if flags["recording"] and episode_tracker is None:
            flags["human_clean"] = None
            episode_tracker = make_episode_quality_tracker(env_uw, writer.n_episodes, episode_arm_offset)
            lifted_hold_steps = 0
            print(
                "[quality] Episode started: "
                f"chicken=({episode_tracker['chicken_start_pos'][0]:+.3f}, "
                f"{episode_tracker['chicken_start_pos'][1]:+.3f}, "
                f"{episode_tracker['chicken_start_pos'][2]:+.3f}), "
                f"yaw={np.rad2deg(episode_tracker['chicken_start_yaw_rad']):+.1f} deg. "
                "Press Y=clean, N=not clean before saving."
            )

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
            if episode_tracker is not None:
                episode_tracker["human_clean"] = flags["human_clean"]
            preview = draw_multi_camera_preview(
                cam_frames,
                flags["recording"],
                ep_timestamp,
                episode_tracker,
                coverage_entries,
                demos_saved,
                args_cli.num_demos,
            )
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
                print("[chicken] dropped during recording; discarding this episode")
                if episode_tracker is not None:
                    episode_tracker["ended_reason"] = "dropped"
                flags["discard"] = True

        # ── record ────────────────────────────────────────────────────────────
        if flags["recording"] and not dropped:
            if cam_frames is None:
                cam_frames = get_camera_frames(env_uw)
            if episode_tracker is not None:
                update_episode_quality_tracker(episode_tracker, env_uw, action_np, rec_steps, ep_timestamp)
                stage = 1 if episode_tracker["success"] else stage
                if episode_tracker["success"]:
                    lifted_hold_steps += 1
                else:
                    lifted_hold_steps = 0
            writer.add_step(obs_dict, action_np, cam_frames,
                            timestamp=ep_timestamp, stage=stage)
            rec_steps    += 1
            ep_timestamp += SIM_STEP_DT

            if rec_steps % 50 == 0:
                print(
                    f"  [REC] {rec_steps} steps  (saved: {demos_saved})  "
                    f"reward={reward:.1f}  "
                    f"close={episode_tracker['first_gripper_close_step'] if episode_tracker else None}  "
                    f"max_lift={episode_tracker['max_lift_above_start'] if episode_tracker else 0.0:.3f} m  "
                    f"success={episode_tracker['success'] if episode_tracker else False}"
                )

            if (
                args_cli.auto_stop_on_lift
                and lifted_hold_steps >= max(args_cli.auto_stop_hold_steps, 1)
            ):
                print(
                    "\n[quality] Lift held above threshold — auto-saving "
                    f"(max_lift={episode_tracker['max_lift_above_start']:.3f} m)"
                )
                if episode_tracker is not None:
                    episode_tracker["ended_reason"] = "auto_lift_success"
                flags["save"] = True

            if args_cli.episode_steps > 0 and rec_steps >= args_cli.episode_steps:
                print("\n[INFO] Max episode length — auto-saving")
                if episode_tracker is not None:
                    episode_tracker["ended_reason"] = "max_episode_steps"
                flags["save"] = True

        # ── save ──────────────────────────────────────────────────────────────
        if flags["save"]:
            if episode_tracker is not None:
                episode_tracker["human_clean"] = flags["human_clean"]
                episode_tracker["ended_reason"] = episode_tracker["ended_reason"] or "manual_save"
            saved_metadata = dict(episode_tracker) if episode_tracker is not None else None
            if writer.save_episode(saved_metadata):
                demos_saved += 1
                if saved_metadata is not None:
                    coverage_entries.append(saved_metadata)
                    save_coverage_map(out_root, coverage_entries)
            flags.update({"recording": False, "save": False})
            rec_steps    = 0
            ep_timestamp = 0.0
            episode_tracker = None
            lifted_hold_steps = 0
            episode_arm_offset = sample_episode_arm_offset(rng)
            last_arm_joints = _reset_env(
                env, env_uw, gello, rng, perpendicular_reference_joints, episode_arm_offset
            ).astype(np.float32).copy()
            print(f"  → {demos_saved} episodes saved. Press C for next.\n")

        # ── discard ───────────────────────────────────────────────────────────
        if flags["discard"]:
            writer.discard_episode()
            flags.update({"recording": False, "discard": False})
            rec_steps    = 0
            ep_timestamp = 0.0
            episode_tracker = None
            lifted_hold_steps = 0
            episode_arm_offset = sample_episode_arm_offset(rng)
            last_arm_joints = _reset_env(
                env, env_uw, gello, rng, perpendicular_reference_joints, episode_arm_offset
            ).astype(np.float32).copy()
            print("  → Discarded. Press C for a new episode.\n")

        # ── auto-reset on env termination ─────────────────────────────────────
        if terminated or truncated:
            if flags["recording"] and writer.ep_len > 0:
                print("\n[WARN] Episode terminated early — auto-saving")
                if episode_tracker is not None:
                    episode_tracker["human_clean"] = flags["human_clean"]
                    episode_tracker["ended_reason"] = "env_terminated"
                saved_metadata = dict(episode_tracker) if episode_tracker is not None else None
                if writer.save_episode(saved_metadata):
                    demos_saved += 1
                    if saved_metadata is not None:
                        coverage_entries.append(saved_metadata)
                        save_coverage_map(out_root, coverage_entries)
                flags["recording"] = False
                rec_steps    = 0
                ep_timestamp = 0.0
                episode_tracker = None
                lifted_hold_steps = 0
            episode_arm_offset = sample_episode_arm_offset(rng)
            last_arm_joints = _reset_env(
                env, env_uw, gello, rng, perpendicular_reference_joints, episode_arm_offset
            ).astype(np.float32).copy()

        loop_step += 1

    # ── cleanup ───────────────────────────────────────────────────────────────
    kb.close()
    print("[writer] Waiting for queued zarr/video writes to finish...")
    writer.close()
    if _HAS_CV2:
        cv2.destroyAllWindows()
    env.close()
    print(f"\nDone. {demos_saved} episodes in {args_cli.out_dir}/replay_buffer.zarr/")


def _reset_env(
    env,
    env_uw,
    gello,
    rng: np.random.Generator,
    perpendicular_reference_joints: np.ndarray,
    episode_arm_offset: np.ndarray,
):
    env.reset()
    if np.max(np.abs(episode_arm_offset)) > 0.0:
        print(f"[reset] Episode arm offset deg: {np.rad2deg(episode_arm_offset)}")
    robot = env_uw.scene["robot"]
    ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    current = robot.data.joint_pos[0, ids].cpu().numpy().astype(np.float32)
    current = keep_tool_perpendicular_targets(current, perpendicular_reference_joints)
    chicken_target_pos, chicken_target_quat = place_chicken_on_table(env_uw, rng, reason="episode reset")
    settle_chicken_reset_pose(env, env_uw, current, chicken_target_pos, chicken_target_quat)
    g = keep_tool_perpendicular_targets(
        snap_to_gello(env_uw, gello) + episode_arm_offset,
        perpendicular_reference_joints,
    )
    for _ in range(120):
        current = rate_limit_arm_targets(current, g, MAX_ARM_TARGET_STEP_RAD)
        hold_arm_step(env, env_uw, current)
    return current


if __name__ == "__main__":
    main()
    simulation_app.close()
