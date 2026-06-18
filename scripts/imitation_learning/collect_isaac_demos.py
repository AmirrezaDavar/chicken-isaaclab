#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""
Collect teleoperated demonstrations in Isaac Sim using a GELLO device and save
them to a Zarr replay buffer in ChicGrasp-compatible format for diffusion-policy
training.

GELLO output:  7D  →  [j0..j5 arm joints (rad), gripper_fraction [0,1]]
Isaac action:  8D  →  [j0..j5 absolute joint positions, left_jaw_binary (±1), right_jaw_binary (±1)]

Controls:
  GELLO handle        → arm joint positions (direct joint-space)
  GELLO trigger       → gripper open/close (all 4 jaws together)
  C key               → START recording
  S key               → STOP recording + SAVE episode
  Backspace key       → DISCARD current buffered episode
  Q key               → quit

Zarr layout (ChicGrasp format):
  data/
    obs     : (T, 13)  float32  — [ee_pos(3), ee_euler(3), ck_pos(3), ck_euler(3), gripper(1)]
    action  : (T, 8)   float32  — [j0..j5(6), left_jaw_binary(1), right_jaw_binary(1)]
  meta/
    episode_ends : (N,) int64

Usage (activate isaaclab env first):
  cd /home/wanglab22/3_chicken-isaaclab
  python scripts/imitation_learning/collect_isaac_demos.py \\
      --out_dir /home/wanglab22/ChicGrasp/data/isaac_chicken \\
      --num_demos 50
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib
import sys
import os

# ── GELLO package path (dynamixel_sdk must be installed in this env) ──────────
GELLO_SOFTWARE_DIR = (
    "/home/wanglab22/1_gello_software"
    "(pressure+gelsight+tele speed alighnment))/gello_software"
)
if GELLO_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, GELLO_SOFTWARE_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Isaac Sim chicken-lift demos via GELLO.")
parser.add_argument("--out_dir",       type=str,  default="./data/isaac_chicken",
                    help="Output directory for Zarr dataset.")
parser.add_argument("--num_demos",     type=int,  default=0,
                    help="Number of demos to collect (0 = infinite).")
parser.add_argument("--episode_steps", type=int,  default=300,
                    help="Max steps per episode before auto-save.")
parser.add_argument("--gello_port",    type=str,  default=None,
                    help="GELLO USB serial port (auto-detected if not given).")
parser.add_argument("--calib_path",    type=str,
                    default=os.path.join(GELLO_SOFTWARE_DIR, "gello_calibration.json"),
                    help="Path to gello_calibration.json.")
parser.add_argument("--diagnose",      action="store_true",
                    help="Diagnose mode: freeze sim robot, print GELLO vs sim joints. "
                         "Move GELLO and sim independently to compare joint angles.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True   # required for CameraCfg sensor

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after sim is live."""

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
    print("[WARN] opencv-python not found — camera live view disabled. "
          "Run: pip install opencv-python")

import numcodecs

# ─────────────────────────────────────────────────────────────────────────────
TASK_ID    = "Isaac-Lift-Chicken-UR10e-CustomGripper-GELLO-v0"
OBS_DIM    = 13  # ee_pos(3) + ee_euler(3) + ck_pos(3) + ck_euler(3) + gripper(1)
ACTION_DIM = 8   # arm_joints(6) + left_jaw_binary(1) + right_jaw_binary(1)
CAM_H      = 480
CAM_W      = 640

# Gripper fraction threshold: frac < THRESH → open, frac >= THRESH → close
GRIPPER_THRESH = 0.5

# ── GELLO → sim joint correction ─────────────────────────────────────────────
# Fill these in after running --diagnose:
#   1. Run:  python collect_isaac_demos.py --diagnose
#   2. Hold GELLO in its natural home pose (arm hanging down / relaxed)
#   3. Move sim robot (via Isaac Sim viewport) to the matching pose
#   4. Read the table: for each joint where GELLO(°) and SIM(°) differ:
#        - Δ ≈ ±180°  →  set SIGN to -1
#        - Δ ≈ constant offset  →  set OFFSET to that value (radians)
# Joint order: [pan, lift, elbow, wrist1, wrist2, wrist3]
GELLO_SIGNS   = np.array([1.0,  1.0,  1.0,  1.0,  1.0,  1.0], dtype=np.float32)
GELLO_OFFSETS = np.array([0.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# GELLO reader
# ─────────────────────────────────────────────────────────────────────────────

def _find_gello_port() -> str:
    """Auto-detect the GELLO FTDI serial port."""
    from glob import glob
    ports = glob("/dev/serial/by-id/*FTDI*")
    if not ports:
        raise RuntimeError(
            "No FTDI serial device found. Plug in the GELLO USB cable and retry."
        )
    if len(ports) > 1:
        print(f"[GELLO] Multiple FTDI ports found: {ports}")
        print(f"[GELLO] Using first: {ports[0]}")
    return ports[0]


class GelloReader:
    """Reads 7D joint state from a GELLO arm (6 arm joints + 1 gripper fraction)."""

    def __init__(self, port: str, calib_path: str):
        from gello.robots.dynamixel import DynamixelRobot

        with open(calib_path) as f:
            calib = json.load(f)

        joint_offsets = calib["offsets"]   # 6 arm offsets from latest calibration
        joint_signs   = calib["signs"]     # 6 arm signs

        # Gripper open/close positions from calibration JSON.
        # gello_get_offset.py records:
        #   open  = raw_reading_deg - 0.2  → stored as gripper_offset_deg
        #   close = raw_reading_deg - 42   = open - 41.8
        g_open_deg  = calib["gripper_offset_deg"]
        g_close_deg = g_open_deg - 41.8
        gripper_config = (7, g_open_deg, g_close_deg)
        print(f"[GELLO] Gripper open={g_open_deg:.2f}°  close={g_close_deg:.2f}°")

        print(f"[GELLO] Connecting to port: {port}")
        self._robot = DynamixelRobot(
            joint_ids=(1, 2, 3, 4, 5, 6),
            joint_offsets=joint_offsets,
            joint_signs=joint_signs,
            real=True,
            port=port,
            gripper_config=gripper_config,
        )
        print("[GELLO] Connected.")

    def get_joints(self) -> np.ndarray:
        """Return 7D: [j0..j5 in radians, gripper_fraction in [0,1]]."""
        return self._robot.get_joint_state()   # shape (7,)

    def get_arm_joints(self) -> np.ndarray:
        return self.get_joints()[:6]

    def get_gripper_frac(self) -> float:
        return float(self.get_joints()[6])


# ─────────────────────────────────────────────────────────────────────────────
# Keyboard handler (direct Carb subscription — no SE3 key conflicts)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleKeyboard:
    def __init__(self):
        self._input     = carb.input.acquire_input_interface()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._keyboard  = self._appwindow.get_keyboard()
        self._callbacks = {}
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_event
        )

    def add_callback(self, key: str, func):
        self._callbacks[key.upper()] = func

    def _on_event(self, event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            name = event.input.name
            if name in self._callbacks:
                self._callbacks[name]()
        return True

    def close(self):
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)


# ─────────────────────────────────────────────────────────────────────────────
# Observation extraction
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_euler(quat_wxyz: np.ndarray) -> np.ndarray:
    t = torch.tensor(quat_wxyz, dtype=torch.float32).unsqueeze(0)
    r, p, y = euler_xyz_from_quat(t)
    return np.array([r.item(), p.item(), y.item()], dtype=np.float32)


def extract_obs(env_uw) -> np.ndarray:
    """13D: [ee_pos(3), ee_euler(3), chicken_pos(3), chicken_euler(3), gripper_frac(1)]"""
    scene = env_uw.scene

    ee_pos_w  = scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    ee_quat_w = scene["ee_frame"].data.target_quat_w[0, 0].cpu().numpy()
    ck_pos_w  = scene["chicken"].data.root_pos_w[0].cpu().numpy()
    ck_quat_w = scene["chicken"].data.root_quat_w[0].cpu().numpy()
    rb_pos_w  = scene["robot"].data.root_pos_w[0].cpu().numpy()

    ee_pos_r = ee_pos_w - rb_pos_w
    ck_pos_r = ck_pos_w - rb_pos_w

    robot = scene["robot"]
    gripper_ids, _ = robot.find_joints("PrismaticJoint.*")
    gpos = robot.data.joint_pos[0, gripper_ids].cpu().numpy()
    gripper_frac = float(np.clip((gpos - 0.0) / (-0.0093 - 0.0), 0.0, 1.0).mean())

    return np.concatenate([
        ee_pos_r,
        _quat_to_euler(ee_quat_w),
        ck_pos_r,
        _quat_to_euler(ck_quat_w),
        [gripper_frac],
    ]).astype(np.float32)


_ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def get_sim_arm_joints(env_uw) -> np.ndarray:
    """Read the 6 arm joint positions currently in sim."""
    robot = env_uw.scene["robot"]
    ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    return robot.data.joint_pos[0, ids].cpu().numpy()


def snap_to_gello(env_uw, gello: "GelloReader") -> np.ndarray:
    """Teleport sim robot directly to GELLO's current position (no interpolation).

    Returns the corrected GELLO arm joint positions (6-D, radians).
    """
    gello_pos = gello.get_arm_joints() * GELLO_SIGNS + GELLO_OFFSETS
    robot = env_uw.scene["robot"]
    ids, _ = robot.find_joints(_ARM_JOINT_NAMES)
    pos_t = torch.tensor(gello_pos, device=env_uw.device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_position_to_sim(pos_t, joint_ids=ids)
    return gello_pos


def get_camera_frame(env_uw) -> np.ndarray:
    """Return the gripper camera image as (H, W, 3) uint8 RGB."""
    rgb_t = env_uw.scene["camera"].data.output["rgb"][0, :, :, :3]  # (H, W, 3) float or uint8
    rgb_np = rgb_t.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
    return rgb_np


_JOINT_NAMES = ["pan", "lift", "elbow", "wrist1", "wrist2", "wrist3"]


def print_joint_table(gello_raw: np.ndarray, sim_joints: np.ndarray):
    """Print a side-by-side comparison of GELLO and sim joint angles."""
    corrected = gello_raw * GELLO_SIGNS + GELLO_OFFSETS
    print("\033[2J\033[H", end="")   # clear terminal
    print("─" * 66)
    print(f"  {'Joint':<10}  {'GELLO raw(°)':>12}  {'Corrected(°)':>12}  {'SIM(°)':>10}  {'Δ(°)':>8}")
    print("─" * 66)
    for i, name in enumerate(_JOINT_NAMES):
        g = np.rad2deg(gello_raw[i])
        c = np.rad2deg(corrected[i])
        s = np.rad2deg(sim_joints[i])
        d = c - s
        flag = "  ← fix" if abs(d) > 10 else ""
        print(f"  {name:<10}  {g:>+12.2f}  {c:>+12.2f}  {s:>+10.2f}  {d:>+8.2f}{flag}")
    print("─" * 66)
    print("  Joints marked '← fix': adjust GELLO_SIGNS or GELLO_OFFSETS")
    print("  Δ ≈ ±180° → flip sign  |  Δ ≈ constant → add offset (rad)")
    print("  Press Q to exit.\n")


def run_diagnose_loop(env, env_uw, gello):
    """Drive sim robot with GELLO and stream joint comparison table.

    Move the GELLO and watch the robot follow — joints marked '← fix' are
    moving in the wrong direction.  Edit GELLO_SIGNS / GELLO_OFFSETS then
    re-run --diagnose until all Δ values stay near 0.
    """
    kb = SimpleKeyboard()
    quit_flag = {"v": False}
    kb.add_callback("Q", lambda: quit_flag.update({"v": True}))

    env.reset()
    step = 0

    print("\n[DIAGNOSE] Robot follows GELLO. Move GELLO to see directions.")
    print("[DIAGNOSE] Press Q to quit.\n")

    while simulation_app.is_running() and not quit_flag["v"]:
        gello_raw     = gello.get_arm_joints()
        corrected     = gello_raw * GELLO_SIGNS + GELLO_OFFSETS
        gripper_frac  = float(gello.get_gripper_frac())
        gripper_bin   = 1.0 if gripper_frac < GRIPPER_THRESH else -1.0

        act = np.array([*corrected, gripper_bin, gripper_bin], dtype=np.float32)
        env.step(torch.tensor(act, device=env_uw.device).unsqueeze(0))

        # Refresh table every 30 steps (~0.6 s)
        if step % 30 == 0:
            sim_joints = get_sim_arm_joints(env_uw)
            print_joint_table(gello_raw, sim_joints)

        step += 1

    kb.close()


# ─────────────────────────────────────────────────────────────────────────────
# Zarr writer
# ─────────────────────────────────────────────────────────────────────────────

class ZarrDemoWriter:
    def __init__(self, out_dir: str):
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        zarr_path = out / "replay_buffer.zarr"
        store = zarr.DirectoryStore(str(zarr_path))
        self.root = zarr.group(store=store, overwrite=False)
        data = self.root.require_group("data", overwrite=False)
        meta = self.root.require_group("meta", overwrite=False)

        # Wipe stale store if action dim changed and no episodes recorded
        if "action" in data and data["action"].shape[-1] != ACTION_DIM:
            n_ep = len(meta["episode_ends"]) if "episode_ends" in meta else 0
            if n_ep == 0:
                print(f"[zarr] Recreating store: action dim {data['action'].shape[-1]} → {ACTION_DIM}")
                self.root = zarr.group(store=store, overwrite=True)
                data = self.root.require_group("data", overwrite=False)
                meta = self.root.require_group("meta", overwrite=False)
            else:
                raise RuntimeError(
                    f"Existing store has action dim {data['action'].shape[-1]} but "
                    f"script expects {ACTION_DIM}. Delete {zarr_path} manually."
                )

        def _open_or_create(group, name, init_shape, dtype, chunks, **kw):
            if name in group:
                return group[name]
            return group.require_dataset(name, shape=init_shape, dtype=dtype, chunks=chunks, **kw)

        self.obs_arr  = _open_or_create(data, "obs",           (0, OBS_DIM),    "float32", (1000, OBS_DIM))
        self.act_arr  = _open_or_create(data, "action",        (0, ACTION_DIM), "float32", (1000, ACTION_DIM))
        self.cam_arr  = _open_or_create(
            data, "camera", (0, CAM_H, CAM_W, 3), "uint8", (50, CAM_H, CAM_W, 3),
            compressor=numcodecs.Blosc(cname="lz4", clevel=3),
        )
        self.ends_arr = _open_or_create(meta, "episode_ends",  (0,),            "int64",   (500,))

        self._ep_obs:  list = []
        self._ep_acts: list = []
        self._ep_cams: list = []
        self._total_steps = int(self.ends_arr[-1]) if len(self.ends_arr) > 0 else 0
        self._n_episodes  = len(self.ends_arr)

    @property
    def n_episodes(self) -> int:
        return self._n_episodes

    @property
    def ep_len(self) -> int:
        return len(self._ep_obs)

    def add_step(self, obs: np.ndarray, action: np.ndarray, camera: np.ndarray | None = None):
        self._ep_obs.append(obs)
        self._ep_acts.append(action)
        if camera is not None:
            self._ep_cams.append(camera)

    def save_episode(self) -> bool:
        if not self._ep_obs:
            print("  [zarr] nothing to save (episode is empty)")
            return False
        T         = len(self._ep_obs)
        prev_T    = self._total_steps
        obs_block = np.stack(self._ep_obs,  axis=0)
        act_block = np.stack(self._ep_acts, axis=0)

        self.obs_arr.resize((prev_T + T, OBS_DIM))
        self.act_arr.resize((prev_T + T, ACTION_DIM))
        self.obs_arr[prev_T:prev_T + T] = obs_block
        self.act_arr[prev_T:prev_T + T] = act_block

        if self._ep_cams:
            cam_block = np.stack(self._ep_cams, axis=0)
            self.cam_arr.resize((prev_T + T, CAM_H, CAM_W, 3))
            self.cam_arr[prev_T:prev_T + T] = cam_block

        self._total_steps += T
        self._n_episodes  += 1
        self.ends_arr.resize((self._n_episodes,))
        self.ends_arr[self._n_episodes - 1] = self._total_steps

        self._ep_obs.clear()
        self._ep_acts.clear()
        self._ep_cams.clear()
        print(f"  [zarr] saved episode {self._n_episodes}  ({T} steps, total {self._total_steps})")
        return True

    def discard_episode(self):
        n = len(self._ep_obs)
        self._ep_obs.clear()
        self._ep_acts.clear()
        self._ep_cams.clear()
        print(f"  [zarr] discarded {n} buffered steps")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── env ──────────────────────────────────────────────────────────────────
    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = 10000.0
    env = gym.make(TASK_ID, cfg=env_cfg)
    env_uw = env.unwrapped

    print(f"[INFO] obs space  : {env.observation_space}")
    print(f"[INFO] act space  : {env.action_space}")

    # ── GELLO ────────────────────────────────────────────────────────────────
    gello_port = args_cli.gello_port or _find_gello_port()
    gello = GelloReader(port=gello_port, calib_path=args_cli.calib_path)

    # ── diagnose mode: freeze sim, stream joint comparison, then exit ─────────
    if args_cli.diagnose:
        run_diagnose_loop(env, env_uw, gello)
        env.close()
        return

    # ── Keyboard ─────────────────────────────────────────────────────────────
    kb = SimpleKeyboard()

    flags = {
        "is_recording":   False,
        "should_save":    False,
        "should_discard": False,
        "quit":           False,
    }

    kb.add_callback("C",         lambda: flags.update({"is_recording": True})
                    or print("\n[REC] Recording started — press S to save, Backspace to discard"))
    kb.add_callback("S",         lambda: flags.update({"should_save": True}))
    kb.add_callback("BACKSPACE", lambda: flags.update({"should_discard": True}))
    kb.add_callback("Q",         lambda: flags.update({"quit": True}))

    # ── Zarr writer ───────────────────────────────────────────────────────────
    writer = ZarrDemoWriter(args_cli.out_dir)
    print(f"\n[INFO] Dataset: {args_cli.out_dir}/replay_buffer.zarr")
    print(f"[INFO] Resuming from {writer.n_episodes} existing episodes")

    print("\n" + "="*60)
    print("Controls:")
    print("  GELLO handle  → arm joint positions")
    print("  GELLO trigger → all 4 gripper jaws open/close")
    print("  C             → START recording")
    print("  S             → STOP + SAVE episode")
    print("  Backspace     → discard current episode")
    print("  Q             → quit")
    print("="*60 + "\n")

    # ── reset and snap sim robot to GELLO's current position ─────────────────
    env.reset()
    print("[GELLO] Snapping sim robot to GELLO position...")
    gello_start = snap_to_gello(env_uw, gello)
    # Let physics settle at the new position (no visible movement)
    for _ in range(10):
        act = np.array([*gello_start, 1.0, 1.0], dtype=np.float32)
        env.step(torch.tensor(act, device=env_uw.device).unsqueeze(0))
    print("[GELLO] Ready. Press C to start recording.")

    # ── main loop ─────────────────────────────────────────────────────────────
    demos_saved = writer.n_episodes
    rec_steps   = 0

    while simulation_app.is_running() and not flags["quit"]:
        if args_cli.num_demos > 0 and demos_saved >= args_cli.num_demos:
            print(f"\nTarget of {args_cli.num_demos} demos reached. Exiting.")
            break

        # ── read GELLO ────────────────────────────────────────────────────────
        gello_state   = gello.get_joints()             # (7,)
        arm_joints_7d = gello_state[:6].astype(np.float32) * GELLO_SIGNS + GELLO_OFFSETS
        gripper_frac  = float(gello_state[6])

        # Binary gripper: open when trigger < threshold
        gripper_binary = 1.0 if gripper_frac < GRIPPER_THRESH else -1.0

        # ── assemble 8D action (both jaws share the GELLO trigger) ───────────
        action_np = np.array([*arm_joints_7d, gripper_binary, gripper_binary], dtype=np.float32)
        action_t  = torch.tensor(action_np, device=env_uw.device).unsqueeze(0)

        # ── observe + camera (before step) ───────────────────────────────────
        obs_vec   = extract_obs(env_uw)
        cam_frame = get_camera_frame(env_uw)

        # ── live camera view ──────────────────────────────────────────────────
        if _HAS_CV2:
            cv2.imshow("RealSense Camera", cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # ── step sim ──────────────────────────────────────────────────────────
        _, _, terminated, truncated, _ = env.step(action_t)

        # ── record ────────────────────────────────────────────────────────────
        if flags["is_recording"]:
            writer.add_step(obs_vec, action_np, cam_frame)
            rec_steps += 1

            if rec_steps % 50 == 0:
                print(f"  [REC] {rec_steps} steps  "
                      f"(saved: {demos_saved}  target: "
                      f"{'∞' if args_cli.num_demos == 0 else args_cli.num_demos})")

            if rec_steps >= args_cli.episode_steps:
                print(f"\n[INFO] Max episode length reached — auto-saving")
                flags["should_save"] = True

        # ── save ──────────────────────────────────────────────────────────────
        if flags["should_save"]:
            if writer.save_episode():
                demos_saved += 1
            flags["is_recording"] = False
            flags["should_save"]  = False
            rec_steps = 0
            env.reset()
            gello_now = snap_to_gello(env_uw, gello)
            for _ in range(5):
                act_r = np.array([*gello_now, 1.0, 1.0], dtype=np.float32)
                env.step(torch.tensor(act_r, device=env_uw.device).unsqueeze(0))
            print(f"  → Episode saved. {demos_saved} total. Press C to start next.\n")

        # ── discard ───────────────────────────────────────────────────────────
        if flags["should_discard"]:
            writer.discard_episode()
            flags["is_recording"]   = False
            flags["should_discard"] = False
            rec_steps = 0
            env.reset()
            gello_now = snap_to_gello(env_uw, gello)
            for _ in range(5):
                act_r = np.array([*gello_now, 1.0, 1.0], dtype=np.float32)
                env.step(torch.tensor(act_r, device=env_uw.device).unsqueeze(0))
            print("  → Episode discarded. Press C to start a new one.\n")

        # ── handle env termination ────────────────────────────────────────────
        if terminated or truncated:
            if flags["is_recording"] and writer.ep_len > 0:
                print("\n[WARN] Episode terminated early — auto-saving")
                if writer.save_episode():
                    demos_saved += 1
                flags["is_recording"] = False
                rec_steps = 0
            env.reset()
            gello_now = snap_to_gello(env_uw, gello)
            for _ in range(5):
                act_r = np.array([*gello_now, 1.0, 1.0], dtype=np.float32)
                env.step(torch.tensor(act_r, device=env_uw.device).unsqueeze(0))

    # ── cleanup ───────────────────────────────────────────────────────────────
    kb.close()
    if _HAS_CV2:
        cv2.destroyAllWindows()
    env.close()
    print(f"\nDone. {demos_saved} episodes saved to {args_cli.out_dir}/replay_buffer.zarr")


if __name__ == "__main__":
    main()
    simulation_app.close()
