#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate an RGB+state ChicGrasp Diffusion Policy checkpoint in Isaac Lab."""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

from isaaclab.app import AppLauncher


GELLO_SOFTWARE_DIR = (
    "/home/wanglab22/1_gello_software"
    "(pressure+gelsight+tele speed alighnment))/gello_software"
)
if GELLO_SOFTWARE_DIR not in sys.path:
    sys.path.insert(0, GELLO_SOFTWARE_DIR)

CHICGRASP_ROOT = pathlib.Path("/home/wanglab22/ChicGrasp")
DEFAULT_IMAGE_KEYS = ["camera_rgb", "camera_left_rgb", "camera_right_rgb"]
IMAGE_KEY_TO_SCENE_CAMERA = {
    "camera_rgb": "camera",
    "camera_left_rgb": "left_camera",
    "camera_right_rgb": "right_camera",
}

parser = argparse.ArgumentParser(description="Evaluate RGB+state ChicGrasp DP checkpoint in Isaac.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to latest.ckpt from RGB+state training.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Chicken-UR10e-CustomGripper-GELLO-v0")
parser.add_argument("--num_episodes", type=int, default=1)
parser.add_argument("--episode_steps", type=int, default=300)
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--device_policy", type=str, default="cuda:0")
parser.add_argument("--chicgrasp_root", type=str, default=str(CHICGRASP_ROOT))
parser.add_argument("--image_keys", type=str, nargs="+", default=None,
                    help="RGB observation keys to feed. Defaults to RGB keys saved in the checkpoint config.")
parser.add_argument("--image_key", type=str, default=None,
                    help="Backward-compatible single RGB key override. Prefer --image_keys.")
parser.add_argument("--state_key", type=str, default="state")
parser.add_argument("--action_stride", type=int, default=0, help="0 uses checkpoint n_action_steps.")
parser.add_argument("--no_live_camera", action="store_true", help="Disable the OpenCV multi-camera preview.")
parser.add_argument("--preview_stride", type=int, default=1, help="Show one preview frame every N sim steps.")
parser.add_argument("--preview_width", type=int, default=1280, help="Initial live camera window width.")
parser.add_argument("--preview_height", type=int, default=720, help="Initial live camera window height.")
parser.add_argument("--no_gello", action="store_true", help="Disable GELLO manual staging while policy is stopped.")
parser.add_argument(
    "--absolute_gello",
    action="store_true",
    help="Use raw GELLO absolute joint targets during staging. Default is safer relative staging.",
)
parser.add_argument("--gello_port", type=str, default=None)
parser.add_argument(
    "--calib_path",
    type=str,
    default=os.path.join(GELLO_SOFTWARE_DIR, "gello_calibration.json"),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import dill  # noqa: E402
import gymnasium as gym  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import carb.input  # noqa: E402
import omni.appwindow  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

try:
    import cv2  # noqa: E402
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    print("[WARN] opencv-python not found; live evaluation camera preview disabled.")


chicgrasp_root = pathlib.Path(args_cli.chicgrasp_root)
if str(chicgrasp_root) not in sys.path:
    sys.path.insert(0, str(chicgrasp_root))

OmegaConf.register_new_resolver("eval", eval, replace=True)

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_THRESH = 0.5
ARM_IDLE_DEADBAND_RAD = 0.01
MAX_ARM_TARGET_STEP_RAD = 0.025
GELLO_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
GELLO_OFFSETS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
_SHOULDER_LIFT_ID = 1
_ELBOW_ID = 2
_WRIST_PITCH_ID = 3
_WRIST_ROLL_ID = 4
TOOL_PITCH_OFFSET_DEG = 9.5
PREVIEW_WINDOW_NAME = "Policy Evaluation Cameras"
PREVIEW_TILE_W = 640
PREVIEW_TILE_H = 360
PREVIEW_LABEL_H = 34
PREVIEW_LABELS = [
    ("camera_rgb", "WRIST"),
    ("camera_left_rgb", "LEFT TABLE"),
    ("camera_right_rgb", "RIGHT TABLE"),
    ("__empty__", ""),
]


class SimpleKeyboard:
    def __init__(self):
        self._input = carb.input.acquire_input_interface()
        self._appwindow = omni.appwindow.get_default_app_window()
        self._keyboard = self._appwindow.get_keyboard()
        self._callbacks = {}
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)

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


def _find_gello_port() -> str:
    from glob import glob

    ports = glob("/dev/serial/by-id/*FTDI*")
    if not ports:
        raise RuntimeError("No FTDI serial device found. Use --no_gello to run policy eval without manual staging.")
    if len(ports) > 1:
        print(f"[GELLO] Multiple ports: {ports}. Using {ports[0]}")
    return ports[0]


class GelloReader:
    def __init__(self, port: str, calib_path: str):
        from gello.robots.dynamixel import DynamixelRobot

        with open(calib_path) as f:
            calib = json.load(f)
        g_open = calib["gripper_offset_deg"]
        g_close = g_open - 41.8
        print(f"[GELLO] gripper open={g_open:.2f} deg  close={g_close:.2f} deg")
        print(f"[GELLO] Connecting -> {port}")
        self._robot = DynamixelRobot(
            joint_ids=(1, 2, 3, 4, 5, 6),
            joint_offsets=calib["offsets"],
            joint_signs=calib["signs"],
            real=True,
            port=port,
            gripper_config=(7, g_open, g_close),
        )
        print("[GELLO] Connected. Move GELLO while policy is stopped.")

    def get_joints(self) -> np.ndarray:
        return self._robot.get_joint_state()

    def get_arm_joints(self) -> np.ndarray:
        return self.get_joints()[:6]

    def get_gripper_frac(self) -> float:
        return float(self.get_joints()[6])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    r, p, y = euler_xyz_from_quat(t)
    return np.array([r.item(), p.item(), y.item()], dtype=np.float32)


def extract_state(env_uw) -> np.ndarray:
    scene = env_uw.scene
    robot = scene["robot"]
    rb_pos = robot.data.root_pos_w[0].cpu().numpy()

    ee_pos_w = scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    ee_quat_w = scene["ee_frame"].data.target_quat_w[0, 0].cpu().numpy()
    ee_pos_r = (ee_pos_w - rb_pos).astype(np.float32)
    ee_euler = quat_to_euler(ee_quat_w)

    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    robot_joint = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
    robot_joint_vel = robot.data.joint_vel[0, arm_ids].cpu().numpy().astype(np.float32)

    lj_ids, _ = robot.find_joints(["PrismaticJoint1", "PrismaticJoint2"])
    rj_ids, _ = robot.find_joints(["PrismaticJoint3", "PrismaticJoint4"])
    left_jaw = float(np.clip(robot.data.joint_pos[0, lj_ids].cpu().numpy() / -0.0093, 0.0, 1.0).mean())
    right_jaw = float(np.clip(robot.data.joint_pos[0, rj_ids].cpu().numpy() / -0.0093, 0.0, 1.0).mean())

    return np.concatenate(
        [
            robot_joint,
            robot_joint_vel,
            np.concatenate([ee_pos_r, ee_euler]).astype(np.float32),
            np.array([left_jaw], dtype=np.float32),
            np.array([right_jaw], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)


def get_camera_frame(env_uw, scene_camera_name: str) -> np.ndarray:
    rgb_t = env_uw.scene[scene_camera_name].data.output["rgb"][0, :, :, :3]
    rgb_np = rgb_t.cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255.0).clip(0, 255).astype(np.uint8)
    return rgb_np


def get_camera_frames(env_uw, image_keys: list[str]) -> dict[str, np.ndarray]:
    frames = {}
    for image_key in image_keys:
        scene_camera = IMAGE_KEY_TO_SCENE_CAMERA.get(image_key)
        if scene_camera is None:
            raise KeyError(f"No Isaac scene camera mapping for image key '{image_key}'")
        frames[image_key] = get_camera_frame(env_uw, scene_camera)
    return frames


def image_to_policy_tensor(image: np.ndarray) -> np.ndarray:
    # H,W,C uint8 -> C,H,W float32 [0,1]
    return np.moveaxis(image, -1, 0).astype(np.float32) / 255.0


def load_policy(checkpoint_path: str, device: str):
    try:
        import huggingface_hub

        if not hasattr(huggingface_hub, "cached_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    except Exception:
        pass

    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if cfg.training.use_ema and "ema_model" in payload["state_dicts"] else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    policy.to(torch.device(device))
    policy.eval()
    print(f"[policy] Loaded {checkpoint_path}")
    print(f"[policy] Using checkpoint state_dicts/{state_key}")
    print(f"[policy] n_obs_steps={policy.n_obs_steps}, n_action_steps={policy.n_action_steps}")
    return policy, cfg


def get_rgb_obs_keys_from_cfg(cfg) -> list[str]:
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    obs_meta = shape_meta["obs"]
    return [key for key, meta in obs_meta.items() if meta.get("type") == "rgb"]


def make_hold_action(env_uw, last_action: np.ndarray | None = None) -> np.ndarray:
    robot = env_uw.scene["robot"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    arm_joints = robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)
    gripper = last_action[6:8].astype(np.float32) if last_action is not None else np.array([1.0, 1.0], dtype=np.float32)
    return np.array([*arm_joints, *gripper], dtype=np.float32)


def get_sim_arm_joints(env_uw) -> np.ndarray:
    robot = env_uw.scene["robot"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
    return robot.data.joint_pos[0, arm_ids].cpu().numpy().astype(np.float32)


def get_gripper_tip_z_w(env_uw) -> float:
    return float(env_uw.scene["ee_frame"].data.target_pos_w[0, 0, 2].item())


def rate_limit_arm_targets(prev: np.ndarray, target: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.clip(target - prev, -max_step, max_step)
    return (prev + delta).astype(np.float32)


def keep_tool_perpendicular_targets(target: np.ndarray, reference_joints: np.ndarray) -> np.ndarray:
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


def clamp_arm_target_to_gripper_z_limit(
    env_uw,
    current: np.ndarray,
    target: np.ndarray,
    z_limit_w: float | None,
) -> tuple[np.ndarray, bool]:
    if z_limit_w is None or get_gripper_tip_z_w(env_uw) < z_limit_w:
        return target, False

    robot = env_uw.scene["robot"]
    arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES)
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


def snap_to_gello(gello: GelloReader) -> np.ndarray:
    return (gello.get_arm_joints().astype(np.float32) * GELLO_SIGNS + GELLO_OFFSETS).astype(np.float32)


def make_gello_action(
    env_uw,
    gello: GelloReader,
    last_arm_joints: np.ndarray,
    perpendicular_reference_joints: np.ndarray,
    gripper_z_limit_w: float | None,
    gello_anchor_joints: np.ndarray | None = None,
    sim_anchor_joints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    gello_state = gello.get_joints()
    raw_arm_joints = gello_state[:6].astype(np.float32) * GELLO_SIGNS + GELLO_OFFSETS
    if gello_anchor_joints is not None and sim_anchor_joints is not None:
        arm_joints = sim_anchor_joints + (raw_arm_joints - gello_anchor_joints)
    else:
        arm_joints = raw_arm_joints
    arm_joints = keep_tool_perpendicular_targets(arm_joints, perpendicular_reference_joints)
    arm_joints, z_limited = clamp_arm_target_to_gripper_z_limit(env_uw, last_arm_joints, arm_joints, gripper_z_limit_w)

    if np.max(np.abs(arm_joints - last_arm_joints)) < ARM_IDLE_DEADBAND_RAD:
        arm_joints = last_arm_joints.copy()
    else:
        arm_joints = rate_limit_arm_targets(last_arm_joints, arm_joints, MAX_ARM_TARGET_STEP_RAD)

    gripper_frac = float(gello_state[6])
    gripper_bin = 1.0 if gripper_frac < GRIPPER_THRESH else -1.0
    action = np.array([*arm_joints, gripper_bin, gripper_bin], dtype=np.float32)
    return action, arm_joints.astype(np.float32), z_limited


def _draw_preview_tile(frame: np.ndarray | None, label: str) -> np.ndarray:
    if frame is None:
        tile = np.zeros((PREVIEW_TILE_H, PREVIEW_TILE_W, 3), dtype=np.uint8)
        if label:
            cv2.putText(tile, label, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 225, 225), 2, cv2.LINE_AA)
        cv2.putText(tile, "NO CAMERA", (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (120, 120, 120), 2, cv2.LINE_AA)
        return tile

    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tile = cv2.resize(bgr, (PREVIEW_TILE_W, PREVIEW_TILE_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (PREVIEW_TILE_W, PREVIEW_LABEL_H), (12, 12, 12), -1)
    if label:
        cv2.putText(tile, label, (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    return tile


def draw_multi_camera_preview(frames: dict[str, np.ndarray], running: bool, episode: int, step: int, elapsed_s: float) -> np.ndarray:
    tiles = [_draw_preview_tile(frames.get(key), label) for key, label in PREVIEW_LABELS]
    preview = np.vstack((np.hstack((tiles[0], tiles[1])), np.hstack((tiles[2], tiles[3]))))
    if running:
        mins = int(elapsed_s // 60)
        secs = int(elapsed_s % 60)
        cv2.circle(preview, (24, 52), 9, (0, 0, 255), -1, cv2.LINE_AA)
        label = f"EVAL ep {episode} step {step}  {mins:02d}:{secs:02d}"
        cv2.putText(preview, label, (42, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(preview, "READY - C/SPACE START, S STOP", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    return preview


def create_resizable_preview_window(width: int, height: int) -> list[int]:
    flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
    cv2.namedWindow(PREVIEW_WINDOW_NAME, flags)
    width = max(int(width), 640)
    height = max(int(height), 360)
    cv2.resizeWindow(PREVIEW_WINDOW_NAME, width, height)
    return [width, height]


def handle_preview_window_key(key: int, window_size: list[int], flags: dict[str, bool]) -> None:
    if key in (ord("c"), ord("C"), ord(" ")):
        if not flags["running"]:
            flags["running"] = True
            flags["start_requested"] = True
        return
    if key in (ord("s"), ord("S")):
        flags["running"] = False
        return
    if key in (ord("q"), ord("Q"), 27):
        flags["quit"] = True
        return

    if key in (ord("+"), ord("=")):
        scale = 1.15
    elif key in (ord("-"), ord("_")):
        scale = 1.0 / 1.15
    else:
        return

    window_size[0] = max(640, int(window_size[0] * scale))
    window_size[1] = max(360, int(window_size[1] * scale))
    cv2.resizeWindow(PREVIEW_WINDOW_NAME, window_size[0], window_size[1])


def init_histories(env_uw, image_keys: list[str], n_obs_steps: int):
    state_hist = collections.deque(maxlen=n_obs_steps)
    image_hists = {image_key: collections.deque(maxlen=n_obs_steps) for image_key in image_keys}
    for _ in range(n_obs_steps):
        state_hist.append(extract_state(env_uw))
        for image_key, frame in get_camera_frames(env_uw, image_keys).items():
            image_hists[image_key].append(image_to_policy_tensor(frame))
    return state_hist, image_hists


def predict_action_queue(policy, state_hist, image_hists, image_keys: list[str], stride: int) -> list[np.ndarray]:
    obs = {
        args_cli.state_key: torch.from_numpy(np.stack(list(state_hist), axis=0)[None]).to(args_cli.device_policy),
    }
    for image_key in image_keys:
        obs[image_key] = torch.from_numpy(np.stack(list(image_hists[image_key]), axis=0)[None]).to(args_cli.device_policy)
    with torch.no_grad():
        result = policy.predict_action(obs)
    actions = result["action"][0].detach().cpu().numpy().astype(np.float32)
    return [actions[i] for i in range(min(stride, len(actions)))]


def threshold_policy_gripper(action: np.ndarray) -> np.ndarray:
    action = action.astype(np.float32, copy=True)
    action[6:8] = np.where(action[6:8] >= 0.0, 1.0, -1.0).astype(np.float32)
    return action


def step_env(env, env_uw, action: np.ndarray):
    action_t = torch.tensor(action, dtype=torch.float32, device=env_uw.device).unsqueeze(0)
    return env.step(action_t)


def main() -> None:
    checkpoint_path = pathlib.Path(args_cli.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not _HAS_CV2 and not args_cli.no_live_camera:
        print("[WARN] Live camera preview requested, but cv2 is unavailable.")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    env_cfg.episode_length_s = 10000.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_uw = env.unwrapped

    gello = None
    if not args_cli.no_gello:
        gello_port = args_cli.gello_port or _find_gello_port()
        gello = GelloReader(port=gello_port, calib_path=args_cli.calib_path)

    policy, cfg = load_policy(str(checkpoint_path), args_cli.device_policy)
    if args_cli.image_key is not None:
        image_keys = [args_cli.image_key]
    elif args_cli.image_keys is not None:
        image_keys = list(args_cli.image_keys)
    else:
        image_keys = get_rgb_obs_keys_from_cfg(cfg) or DEFAULT_IMAGE_KEYS
    print(f"[policy] RGB observation keys: {image_keys}")

    n_obs_steps = int(policy.n_obs_steps)
    stride = int(args_cli.action_stride) if args_cli.action_stride > 0 else int(policy.n_action_steps)

    flags = {"running": False, "quit": False, "start_requested": False, "reanchor_gello": True}

    kb = SimpleKeyboard()

    def start_policy():
        if not flags["running"]:
            flags["running"] = True
            flags["start_requested"] = True
            print("[eval] START")

    def stop_policy():
        if flags["running"]:
            flags["running"] = False
            flags["reanchor_gello"] = True
            print("[eval] STOP")

    def quit_eval():
        flags["quit"] = True
        print("[eval] QUIT")

    kb.add_callback("C", start_policy)
    kb.add_callback("SPACE", start_policy)
    kb.add_callback("S", stop_policy)
    kb.add_callback("Q", quit_eval)

    preview_allowed = _HAS_CV2 and not args_cli.no_live_camera
    preview_window_size = None
    loop_step = 0
    ep = 0
    step = 0
    last_action: np.ndarray | None = None
    last_arm_joints = get_sim_arm_joints(env_uw)
    perpendicular_reference_joints = last_arm_joints.copy()
    gello_anchor_joints: np.ndarray | None = None
    sim_anchor_joints: np.ndarray | None = None
    action_queue: list[np.ndarray] = []
    run_started_at = time.monotonic()

    def reset_episode(next_ep: int):
        nonlocal last_arm_joints, last_action, perpendicular_reference_joints, gello_anchor_joints, sim_anchor_joints
        env.reset()
        hold = make_hold_action(env_uw)
        for _ in range(args_cli.settle_steps):
            step_env(env, env_uw, hold)
        last_arm_joints = get_sim_arm_joints(env_uw)
        perpendicular_reference_joints = last_arm_joints.copy()
        last_action = None
        gello_anchor_joints = None
        sim_anchor_joints = None
        flags["reanchor_gello"] = True
        state_hist, image_hists = init_histories(env_uw, image_keys, n_obs_steps)
        if gello is not None:
            mode = "absolute" if args_cli.absolute_gello else "relative"
            print(
                f"[eval] Episode {next_ep} ready. GELLO staging is {mode}. "
                "Move with GELLO, then press C/Space. S stops policy, Q quits."
            )
        else:
            print(f"[eval] Episode {next_ep} ready. Press C/Space to start, S to stop, Q to quit.")
        return state_hist, image_hists

    state_hist, image_hists = reset_episode(ep)

    try:
        while simulation_app.is_running() and not flags["quit"] and ep < args_cli.num_episodes:
            frames = None
            if preview_allowed and loop_step % max(args_cli.preview_stride, 1) == 0:
                frames = get_camera_frames(env_uw, image_keys)
                if preview_window_size is None:
                    preview_window_size = create_resizable_preview_window(args_cli.preview_width, args_cli.preview_height)
                elapsed_s = time.monotonic() - run_started_at if flags["running"] else 0.0
                preview = draw_multi_camera_preview(frames, flags["running"], ep, step, elapsed_s)
                cv2.imshow(PREVIEW_WINDOW_NAME, preview)
                handle_preview_window_key(cv2.waitKey(1) & 0xFF, preview_window_size, flags)
            elif preview_allowed and preview_window_size is not None:
                handle_preview_window_key(cv2.waitKey(1) & 0xFF, preview_window_size, flags)

            if not flags["running"]:
                if gello is not None:
                    if flags["reanchor_gello"] or gello_anchor_joints is None or sim_anchor_joints is None:
                        gello_anchor_joints = snap_to_gello(gello)
                        sim_anchor_joints = get_sim_arm_joints(env_uw)
                        last_arm_joints = sim_anchor_joints.copy()
                        flags["reanchor_gello"] = False
                        if not args_cli.absolute_gello:
                            print("[GELLO] Relative staging anchored at current sim pose.")
                    hold_action, last_arm_joints, z_limited = make_gello_action(
                        env_uw,
                        gello,
                        last_arm_joints,
                        perpendicular_reference_joints,
                        None,
                        None if args_cli.absolute_gello else gello_anchor_joints,
                        None if args_cli.absolute_gello else sim_anchor_joints,
                    )
                else:
                    hold_action = make_hold_action(env_uw, last_action)
                last_action = hold_action.copy()
                step_env(env, env_uw, hold_action)
                state_hist.append(extract_state(env_uw))
                current_frames = frames if frames is not None else get_camera_frames(env_uw, image_keys)
                for image_key, frame in current_frames.items():
                    image_hists[image_key].append(image_to_policy_tensor(frame))
                action_queue.clear()
                loop_step += 1
                continue

            if flags["start_requested"]:
                action_queue.clear()
                state_hist, image_hists = init_histories(env_uw, image_keys, n_obs_steps)
                run_started_at = time.monotonic()
                flags["start_requested"] = False
                print(f"[eval] Episode {ep} start")

            if step == 0 and not action_queue:
                run_started_at = time.monotonic()

            if not action_queue:
                action_queue.extend(predict_action_queue(policy, state_hist, image_hists, image_keys, stride))

            raw_action = action_queue.pop(0)
            action = threshold_policy_gripper(raw_action)
            last_action = action.copy()
            _, _, terminated, truncated, _ = step_env(env, env_uw, action)
            state_hist.append(extract_state(env_uw))
            current_frames = frames if frames is not None else get_camera_frames(env_uw, image_keys)
            for image_key, frame in current_frames.items():
                image_hists[image_key].append(image_to_policy_tensor(frame))

            if step % 25 == 0:
                print(
                    f"[eval] ep={ep} step={step} "
                    f"raw_grip={np.round(raw_action[6:8], 3)} "
                    f"action={np.round(action, 3)}"
                )

            step += 1
            loop_step += 1
            terminated_bool = bool(torch.as_tensor(terminated).any().item())
            truncated_bool = bool(torch.as_tensor(truncated).any().item())
            if step >= args_cli.episode_steps or terminated_bool or truncated_bool:
                print(f"[eval] Episode {ep} finished at step {step}.")
                ep += 1
                step = 0
                last_action = None
                action_queue.clear()
                flags["running"] = False
                if ep < args_cli.num_episodes:
                    state_hist, image_hists = reset_episode(ep)
    finally:
        kb.close()
        env.close()
        if preview_allowed:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    simulation_app.close()
