# SPDX-License-Identifier: BSD-3-Clause
"""Interactive ClassHumanoid demo for a local RSL-RL checkpoint.

Usage:
    ./isaaclab.sh -p my_scripts/class_humanoid_policy_teleop.py \
      --task Isaac-Velocity-Rough-ClassHumanoid-Play-v0 \
      --checkpoint logs/rsl_rl/class_humanoid_rough/<run_dir>/model_14999.pt \
      --num_envs 25
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# Import RSL-RL CLI utilities from Isaac Lab scripts.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO_ROOT)
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Interactive ClassHumanoid locomotion with a local checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-ClassHumanoid-Play-v0",
    help="Task used to build the environment and policy config.",
)
parser.add_argument("--num_envs", type=int, default=25, help="Number of environments.")
parser.add_argument("--forward_speed", type=float, default=1.0, help="Forward command magnitude.")
parser.add_argument("--turn_speed", type=float, default=1.0, help="Turning command magnitude.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.checkpoint is None:
    parser.error("--checkpoint is required (path to model_XXXX.pt).")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

if not args_cli.headless:
    import carb
    import omni
    from omni.kit.viewport.utility import get_viewport_from_window_name
    from omni.kit.viewport.utility.camera_state import ViewportCameraState
    from pxr import Gf, Sdf

    from isaaclab.sim.utils.stage import get_current_stage


class ClassHumanoidInteractiveDemo:
    """H1 demo style controller for ClassHumanoid policies."""

    def __init__(self):
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)

        # Keep command source stable; keyboard override drives selected robot.
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.episode_length_s = 1.0e6

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device

        checkpoint_path = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint_path)
        self.policy = runner.get_inference_policy(device=self.device)
        self.policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic

        obs_mgr = self.env.unwrapped.observation_manager
        if not obs_mgr.group_obs_concatenate.get("policy", False):
            raise RuntimeError("Expected concatenated 'policy' observations.")
        names = obs_mgr.active_terms["policy"]
        dims = obs_mgr.group_obs_term_dim["policy"]
        self.cmd_slice = self._find_term_slice(names, dims, "velocity_commands")
        self.cmd_dim = self.cmd_slice[1] - self.cmd_slice[0]
        print(f"[INFO] velocity_commands obs slice: {self.cmd_slice}, dim={self.cmd_dim}")

        self.commands = torch.zeros(self.env.unwrapped.num_envs, self.cmd_dim, device=self.device)
        self._sync_commands_from_manager()
        self._requested_reset = False
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

        if not args_cli.headless:
            self._set_up_keyboard()
            self._prim_selection = omni.usd.get_context().get_selection()
            self._create_camera()
            self._set_default_overview_camera()
            print(
                "[INFO] Controls: click robot -> UP/LEFT/RIGHT/DOWN (or W/A/D/S), C view switch, ESC deselect, R reset."
            )
        else:
            print("[WARN] Headless mode: interactive keyboard/camera controls are disabled.")

    @staticmethod
    def _find_term_slice(term_names: list[str], term_dims: list[tuple[int, ...]], term_name: str) -> tuple[int, int]:
        offset = 0
        for name, dims in zip(term_names, term_dims):
            term_size = int(torch.tensor(dims, dtype=torch.int64).prod().item())
            if name == term_name:
                return offset, offset + term_size
            offset += term_size
        raise RuntimeError(f"Observation term '{term_name}' not found. Found terms: {term_names}")

    def _make_command_vector(self, forward: float, turn: float) -> torch.Tensor:
        cmd = torch.zeros(self.cmd_dim, device=self.device)
        if self.cmd_dim >= 1:
            cmd[0] = forward
        if self.cmd_dim >= 2:
            cmd[1] = 0.0
        # For 3D commands this is yaw-rate; for 4D (legacy) mimic h1_locomotion layout.
        if self.cmd_dim == 3:
            cmd[2] = turn
        elif self.cmd_dim >= 4:
            cmd[2] = 0.0
            cmd[3] = turn
        return cmd

    def _sync_commands_from_manager(self):
        manager_cmd = self.env.unwrapped.command_manager.get_command("base_velocity")
        self.commands.zero_()
        num_cols = min(self.cmd_dim, manager_cmd.shape[1])
        self.commands[:, :num_cols] = manager_cmd[:, :num_cols]

    def _create_camera(self):
        stage = get_current_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def _set_default_overview_camera(self):
        """Set a deterministic initial view so the scene is visible before robot selection."""
        self.env.unwrapped.sim.set_camera_view(eye=[3.4, -1.6, 2.0], target=[0.0, 0.0, 0.9])

    def _set_up_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        fwd = float(args_cli.forward_speed)
        turn = float(args_cli.turn_speed)
        self._key_to_command = {
            "UP": self._make_command_vector(fwd, 0.0),
            "DOWN": self._make_command_vector(0.0, 0.0),
            "LEFT": self._make_command_vector(fwd, turn),
            "RIGHT": self._make_command_vector(fwd, -turn),
            "W": self._make_command_vector(fwd, 0.0),
            "S": self._make_command_vector(0.0, 0.0),
            "A": self._make_command_vector(fwd, turn),
            "D": self._make_command_vector(fwd, -turn),
            "ZEROS": self._make_command_vector(0.0, 0.0),
        }

    def _on_keyboard_event(self, event):
        raw_input = getattr(event, "input", None)
        if raw_input is None:
            return True

        if isinstance(raw_input, str):
            key = raw_input.upper()
        else:
            key_name = getattr(raw_input, "name", None)
            key = (str(key_name) if key_name is not None else str(raw_input)).upper()
        # Normalize names like "KeyboardInput.UP" -> "UP"
        if "." in key:
            key = key.split(".")[-1]

        event_type = getattr(event, "type", None)
        is_press = event_type == carb.input.KeyboardEventType.KEY_PRESS
        is_release = event_type == carb.input.KeyboardEventType.KEY_RELEASE
        if not (is_press or is_release):
            # Fallback for backends exposing string event type.
            type_name = str(event_type).upper()
            is_press = "PRESS" in type_name
            is_release = "RELEASE" in type_name

        if is_press:
            if key in self._key_to_command:
                if self._selected_id is not None:
                    self.commands[self._selected_id] = self._key_to_command[key]
            elif key == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif key == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
            elif key == "R":
                self._requested_reset = True
        elif is_release:
            if key in self._key_to_command and self._selected_id is not None:
                self.commands[self._selected_id] = self._key_to_command["ZEROS"]
        return True

    def _update_camera(self):
        robot = self.env.unwrapped.scene["robot"]
        base_pos = robot.data.root_pos_w[self._selected_id, :]
        base_quat = robot.data.root_quat_w[self._selected_id, :]
        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
            self._set_default_overview_camera()
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please select only one robot.")
        else:
            prim_path_parts = selected_prim_paths[0].split("/")
            if len(prim_path_parts) >= 4 and prim_path_parts[3].startswith("env_"):
                env_id = int(prim_path_parts[3][4:])
                if 0 <= env_id < self.env.unwrapped.num_envs:
                    self._selected_id = env_id
                    if self._previous_selected_id != self._selected_id:
                        self.viewport.set_active_camera(self.camera_path)
                    self._update_camera()
                else:
                    print(f"Selected env id {env_id} is out of range.")
            else:
                print("The selected prim was not a ClassHumanoid robot.")

        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            self._sync_commands_from_manager()

    def _apply_command_override(self, obs):
        """Overwrite velocity command observation with keyboard command.

        `RslRlVecEnvWrapper` can return either a plain Tensor or a TensorDict.
        For inference tensors we clone first, then write.
        """
        start, end = self.cmd_slice

        if isinstance(obs, torch.Tensor):
            if torch.is_inference(obs):
                obs = obs.clone()
            obs[:, start:end] = self.commands
            return obs

        if hasattr(obs, "get") and hasattr(obs, "set"):
            policy_obs = obs.get("policy")
            if policy_obs is None:
                raise RuntimeError("Expected observation key 'policy' in TensorDict observation.")
            # Always clone to avoid in-place writes on inference tensors.
            policy_obs = policy_obs.clone()
            policy_obs[:, start:end] = self.commands
            obs = obs.set("policy", policy_obs)
            return obs

        raise TypeError(f"Unsupported observation type for command override: {type(obs)}")

    def run(self):
        obs, _ = self.env.reset()
        self._sync_commands_from_manager()
        obs = self._apply_command_override(obs)

        while simulation_app.is_running():
            if not args_cli.headless:
                self.update_selected_object()

            if self._requested_reset:
                obs, _ = self.env.reset()
                self._sync_commands_from_manager()
                obs = self._apply_command_override(obs)
                self._requested_reset = False
                continue

            with torch.inference_mode():
                actions = self.policy(obs)
                obs, _, dones, _ = self.env.step(actions)
                if hasattr(self.policy_nn, "reset"):
                    self.policy_nn.reset(dones)

            # Same idea as h1_locomotion: overwrite command observations for next action.
            obs = self._apply_command_override(obs)

        self.env.close()


def main():
    app = ClassHumanoidInteractiveDemo()
    app.run()


if __name__ == "__main__":
    main()
    simulation_app.close()
