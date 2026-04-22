# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom command/reward/observation terms for class humanoid primitive skills."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _estimate_target_pos_from_depth(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    data_type: str,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    """Estimate a coarse target position in base frame from the nearest depth pixel.

    The estimate uses depth and pixel coordinates only. It is intentionally simple and robust:
    x is forward range, y/z are lateral/vertical offsets inferred from image coordinates.
    """
    sensor = env.scene.sensors[sensor_name]
    depth = sensor.data.output[data_type].squeeze(-1)  # (N, H, W)
    num_envs, height, width = depth.shape
    device = depth.device

    valid = torch.isfinite(depth) & (depth > min_depth)
    large_val = torch.full_like(depth, max_depth + 10.0)
    masked_depth = torch.where(valid, depth, large_val)

    flat = masked_depth.reshape(num_envs, -1)
    flat_idx = torch.argmin(flat, dim=1)
    min_depth_vals = flat[torch.arange(num_envs, device=device), flat_idx]
    min_depth_vals = torch.clamp(min_depth_vals, min=min_depth, max=max_depth)

    u = (flat_idx % width).float()
    v = (flat_idx // width).float()
    u_norm = (u / max(float(width - 1), 1.0)) - 0.5
    v_norm = (v / max(float(height - 1), 1.0)) - 0.5

    # Approximate camera->base projection: x forward, y left, z up.
    x = min_depth_vals
    y = -u_norm * min_depth_vals
    z = -v_norm * min_depth_vals

    estimate = torch.stack((x, y, z), dim=1)
    estimate[~torch.isfinite(estimate)] = 0.0
    return estimate


@configclass
class UniformPelvisHeightCommandCfg(CommandTermCfg):
    class_type: type = None
    resampling_time_range: tuple[float, float] = (1.5, 3.0)
    min_height: float = 0.62
    max_height: float = 0.75
    asset_name: str = "robot"

    def __post_init__(self):
        self.class_type = UniformPelvisHeightCommand


class UniformPelvisHeightCommand(CommandTerm):
    cfg: UniformPelvisHeightCommandCfg

    def __init__(self, cfg: UniformPelvisHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.asset_name]
        self._command = torch.full((self.num_envs, 1), (cfg.min_height + cfg.max_height) * 0.5, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["height_error"] = torch.abs(self.asset.data.root_pos_w[:, 2] - self._command[:, 0])

    def _resample_command(self, env_ids: Sequence[int]):
        self._command[env_ids, 0] = self._command[env_ids, 0].uniform_(self.cfg.min_height, self.cfg.max_height)

    def _update_command(self):
        pass


@configclass
class AlternatingFootCommandCfg(CommandTermCfg):
    class_type: type = None
    resampling_time_range: tuple[float, float] = (0.7, 1.1)
    start_with_right: bool = True

    def __post_init__(self):
        self.class_type = AlternatingFootCommand


class AlternatingFootCommand(CommandTerm):
    cfg: AlternatingFootCommandCfg

    def __init__(self, cfg: AlternatingFootCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        init_val = 1.0 if cfg.start_with_right else -1.0
        self._command = torch.full((self.num_envs, 1), init_val, device=self.device)
        self.metrics["swing_ratio_right"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["swing_ratio_right"] = (self._command[:, 0] > 0.0).float()

    def _resample_command(self, env_ids: Sequence[int]):
        # `CommandTerm.reset()` zeros `command_counter` before the first resample of each episode.
        # Use that to restart the gait phase deterministically instead of toggling from stale episode state.
        default_sign = 1.0 if self.cfg.start_with_right else -1.0
        is_reset_resample = self.command_counter[env_ids] == 0
        toggled_sign = -self._command[env_ids, 0]
        self._command[env_ids, 0] = torch.where(
            is_reset_resample,
            torch.full_like(toggled_sign, default_sign),
            toggled_sign,
        )

    def _update_command(self):
        pass


@configclass
class StepTargetFootCommandCfg(CommandTermCfg):
    class_type: type = None
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    x_range: tuple[float, float] = (0.10, 0.22)
    y_abs_range: tuple[float, float] = (0.06, 0.14)
    swing_command_name: str = "swing_foot"

    def __post_init__(self):
        self.class_type = StepTargetFootCommand


class StepTargetFootCommand(CommandTerm):
    cfg: StepTargetFootCommandCfg

    def __init__(self, cfg: StepTargetFootCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_swing_sign = torch.zeros(self.num_envs, device=self.device)
        self.metrics["target_step_xy_norm"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["target_step_xy_norm"] = torch.norm(self._command, dim=1)

    def _current_swing_sign(self) -> torch.Tensor:
        swing = self._env.command_manager.get_command(self.cfg.swing_command_name)[:, 0]
        return torch.where(swing > 0.0, -1.0, 1.0)

    def _sample_target(self, env_ids: Sequence[int], swing_sign: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        if swing_sign is None:
            swing_sign = self._current_swing_sign()[env_ids]
        x_low, x_high = self.cfg.x_range
        if x_low == x_high:
            self._command[env_ids, 0] = x_low
        else:
            self._command[env_ids, 0] = self._command[env_ids, 0].uniform_(x_low, x_high)

        y_low, y_high = self.cfg.y_abs_range
        if y_low == y_high:
            self._command[env_ids, 1] = y_low * swing_sign
        else:
            self._command[env_ids, 1] = self._command[env_ids, 1].uniform_(y_low, y_high) * swing_sign
        self._last_swing_sign[env_ids] = swing_sign

    def _resample_command(self, env_ids: Sequence[int]):
        try:
            self._sample_target(env_ids, self._current_swing_sign()[env_ids])
        except Exception:
            rnd_sign = torch.where(torch.rand(len(env_ids), device=self.device) > 0.5, 1.0, -1.0)
            self._sample_target(env_ids, rnd_sign)

    def _update_command(self):
        swing_sign = self._current_swing_sign()
        changed_env_ids = (swing_sign != self._last_swing_sign).nonzero().flatten()
        if len(changed_env_ids) > 0:
            # Keep target placement locked to the swing-leg phase transition so the command semantics stay atomic.
            self._sample_target(changed_env_ids, swing_sign[changed_env_ids])


@configclass
class DepthTargetPosCommandCfg(CommandTermCfg):
    class_type: type = None
    resampling_time_range: tuple[float, float] = (0.5, 0.5)
    sensor_name: str = "depth_camera"
    data_type: str = "distance_to_image_plane"
    min_depth: float = 0.20
    max_depth: float = 2.00
    smooth_factor: float = 0.8

    def __post_init__(self):
        self.class_type = DepthTargetPosCommand


class DepthTargetPosCommand(CommandTerm):
    cfg: DepthTargetPosCommandCfg

    def __init__(self, cfg: DepthTargetPosCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 3), device=self.device)
        self.metrics["depth_target_norm"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["depth_target_norm"] = torch.norm(self._command, dim=1)

    def _resample_command(self, env_ids: Sequence[int]):
        # This command is updated from sensor data every step.
        self._update_command()

    def _update_command(self):
        est = _estimate_target_pos_from_depth(
            self._env, self.cfg.sensor_name, self.cfg.data_type, self.cfg.min_depth, self.cfg.max_depth
        )
        alpha = torch.clamp(torch.tensor(self.cfg.smooth_factor, device=self.device), 0.0, 1.0)
        self._command = alpha * self._command + (1.0 - alpha) * est


def base_rpy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Base orientation in roll/pitch/yaw."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack((roll, pitch, yaw), dim=1)


def binary_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Binary foot contacts for observation."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    contacts = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return contacts.float()


def illegal_contact_below_height(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    max_height: float,
) -> torch.Tensor:
    """Terminate only when a monitored body has significant contact while actually near the ground.

    For this humanoid asset, broad contact sensing can report large support-reaction forces on torso and hip bodies
    even when only the feet are touching the ground. Filtering by body-frame height removes these false positives
    while keeping contact-based fall detection for limbs or torso segments that truly reach the floor.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    contact_mask = torch.max(torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    near_ground_mask = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] < max_height
    return torch.any(contact_mask & near_ground_mask, dim=1)


def pelvis_height_error_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    target_h = env.command_manager.get_command(command_name)[:, 0]
    return torch.square(asset.data.root_pos_w[:, 2] - target_h)


def torso_tilt_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Equivalent to flat-orientation cost, kept explicit for readability."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def base_lin_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize horizontal base drift for in-place behaviors."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w[:, :2]), dim=1)


def base_ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize yaw spinning for in-place behaviors."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_w[:, 2])


def semantic_signed_joint_pos(
    env: ManagerBasedRLEnv,
    joint_signs: Sequence[float],
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return joint positions multiplied by task-level semantic signs.

    This is used when the robot asset uses different raw sign conventions for left/right joints.
    For example, if left-knee flexion is positive but right-knee flexion is negative in raw joint space,
    this helper converts both to a common "positive means flexion" convention.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_signs_tensor = torch.as_tensor(joint_signs, device=env.device, dtype=joint_pos.dtype)
    if joint_signs_tensor.numel() != len(asset_cfg.joint_ids):
        raise ValueError(
            f"Expected {len(asset_cfg.joint_ids)} semantic joint signs for {asset_cfg.joint_names}, "
            f"received {joint_signs_tensor.numel()}."
        )
    return joint_pos * joint_signs_tensor.unsqueeze(0)


def _resolve_articulation_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> list[int]:
    """Resolve joint ids from a scene entity config into a concrete Python list."""
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return list(range(asset.num_joints))
    if isinstance(joint_ids, slice):
        return list(range(asset.num_joints))[joint_ids]
    if isinstance(joint_ids, torch.Tensor):
        return joint_ids.tolist()
    return list(joint_ids)


def _resolve_env_ids(
    env_ids: Sequence[int] | slice | torch.Tensor | None,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Convert environment ids into a dense long tensor on the correct device."""
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long).flatten()
    return torch.as_tensor(list(env_ids), device=device, dtype=torch.long)


def class_humanoid_joint_semantic_signs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return semantic sign multipliers for class humanoid joints.

    Convention:
    - left-side joints keep their raw sign
    - right-side joints are multiplied by -1
    - center / unmatched joints keep +1

    This makes paired left/right joints use a shared semantic direction in observations/rewards.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_articulation_joint_ids(asset, asset_cfg)
    joint_names = [asset.data.joint_names[i] for i in joint_ids]
    joint_signs = [-1.0 if name.startswith("Right_") else 1.0 for name in joint_names]
    return torch.tensor(joint_signs, device=env.device, dtype=asset.data.joint_pos.dtype)


def class_humanoid_joint_pos_rel_semantic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions relative to default, converted into class-humanoid semantic signs."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_articulation_joint_ids(asset, asset_cfg)
    joint_signs = class_humanoid_joint_semantic_signs(env, asset_cfg)
    joint_pos_rel = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return joint_pos_rel * joint_signs.unsqueeze(0)


def class_humanoid_joint_vel_rel_semantic(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocities relative to default, converted into class-humanoid semantic signs."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = _resolve_articulation_joint_ids(asset, asset_cfg)
    joint_signs = class_humanoid_joint_semantic_signs(env, asset_cfg)
    joint_vel_rel = asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]
    return joint_vel_rel * joint_signs.unsqueeze(0)


def class_humanoid_knee_flexion(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
) -> torch.Tensor:
    """Return knee flexion where positive always means bending the knee."""
    return semantic_signed_joint_pos(env, joint_signs=(1.0, -1.0), asset_cfg=asset_cfg)


def class_humanoid_hip_pitch_flexion(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch_RS04", "Right_Hip_Pitch_RS04"]),
) -> torch.Tensor:
    """Return hip-pitch flexion where positive always means lifting the thigh forward."""
    return semantic_signed_joint_pos(env, joint_signs=(1.0, -1.0), asset_cfg=asset_cfg)


def _selected_and_support_leg_indices(env: ManagerBasedRLEnv, swing_command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve selected swing leg and support leg indices from the swing-foot command."""
    swing = env.command_manager.get_command(swing_command_name)[:, 0] > 0.0
    selected_idx = torch.where(
        swing,
        torch.ones(env.num_envs, dtype=torch.long, device=env.device),
        torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    )
    support_idx = 1 - selected_idx
    return selected_idx, support_idx


def _foot_positions_in_base_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    """Return left/right foot positions expressed in the robot base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1).reshape(-1, 3)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1).reshape(-1, 4)
    foot_pos_b, _ = math_utils.subtract_frame_transforms(
        root_pos_w,
        root_quat_w,
        foot_pos_w.reshape(-1, 3),
    )
    return foot_pos_b.reshape(env.num_envs, len(asset_cfg.body_ids), 3)


def _base_position_in_env_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return base position expressed in each environment frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def _feet_midpoint_in_env_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    """Return the midpoint between both feet expressed in each environment frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    return foot_pos_w.mean(dim=1) - env.scene.env_origins[:, :2]


class BaseXYFromResetObservation(ManagerTermBase):
    """Observe base XY displacement relative to the episode-reset pose."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.initial_base_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        env_ids = _resolve_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() == 0:
            return
        base_pos_env = _base_position_in_env_frame(self._env, self.asset_cfg)
        self.initial_base_xy[env_ids] = base_pos_env[env_ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        base_pos_env = _base_position_in_env_frame(env, asset_cfg)
        return (base_pos_env[:, :2] - self.initial_base_xy).clone()


class BaseResetPositionPenalty(ManagerTermBase):
    """Penalize horizontal base drift once it leaves a small in-place deadband."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.initial_base_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        env_ids = _resolve_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() == 0:
            return
        base_pos_env = _base_position_in_env_frame(self._env, self.asset_cfg)
        self.initial_base_xy[env_ids] = base_pos_env[env_ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        deadband: float = 0.05,
        std: float = 0.08,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        base_pos_env = _base_position_in_env_frame(env, asset_cfg)
        error = torch.norm(base_pos_env[:, :2] - self.initial_base_xy, dim=1)
        excess = torch.clamp(error - deadband, min=0.0)
        return torch.square(excess / max(std, 1.0e-6))


class BaseResetOutwardVelocityPenalty(ManagerTermBase):
    """Penalize only outward motion away from the reset anchor, not corrective return motion."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.initial_base_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        env_ids = _resolve_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() == 0:
            return
        base_pos_env = _base_position_in_env_frame(self._env, self.asset_cfg)
        self.initial_base_xy[env_ids] = base_pos_env[env_ids, :2]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        deadband: float = 0.03,
        distance_scale: float = 0.10,
        vel_scale: float = 0.20,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        base_pos_env = _base_position_in_env_frame(env, asset_cfg)
        offset = base_pos_env[:, :2] - self.initial_base_xy
        distance = torch.norm(offset, dim=1)
        radial_dir = offset / torch.clamp(distance.unsqueeze(1), min=1.0e-6)
        radial_speed = torch.sum(asset.data.root_lin_vel_w[:, :2] * radial_dir, dim=1)
        outward_speed = torch.clamp(radial_speed, min=0.0)
        active = (distance > deadband).float()
        distance_gate = torch.clamp((distance - deadband) / max(distance_scale, 1.0e-6), 0.0, 1.0)
        return active * distance_gate * torch.square(outward_speed / max(vel_scale, 1.0e-6))


class FeetMidpointResetPenalty(ManagerTermBase):
    """Penalize the walking pattern drifting away by anchoring the feet midpoint to reset."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get(
            "asset_cfg", SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"])
        )
        self.initial_midpoint_xy = torch.zeros((env.num_envs, 2), device=env.device)
        self.reset()

    def reset(self, env_ids: Sequence[int] | slice | torch.Tensor | None = None) -> None:
        env_ids = _resolve_env_ids(env_ids, self.num_envs, self.device)
        if env_ids.numel() == 0:
            return
        feet_midpoint_env = _feet_midpoint_in_env_frame(self._env, self.asset_cfg)
        self.initial_midpoint_xy[env_ids] = feet_midpoint_env[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        deadband: float = 0.04,
        std: float = 0.08,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
    ) -> torch.Tensor:
        feet_midpoint_env = _feet_midpoint_in_env_frame(env, asset_cfg)
        error = torch.norm(feet_midpoint_env - self.initial_midpoint_xy, dim=1)
        excess = torch.clamp(error - deadband, min=0.0)
        return torch.square(excess / max(std, 1.0e-6))


def _selected_and_support_contacts(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return swing/support contact indicators for the commanded step phase."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[
        0
    ] > threshold
    selected_idx, support_idx = _selected_and_support_leg_indices(env, swing_command_name)
    env_ids = torch.arange(env.num_envs, device=env.device)
    swing_contact = contacts[env_ids, selected_idx].float()
    support_contact = contacts[env_ids, support_idx].float()
    return swing_contact, support_contact


def selected_foot_step_error(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    target_command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    """Return XY step placement error for selected swing foot."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    selected_idx, _ = _selected_and_support_leg_indices(env, swing_command_name)
    selected_pos_w = foot_pos_w[torch.arange(env.num_envs, device=env.device), selected_idx]
    target_xy_b = env.command_manager.get_command(target_command_name)[:, :2]
    target_pos_b = torch.cat([target_xy_b, torch.zeros((env.num_envs, 1), device=env.device)], dim=1)
    target_pos_w, _ = math_utils.combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, target_pos_b)
    return torch.norm(selected_pos_w[:, :2] - target_pos_w[:, :2], dim=1)


def selected_foot_step_reward_tanh(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    target_command_name: str,
    std: float = 0.07,
    sensor_cfg: SceneEntityCfg | None = None,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    err = selected_foot_step_error(env, swing_command_name, target_command_name, asset_cfg)
    reward = 1.0 - torch.tanh(err / std)
    if sensor_cfg is not None:
        swing_contact, support_contact = _selected_and_support_contacts(env, swing_command_name, sensor_cfg, threshold)
        reward = reward * (1.0 - swing_contact) * support_contact
    return reward


def feet_lateral_order_penalty(
    env: ManagerBasedRLEnv,
    min_separation: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    """Penalize left/right foot crossing in the robot base frame.

    A healthy in-place step should keep the left foot on the robot's left side and the right foot on the
    robot's right side. If the lateral ordering collapses or swaps, this penalty increases.
    """
    foot_pos_b = _foot_positions_in_base_frame(env, asset_cfg)
    lateral_separation = foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1]
    return torch.clamp(min_separation - lateral_separation, min=0.0)


def swing_foot_base_clearance_reward(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    sensor_cfg: SceneEntityCfg,
    start_height: float = 0.03,
    target_height: float = 0.10,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    """Reward swing-foot lift in the robot base frame.

    This captures whether the leg is actually being lifted relative to the torso, not just whether the
    whole robot moved upward.
    """
    foot_pos_b = _foot_positions_in_base_frame(env, asset_cfg)
    selected_idx, _ = _selected_and_support_leg_indices(env, swing_command_name)
    env_ids = torch.arange(env.num_envs, device=env.device)
    swing_height = foot_pos_b[env_ids, selected_idx, 2]
    swing_contact, support_contact = _selected_and_support_contacts(env, swing_command_name, sensor_cfg, threshold)
    active_swing = (1.0 - swing_contact) * support_contact
    height_span = max(target_height - start_height, 1.0e-6)
    reward = torch.clamp((swing_height - start_height) / height_span, 0.0, 1.0)
    return reward * active_swing


def swing_support_height_difference_reward(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    sensor_cfg: SceneEntityCfg,
    start_height_diff: float = 0.02,
    target_height_diff: float = 0.08,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    """Reward the swing foot being higher than the support foot in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    selected_idx, support_idx = _selected_and_support_leg_indices(env, swing_command_name)
    env_ids = torch.arange(env.num_envs, device=env.device)
    swing_height = foot_pos_w[env_ids, selected_idx, 2]
    support_height = foot_pos_w[env_ids, support_idx, 2]
    swing_contact, support_contact = _selected_and_support_contacts(env, swing_command_name, sensor_cfg, threshold)
    active_swing = (1.0 - swing_contact) * support_contact
    height_diff = swing_height - support_height
    diff_span = max(target_height_diff - start_height_diff, 1.0e-6)
    reward = torch.clamp((height_diff - start_height_diff) / diff_span, 0.0, 1.0)
    return reward * active_swing


def selected_swing_knee_angle_range_reward(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    min_angle: float = 0.60,
    max_angle: float = 1.30,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
) -> torch.Tensor:
    """Reward the commanded swing knee for flexing into a target range.

    The reward uses semantic knee flexion, not raw joint sign, so left/right knees can share one target range.
    A value of 1.0 is returned inside the range and decays smoothly outside it.
    """
    knee_flexion = class_humanoid_knee_flexion(env, asset_cfg)
    selected_idx, _ = _selected_and_support_leg_indices(env, swing_command_name)
    selected_knee = knee_flexion[torch.arange(env.num_envs, device=env.device), selected_idx]
    below = torch.clamp(min_angle - selected_knee, min=0.0)
    above = torch.clamp(selected_knee - max_angle, min=0.0)
    range_error = below + above
    return 1.0 - torch.tanh(range_error / std)


def selected_swing_knee_min_flex_reward(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    target_angle: float = 0.60,
    start_angle: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
) -> torch.Tensor:
    """Reward the commanded swing knee for reaching a minimum useful flexion amount.

    Unlike the range reward, this ramps up from a nearly straight leg so the policy sees
    learning signal before it discovers the final preferred knee-bend range.
    """
    knee_flexion = class_humanoid_knee_flexion(env, asset_cfg)
    selected_idx, _ = _selected_and_support_leg_indices(env, swing_command_name)
    selected_knee = knee_flexion[torch.arange(env.num_envs, device=env.device), selected_idx]
    flex_span = max(target_angle - start_angle, 1.0e-6)
    return torch.clamp((selected_knee - start_angle) / flex_span, 0.0, 1.0)


def support_knee_straight_reward(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    max_angle: float = 0.35,
    std: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
) -> torch.Tensor:
    """Reward keeping the support knee relatively straight.

    This counters the bilateral crouch solution where both knees bend together and the robot
    shuffles in place without a clear stance leg.
    """
    knee_flexion = class_humanoid_knee_flexion(env, asset_cfg)
    _, support_idx = _selected_and_support_leg_indices(env, swing_command_name)
    support_knee = knee_flexion[torch.arange(env.num_envs, device=env.device), support_idx]
    flex_excess = torch.clamp(support_knee - max_angle, min=0.0)
    return 1.0 - torch.tanh(flex_excess / std)


def hip_only_swing_penalty(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    hip_threshold: float = 0.35,
    hip_saturation: float = 0.80,
    knee_threshold: float = 0.30,
    knee_floor: float = 0.0,
    hip_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Hip_Pitch_RS04", "Right_Hip_Pitch_RS04"]),
    knee_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
) -> torch.Tensor:
    """Penalize using large swing-hip flexion without enough swing-knee bend."""
    hip_flexion = class_humanoid_hip_pitch_flexion(env, hip_cfg)
    knee_flexion = class_humanoid_knee_flexion(env, knee_cfg)
    selected_idx, _ = _selected_and_support_leg_indices(env, swing_command_name)
    env_ids = torch.arange(env.num_envs, device=env.device)
    swing_hip = hip_flexion[env_ids, selected_idx]
    swing_knee = knee_flexion[env_ids, selected_idx]
    hip_span = max(hip_saturation - hip_threshold, 1.0e-6)
    knee_span = max(knee_threshold - knee_floor, 1.0e-6)
    hip_excess = torch.clamp((swing_hip - hip_threshold) / hip_span, 0.0, 1.0)
    knee_deficit = torch.clamp((knee_threshold - swing_knee) / knee_span, 0.0, 1.0)
    return hip_excess * knee_deficit


def swing_leg_shortening_reward(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    target_margin: float = 0.08,
    std: float = 0.04,
    hip_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["HipYoke_Left_1", "HipYoke_Right_1"]),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    """Reward the swing leg for being shorter than the support leg.

    The shortening is measured geometrically as the hip-to-foot distance on each side. This avoids
    requiring a fragile absolute target length and instead encourages the swing leg to tuck relative
    to the planted leg.
    """
    asset: Articulation = env.scene[hip_cfg.name]
    hip_pos_w = asset.data.body_pos_w[:, hip_cfg.body_ids, :3]
    foot_pos_w = asset.data.body_pos_w[:, foot_cfg.body_ids, :3]

    leg_lengths = torch.norm(foot_pos_w - hip_pos_w, dim=2)
    selected_idx, support_idx = _selected_and_support_leg_indices(env, swing_command_name)
    swing_length = leg_lengths[torch.arange(env.num_envs, device=env.device), selected_idx]
    support_length = leg_lengths[torch.arange(env.num_envs, device=env.device), support_idx]
    shortening_margin = support_length - swing_length
    margin_error = torch.clamp(target_margin - shortening_margin, min=0.0)
    return 1.0 - torch.tanh(margin_error / std)


def alternating_contact_pattern_reward(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward matching the commanded alternating support pattern.

    A reward of 1 is given when the commanded swing foot is off the ground and the support foot remains planted.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[
        0
    ] > threshold
    selected_idx, support_idx = _selected_and_support_leg_indices(env, swing_command_name)
    swing_contact = contacts[torch.arange(env.num_envs, device=env.device), selected_idx].float()
    support_contact = contacts[torch.arange(env.num_envs, device=env.device), support_idx].float()
    return (1.0 - swing_contact) * support_contact


def swing_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize the commanded swing foot remaining in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[
        0
    ] > threshold
    selected_idx, _ = _selected_and_support_leg_indices(env, swing_command_name)
    return contacts[torch.arange(env.num_envs, device=env.device), selected_idx].float()


def support_foot_slip_penalty(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize support-foot slip while in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contacts = torch.max(torch.norm(contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[
        0
    ] > threshold

    _, support_idx = _selected_and_support_leg_indices(env, swing_command_name)
    support_speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2][
        torch.arange(env.num_envs, device=env.device), support_idx
    ].norm(dim=1)
    support_contact = contacts[torch.arange(env.num_envs, device=env.device), support_idx].float()
    return support_speed * support_contact


def end_effector_object_distance(
    env: ManagerBasedRLEnv,
    target_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Wrist_Right_1"]),
) -> torch.Tensor:
    """Distance between right wrist and target object center."""
    robot: Articulation = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_name]
    ee_pos = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    target_pos = target.data.root_pos_w[:, :3]
    return torch.norm(ee_pos - target_pos, dim=1)


def end_effector_object_reward_tanh(
    env: ManagerBasedRLEnv,
    target_name: str,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Wrist_Right_1"]),
) -> torch.Tensor:
    dist = end_effector_object_distance(env, target_name, asset_cfg)
    return 1.0 - torch.tanh(dist / std)


def target_pos_base(
    env: ManagerBasedRLEnv,
    target_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ground-truth target position in robot base frame (for critic/debug)."""
    robot: RigidObject = env.scene[asset_cfg.name]
    target: RigidObject = env.scene[target_name]
    rel_pos_w = target.data.root_pos_w[:, :3] - robot.data.root_pos_w
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)


def target_pos_base_from_depth(
    env: ManagerBasedRLEnv,
    sensor_name: str = "depth_camera",
    data_type: str = "distance_to_image_plane",
    min_depth: float = 0.20,
    max_depth: float = 2.00,
) -> torch.Tensor:
    """Depth-only target estimate in base frame."""
    return _estimate_target_pos_from_depth(env, sensor_name, data_type, min_depth, max_depth)
