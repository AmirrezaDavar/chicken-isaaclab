# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command, observation, and reward helpers for Class Humanoid stepping."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from . import common_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
            self._sample_target(changed_env_ids, swing_sign[changed_env_ids])


def _resolve_env_ids(
    env_ids: Sequence[int] | slice | torch.Tensor | None,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)[env_ids]
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long).flatten()
    return torch.as_tensor(list(env_ids), device=device, dtype=torch.long)


def _selected_and_support_leg_indices(env: ManagerBasedRLEnv, swing_command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
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
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1).reshape(-1, 3)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1).reshape(-1, 4)
    foot_pos_b, _ = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, foot_pos_w.reshape(-1, 3))
    return foot_pos_b.reshape(env.num_envs, len(asset_cfg.body_ids), 3)


def _base_position_in_env_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def _feet_midpoint_in_env_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
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
    """Penalize the stepping pattern drifting away by anchoring the feet midpoint to reset."""

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


def selected_foot_step_error_penalty(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    target_command_name: str,
    std: float = 0.10,
    max_error: float = 0.45,
    sensor_cfg: SceneEntityCfg | None = None,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    err = torch.clamp(selected_foot_step_error(env, swing_command_name, target_command_name, asset_cfg), max=max_error)
    penalty = torch.square(err / max(std, 1.0e-6))
    if sensor_cfg is not None:
        swing_contact, support_contact = _selected_and_support_contacts(env, swing_command_name, sensor_cfg, threshold)
        penalty = penalty * (1.0 - swing_contact) * support_contact
    return penalty


def feet_lateral_order_penalty(
    env: ManagerBasedRLEnv,
    min_separation: float = 0.04,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
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


def swing_support_height_excess_penalty(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    sensor_cfg: SceneEntityCfg,
    max_height_diff: float = 0.16,
    std: float = 0.08,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Foot_Left_1", "Foot_Right_1"]),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :3]
    selected_idx, support_idx = _selected_and_support_leg_indices(env, swing_command_name)
    env_ids = torch.arange(env.num_envs, device=env.device)
    swing_height = foot_pos_w[env_ids, selected_idx, 2]
    support_height = foot_pos_w[env_ids, support_idx, 2]
    swing_contact, support_contact = _selected_and_support_contacts(env, swing_command_name, sensor_cfg, threshold)
    active_swing = (1.0 - swing_contact) * support_contact
    excess = torch.clamp((swing_height - support_height) - max_height_diff, min=0.0)
    return torch.square(excess / max(std, 1.0e-6)) * active_swing


def root_height_below_penalty(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.68,
    std: float = 0.08,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    deficit = torch.clamp(minimum_height - asset.data.root_pos_w[:, 2], min=0.0)
    return torch.square(deficit / max(std, 1.0e-6))


def selected_swing_knee_angle_range_reward(
    env: ManagerBasedRLEnv,
    swing_command_name: str,
    min_angle: float = 0.60,
    max_angle: float = 1.30,
    std: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Left_Knee_RS04", "Right_Knee_RS04"]),
) -> torch.Tensor:
    knee_flexion = common_mdp.class_humanoid_knee_flexion(env, asset_cfg)
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
    knee_flexion = common_mdp.class_humanoid_knee_flexion(env, asset_cfg)
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
    knee_flexion = common_mdp.class_humanoid_knee_flexion(env, asset_cfg)
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
    hip_flexion = common_mdp.class_humanoid_hip_pitch_flexion(env, hip_cfg)
    knee_flexion = common_mdp.class_humanoid_knee_flexion(env, knee_cfg)
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
