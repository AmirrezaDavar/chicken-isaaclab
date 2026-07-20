#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Train ChicGrasp Diffusion Policy using Isaac chicken multi-camera RGB + low-dimensional state.

This script installs the tiny Isaac-specific zarr dataset/config bridge into
the separate ChicGrasp-IsaacChicken checkout, then launches its official
train.py. The collected zarr layout remains:

  replay_buffer.zarr/
    data/action
    data/camera_rgb
    data/camera_left_rgb
    data/camera_right_rgb
    data/state
    meta/episode_ends

Example smoke test:
  conda activate robodiff
  python scripts/imitation_learning/02_train_chicken_rgb_state_policy.py \
      --zarr_path /home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
      --num_epochs 5 --max_train_steps 20 --max_val_steps 5 --batch_size 1 \
      --logging_mode offline

Full run:
  conda activate robodiff
  python scripts/imitation_learning/02_train_chicken_rgb_state_policy.py \
      --zarr_path /home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr \
      --num_epochs 450 --batch_size 2 --logging_mode offline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import zarr


DEFAULT_CHICGRASP_ROOT = Path("/home/wanglab22/ChicGrasp")
DEFAULT_ZARR = Path("/home/wanglab22/3_chicken-isaaclab/data/chicken_rgb_state/replay_buffer.zarr")
DEFAULT_IMAGE_KEYS = ["camera_rgb", "camera_left_rgb", "camera_right_rgb"]


DATASET_CODE = r'''
"""Isaac chicken multi-camera RGB+state zarr dataset for Diffusion Policy."""

from __future__ import annotations

from typing import Dict, Optional
import copy

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class IsaacChickenImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        zarr_path: str,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        n_obs_steps: Optional[int] = 2,
        seed: int = 42,
        val_ratio: float = 0.1,
        max_train_episodes: Optional[int] = None,
        image_keys: Optional[list[str]] = None,
        state_key: str = "state",
        action_key: str = "action",
    ):
        super().__init__()

        if image_keys is None:
            image_keys = ["camera_rgb", "camera_left_rgb", "camera_right_rgb"]
        if isinstance(image_keys, str):
            image_keys = [image_keys]
        self.image_keys = list(image_keys)
        self.state_key = state_key
        self.action_key = action_key
        self.shape_meta = shape_meta
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Keep the zarr arrays disk-backed. Loading three RGB streams into RAM
        # can exceed 30+ GB before the first optimizer step.
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(~val_mask, max_n=max_train_episodes, seed=seed)

        key_first_k = {}
        if n_obs_steps is not None:
            for image_key in self.image_keys:
                key_first_k[image_key] = n_obs_steps
            key_first_k[state_key] = n_obs_steps

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        self.train_mask = train_mask

    def get_validation_dataset(self) -> "IsaacChickenImageDataset":
        val = copy.copy(self)
        val.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k={
                **{image_key: self.n_obs_steps for image_key in self.image_keys},
                self.state_key: self.n_obs_steps,
            },
        )
        val.train_mask = ~self.train_mask
        return val

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer[self.action_key]
        )
        normalizer[self.state_key] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer[self.state_key]
        )
        for image_key in self.image_keys:
            normalizer[image_key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(idx)
        obs_slice = slice(self.n_obs_steps)

        obs = {
            self.state_key: sample[self.state_key][obs_slice].astype(np.float32),
        }
        for image_key in self.image_keys:
            image = sample[image_key][obs_slice]
            if image.dtype != np.uint8:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            obs[image_key] = np.moveaxis(image, -1, 1).astype(np.float32) / 255.0

        data = {
            "obs": obs,
            "action": sample[self.action_key].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)
'''


NULL_IMAGE_RUNNER_CODE = r'''
"""No-op image runner for offline-only Isaac chicken training."""

from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NullImageRunner(BaseImageRunner):
    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
'''


def _inspect_zarr(
    zarr_path: Path,
    image_keys: list[str],
    state_key: str,
    action_key: str,
) -> tuple[dict[str, tuple[int, int]], int, int]:
    root = zarr.open_group(str(zarr_path), mode="r")
    data = root["data"]
    for key in (*image_keys, state_key, action_key):
        if key not in data:
            raise SystemExit(f"Missing data/{key} in {zarr_path}")
    state_shape = data[state_key].shape
    action_shape = data[action_key].shape
    image_shapes = {}
    for image_key in image_keys:
        image_shape = data[image_key].shape
        if len(image_shape) != 4 or image_shape[-1] != 3:
            raise SystemExit(f"data/{image_key} must be (T,H,W,3), got {image_shape}")
        if image_shape[0] != state_shape[0]:
            raise SystemExit(f"data/{image_key} has {image_shape[0]} steps but data/{state_key} has {state_shape[0]}")
        image_shapes[image_key] = (int(image_shape[1]), int(image_shape[2]))
    if state_shape[1] != 20:
        raise SystemExit(f"data/{state_key} must be (T,20), got {state_shape}")
    if action_shape[1] != 8:
        raise SystemExit(f"data/{action_key} must be (T,8), got {action_shape}")
    if action_shape[0] != state_shape[0]:
        raise SystemExit(f"data/{action_key} has {action_shape[0]} steps but data/{state_key} has {state_shape[0]}")
    return image_shapes, int(state_shape[1]), int(action_shape[1])


def _write_if_changed(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = textwrap.dedent(text).lstrip()
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)
    print(f"[install] wrote {path}")


def install_chicgrasp_bridge(
    chicgrasp_root: Path,
    zarr_path: Path,
    image_keys: list[str],
    state_key: str,
    action_key: str,
    image_shapes: dict[str, tuple[int, int]],
) -> None:
    _write_if_changed(
        chicgrasp_root / "diffusion_policy/dataset/isaac_chicken_image_dataset.py",
        DATASET_CODE,
    )
    _write_if_changed(
        chicgrasp_root / "diffusion_policy/env_runner/null_image_runner.py",
        NULL_IMAGE_RUNNER_CODE,
    )

    image_shape_meta = "\n".join(
        f"""        {image_key}:
          shape: [3, {image_shapes[image_key][0]}, {image_shapes[image_key][1]}]
          type: rgb"""
        for image_key in image_keys
    )
    image_keys_yaml = "\n".join(f"        - {image_key}" for image_key in image_keys)

    task_yaml = f"""
    name: isaac_chicken_image

    shape_meta: &shape_meta
      obs:
{image_shape_meta}
        {state_key}:
          shape: [20]
          type: low_dim
      action:
        shape: [8]

    env_runner:
      _target_: diffusion_policy.env_runner.null_image_runner.NullImageRunner

    dataset:
      _target_: diffusion_policy.dataset.isaac_chicken_image_dataset.IsaacChickenImageDataset
      shape_meta: *shape_meta
      zarr_path: {zarr_path}
      image_keys:
{image_keys_yaml}
      state_key: {state_key}
      action_key: {action_key}
      horizon: ${{horizon}}
      pad_before: ${{eval:'${{n_obs_steps}}-1+${{n_latency_steps}}'}}
      pad_after: ${{eval:'${{n_action_steps}}-1'}}
      n_obs_steps: ${{dataset_obs_steps}}
      seed: ${{training.seed}}
      val_ratio: 0.1
      max_train_episodes: null
    """
    _write_if_changed(
        chicgrasp_root / "diffusion_policy/config/task/isaac_chicken_image.yaml",
        task_yaml,
    )

    train_yaml = f"""
    defaults:
      - _self_
      - task: isaac_chicken_image

    name: train_diffusion_unet_image_isaac_chicken
    _target_: diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace

    task_name: ${{task.name}}
    shape_meta: ${{task.shape_meta}}
    exp_name: "rgb_state"

    horizon: 16
    n_obs_steps: 2
    n_action_steps: 8
    n_latency_steps: 0
    dataset_obs_steps: ${{n_obs_steps}}
    past_action_visible: False
    keypoint_visible_rate: 1.0
    obs_as_global_cond: True

    policy:
      _target_: diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy
      shape_meta: ${{shape_meta}}
      noise_scheduler:
        _target_: diffusers.schedulers.scheduling_ddpm.DDPMScheduler
        num_train_timesteps: 100
        beta_start: 0.0001
        beta_end: 0.02
        beta_schedule: squaredcos_cap_v2
        variance_type: fixed_small
        clip_sample: True
        prediction_type: epsilon
      obs_encoder:
        _target_: diffusion_policy.model.vision.multi_image_obs_encoder.MultiImageObsEncoder
        shape_meta: ${{shape_meta}}
        rgb_model:
          _target_: diffusion_policy.model.vision.model_getter.get_resnet
          name: resnet18
          weights: null
        resize_shape: [120, 160]
        crop_shape: [108, 144]
        random_crop: True
        use_group_norm: True
        share_rgb_model: True
        imagenet_norm: True
      horizon: ${{horizon}}
      n_action_steps: ${{eval:'${{n_action_steps}}+${{n_latency_steps}}'}}
      n_obs_steps: ${{n_obs_steps}}
      num_inference_steps: 100
      obs_as_global_cond: ${{obs_as_global_cond}}
      diffusion_step_embed_dim: 128
      down_dims: [128, 256, 512]
      kernel_size: 5
      n_groups: 8
      cond_predict_scale: True

    ema:
      _target_: diffusion_policy.model.diffusion.ema_model.EMAModel
      update_after_step: 0
      inv_gamma: 1.0
      power: 0.75
      min_value: 0.0
      max_value: 0.9999

    dataloader:
      batch_size: 1
      num_workers: 4
      shuffle: True
      pin_memory: True
      persistent_workers: False

    val_dataloader:
      batch_size: 1
      num_workers: 4
      shuffle: False
      pin_memory: True
      persistent_workers: False

    optimizer:
      _target_: torch.optim.AdamW
      lr: 1.0e-4
      betas: [0.95, 0.999]
      eps: 1.0e-8
      weight_decay: 1.0e-6

    training:
      device: "cuda:0"
      seed: 42
      debug: False
      resume: True
      lr_scheduler: cosine
      lr_warmup_steps: 500
      num_epochs: 450
      gradient_accumulate_every: 1
      use_ema: True
      freeze_encoder: False
      rollout_every: 9999
      checkpoint_every: 50
      val_every: 1
      sample_every: 5
      max_train_steps: null
      max_val_steps: null
      tqdm_interval_sec: 1.0

    logging:
      project: isaac_chicken_diffusion
      resume: True
      mode: offline
      name: ${{now:%Y.%m.%d-%H.%M.%S}}_${{name}}_${{task_name}}
      tags: ["${{name}}", "${{task_name}}", "${{exp_name}}"]
      id: null
      group: null

    checkpoint:
      topk:
        monitor_key: train_loss
        mode: min
        k: 5
        format_str: 'epoch={{epoch:04d}}-train_loss={{train_loss:.3f}}.ckpt'
      save_last_ckpt: True
      save_last_snapshot: False

    multi_run:
      run_dir: data/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}
      wandb_name_base: ${{now:%Y.%m.%d-%H.%M.%S}}_${{name}}_${{task_name}}

    hydra:
      job:
        override_dirname: ${{name}}
      run:
        dir: data/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}
      sweep:
        dir: data/outputs/${{now:%Y.%m.%d}}/${{now:%H.%M.%S}}_${{name}}_${{task_name}}
        subdir: ${{hydra.job.num}}
    """
    _write_if_changed(
        chicgrasp_root / "diffusion_policy/config/train_diffusion_unet_image_isaac_chicken_workspace.yaml",
        train_yaml,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RGB+state Diffusion Policy for Isaac chicken grasping.")
    parser.add_argument("--chicgrasp_root", type=Path, default=DEFAULT_CHICGRASP_ROOT)
    parser.add_argument("--zarr_path", type=Path, default=DEFAULT_ZARR)
    parser.add_argument("--image_keys", type=str, nargs="+", default=DEFAULT_IMAGE_KEYS,
                        help="RGB zarr keys to train with. Defaults to all three Isaac cameras.")
    parser.add_argument("--image_key", type=str, default=None,
                        help="Backward-compatible single RGB key override. Prefer --image_keys.")
    parser.add_argument("--state_key", type=str, default="state")
    parser.add_argument("--action_key", type=str, default="action")
    parser.add_argument("--num_epochs", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logging_mode", type=str, default="offline", choices=["offline", "online", "disabled"])
    parser.add_argument("--dry_run", action="store_true", help="Install configs and print the train command only.")
    args = parser.parse_args()

    chicgrasp_root = args.chicgrasp_root.expanduser().resolve()
    zarr_path = args.zarr_path.expanduser().resolve()
    if not (chicgrasp_root / "train.py").exists():
        raise SystemExit(f"Missing ChicGrasp train.py under {chicgrasp_root}")
    if not zarr_path.exists():
        raise SystemExit(f"Missing zarr dataset: {zarr_path}")

    image_keys = [args.image_key] if args.image_key is not None else list(args.image_keys)
    image_shapes, _, _ = _inspect_zarr(zarr_path, image_keys, args.state_key, args.action_key)
    install_chicgrasp_bridge(
        chicgrasp_root=chicgrasp_root,
        zarr_path=zarr_path,
        image_keys=image_keys,
        state_key=args.state_key,
        action_key=args.action_key,
        image_shapes=image_shapes,
    )

    cmd = [
        sys.executable,
        "train.py",
        "--config-name=train_diffusion_unet_image_isaac_chicken_workspace",
        f"task.dataset.zarr_path={zarr_path}",
        f"task.dataset.val_ratio={args.val_ratio}",
        f"dataloader.batch_size={args.batch_size}",
        f"val_dataloader.batch_size={args.batch_size}",
        f"dataloader.num_workers={args.num_workers}",
        f"val_dataloader.num_workers={args.num_workers}",
        f"training.device={args.device}",
        f"training.num_epochs={args.num_epochs}",
        f"logging.mode={args.logging_mode}",
    ]
    if args.max_train_steps is not None:
        cmd.append(f"training.max_train_steps={args.max_train_steps}")
    if args.max_val_steps is not None:
        cmd.append(f"training.max_val_steps={args.max_val_steps}")
    if args.num_workers == 0:
        cmd.extend(["dataloader.persistent_workers=False", "val_dataloader.persistent_workers=False"])

    print("[train] " + " ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=str(chicgrasp_root), check=True)


if __name__ == "__main__":
    main()
