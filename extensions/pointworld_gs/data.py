# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data loading for one-scene PointWorld-GS overfit experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import torch
import webdataset as wds

from dataset_components.decoders import build_flow_sample, decode_data
from dataset_components.transforms import (
    assert_camera_payload_resolution,
    center_shift,
    compute_helper_variables,
    enforce_max_num_points,
    filter_within_bounds,
    grid_sample_transform,
    make_gt_copy,
    normalize_colors,
)
from robot_sampler import RobotSampler as TorchRobotSampler
from utils import resolve_robot_urdf


@dataclass
class SceneBundle:
    key: str
    positions: torch.Tensor
    colors: torch.Tensor
    initial_rgb: torch.Tensor
    intrinsic: torch.Tensor
    extrinsic: torch.Tensor
    dynamic_mask: torch.Tensor
    target_rgb: torch.Tensor | None
    selected_camera: str


def list_shards(data_dir: str, split: str) -> list[str]:
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    shards = sorted(str(path) for path in split_dir.glob("*.tar") if path.stat().st_size >= 1024 * 1024)
    if not shards:
        raise FileNotFoundError(f"No non-empty WebDataset shards found under {split_dir}")
    return shards


def load_raw_sample(data_dir: str, split: str, sample_index: int, domain: str) -> dict:
    shards = list_shards(data_dir, split)
    dataset = wds.WebDataset(shards, shardshuffle=False, handler=wds.warn_and_continue).map(
        partial(decode_data, domain=domain),
        handler=wds.warn_and_continue,
    )
    for idx, sample in enumerate(dataset):
        if idx == sample_index:
            return sample
    raise IndexError(f"Sample index {sample_index} was not found in {len(shards)} shard(s)")


def _camera_prefixes_from_sample(sample: dict) -> list[str]:
    prefixes = set()
    for key in sample.keys():
        if key.endswith("_initial_rgb"):
            prefixes.add(key[: -len("_initial_rgb")])
        elif "_scene_flows" in key:
            prefixes.add(key.split("_scene_flows")[0])
    return sorted(prefixes)


def choose_camera_prefix(sample: dict, seed: int) -> str:
    prefixes = _camera_prefixes_from_sample(sample)
    if not prefixes:
        raise RuntimeError("No camera prefixes found in decoded sample")
    rng = np.random.RandomState(seed)
    return str(rng.choice(prefixes, size=1)[0])


def _format_key_template(template: str, camera_prefix: str, frame_idx: int) -> str:
    return template.format(camera=camera_prefix, cam=camera_prefix, t=frame_idx, frame=frame_idx)


def _decode_rgb_value(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"Expected RGB array with shape HxWx3, got {value.shape}")
        if value.dtype != np.uint8:
            value = np.clip(value, 0, 255).astype(np.uint8)
        return value
    if isinstance(value, (bytes, bytearray)):
        decoded = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to decode RGB image bytes")
        return decoded[..., ::-1]
    raise TypeError(f"Unsupported RGB value type: {type(value)}")


def load_future_rgb_from_wds(
    raw_sample: dict,
    camera_prefix: str,
    template: str | None,
    max_frames: int | None,
) -> np.ndarray | None:
    """Load future RGB frames from an extended WDS sample if present.

    If `template` is set, keys are read as
    `template.format(camera=<prefix>, t=<frame>)`. If unset, this tries a small
    set of common conventions. The current public PointWorld release generally
    does not include these keys; this is a hook for regenerated datasets.
    """
    if template:
        frames = []
        limit = max_frames if max_frames is not None else 10_000
        for t in range(limit):
            key = _format_key_template(template, camera_prefix, t)
            if key not in raw_sample:
                break
            frames.append(_decode_rgb_value(raw_sample[key]))
        return np.stack(frames, axis=0) if frames else None

    patterns = [
        re.compile(rf"^{re.escape(camera_prefix)}_(?:rgb|image|frame)_(\d+)$"),
        re.compile(rf"^{re.escape(camera_prefix)}_(?:rgb|image|frame)_(\d+)\.jpg$"),
        re.compile(rf"^{re.escape(camera_prefix)}_(?:rgb|image|frame)_(\d+)\.png$"),
    ]
    found: list[tuple[int, str]] = []
    for key in raw_sample.keys():
        for pattern in patterns:
            match = pattern.match(key)
            if match:
                found.append((int(match.group(1)), key))
                break
    if not found:
        return None
    found.sort(key=lambda item: item[0])
    if max_frames is not None:
        found = found[:max_frames]
    return np.stack([_decode_rgb_value(raw_sample[key]) for _, key in found], axis=0)


def load_future_rgb_from_dir(path: str | None, max_frames: int | None) -> np.ndarray | None:
    if not path:
        return None
    frame_dir = Path(path)
    if not frame_dir.exists():
        raise FileNotFoundError(f"Future RGB directory does not exist: {frame_dir}")
    image_paths = sorted(
        p
        for p in frame_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if max_frames is not None:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise FileNotFoundError(f"No RGB images found in {frame_dir}")
    frames = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {image_path}")
        frames.append(image[..., ::-1])
    return np.stack(frames, axis=0)


def resize_rgb_tensor(rgb: torch.Tensor, intrinsic: torch.Tensor | None, render_scale: float) -> tuple[torch.Tensor, torch.Tensor | None]:
    if render_scale == 1.0:
        return rgb, intrinsic
    if render_scale <= 0.0:
        raise ValueError("--render_scale must be positive")

    if rgb.ndim == 3:
        rgb_batched = rgb[None]
    elif rgb.ndim == 4:
        rgb_batched = rgb
    else:
        raise ValueError(f"Expected RGB tensor HxWx3 or TxHxWx3, got {tuple(rgb.shape)}")

    resized = []
    for frame in rgb_batched:
        rgb_np = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        height, width = rgb_np.shape[:2]
        new_size = (max(1, int(round(width * render_scale))), max(1, int(round(height * render_scale))))
        resized_np = cv2.resize(rgb_np, new_size, interpolation=cv2.INTER_AREA)
        resized.append(torch.as_tensor(resized_np, device=rgb.device, dtype=torch.float32) / 255.0)
    resized_t = torch.stack(resized, dim=0)
    if rgb.ndim == 3:
        resized_t = resized_t[0]

    if intrinsic is not None:
        intr = intrinsic.clone()
        intr[0, :] *= render_scale
        intr[1, :] *= render_scale
    else:
        intr = None
    return resized_t, intr


def resize_rgb_to_hw(rgb: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize one RGB image or a stack of RGB images to a fixed H,W."""
    if rgb.ndim == 3:
        rgb_batched = rgb[None]
    elif rgb.ndim == 4:
        rgb_batched = rgb
    else:
        raise ValueError(f"Expected RGB tensor HxWx3 or TxHxWx3, got {tuple(rgb.shape)}")

    resized = []
    for frame in rgb_batched:
        rgb_np = (frame.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        resized_np = cv2.resize(rgb_np, (width, height), interpolation=cv2.INTER_AREA)
        resized.append(torch.as_tensor(resized_np, device=rgb.device, dtype=torch.float32) / 255.0)
    resized_t = torch.stack(resized, dim=0)
    return resized_t[0] if rgb.ndim == 3 else resized_t


def prepare_release_sample(raw: dict, args, selected_camera: str) -> dict:
    robot_sampler = TorchRobotSampler(
        urdf_path=resolve_robot_urdf(args.domain),
        gripper_only=False,
        device="cpu",
    )

    sample = build_flow_sample(
        raw,
        domain=args.domain,
        robot_sampler=robot_sampler,
        max_robot_points=args.max_robot_points,
        deterministic=True,
        seed=args.seed,
        force_single_arm=False,
    )

    # Avoid random camera selection here so that future-RGB extraction and
    # release camera preprocessing use the same view.
    for key in list(sample.keys()):
        if key.startswith(f"{selected_camera}_"):
            sample[f"cam0_{key[len(selected_camera) + 1:]}"] = sample[key]

    scene_attributes = set()
    for key in list(sample.keys()):
        if key.startswith(f"{selected_camera}_scene_"):
            scene_attributes.add(key[len(selected_camera) + 1 :])
    for attr in scene_attributes:
        sample[attr] = sample[f"{selected_camera}_{attr}"]

    sample = center_shift(sample)
    sample = filter_within_bounds(sample)
    sample = assert_camera_payload_resolution(sample, expected_hw=(180, 320))
    sample = grid_sample_transform(sample, grid_size=args.grid_size, mode="test")
    sample = enforce_max_num_points(
        sample,
        max_scene_points=args.max_scene_points,
        deterministic=True,
        seed=args.seed,
    )
    sample = center_shift(sample)
    sample = normalize_colors(sample)
    sample = make_gt_copy(sample)
    sample = compute_helper_variables(
        sample,
        max_relative_movement=args.max_relative_movement,
        domain=args.domain,
    )
    return sample


def _to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def load_scene_bundle(args) -> SceneBundle:
    device = torch.device(args.device)
    raw = load_raw_sample(args.data_dir, args.split, args.sample_index, args.domain)
    selected_camera = args.camera_prefix or choose_camera_prefix(raw, args.seed)

    future_rgb = load_future_rgb_from_dir(args.future_rgb_dir, args.max_frames)
    if future_rgb is None:
        future_rgb = load_future_rgb_from_wds(raw, selected_camera, args.future_rgb_key_template, args.max_frames)

    sample = prepare_release_sample(raw, args, selected_camera)
    positions = _to_tensor(sample["gt_scene_flows"], device)
    colors = _to_tensor(sample["scene_colors"], device).clamp(0.0, 1.0)
    if args.max_frames is not None:
        positions = positions[: args.max_frames]
        colors = colors[: args.max_frames]

    initial_rgb = _to_tensor(sample["cam0_initial_rgb"], device) / 255.0
    intrinsic = _to_tensor(sample["cam0_intrinsic"], device)
    extrinsic = _to_tensor(sample["cam0_extrinsic"], device)
    initial_rgb, intrinsic = resize_rgb_tensor(initial_rgb, intrinsic, args.render_scale)

    target_rgb_t = None
    if future_rgb is not None:
        target_rgb_t = _to_tensor(future_rgb, device) / 255.0
        if args.max_frames is not None:
            target_rgb_t = target_rgb_t[: args.max_frames]
        target_rgb_t = resize_rgb_to_hw(target_rgb_t, int(initial_rgb.shape[0]), int(initial_rgb.shape[1]))
        frames = min(target_rgb_t.shape[0], positions.shape[0])
        target_rgb_t = target_rgb_t[:frames]
        positions = positions[:frames]
        colors = colors[:frames]

    displacement = positions - positions[:1]
    dynamic_mask = displacement.norm(dim=-1).amax(dim=0) > args.dynamic_threshold
    key = str(sample.get("__key__", f"{args.domain}:{args.sample_index}"))
    return SceneBundle(
        key=key,
        positions=positions,
        colors=colors,
        initial_rgb=initial_rgb,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        dynamic_mask=dynamic_mask,
        target_rgb=target_rgb_t,
        selected_camera=selected_camera,
    )
