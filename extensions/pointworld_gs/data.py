# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data loading for one-scene PointWorld-GS overfit experiments."""

from __future__ import annotations

import re
import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import webdataset as wds

import transform_utils


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
    del domain
    dataset = wds.WebDataset(shards, shardshuffle=False, handler=wds.warn_and_continue)
    for idx, sample in enumerate(dataset):
        if idx == sample_index:
            return sample
    raise IndexError(f"Sample index {sample_index} was not found in {len(shards)} shard(s)")


def _camera_prefixes_from_sample(sample: dict) -> list[str]:
    prefixes = set()
    for key in sample.keys():
        if key.endswith("_initial_rgb.jpg") or key.endswith("_initial_rgb.png"):
            prefixes.add(key.rsplit("_initial_rgb.", 1)[0])
        elif "_scene_flows" in key:
            prefixes.add(key.split("_scene_flows")[0])
        elif "_local_scene_points" in key:
            prefixes.add(key.split("_local_scene_points")[0])
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


def _load_npy_value(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        with io.BytesIO(bytes(value)) as f:
            return np.load(f, allow_pickle=False)
    raise TypeError(f"Unsupported npy value type: {type(value)}")


def _load_pickle_value(value):
    if isinstance(value, (bytes, bytearray, memoryview)):
        return pickle.loads(bytes(value))
    raise TypeError(f"Unsupported pickle value type: {type(value)}")


def _get_sample_value(sample: dict, *candidates: str, required: bool = True):
    for key in candidates:
        if key in sample:
            return sample[key]
    if required:
        raise KeyError(f"Missing any of keys: {candidates}")
    return None


def _to_uint8_colors(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors)
    if arr.dtype == np.uint8:
        return arr
    if arr.size == 0:
        return arr.astype(np.uint8)
    arr_f = arr.astype(np.float32)
    if float(np.nanmax(arr_f)) <= 1.0 + 1e-6:
        arr_f = arr_f * 255.0
    return np.clip(arr_f, 0.0, 255.0).astype(np.uint8)


def _ensure_temporal_colors(colors: np.ndarray, num_frames: int) -> np.ndarray:
    colors = _to_uint8_colors(colors)
    if colors.ndim == 2:
        if colors.shape[1] != 3:
            raise ValueError(f"scene colors must have shape N,3 or T,N,3, got {colors.shape}")
        return np.broadcast_to(colors[None], (num_frames, colors.shape[0], 3)).copy()
    if colors.ndim == 3 and colors.shape[0] == num_frames and colors.shape[2] == 3:
        return colors
    raise ValueError(f"scene colors must have shape N,3 or T,N,3 aligned to T={num_frames}, got {colors.shape}")


def _subsample_points(
    positions: np.ndarray,
    colors: np.ndarray,
    max_points: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_points is None or max_points <= 0 or positions.shape[1] <= max_points:
        return positions, colors
    rng = np.random.RandomState(seed)
    keep = np.sort(rng.choice(positions.shape[1], size=max_points, replace=False))
    return positions[:, keep], colors[:, keep]


def _load_camera_rgb(sample: dict, camera_prefix: str) -> np.ndarray:
    value = _get_sample_value(
        sample,
        f"{camera_prefix}_initial_rgb.jpg",
        f"{camera_prefix}_initial_rgb.png",
    )
    return _decode_rgb_value(value)


def _load_droid_scene(sample: dict, camera_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    positions = _load_npy_value(_get_sample_value(sample, f"{camera_prefix}_scene_flows.npy")).astype(np.float32)
    colors = _load_npy_value(_get_sample_value(sample, f"{camera_prefix}_scene_colors.npy"))
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"{camera_prefix}_scene_flows must have shape T,N,3, got {positions.shape}")
    return positions, _ensure_temporal_colors(colors, int(positions.shape[0]))


def _load_behavior_scene(sample: dict, camera_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    local_points = _load_pickle_value(_get_sample_value(sample, f"{camera_prefix}_local_scene_points.pyd"))
    local_colors = _load_pickle_value(_get_sample_value(sample, f"{camera_prefix}_local_scene_colors.pyd"))
    trajectories = _load_pickle_value(_get_sample_value(sample, f"{camera_prefix}_scene_mesh_trajectories.pyd"))

    mesh_names = sorted(set(local_points.keys()) & set(local_colors.keys()) & set(trajectories.keys()))
    if not mesh_names:
        raise RuntimeError(f"No common behavior mesh payloads found for {camera_prefix}")

    all_points = []
    all_colors = []
    num_frames = None
    for mesh_name in mesh_names:
        points = np.asarray(local_points[mesh_name], dtype=np.float32)
        colors = _to_uint8_colors(np.asarray(local_colors[mesh_name]))
        poses = np.asarray(trajectories[mesh_name], dtype=np.float32)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError(f"local_scene_points[{mesh_name}] must be N,3, got {points.shape}")
        if colors.ndim != 2 or colors.shape[-1] != 3:
            raise ValueError(f"local_scene_colors[{mesh_name}] must be N,3, got {colors.shape}")
        if poses.ndim != 2 or poses.shape[-1] != 7:
            raise ValueError(f"scene_mesh_trajectories[{mesh_name}] must be T,7, got {poses.shape}")

        pose_mats = np.asarray(transform_utils.convert_pose_quat2mat(poses), dtype=np.float32)
        mesh_frames = int(pose_mats.shape[0])
        if num_frames is None:
            num_frames = mesh_frames
        elif mesh_frames != num_frames:
            raise ValueError(f"Inconsistent behavior trajectory length: {mesh_frames} vs {num_frames}")

        world_points = np.einsum("tij,nj->tni", pose_mats[:, :3, :3], points) + pose_mats[:, None, :3, 3]
        all_points.append(world_points.astype(np.float32, copy=False))
        all_colors.append(np.broadcast_to(colors[None], (mesh_frames, colors.shape[0], 3)).copy())

    return np.concatenate(all_points, axis=1), np.concatenate(all_colors, axis=1)


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


def _to_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, device=device, dtype=dtype)


def load_scene_bundle(args) -> SceneBundle:
    device = torch.device(args.device)
    raw = load_raw_sample(args.data_dir, args.split, args.sample_index, args.domain)
    selected_camera = args.camera_prefix or choose_camera_prefix(raw, args.seed)

    future_rgb = load_future_rgb_from_dir(args.future_rgb_dir, args.max_frames)
    if future_rgb is None:
        future_rgb = load_future_rgb_from_wds(raw, selected_camera, args.future_rgb_key_template, args.max_frames)

    if args.domain == "droid":
        positions_np, colors_np = _load_droid_scene(raw, selected_camera)
    elif args.domain == "behavior":
        positions_np, colors_np = _load_behavior_scene(raw, selected_camera)
    else:
        raise ValueError(f"Unsupported domain: {args.domain}")

    positions_np, colors_np = _subsample_points(positions_np, colors_np, args.max_scene_points, args.seed)
    positions = _to_tensor(positions_np, device)
    colors = _to_tensor(colors_np, device).clamp(0.0, 255.0) / 255.0
    if args.max_frames is not None:
        positions = positions[: args.max_frames]
        colors = colors[: args.max_frames]

    initial_rgb = _to_tensor(_load_camera_rgb(raw, selected_camera), device) / 255.0
    intrinsic = _to_tensor(_load_npy_value(_get_sample_value(raw, f"{selected_camera}_intrinsic.npy")), device)
    extrinsic = _to_tensor(_load_npy_value(_get_sample_value(raw, f"{selected_camera}_extrinsic.npy")), device)
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
    key = str(raw.get("__key__", f"{args.domain}:{args.sample_index}"))
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
