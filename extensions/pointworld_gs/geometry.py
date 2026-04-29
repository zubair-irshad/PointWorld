# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static/dynamic geometry factorization for PointWorld-GS overfit runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GeometryState:
    positions: torch.Tensor
    dynamic_mask: torch.Tensor
    rigid_cluster_ids: torch.Tensor
    robot_mask: torch.Tensor
    stats: dict[str, int | float | str]


def compute_dynamic_mask(positions: torch.Tensor, threshold_m: float) -> torch.Tensor:
    displacement = positions - positions[:1]
    return displacement.norm(dim=-1).amax(dim=0) > float(threshold_m)


def _load_bool_mask(path: str | None, num_points: int, device: torch.device) -> torch.Tensor | None:
    if not path:
        return None
    import numpy as np

    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        key = "mask" if "mask" in payload else "robot_mask" if "robot_mask" in payload else "dynamic_mask"
        if key not in payload:
            raise KeyError(f"{path} must contain 'mask', 'robot_mask', or 'dynamic_mask'")
        array = payload[key]
    else:
        array = payload
    mask = torch.as_tensor(array, device=device).bool().reshape(-1)
    if mask.numel() != num_points:
        raise ValueError(f"Mask {path} has {mask.numel()} points, expected {num_points}")
    return mask


def load_dynamic_mask_override(path: str | None, num_points: int, device: torch.device) -> torch.Tensor | None:
    return _load_bool_mask(path, num_points, device)


def load_robot_mask(path: str | None, num_points: int, device: torch.device) -> torch.Tensor:
    mask = _load_bool_mask(path, num_points, device)
    if mask is None:
        return torch.zeros(num_points, device=device, dtype=torch.bool)
    return mask


def _stable_unique_inverse(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # torch.unique with dim=0 is deterministic for identical input and avoids
    # bringing large point sets back to Python object dictionaries.
    unique, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    return unique, inverse


def cluster_dynamic_points(
    positions: torch.Tensor,
    dynamic_mask: torch.Tensor,
    spatial_voxel_m: float,
    motion_voxel_m: float,
    min_cluster_points: int,
) -> torch.Tensor:
    """Assign local rigid-cluster ids to dynamic points.

    This intentionally makes conservative local motion clusters rather than
    semantic object instances. Nonrigid cloth/articulated leftovers can remain
    per-point deformable by failing the min-cluster threshold.
    """
    num_points = positions.shape[1]
    cluster_ids = torch.full((num_points,), -1, device=positions.device, dtype=torch.long)
    dyn_idx = torch.nonzero(dynamic_mask, as_tuple=False).reshape(-1)
    if dyn_idx.numel() == 0:
        return cluster_ids

    p0 = positions[0, dyn_idx]
    disp = positions[-1, dyn_idx] - positions[0, dyn_idx]
    spatial_key = torch.floor(p0 / max(float(spatial_voxel_m), 1e-6)).to(torch.long)
    motion_key = torch.floor(disp / max(float(motion_voxel_m), 1e-6)).to(torch.long)
    keys = torch.cat([spatial_key, motion_key], dim=-1)
    _, inverse = _stable_unique_inverse(keys)
    counts = torch.bincount(inverse)
    valid_local = counts[inverse] >= int(min_cluster_points)
    if valid_local.sum() == 0:
        return cluster_ids

    valid_inverse = inverse[valid_local]
    valid_dyn_idx = dyn_idx[valid_local]
    valid_unique = torch.unique(valid_inverse, sorted=True)
    remap = torch.full((int(inverse.max().item()) + 1,), -1, device=positions.device, dtype=torch.long)
    remap[valid_unique] = torch.arange(valid_unique.numel(), device=positions.device, dtype=torch.long)
    cluster_ids[valid_dyn_idx] = remap[valid_inverse]
    return cluster_ids


def _fit_rigid_transform(src: torch.Tensor, dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    src_centroid = src.mean(dim=0)
    dst_centroid = dst.mean(dim=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    cov = src_centered.T @ dst_centered / max(int(src.shape[0]), 1)
    u, _, vh = torch.linalg.svd(cov, full_matrices=False)
    rot = vh.T @ u.T
    if torch.linalg.det(rot) < 0:
        vh = vh.clone()
        vh[-1] *= -1.0
        rot = vh.T @ u.T
    trans = dst_centroid - src_centroid @ rot.T
    return rot, trans


def apply_rigid_clusters(positions: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
    out = positions.clone()
    valid_clusters = torch.unique(cluster_ids[cluster_ids >= 0], sorted=True)
    if valid_clusters.numel() == 0:
        return out

    canonical = positions[0]
    for cluster_id in valid_clusters.tolist():
        mask = cluster_ids == int(cluster_id)
        if int(mask.sum().item()) < 3:
            continue
        src = canonical[mask]
        for frame_idx in range(positions.shape[0]):
            rot, trans = _fit_rigid_transform(src, positions[frame_idx, mask])
            out[frame_idx, mask] = src @ rot.T + trans
    return out


def factorize_geometry(
    positions: torch.Tensor,
    dynamic_threshold_m: float,
    mode: str,
    dynamic_mask_override_path: str | None = None,
    robot_mask_path: str | None = None,
    cluster_spatial_voxel_m: float = 0.06,
    cluster_motion_voxel_m: float = 0.01,
    min_cluster_points: int = 32,
) -> GeometryState:
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape T,N,3, got {tuple(positions.shape)}")
    if mode not in {"gt", "static", "static_dynamic", "rigid_clusters"}:
        raise ValueError(f"Unsupported geometry mode: {mode}")

    num_points = int(positions.shape[1])
    device = positions.device
    dynamic_mask = compute_dynamic_mask(positions, dynamic_threshold_m)
    override = load_dynamic_mask_override(dynamic_mask_override_path, num_points, device)
    if override is not None:
        dynamic_mask = override
    robot_mask = load_robot_mask(robot_mask_path, num_points, device)

    rigid_cluster_ids = torch.full((num_points,), -1, device=device, dtype=torch.long)
    factored = positions
    if mode == "static":
        factored = positions[:1].expand_as(positions).clone()
    elif mode == "static_dynamic":
        factored = positions.clone()
        factored[:, ~dynamic_mask] = positions[:1, ~dynamic_mask].expand(positions.shape[0], -1, -1)
    elif mode == "rigid_clusters":
        factored = positions.clone()
        factored[:, ~dynamic_mask] = positions[:1, ~dynamic_mask].expand(positions.shape[0], -1, -1)
        rigid_cluster_ids = cluster_dynamic_points(
            positions,
            dynamic_mask | robot_mask,
            spatial_voxel_m=cluster_spatial_voxel_m,
            motion_voxel_m=cluster_motion_voxel_m,
            min_cluster_points=min_cluster_points,
        )
        rigid_positions = apply_rigid_clusters(positions, rigid_cluster_ids)
        rigid_mask = rigid_cluster_ids >= 0
        factored[:, rigid_mask] = rigid_positions[:, rigid_mask]

    num_rigid_clusters = int(torch.unique(rigid_cluster_ids[rigid_cluster_ids >= 0]).numel())
    num_rigid_points = int((rigid_cluster_ids >= 0).sum().item())
    num_dynamic = int(dynamic_mask.sum().item())
    stats: dict[str, int | float | str] = {
        "geometry_mode": mode,
        "num_points": num_points,
        "num_static_points": int((~dynamic_mask).sum().item()),
        "num_dynamic_points": num_dynamic,
        "num_robot_points": int(robot_mask.sum().item()),
        "num_rigid_clusters": num_rigid_clusters,
        "num_rigid_points": num_rigid_points,
        "num_deformable_dynamic_points": int((dynamic_mask & (rigid_cluster_ids < 0)).sum().item()),
        "dynamic_threshold_m": float(dynamic_threshold_m),
        "cluster_spatial_voxel_m": float(cluster_spatial_voxel_m),
        "cluster_motion_voxel_m": float(cluster_motion_voxel_m),
        "min_cluster_points": int(min_cluster_points),
    }
    return GeometryState(
        positions=factored.contiguous(),
        dynamic_mask=dynamic_mask,
        rigid_cluster_ids=rigid_cluster_ids,
        robot_mask=robot_mask,
        stats=stats,
    )
