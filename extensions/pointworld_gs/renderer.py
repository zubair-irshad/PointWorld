# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differentiable Gaussian renderers for PointWorld-GS experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class RenderOutput:
    image: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    accum_weight: torch.Tensor
    projected_xy: torch.Tensor
    projected_depth: torch.Tensor
    valid_points: torch.Tensor


def _empty_render(
    height: int,
    width: int,
    bg_color: torch.Tensor,
    points: torch.Tensor,
    pixels: torch.Tensor | None = None,
    depth: torch.Tensor | None = None,
) -> RenderOutput:
    image = bg_color.view(1, 1, 3).expand(height, width, 3).clone()
    alpha = torch.zeros((height, width, 1), device=points.device, dtype=points.dtype)
    depth_img = torch.zeros((height, width, 1), device=points.device, dtype=points.dtype)
    weight = torch.zeros((height, width, 1), device=points.device, dtype=points.dtype)
    if pixels is None:
        pixels = torch.empty((points.shape[0], 2), device=points.device, dtype=points.dtype)
    if depth is None:
        depth = torch.empty((points.shape[0],), device=points.device, dtype=points.dtype)
    valid = torch.zeros((points.shape[0],), device=points.device, dtype=torch.bool)
    return RenderOutput(image, alpha, depth_img, weight, pixels, depth, valid)


def project_points(
    points: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world-frame points into one calibrated camera."""
    ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
    points_h = torch.cat([points, ones], dim=-1)
    cam = (extrinsic @ points_h.T).T[:, :3]
    pix_h = (intrinsic @ cam.T).T
    z = pix_h[:, 2].clamp(min=1e-6)
    pixels = pix_h[:, :2] / z[:, None]
    return pixels, cam[:, 2]


def render_gaussians_gsplat(
    points: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    height: int,
    width: int,
    bg_color: torch.Tensor | None = None,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    packed: bool = True,
    rasterize_mode: str = "classic",
    absgrad: bool = False,
) -> RenderOutput:
    """Render true 3D Gaussians with gsplat.

    gsplat expects 3D means, anisotropic scales, and wxyz quaternions in world
    coordinates. The PointWorld camera extrinsics are already world-to-camera
    matrices, which matches the gsplat `viewmats` convention.
    """
    try:
        from gsplat.rendering import rasterization
    except ImportError as exc:
        raise ImportError(
            "Renderer backend 'gsplat' requires the gsplat package. Install it in "
            "the active PointWorld environment, then rerun with --renderer gsplat."
        ) from exc

    if bg_color is None:
        bg_color = torch.zeros(3, device=points.device, dtype=points.dtype)
    if points.numel() == 0:
        return _empty_render(height, width, bg_color, points)

    # gsplat is CUDA-oriented; fail clearly rather than silently falling back.
    if not points.is_cuda:
        raise RuntimeError("--renderer gsplat requires CUDA tensors; use --device cuda or --renderer surfel")

    means = points.contiguous()
    colors = colors.contiguous().clamp(0.0, 1.0)
    opacities = opacity.reshape(-1).contiguous().clamp(0.0, 1.0)
    scales = scales.contiguous().clamp(min=1e-5)
    quats = quats.contiguous()
    viewmats = extrinsic.to(device=points.device, dtype=points.dtype).reshape(1, 4, 4).contiguous()
    Ks = intrinsic.to(device=points.device, dtype=points.dtype).reshape(1, 3, 3).contiguous()
    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=int(width),
        height=int(height),
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        radius_clip=float(radius_clip),
        eps2d=float(eps2d),
        sh_degree=None,
        packed=bool(packed),
        backgrounds=None,
        render_mode="RGB+ED",
        rasterize_mode=rasterize_mode,
        absgrad=bool(absgrad),
    )

    rendered = render_colors[0]
    alpha = render_alphas[0].clamp(0.0, 1.0)
    image = (rendered[..., :3] + bg_color.view(1, 1, 3) * (1.0 - alpha)).clamp(0.0, 1.0)
    depth_img = rendered[..., 3:4]
    pixels, depth = project_points(points, intrinsic, extrinsic)
    valid = (
        torch.isfinite(pixels).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > near_plane)
        & (depth < far_plane)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return RenderOutput(
        image=image,
        alpha=alpha,
        depth=depth_img,
        accum_weight=alpha,
        projected_xy=pixels,
        projected_depth=depth,
        valid_points=valid,
    )


def render_gaussian_surfels(
    points: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    intrinsic: torch.Tensor,
    extrinsic: torch.Tensor,
    height: int,
    width: int,
    sigma_px: torch.Tensor | float,
    bg_color: torch.Tensor | None = None,
    depth_temperature: float = 0.0,
) -> RenderOutput:
    """Render screen-space Gaussian surfels with scatter_add.

    This is a research/debug renderer, not a production 3DGS rasterizer. It is
    differentiable with respect to color, opacity, and sigma. It deliberately
    keeps the interface close to 3DGS renderers so it can be replaced with
    `gsplat` or `diff-gaussian-rasterization` later.

    `depth_temperature > 0` softly favors nearer surfels at each pixel. It is not
    exact alpha compositing, but it improves over a pure unoccluded average for
    dense point clouds while keeping the code dependency-free.
    """
    if bg_color is None:
        bg_color = torch.zeros(3, device=points.device, dtype=points.dtype)
    if not torch.is_tensor(sigma_px):
        sigma = torch.tensor(float(sigma_px), device=points.device, dtype=points.dtype)
    else:
        sigma = sigma_px.to(device=points.device, dtype=points.dtype).clamp(min=0.25)

    pixels, depth = project_points(points, intrinsic, extrinsic)
    valid = (
        torch.isfinite(pixels).all(dim=-1)
        & torch.isfinite(depth)
        & (depth > 0)
        & (pixels[:, 0] >= -3.0 * sigma)
        & (pixels[:, 0] < width + 3.0 * sigma)
        & (pixels[:, 1] >= -3.0 * sigma)
        & (pixels[:, 1] < height + 3.0 * sigma)
    )
    if valid.sum() == 0:
        return _empty_render(height, width, bg_color, points, pixels=pixels, depth=depth)

    pixels_v = pixels[valid]
    depth_v = depth[valid]
    colors_v = colors[valid]
    opacity_v = opacity[valid].reshape(-1).clamp(0.0, 1.0)

    radius = int(max(1, math.ceil(3.0 * float(sigma.detach().cpu()))))
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=points.device),
        torch.arange(-radius, radius + 1, device=points.device),
        indexing="ij",
    )
    offsets = torch.stack([offsets_x.reshape(-1), offsets_y.reshape(-1)], dim=-1).to(points.dtype)

    centers = pixels_v.round().to(torch.long)
    candidate_xy = centers[:, None, :] + offsets[None].to(torch.long)
    pixel_xy = candidate_xy.to(points.dtype)
    delta = pixel_xy - pixels_v[:, None, :]
    weights = torch.exp(-0.5 * (delta.square().sum(dim=-1) / sigma.square())) * opacity_v[:, None]
    if depth_temperature > 0:
        weights = weights * torch.exp(-depth_temperature * depth_v[:, None])

    inside = (
        (candidate_xy[..., 0] >= 0)
        & (candidate_xy[..., 0] < width)
        & (candidate_xy[..., 1] >= 0)
        & (candidate_xy[..., 1] < height)
    )
    flat_idx = (candidate_xy[..., 1] * width + candidate_xy[..., 0]).reshape(-1)
    flat_inside = inside.reshape(-1)
    flat_weights = weights.reshape(-1)[flat_inside]
    flat_idx = flat_idx[flat_inside]

    weighted_colors = (weights[..., None] * colors_v[:, None, :]).reshape(-1, 3)[flat_inside]
    weighted_depth = (weights * depth_v[:, None]).reshape(-1)[flat_inside]

    accum_w = torch.zeros((height * width,), device=points.device, dtype=points.dtype)
    accum_rgb = torch.zeros((height * width, 3), device=points.device, dtype=points.dtype)
    accum_depth = torch.zeros((height * width,), device=points.device, dtype=points.dtype)
    accum_w.scatter_add_(0, flat_idx, flat_weights)
    accum_rgb.scatter_add_(0, flat_idx[:, None].expand(-1, 3), weighted_colors)
    accum_depth.scatter_add_(0, flat_idx, weighted_depth)

    alpha = 1.0 - torch.exp(-accum_w).reshape(height, width, 1)
    image = accum_rgb / accum_w.clamp(min=1e-6).reshape(-1, 1)
    image = image.reshape(height, width, 3)
    depth_img = (accum_depth / accum_w.clamp(min=1e-6)).reshape(height, width, 1)

    image = image * alpha + bg_color.view(1, 1, 3) * (1.0 - alpha)
    return RenderOutput(
        image=image.clamp(0.0, 1.0),
        alpha=alpha.clamp(0.0, 1.0),
        depth=depth_img,
        accum_weight=accum_w.reshape(height, width, 1),
        projected_xy=pixels,
        projected_depth=depth,
        valid_points=valid,
    )
