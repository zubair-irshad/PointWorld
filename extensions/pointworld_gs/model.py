# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time-dependent Gaussian appearance models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_quat_wxyz(quats: torch.Tensor) -> torch.Tensor:
    return quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class TimeDependentGaussianAppearance(nn.Module):
    """Per-scene time-dependent color, opacity, and emission parameters.

    Geometry is external: PointWorld or ground-truth trajectories move Gaussian
    centers. This module only owns appearance/rasterization attributes.

    appearance_mode:
      - full: every point can change color/opacity over time.
      - dynamic_only: only geometrically moving points can change appearance.
      - static_dynamic: separate static and dynamic bias terms, plus residuals.
    """

    def __init__(
        self,
        initial_colors: torch.Tensor,
        num_frames: int,
        init_opacity: float,
        sigma_px: float,
        enable_emission: bool,
        dynamic_mask: torch.Tensor | None = None,
        appearance_mode: str = "full",
        init_scale_m: float = 0.01,
        learn_scales: bool = True,
        learn_rotations: bool = True,
    ) -> None:
        super().__init__()
        if appearance_mode not in {"full", "dynamic_only", "static_dynamic"}:
            raise ValueError(f"Unsupported appearance_mode: {appearance_mode}")

        eps = 1e-4
        init = initial_colors.clamp(eps, 1.0 - eps)
        self.register_buffer("base_color_logit", torch.logit(init))
        self.register_buffer(
            "dynamic_mask",
            torch.ones(initial_colors.shape[0], dtype=torch.bool)
            if dynamic_mask is None
            else dynamic_mask.bool().clone(),
        )
        self.appearance_mode = appearance_mode

        num_points = initial_colors.shape[0]
        self.color_delta = nn.Parameter(torch.zeros(num_frames, num_points, 3))

        opacity_logit = math.log(init_opacity / max(1.0 - init_opacity, eps))
        self.opacity_logit = nn.Parameter(torch.full((num_points, 1), opacity_logit))
        self.opacity_delta = nn.Parameter(torch.zeros(num_frames, num_points, 1))

        if appearance_mode == "static_dynamic":
            self.static_color_bias = nn.Parameter(torch.zeros(num_frames, 1, 3))
            self.dynamic_color_bias = nn.Parameter(torch.zeros(num_frames, 1, 3))
        else:
            self.register_parameter("static_color_bias", None)
            self.register_parameter("dynamic_color_bias", None)

        self.log_sigma_px = nn.Parameter(torch.tensor(math.log(sigma_px), dtype=torch.float32))
        init_log_scales = torch.full((num_points, 3), math.log(init_scale_m), dtype=torch.float32)
        if learn_scales:
            self.log_scales = nn.Parameter(init_log_scales)
        else:
            self.register_buffer("log_scales", init_log_scales)

        init_quats = torch.zeros(num_points, 4, dtype=torch.float32)
        init_quats[:, 0] = 1.0
        if learn_rotations:
            self.quat_raw = nn.Parameter(init_quats)
        else:
            self.register_buffer("quat_raw", init_quats)

        self.enable_emission = enable_emission
        if enable_emission:
            self.emission_logit = nn.Parameter(torch.full((num_frames, num_points, 3), -8.0))
        else:
            self.register_parameter("emission_logit", None)

    def _masked_delta(self, delta: torch.Tensor) -> torch.Tensor:
        if self.appearance_mode == "dynamic_only":
            return delta * self.dynamic_mask.view(1, -1, 1).to(delta.dtype)
        return delta

    def forward(self, frame_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        color_delta = self._masked_delta(self.color_delta[frame_idx : frame_idx + 1]).squeeze(0)
        if self.appearance_mode == "static_dynamic":
            dynamic = self.dynamic_mask.view(-1, 1).to(color_delta.dtype)
            static_bias = self.static_color_bias[frame_idx].expand_as(color_delta)
            dynamic_bias = self.dynamic_color_bias[frame_idx].expand_as(color_delta)
            color_delta = color_delta + static_bias * (1.0 - dynamic) + dynamic_bias * dynamic

        rgb = torch.sigmoid(self.base_color_logit + color_delta)
        opacity_delta = self._masked_delta(self.opacity_delta[frame_idx : frame_idx + 1]).squeeze(0)
        opacity = torch.sigmoid(self.opacity_logit + opacity_delta)

        if self.enable_emission:
            emission_delta = self._masked_delta(self.emission_logit[frame_idx : frame_idx + 1]).squeeze(0)
            emission = F.softplus(emission_delta)
        else:
            emission = torch.zeros_like(rgb)

        sigma = self.log_sigma_px.exp().clamp(0.25, 12.0)
        return (rgb + emission).clamp(0.0, 1.0), opacity, sigma, emission

    def gaussian_geometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        scales = self.log_scales.exp().clamp(min=1e-5, max=0.25)
        quats = _normalize_quat_wxyz(self.quat_raw)
        return scales, quats

    def gaussian_scale_regularization(self, target_scale_m: float) -> torch.Tensor:
        target = math.log(target_scale_m)
        return (self.log_scales - target).square().mean()

    def gaussian_rotation_regularization(self) -> torch.Tensor:
        identity = self.quat_raw.new_zeros(self.quat_raw.shape)
        identity[:, 0] = 1.0
        return (_normalize_quat_wxyz(self.quat_raw) - identity).square().mean()

    def temporal_regularization(self) -> torch.Tensor:
        color = self.color_delta
        if color.shape[0] <= 1:
            return color.new_zeros(())
        return color[1:].sub(color[:-1]).square().mean()

    def appearance_magnitude(self) -> torch.Tensor:
        return self.color_delta.square().mean()
