# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visualization helpers for PointWorld-GS overfit experiments."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def tensor_to_u8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)


def write_rgb(path: Path, image: torch.Tensor) -> None:
    rgb = tensor_to_u8(image)
    cv2.imwrite(str(path), rgb[..., ::-1])


def colorize_mask(mask: torch.Tensor, color=(255, 80, 30)) -> torch.Tensor:
    mask = mask.float().clamp(0.0, 1.0)
    if mask.ndim == 2:
        mask = mask[..., None]
    rgb = torch.tensor(color, dtype=mask.dtype, device=mask.device).view(1, 1, 3) / 255.0
    return mask * rgb


def make_grid(images: list[torch.Tensor], labels: list[str]) -> np.ndarray:
    rendered = []
    for image, label in zip(images, labels):
        rgb = tensor_to_u8(image)
        cv2.putText(rgb, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(rgb, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(rgb)
    return np.concatenate(rendered, axis=1)


def save_training_grid(
    output_dir: Path,
    step: int,
    frame_idx: int,
    source: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    alpha: torch.Tensor,
    residual: torch.Tensor,
) -> None:
    alpha_rgb = alpha.expand(-1, -1, 3)
    grid = make_grid(
        [source, target, pred, residual, alpha_rgb],
        ["initial rgb", f"target t={frame_idx}", f"pred step={step}", "|error|", "pred alpha"],
    )
    cv2.imwrite(str(output_dir / f"train_step_{step:06d}_frame_{frame_idx:02d}.png"), grid[..., ::-1])

