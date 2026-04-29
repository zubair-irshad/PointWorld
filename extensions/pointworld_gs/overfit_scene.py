# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Overfit time-dependent Gaussian appearance to one PointWorld scene."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from extensions.pointworld_gs.data import SceneBundle, load_scene_bundle
from extensions.pointworld_gs.model import TimeDependentGaussianAppearance
from extensions.pointworld_gs.renderer import RenderOutput, render_gaussian_surfels, render_gaussians_gsplat
from extensions.pointworld_gs.viz import save_training_grid, write_rgb


def _init_wandb(args: argparse.Namespace, output_dir: Path):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Install wandb or rerun without --wandb") from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        mode=args.wandb_mode,
        dir=str(output_dir),
        config=vars(args),
    )


def _wandb_image(image: torch.Tensor, caption: str):
    import wandb

    array = (image.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return wandb.Image(array, caption=caption)


def render_frame(
    bundle: SceneBundle,
    geometry: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    sigma: torch.Tensor | float,
    args: argparse.Namespace,
    scales: torch.Tensor | None = None,
    quats: torch.Tensor | None = None,
) -> RenderOutput:
    height, width = bundle.initial_rgb.shape[:2]
    bg = torch.tensor(args.bg_color, device=geometry.device, dtype=geometry.dtype)
    if args.renderer == "gsplat":
        if scales is None or quats is None:
            raise ValueError("renderer=gsplat requires scales and quats")
        return render_gaussians_gsplat(
            geometry,
            colors,
            opacity,
            scales,
            quats,
            bundle.intrinsic,
            bundle.extrinsic,
            height,
            width,
            bg_color=bg,
            near_plane=args.gsplat_near_plane,
            far_plane=args.gsplat_far_plane,
            radius_clip=args.gsplat_radius_clip,
            eps2d=args.gsplat_eps2d,
            packed=args.gsplat_packed,
            rasterize_mode=args.gsplat_rasterize_mode,
            absgrad=args.gsplat_absgrad,
        )

    return render_gaussian_surfels(
        geometry,
        colors,
        opacity,
        bundle.intrinsic,
        bundle.extrinsic,
        height,
        width,
        sigma,
        bg_color=bg,
        depth_temperature=args.depth_temperature,
    )


def load_geometry_override(path: str | None, bundle: SceneBundle, device: torch.device) -> torch.Tensor | None:
    if not path:
        return None
    geometry_path = Path(path)
    if not geometry_path.exists():
        raise FileNotFoundError(f"Geometry override does not exist: {geometry_path}")
    if geometry_path.suffix == ".npy":
        array = np.load(geometry_path)
    elif geometry_path.suffix == ".npz":
        payload = np.load(geometry_path)
        key = "positions" if "positions" in payload else "scene_flows"
        if key not in payload:
            raise KeyError(f"{geometry_path} must contain 'positions' or 'scene_flows'")
        array = payload[key]
    else:
        raise ValueError("--geometry_path must point to .npy or .npz")

    geometry = torch.as_tensor(array, device=device, dtype=torch.float32)
    if geometry.ndim != 3 or geometry.shape[-1] != 3:
        raise ValueError(f"Geometry override must have shape T,N,3, got {tuple(geometry.shape)}")
    if geometry.shape[1] != bundle.positions.shape[1]:
        raise ValueError(
            f"Geometry override has N={geometry.shape[1]} points, expected {bundle.positions.shape[1]}"
        )
    frames = min(geometry.shape[0], bundle.positions.shape[0])
    return geometry[:frames]


@torch.no_grad()
def precompute_rendered_point_targets(
    bundle: SceneBundle,
    geometry: torch.Tensor,
    args: argparse.Namespace,
    model: TimeDependentGaussianAppearance,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    alphas = []
    opacity = torch.full((geometry.shape[1], 1), args.init_opacity, device=geometry.device)
    scales, quats = model.gaussian_geometry()
    for t in range(geometry.shape[0]):
        rendered = render_frame(bundle, geometry[t], bundle.colors[t], opacity, args.sigma_px, args, scales, quats)
        targets.append(rendered.image)
        alphas.append(rendered.alpha)
    return torch.stack(targets, dim=0), torch.stack(alphas, dim=0)


def choose_targets(
    bundle: SceneBundle,
    geometry: torch.Tensor,
    args: argparse.Namespace,
    model: TimeDependentGaussianAppearance,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    if args.target_mode == "future_rgb":
        if bundle.target_rgb is None:
            raise RuntimeError("target_mode=future_rgb but no future RGB frames were found")
        mask = torch.ones((*bundle.target_rgb.shape[1:3], 1), device=bundle.target_rgb.device)
        return bundle.target_rgb, mask[None].expand(bundle.target_rgb.shape[0], -1, -1, -1), "future_rgb"

    if args.target_mode == "auto" and bundle.target_rgb is not None:
        mask = torch.ones((*bundle.target_rgb.shape[1:3], 1), device=bundle.target_rgb.device)
        return bundle.target_rgb, mask[None].expand(bundle.target_rgb.shape[0], -1, -1, -1), "future_rgb"

    target_images, target_alpha = precompute_rendered_point_targets(bundle, geometry, args, model)
    return target_images, target_alpha, "rendered_points"


def validate_target_shape(bundle: SceneBundle, target_images: torch.Tensor) -> None:
    expected = tuple(bundle.initial_rgb.shape[:2])
    actual = tuple(target_images.shape[1:3])
    if actual != expected:
        raise RuntimeError(f"Target image shape {actual} does not match render shape {expected}")


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((pred - target).abs() * mask).sum() / (mask.sum() * pred.shape[-1]).clamp(min=1.0)


def psnr_from_l1(l1: torch.Tensor) -> torch.Tensor:
    mse_est = l1.square().clamp(min=1e-10)
    return -10.0 * torch.log10(mse_est)


def save_metadata(
    output_dir: Path,
    bundle: SceneBundle,
    geometry: torch.Tensor,
    args: argparse.Namespace,
    target_source: str,
) -> None:
    metadata = {
        "scene_key": bundle.key,
        "domain": args.domain,
        "split": args.split,
        "sample_index": args.sample_index,
        "selected_camera": bundle.selected_camera,
        "target_source": target_source,
        "num_frames": int(geometry.shape[0]),
        "num_points": int(bundle.positions.shape[1]),
        "num_dynamic_points": int(bundle.dynamic_mask.sum().item()),
        "args": vars(args),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train(args: argparse.Namespace) -> None:
    if args.renderer == "gsplat" and not args.device.startswith("cuda"):
        raise RuntimeError("--renderer gsplat requires --device cuda because gsplat rasterization is CUDA-backed")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_scene_bundle(args)
    geometry = load_geometry_override(args.geometry_path, bundle, bundle.positions.device)
    if geometry is None:
        geometry = bundle.positions
    if args.geometry == "static":
        geometry = geometry[:1].expand_as(geometry)

    model = TimeDependentGaussianAppearance(
        initial_colors=bundle.colors[0],
        num_frames=geometry.shape[0],
        init_opacity=args.init_opacity,
        sigma_px=args.sigma_px,
        enable_emission=args.enable_emission,
        dynamic_mask=bundle.dynamic_mask,
        appearance_mode=args.appearance_mode,
        init_scale_m=args.init_scale_m,
        learn_scales=not args.freeze_scales,
        learn_rotations=not args.freeze_rotations,
    ).to(bundle.positions.device)

    target_images, target_mask, target_source = choose_targets(bundle, geometry, args, model)
    validate_target_shape(bundle, target_images)
    save_metadata(output_dir, bundle, geometry, args, target_source)
    wandb_run = _init_wandb(args, output_dir)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    frame_order = torch.arange(geometry.shape[0], device=bundle.positions.device)
    metrics_path = output_dir / "metrics.csv"

    print(
        f"Loaded {bundle.key}: domain={args.domain} camera={bundle.selected_camera} "
        f"T={geometry.shape[0]} N={geometry.shape[1]} "
        f"dynamic_points={int(bundle.dynamic_mask.sum())}/{bundle.dynamic_mask.numel()} "
        f"target={target_source} render={bundle.initial_rgb.shape[1]}x{bundle.initial_rgb.shape[0]}"
    )
    print(f"Writing outputs to {output_dir}")

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "frame",
                "loss",
                "image_l1",
                "point_l1",
                "psnr_est",
                "opacity_reg",
                "emission_reg",
                "temporal_reg",
                "scale_reg",
                "rotation_reg",
                "sigma_px",
                "mean_scale_m",
            ],
        )
        writer.writeheader()

        for step in range(args.steps + 1):
            frame_idx = int(frame_order[step % len(frame_order)].item())
            pred_colors, opacity, sigma, emission = model(frame_idx)
            scales, quats = model.gaussian_geometry()
            rendered = render_frame(bundle, geometry[frame_idx], pred_colors, opacity, sigma, args, scales, quats)

            target = target_images[frame_idx]
            if target_source == "rendered_points":
                alpha_mask = (target_mask[frame_idx] > args.alpha_loss_threshold).float()
            else:
                alpha_mask = target_mask[frame_idx].float()

            image_l1 = masked_l1(rendered.image, target, alpha_mask)
            point_l1 = (pred_colors - bundle.colors[frame_idx]).abs().mean()
            opacity_reg = (opacity - args.init_opacity).square().mean()
            emission_reg = emission.abs().mean()
            temporal_reg = model.temporal_regularization()
            appearance_reg = model.appearance_magnitude()
            scale_reg = model.gaussian_scale_regularization(args.init_scale_m)
            rotation_reg = model.gaussian_rotation_regularization()

            loss = (
                args.image_loss_weight * image_l1
                + args.point_loss_weight * point_l1
                + args.opacity_reg_weight * opacity_reg
                + args.emission_reg_weight * emission_reg
                + args.temporal_reg_weight * temporal_reg
                + args.appearance_reg_weight * appearance_reg
                + args.scale_reg_weight * scale_reg
                + args.rotation_reg_weight * rotation_reg
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            row = {
                "step": step,
                "frame": frame_idx,
                "loss": float(loss.detach().cpu()),
                "image_l1": float(image_l1.detach().cpu()),
                "point_l1": float(point_l1.detach().cpu()),
                "psnr_est": float(psnr_from_l1(image_l1).detach().cpu()),
                "opacity_reg": float(opacity_reg.detach().cpu()),
                "emission_reg": float(emission_reg.detach().cpu()),
                "temporal_reg": float(temporal_reg.detach().cpu()),
                "scale_reg": float(scale_reg.detach().cpu()),
                "rotation_reg": float(rotation_reg.detach().cpu()),
                "sigma_px": float(sigma.detach().cpu()),
                "mean_scale_m": float(scales.mean().detach().cpu()),
            }
            writer.writerow(row)
            if wandb_run is not None and step % args.wandb_log_every == 0:
                wandb_run.log(row, step=step)
            if step % args.log_every == 0:
                print(
                    f"step={step:06d} frame={frame_idx:02d} loss={row['loss']:.6f} "
                    f"image_l1={row['image_l1']:.6f} point_l1={row['point_l1']:.6f} "
                    f"psnr_est={row['psnr_est']:.2f} sigma={row['sigma_px']:.3f} "
                    f"mean_scale_m={row['mean_scale_m']:.5f}"
                )
            if step % args.viz_every == 0 or step == args.steps:
                residual = (rendered.image - target).abs()
                save_training_grid(
                    output_dir,
                    step,
                    frame_idx,
                    bundle.initial_rgb,
                    target,
                    rendered.image,
                    rendered.alpha,
                    residual,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "images/initial_rgb": _wandb_image(bundle.initial_rgb, "initial rgb"),
                            "images/target": _wandb_image(target, f"target frame {frame_idx}"),
                            "images/prediction": _wandb_image(rendered.image, f"prediction step {step}"),
                            "images/residual": _wandb_image(residual, f"absolute residual step {step}"),
                            "images/alpha": _wandb_image(rendered.alpha.expand(-1, -1, 3), f"alpha step {step}"),
                        },
                        step=step,
                    )

    with torch.no_grad():
        for frame_idx in range(geometry.shape[0]):
            pred_colors, opacity, sigma, emission = model(frame_idx)
            scales, quats = model.gaussian_geometry()
            rendered = render_frame(bundle, geometry[frame_idx], pred_colors, opacity, sigma, args, scales, quats)
            write_rgb(output_dir / f"pred_frame_{frame_idx:02d}.png", rendered.image)
            write_rgb(output_dir / f"target_frame_{frame_idx:02d}.png", target_images[frame_idx])
            write_rgb(output_dir / f"alpha_frame_{frame_idx:02d}.png", rendered.alpha.expand(-1, -1, 3))
            write_rgb(output_dir / f"emission_frame_{frame_idx:02d}.png", emission.mean(dim=0).view(1, 1, 3).expand_as(rendered.image))

    torch.save(
        {
            "model": model.state_dict(),
            "scene_key": bundle.key,
            "domain": args.domain,
            "geometry": args.geometry,
            "target_source": target_source,
            "dynamic_mask": bundle.dynamic_mask.detach().cpu(),
            "args": vars(args),
        },
        output_dir / "time_dependent_gaussians.pt",
    )
    if wandb_run is not None:
        wandb_run.finish()


def parse_bg_color(value: str) -> tuple[float, float, float]:
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--bg_color must be comma-separated RGB, e.g. 0,0,0")
    if any(x < 0.0 or x > 1.0 for x in parts):
        raise argparse.ArgumentTypeError("--bg_color values must be in [0, 1]")
    return tuple(parts)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["droid", "behavior"], required=True)
    parser.add_argument("--data_dir", required=True, help="Path to the domain WDS directory, e.g. /path/to/droid/wds")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--camera_prefix", default=None, help="Optional original camera prefix to use; deterministic choice if unset")
    parser.add_argument("--output_dir", default="outputs/pointworld_gs_overfit")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--viz_every", type=int, default=100)
    parser.add_argument("--wandb", action="store_true", help="Log training metrics and images to Weights & Biases")
    parser.add_argument("--wandb_project", default="pointworld-gs")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb_log_every", type=int, default=1)

    parser.add_argument("--geometry", choices=["gt", "static"], default="gt")
    parser.add_argument("--geometry_path", default=None, help="Optional .npy/.npz T,N,3 positions from a PointWorld prediction")
    parser.add_argument("--appearance_mode", choices=["full", "dynamic_only", "static_dynamic"], default="full")
    parser.add_argument("--target_mode", choices=["auto", "rendered_points", "future_rgb"], default="auto")
    parser.add_argument("--future_rgb_dir", default=None, help="Optional directory of target RGB frames")
    parser.add_argument(
        "--future_rgb_key_template",
        default=None,
        help="Optional WDS key template, e.g. '{camera}_rgb_{t:06d}'",
    )
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_scene_points", type=int, default=4000)
    parser.add_argument("--max_robot_points", type=int, default=500)
    parser.add_argument("--grid_size", type=float, default=0.015)
    parser.add_argument("--max_relative_movement", type=float, default=0.25)
    parser.add_argument("--dynamic_threshold", type=float, default=0.01)

    parser.add_argument("--render_scale", type=float, default=0.5)
    parser.add_argument("--renderer", choices=["gsplat", "surfel"], default="gsplat")
    parser.add_argument("--sigma_px", type=float, default=1.25)
    parser.add_argument("--init_opacity", type=float, default=0.65)
    parser.add_argument("--init_scale_m", type=float, default=0.01)
    parser.add_argument("--freeze_scales", action="store_true")
    parser.add_argument("--freeze_rotations", action="store_true")
    parser.add_argument("--enable_emission", action="store_true")
    parser.add_argument("--alpha_loss_threshold", type=float, default=0.02)
    parser.add_argument("--depth_temperature", type=float, default=0.0)
    parser.add_argument("--bg_color", type=parse_bg_color, default=(0.0, 0.0, 0.0))

    parser.add_argument("--gsplat_near_plane", type=float, default=0.01)
    parser.add_argument("--gsplat_far_plane", type=float, default=1e10)
    parser.add_argument("--gsplat_radius_clip", type=float, default=0.0)
    parser.add_argument("--gsplat_eps2d", type=float, default=0.3)
    parser.add_argument("--gsplat_packed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gsplat_rasterize_mode", choices=["classic", "antialiased"], default="classic")
    parser.add_argument("--gsplat_absgrad", action="store_true")

    parser.add_argument("--image_loss_weight", type=float, default=1.0)
    parser.add_argument("--point_loss_weight", type=float, default=0.25)
    parser.add_argument("--opacity_reg_weight", type=float, default=1e-3)
    parser.add_argument("--emission_reg_weight", type=float, default=1e-3)
    parser.add_argument("--temporal_reg_weight", type=float, default=1e-4)
    parser.add_argument("--appearance_reg_weight", type=float, default=0.0)
    parser.add_argument("--scale_reg_weight", type=float, default=1e-4)
    parser.add_argument("--rotation_reg_weight", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not (0.0 < args.init_opacity < 1.0):
        raise ValueError("--init_opacity must be in (0, 1)")
    if args.init_scale_m <= 0.0:
        raise ValueError("--init_scale_m must be positive")
    train(args)


if __name__ == "__main__":
    main()
