# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Visualize PointWorld-GS WDS samples and 3D flow/RGB alignment.

This script intentionally uses the same WDS scene loader as overfit_scene.py so
sample/camera/frame indexing matches the Gaussian overfit path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from extensions.pointworld_gs.data import SceneBundle, load_scene_bundle
from extensions.pointworld_gs.renderer import project_points


def _tensor_rgb_to_u8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)


def _label(image: np.ndarray, text: str, org: tuple[int, int] = (8, 22)) -> np.ndarray:
    out = np.ascontiguousarray(image.copy())
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image[..., ::-1])


def _open_video(path: Path, fps: float, frame_hw: tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame_hw
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def _colorize_motion(vectors: np.ndarray, max_magnitude: float | None = None) -> np.ndarray:
    """Map 2D flow vectors to RGB with hue=direction and value=magnitude."""
    if vectors.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    dx = vectors[:, 0].astype(np.float32)
    dy = vectors[:, 1].astype(np.float32)
    magnitude = np.sqrt(dx * dx + dy * dy)
    if max_magnitude is None:
        finite_mag = magnitude[np.isfinite(magnitude)]
        max_magnitude = float(np.percentile(finite_mag, 95)) if finite_mag.size else 1.0
    max_magnitude = max(float(max_magnitude), 1e-6)
    angle = (np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)

    hsv = np.zeros((vectors.shape[0], 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.clip(angle * 179.0, 0, 179).astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = np.clip(64.0 + 191.0 * np.minimum(magnitude / max_magnitude, 1.0), 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    return bgr[:, ::-1].copy()


def _time_colors(num_frames: int) -> np.ndarray:
    values = np.linspace(0, 179, max(num_frames, 1), dtype=np.uint8)
    hsv = np.zeros((num_frames, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = values
    hsv[:, 0, 1] = 220
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]
    return bgr[:, ::-1].copy()


@torch.no_grad()
def _project_all(bundle: SceneBundle, positions: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = []
    depths = []
    valids = []
    height, width = bundle.initial_rgb.shape[:2]
    for t in range(positions.shape[0]):
        xy, z = project_points(positions[t], bundle.intrinsic, bundle.extrinsic)
        valid = (
            torch.isfinite(xy).all(dim=-1)
            & torch.isfinite(z)
            & (z > 0.01)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        pixels.append(xy.detach().cpu().numpy().astype(np.float32))
        depths.append(z.detach().cpu().numpy().astype(np.float32))
        valids.append(valid.detach().cpu().numpy().astype(bool))
    return np.stack(pixels, axis=0), np.stack(depths, axis=0), np.stack(valids, axis=0)


def _select_overlay_indices(
    _pixels: np.ndarray,
    valid: np.ndarray,
    positions: np.ndarray,
    max_points: int,
    seed: int,
    strategy: str,
) -> np.ndarray:
    num_points = positions.shape[1]
    if max_points <= 0 or num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)

    visible_count = valid.sum(axis=0)
    visible_any = visible_count > 0
    if not np.any(visible_any):
        visible_any[:] = True

    displacement = np.linalg.norm(positions - positions[:1], axis=-1)
    motion_score = np.nanmax(displacement, axis=0)
    rng = np.random.RandomState(seed)

    candidate = np.nonzero(visible_any)[0]
    if candidate.shape[0] <= max_points:
        return np.sort(candidate.astype(np.int64))

    if strategy == "motion":
        order = candidate[np.argsort(-motion_score[candidate], kind="stable")]
        return np.sort(order[:max_points].astype(np.int64))

    return np.sort(rng.choice(candidate, size=max_points, replace=False).astype(np.int64))


def _draw_points(
    image: np.ndarray,
    xy: np.ndarray,
    colors: np.ndarray,
    valid: np.ndarray,
    radius: int,
    alpha: float,
) -> np.ndarray:
    overlay = image.copy()
    for point, color, is_valid in zip(xy, colors, valid):
        if not is_valid:
            continue
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(overlay, (x, y), radius, tuple(int(c) for c in color), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def _draw_arrows(
    image: np.ndarray,
    xy0: np.ndarray,
    xy1: np.ndarray,
    valid: np.ndarray,
    colors: np.ndarray,
    line_width: int,
    min_px_motion: float,
    alpha: float,
) -> np.ndarray:
    overlay = image.copy()
    for start, end, is_valid, color in zip(xy0, xy1, valid, colors):
        if not is_valid:
            continue
        motion = float(np.linalg.norm(end - start))
        if motion < min_px_motion:
            continue
        p0 = (int(round(float(start[0]))), int(round(float(start[1]))))
        p1 = (int(round(float(end[0]))), int(round(float(end[1]))))
        rgb = tuple(int(c) for c in color)
        cv2.arrowedLine(overlay, p0, p1, rgb, line_width, cv2.LINE_AA, tipLength=0.25)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def _draw_tracks(
    image: np.ndarray,
    pixels: np.ndarray,
    valid: np.ndarray,
    indices: np.ndarray,
    frame_idx: int,
    line_width: int,
    alpha: float,
) -> np.ndarray:
    overlay = image.copy()
    colors = _time_colors(frame_idx + 1)
    for point_idx in indices:
        for t in range(frame_idx):
            if not (valid[t, point_idx] and valid[t + 1, point_idx]):
                continue
            p0 = tuple(np.round(pixels[t, point_idx]).astype(np.int32).tolist())
            p1 = tuple(np.round(pixels[t + 1, point_idx]).astype(np.int32).tolist())
            rgb = tuple(int(c) for c in colors[t + 1])
            cv2.line(overlay, p0, p1, rgb, line_width, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)


def _stats_panel(
    height: int,
    width: int,
    rows: list[tuple[str, str]],
) -> np.ndarray:
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = 24
    for key, value in rows:
        text = f"{key}: {value}"
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (35, 35, 35), 1, cv2.LINE_AA)
        y += 22
    return panel


def _make_rgb_montage(frames: np.ndarray, max_cols: int = 4) -> np.ndarray:
    num_frames, height, width = frames.shape[:3]
    cols = min(max_cols, num_frames)
    rows = int(np.ceil(num_frames / cols))
    canvas = np.full((rows * height, cols * width, 3), 20, dtype=np.uint8)
    for t in range(num_frames):
        r, c = divmod(t, cols)
        tile = _label(frames[t], f"rgb[{t:02d}]")
        canvas[r * height : (r + 1) * height, c * width : (c + 1) * width] = tile
    return canvas


def _frame_rgb_sequence(bundle: SceneBundle) -> tuple[np.ndarray, str]:
    initial = _tensor_rgb_to_u8(bundle.initial_rgb)
    if bundle.target_rgb is None:
        repeated = np.broadcast_to(initial[None], (bundle.positions.shape[0], *initial.shape)).copy()
        return repeated, "initial_rgb_repeated_no_future_rgb"
    return _tensor_rgb_to_u8(bundle.target_rgb), "future_rgb"


def visualize(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_scene_bundle(args)
    positions = bundle.positions
    colors = bundle.colors
    if not args.include_depth_augmented_points and bundle.num_depth_aug_points > 0:
        positions = positions[:, : bundle.num_trajectory_points]
        colors = colors[:, : bundle.num_trajectory_points]

    if args.max_frames is not None:
        positions = positions[: args.max_frames]
        colors = colors[: args.max_frames]
        if bundle.target_rgb is not None:
            bundle.target_rgb = bundle.target_rgb[: args.max_frames]

    rgb_frames, rgb_source = _frame_rgb_sequence(bundle)
    frames = min(positions.shape[0], rgb_frames.shape[0])
    positions = positions[:frames]
    colors = colors[:frames]
    rgb_frames = rgb_frames[:frames]

    pixels, depths, valid = _project_all(bundle, positions)
    positions_np = positions.detach().cpu().numpy().astype(np.float32)
    colors_np = (colors.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    indices = _select_overlay_indices(
        pixels,
        valid,
        positions_np,
        args.max_overlay_points,
        args.seed,
        args.sample_strategy,
    )

    height, width = rgb_frames.shape[1:3]
    video_frames = []
    per_frame_stats = []
    for t in range(frames):
        rgb_t = rgb_frames[t]
        current_valid = valid[t, indices]
        current_points = _draw_points(
            rgb_t,
            pixels[t, indices],
            colors_np[t, indices],
            current_valid,
            radius=args.point_radius,
            alpha=args.point_alpha,
        )

        if t == 0:
            step_valid = current_valid
            step_vectors = np.zeros((indices.shape[0], 2), dtype=np.float32)
            step_overlay = _label(rgb_t, "no t-1 flow for frame 0")
        else:
            step_valid = valid[t - 1, indices] & valid[t, indices]
            step_vectors = pixels[t, indices] - pixels[t - 1, indices]
            step_colors = _colorize_motion(step_vectors, args.max_motion_color_px)
            step_overlay = _draw_arrows(
                rgb_t,
                pixels[t - 1, indices],
                pixels[t, indices],
                step_valid,
                step_colors,
                line_width=args.line_width,
                min_px_motion=args.min_px_motion,
                alpha=args.arrow_alpha,
            )

        chunk_valid = valid[0, indices] & valid[t, indices]
        chunk_vectors = pixels[t, indices] - pixels[0, indices]
        chunk_colors = _colorize_motion(chunk_vectors, args.max_motion_color_px)
        chunk_overlay = _draw_arrows(
            rgb_frames[0],
            pixels[0, indices],
            pixels[t, indices],
            chunk_valid,
            chunk_colors,
            line_width=args.line_width,
            min_px_motion=args.min_px_motion,
            alpha=args.arrow_alpha,
        )
        track_overlay = _draw_tracks(
            rgb_frames[0],
            pixels,
            valid,
            indices,
            t,
            line_width=args.line_width,
            alpha=args.arrow_alpha,
        )

        step_motion = np.linalg.norm(step_vectors[step_valid], axis=-1) if np.any(step_valid) else np.array([])
        chunk_motion = np.linalg.norm(chunk_vectors[chunk_valid], axis=-1) if np.any(chunk_valid) else np.array([])
        frame_stats = {
            "frame": int(t),
            "rgb_source": rgb_source,
            "overlay_points": int(indices.shape[0]),
            "visible_points": int(current_valid.sum()),
            "step_visible_flows": int(step_valid.sum()),
            "chunk_visible_flows": int(chunk_valid.sum()),
            "step_motion_px_mean": float(step_motion.mean()) if step_motion.size else 0.0,
            "step_motion_px_p95": float(np.percentile(step_motion, 95)) if step_motion.size else 0.0,
            "chunk_motion_px_mean": float(chunk_motion.mean()) if chunk_motion.size else 0.0,
            "chunk_motion_px_p95": float(np.percentile(chunk_motion, 95)) if chunk_motion.size else 0.0,
            "depth_median": float(np.nanmedian(depths[t, valid[t]])) if np.any(valid[t]) else 0.0,
        }
        per_frame_stats.append(frame_stats)

        stats = _stats_panel(
            height,
            width,
            [
                ("sample", bundle.key[-38:]),
                ("camera", bundle.selected_camera),
                ("frame", f"{t}/{frames - 1}"),
                ("rgb", rgb_source),
                ("points", f"{positions.shape[1]} total, {indices.shape[0]} drawn"),
                ("visible", f"{frame_stats['visible_points']} drawn"),
                ("step flow", f"{frame_stats['step_visible_flows']} valid, p95 {frame_stats['step_motion_px_p95']:.1f}px"),
                ("chunk flow", f"{frame_stats['chunk_visible_flows']} valid, p95 {frame_stats['chunk_motion_px_p95']:.1f}px"),
                ("median depth", f"{frame_stats['depth_median']:.2f}m"),
            ],
        )

        panels = [
            _label(rgb_t, f"rgb[{t:02d}]"),
            _label(current_points, f"projected 3D points[{t:02d}]"),
            _label(step_overlay, f"2D flow [{max(t - 1, 0):02d}->{t:02d}]"),
            _label(chunk_overlay, f"chunk flow [00->{t:02d}]"),
            _label(track_overlay, f"track history <= {t:02d}"),
            stats,
        ]
        grid = np.concatenate(
            [
                np.concatenate(panels[:3], axis=1),
                np.concatenate(panels[3:], axis=1),
            ],
            axis=0,
        )
        video_frames.append(grid)
        _write_rgb(output_dir / f"frame_{t:02d}_grid.png", grid)
        _write_rgb(output_dir / f"frame_{t:02d}_points.png", current_points)
        _write_rgb(output_dir / f"frame_{t:02d}_step_flow.png", step_overlay)
        _write_rgb(output_dir / f"frame_{t:02d}_chunk_flow.png", chunk_overlay)

    montage = _make_rgb_montage(rgb_frames, max_cols=args.montage_cols)
    _write_rgb(output_dir / "rgb_montage.png", montage)

    if args.write_video and video_frames:
        fps = float(args.fps)
        writer = _open_video(output_dir / "dataset_alignment.mp4", fps, video_frames[0].shape[:2])
        for frame in video_frames:
            writer.write(frame[..., ::-1])
        writer.release()

        rgb_writer = _open_video(output_dir / "rgb_sequence.mp4", fps, rgb_frames[0].shape[:2])
        for frame in rgb_frames:
            rgb_writer.write(frame[..., ::-1])
        rgb_writer.release()

    report = {
        "sample_key": bundle.key,
        "domain": args.domain,
        "split": args.split,
        "sample_index": args.sample_index,
        "camera": bundle.selected_camera,
        "rgb_source": rgb_source,
        "num_frames": int(frames),
        "num_points_visualized": int(positions.shape[1]),
        "num_original_trajectory_points": int(bundle.num_trajectory_points),
        "num_depth_augmented_points": int(bundle.num_depth_aug_points),
        "include_depth_augmented_points": bool(args.include_depth_augmented_points),
        "render_size": [int(width), int(height)],
        "per_frame": per_frame_stats,
    }
    with (output_dir / "alignment_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"Wrote dataset visualization for {bundle.key} camera={bundle.selected_camera} "
        f"T={frames} points={positions.shape[1]} rgb={rgb_source} to {output_dir}"
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["droid", "behavior"], required=True)
    parser.add_argument("--data_dir", required=True, help="Path to WDS root containing train/test shard directories")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--camera_prefix", default=None)
    parser.add_argument("--output_dir", default="outputs/pointworld_gs/dataset_viz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render_scale", type=float, default=0.5)
    parser.add_argument("--max_frames", type=int, default=11)
    parser.add_argument("--max_scene_points", type=int, default=50000)
    parser.add_argument("--dynamic_threshold", type=float, default=0.01)
    parser.add_argument("--future_rgb_dir", default=None)
    parser.add_argument("--future_rgb_key_template", default=None)
    parser.add_argument("--augment_initial_depth_points", action="store_true")
    parser.add_argument("--include_depth_augmented_points", action="store_true")
    parser.add_argument("--depth_point_stride", type=int, default=1)
    parser.add_argument("--max_depth_points", type=int, default=0)
    parser.add_argument("--depth_min_m", type=float, default=0.05)
    parser.add_argument("--depth_max_m", type=float, default=5.0)

    parser.add_argument("--max_overlay_points", type=int, default=2500)
    parser.add_argument("--sample_strategy", choices=["motion", "uniform"], default="motion")
    parser.add_argument("--point_radius", type=int, default=1)
    parser.add_argument("--line_width", type=int, default=1)
    parser.add_argument("--min_px_motion", type=float, default=0.25)
    parser.add_argument("--max_motion_color_px", type=float, default=None)
    parser.add_argument("--point_alpha", type=float, default=0.75)
    parser.add_argument("--arrow_alpha", type=float, default=0.85)
    parser.add_argument("--montage_cols", type=int, default=4)
    parser.add_argument("--write_video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fps", type=float, default=3.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
