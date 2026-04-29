# PointWorld-GS overfit extension

This extension overfits a time-dependent Gaussian appearance model to one DROID
or BEHAVIOR scene using PointWorld-style trajectories.

It is intended as a careful first step toward photometric PointWorld rollouts:

- geometry is supplied by scene point trajectories;
- Gaussian colors, opacity, anisotropic scale, rotation, and optional emission
  are optimized over time;
- targets can be rendered point-color supervision from the current release WDS,
  or real future RGB frames from an extended dataset;
- the default renderer is true CUDA 3DGS via `gsplat`; a pure PyTorch
  screen-space surfel fallback remains available for debugging.

## Why time-dependent Gaussians

The useful representation is not a separate Gaussian scene for every timestep.
Use a canonical Gaussian set at `t=0`, move Gaussian centers with
PointWorld/ground-truth point flow, and let appearance parameters vary with time:

```text
mu_i,t = mu_i,0 + flow_i,t
rgb_i,t, alpha_i,t, emission_i,t = appearance(i, t)
scale_i, quat_i = learned Gaussian covariance parameters
```

Static/dynamic separation is useful, but only if it separates geometry from
appearance. A screen or light can have static geometry and dynamic appearance.
The script exposes:

- `--geometry gt`: move centers with the sample trajectory;
- `--geometry static`: freeze centers for an appearance-only ablation;
- `--geometry static_dynamic`: freeze flow-static centers and keep moving
  centers deformable;
- `--geometry rigid_clusters`: freeze static centers, fit local SE(3) motion to
  flow-derived dynamic clusters, and leave small/nonrigid dynamic leftovers
  deformable;
- `--geometry_path`: use saved `T,N,3` PointWorld-predicted positions from
  `.npy` or `.npz` instead of the sample trajectory;
- `--dynamic_mask_path`: optional boolean point mask from external tools,
  e.g. projected SAM/GroundingDINO/VLM object masks;
- `--robot_mask_path`: optional boolean point mask for robot-owned points. The
  current branch does not generate this automatically from URDF yet, but the
  factorizer accepts it so URDF projection or link-sampled masks can be plugged
  in without changing the trainer;
- `--augment_initial_depth_points`: back-project the selected camera's initial
  RGB-D frame into additional static Gaussians. This is useful for future-RGB
  supervision because the tracked trajectory points may not cover all visible
  pixels;
- `--appearance_mode full`: every point can change appearance;
- `--appearance_mode dynamic_only`: only geometrically moving points can change;
- `--appearance_mode static_dynamic`: learned static/dynamic color biases plus
  per-point residuals.

## Files

- `data.py`: loads one WDS sample, selects a camera, preserves trajectories, and
  optionally reads future RGB targets.
- `geometry.py`: flow-based static/dynamic disentanglement and local rigid
  cluster fitting.
- `model.py`: time-dependent Gaussian appearance parameters.
- `renderer.py`: `gsplat` 3DGS renderer plus pure-PyTorch surfel fallback.
- `viz.py`: image/grid writing helpers.
- `overfit_scene.py`: training entry point.

## Renderer backends

`--renderer gsplat` is the default. It calls
`gsplat.rendering.rasterization` with:

- means: current PointWorld/ground-truth trajectory positions, `T,N,3`;
- quaternions: learned `wxyz` Gaussian rotations;
- scales: learned anisotropic `N,3` metric scales;
- opacities: learned per-Gaussian alpha;
- colors: RGB plus optional direct-radiance/emission residual;
- `viewmats`: the sample's world-to-camera extrinsic;
- `Ks`: the resized sample camera intrinsics.

Install `gsplat` in the same environment as PointWorld and run on CUDA:

```bash
pip install gsplat
```

If the workstation does not have CUDA/`gsplat`, use `--renderer surfel` to run
the older differentiable PyTorch renderer. The surfel renderer ignores learned
3D scale and rotation, so it is only a debugging fallback.

## Current release target mode

The public PointWorld release path is built around initial RGB-D plus per-point
trajectories/colors. In that setting, the script renders target images from the
known future point colors:

```bash
conda activate pointworld-env

python extensions/pointworld_gs/overfit_scene.py \
  --domain droid \
  --data_dir /path/to/droid/wds \
  --split test \
  --sample_index 0 \
  --renderer gsplat \
  --device cuda \
  --max_scene_points 4000 \
  --max_frames 8 \
  --render_scale 0.5 \
  --init_scale_m 0.01 \
  --steps 1000 \
  --enable_emission \
  --target_mode rendered_points \
  --output_dir outputs/pointworld_gs/droid_sample0
```

Add `--wandb` to stream scalar metrics and periodic visualizations:

```bash
wandb login

python extensions/pointworld_gs/overfit_scene.py \
  --domain droid \
  --data_dir /path/to/droid/wds \
  --split test \
  --sample_index 0 \
  --renderer gsplat \
  --device cuda \
  --target_mode auto \
  --max_frames 8 \
  --max_scene_points 4000 \
  --steps 1000 \
  --enable_emission \
  --appearance_mode static_dynamic \
  --wandb \
  --wandb_project pointworld-gs \
  --wandb_run_name droid_sample0
```

For future-RGB overfitting, a better first geometry prior is usually
`static_dynamic` or `rigid_clusters`:

```bash
python extensions/pointworld_gs/overfit_scene.py \
  --domain droid \
  --data_dir /path/to/droid/wds \
  --split test \
  --sample_index 0 \
  --renderer gsplat \
  --device cuda \
  --geometry static_dynamic \
  --augment_initial_depth_points \
  --depth_point_stride 1 \
  --target_mode auto \
  --max_frames 8 \
  --max_scene_points 50000 \
  --render_scale 0.5 \
  --init_scale_m 0.006 \
  --steps 3000 \
  --lr 0.01 \
  --point_loss_weight 0.0 \
  --opacity_reg_weight 0.0 \
  --scale_reg_weight 0.0 \
  --rotation_reg_weight 0.0 \
  --enable_emission \
  --appearance_mode full \
  --wandb \
  --wandb_project pointworld-gs \
  --wandb_run_name droid_sample0_static_dynamic
```

To test local rigid primitives for robot/object-like motion:

```bash
python extensions/pointworld_gs/overfit_scene.py \
  --domain droid \
  --data_dir /path/to/droid/wds \
  --split test \
  --sample_index 0 \
  --renderer gsplat \
  --device cuda \
  --geometry rigid_clusters \
  --augment_initial_depth_points \
  --depth_point_stride 1 \
  --cluster_spatial_voxel_m 0.06 \
  --cluster_motion_voxel_m 0.01 \
  --min_cluster_points 32 \
  --target_mode auto \
  --max_frames 8 \
  --max_scene_points 50000 \
  --render_scale 0.5 \
  --init_scale_m 0.006 \
  --steps 3000 \
  --lr 0.01 \
  --point_loss_weight 0.0 \
  --opacity_reg_weight 0.0 \
  --scale_reg_weight 0.0 \
  --rotation_reg_weight 0.0 \
  --enable_emission \
  --appearance_mode full \
  --wandb \
  --wandb_project pointworld-gs \
  --wandb_run_name droid_sample0_rigid_clusters
```

For BEHAVIOR:

```bash
python extensions/pointworld_gs/overfit_scene.py \
  --domain behavior \
  --data_dir /path/to/behavior/wds \
  --split test \
  --sample_index 0 \
  --renderer gsplat \
  --device cuda \
  --max_scene_points 4000 \
  --max_frames 8 \
  --render_scale 0.5 \
  --init_scale_m 0.01 \
  --steps 1000 \
  --enable_emission \
  --target_mode rendered_points \
  --output_dir outputs/pointworld_gs/behavior_sample0
```

## Future RGB target mode

For actual photometric dynamics, train against future RGB frames. You can point
the script at a directory:

```bash
python extensions/pointworld_gs/overfit_scene.py \
  --domain droid \
  --data_dir /path/to/droid/wds \
  --renderer gsplat \
  --device cuda \
  --future_rgb_dir /path/to/frames_for_the_selected_camera \
  --target_mode future_rgb \
  --max_frames 8 \
  --steps 2000 \
  --enable_emission \
  --appearance_mode static_dynamic \
  --output_dir outputs/pointworld_gs/droid_future_rgb
```

To use saved PointWorld-predicted geometry instead of ground-truth scene
trajectories:

```bash
python extensions/pointworld_gs/overfit_scene.py \
  --domain droid \
  --data_dir /path/to/droid/wds \
  --renderer gsplat \
  --device cuda \
  --geometry_path /path/to/predicted_positions.npz \
  --target_mode rendered_points \
  --steps 1000 \
  --output_dir outputs/pointworld_gs/droid_pred_geometry
```

The geometry file must contain `positions` or `scene_flows` with shape `T,N,3`
and the same point count as the selected processed sample.

Or use an extended WDS key template:

```bash
python extensions/pointworld_gs/overfit_scene.py \
  --domain droid \
  --data_dir /path/to/droid/wds \
  --future_rgb_key_template '{camera}_rgb_{t:06d}' \
  --target_mode future_rgb \
  --max_frames 8 \
  --steps 2000 \
  --enable_emission
```

The template gets `camera`, `cam`, `t`, and `frame` variables. For example,
`{camera}_rgb_{t:06d}` with selected camera `camera_0` reads
`camera_0_rgb_000000`, `camera_0_rgb_000001`, and so on.

## Outputs

The output directory contains:

- `metadata.json`: sample/camera/target configuration;
- `geometry_groups.npz`: dynamic mask, optional robot mask, and rigid-cluster
  ids used for the run;
- `metrics.csv`: per-step loss and PSNR estimate;
- `train_step_*_frame_*.png`: initial/target/prediction/error/alpha grid;
- `pred_frame_*.png`: final rendered predictions;
- `target_frame_*.png`: final target frames;
- `alpha_frame_*.png`: final alpha masks;
- `emission_frame_*.png`: global emission-color summary image;
- `time_dependent_gaussians.pt`: trained appearance parameters and run args.

## 3DGS implementation details

The swap point remains `renderer.py`. The training script only expects
`render_frame(...)` to return image, alpha, and depth-like tensors. The `gsplat`
path now replaces screen-space isotropic kernels with:

- 3D Gaussian means from PointWorld trajectories;
- learned anisotropic scale and rotation;
- opacity;
- RGB color, with optional emission/direct-radiance folded into the rendered
  color;
- `gsplat` camera projection and alpha compositing.

Useful flags:

- `--init_scale_m`: metric Gaussian scale initialization;
- `--freeze_scales`, `--freeze_rotations`: ablate learned covariance;
- `--scale_reg_weight`, `--rotation_reg_weight`: keep the learned covariance
  close to initialization;
- `--gsplat_rasterize_mode antialiased`: use gsplat's antialiased mode;
- `--gsplat_radius_clip`: skip tiny projected Gaussians for speed;
- `--gsplat_eps2d`: minimum projected covariance stabilization;
- `--gsplat_packed/--no-gsplat_packed`: memory/runtime tradeoff.

The model/data/training loop can stay largely the same if SH coefficients are
added later; replace `colors=colors` in `render_gaussians_gsplat(...)` with SH
coefficients and pass `sh_degree`.
