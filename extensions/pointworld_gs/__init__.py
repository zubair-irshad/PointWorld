"""Experimental PointWorld + Gaussian splatting utilities."""

from extensions.pointworld_gs.model import TimeDependentGaussianAppearance
from extensions.pointworld_gs.renderer import (
    RenderOutput,
    project_points,
    render_gaussian_surfels,
    render_gaussians_gsplat,
)

__all__ = [
    "RenderOutput",
    "TimeDependentGaussianAppearance",
    "project_points",
    "render_gaussian_surfels",
    "render_gaussians_gsplat",
]
