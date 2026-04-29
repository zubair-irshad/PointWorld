"""Experimental PointWorld + Gaussian splatting utilities."""

from extensions.pointworld_gs.model import TimeDependentGaussianAppearance
from extensions.pointworld_gs.geometry import GeometryState, factorize_geometry
from extensions.pointworld_gs.renderer import (
    RenderOutput,
    project_points,
    render_gaussian_surfels,
    render_gaussians_gsplat,
)

__all__ = [
    "RenderOutput",
    "GeometryState",
    "TimeDependentGaussianAppearance",
    "factorize_geometry",
    "project_points",
    "render_gaussian_surfels",
    "render_gaussians_gsplat",
]
