"""Compact NumPy implementation of radial-angular projection (RAP)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def rap_project(
    points: Array,
    target_dim: int,
    *,
    scale: float = 1.0,
    seed: int = 2_000_003,
) -> Array:
    """Project Lorentz-model points while preserving their radii.

    Args:
        points: Array with shape ``(n, ambient_dim + 1)``. The first column
            is the time coordinate and the remaining columns are spatial.
        target_dim: Number of spatial dimensions after projection.
        scale: Hyperboloid scale. The curvature is ``-1 / scale**2``.
        seed: Random seed for the Gaussian projection.

    Returns:
        Lorentz-model points with shape ``(n, target_dim + 1)``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("points must have shape (n, ambient_dim + 1)")
    if not 1 <= target_dim <= points.shape[1] - 1:
        raise ValueError("target_dim must be between 1 and ambient_dim")
    if scale <= 0:
        raise ValueError("scale must be positive")

    spatial = points[:, 1:]
    spatial_norm = np.linalg.norm(spatial, axis=1)
    directions = np.divide(
        spatial,
        spatial_norm[:, None],
        out=np.zeros_like(spatial),
        where=spatial_norm[:, None] > 0,
    )

    rng = np.random.default_rng(seed)
    projection = rng.normal(
        scale=1.0 / np.sqrt(target_dim),
        size=(target_dim, spatial.shape[1]),
    )
    projected_directions = directions @ projection.T
    projected_norm = np.linalg.norm(projected_directions, axis=1)
    projected_directions = np.divide(
        projected_directions,
        projected_norm[:, None],
        out=np.zeros_like(projected_directions),
        where=projected_norm[:, None] > 0,
    )

    radii = scale * np.arccosh(np.clip(points[:, 0] / scale, 1.0, None))
    time = scale * np.cosh(radii / scale)
    new_spatial = (
        scale * np.sinh(radii / scale)[:, None] * projected_directions
    )
    return np.column_stack((time, new_spatial))


def sample_hyperbolic_points(
    n_points: int,
    ambient_dim: int,
    *,
    radial_scale: float = 2.0,
    scale: float = 1.0,
    seed: int = 0,
) -> Array:
    """Sample synthetic points on the upper Lorentz hyperboloid."""
    if n_points < 2 or ambient_dim < 1:
        raise ValueError("n_points must be >= 2 and ambient_dim must be >= 1")
    if radial_scale < 0 or scale <= 0:
        raise ValueError("radial_scale must be nonnegative and scale positive")

    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_points, ambient_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    radii = radial_scale * np.abs(rng.normal(size=n_points))

    time = scale * np.cosh(radii / scale)
    spatial = scale * np.sinh(radii / scale)[:, None] * directions
    return np.column_stack((time, spatial))


def pairwise_hyperbolic_distances(points: Array, *, scale: float = 1.0) -> Array:
    """Return the full pairwise geodesic-distance matrix."""
    points = np.asarray(points, dtype=np.float64)
    lorentz_gram = points[:, 1:] @ points[:, 1:].T - np.outer(
        points[:, 0], points[:, 0]
    )
    argument = np.maximum(-lorentz_gram / scale**2, 1.0)
    return scale * np.arccosh(argument)


def hyperbolic_radii(points: Array, *, scale: float = 1.0) -> Array:
    """Return each point's distance from the hyperboloid origin."""
    points = np.asarray(points, dtype=np.float64)
    return scale * np.arccosh(np.clip(points[:, 0] / scale, 1.0, None))
