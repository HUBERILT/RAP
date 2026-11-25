"""Run a small, exact all-pairs RAP demonstration."""

from __future__ import annotations

import argparse

import numpy as np

from rap import (
    hyperbolic_radii,
    pairwise_hyperbolic_distances,
    rap_project,
    sample_hyperbolic_points,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=256)
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--target-dim", type=int, default=32)
    parser.add_argument("--radial-scale", type=float, default=2.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--projection-seed", type=int, default=2_000_003)
    args = parser.parse_args()

    points = sample_hyperbolic_points(
        args.n_points,
        args.ambient_dim,
        radial_scale=args.radial_scale,
        scale=args.scale,
        seed=args.data_seed,
    )
    projected = rap_project(
        points,
        args.target_dim,
        scale=args.scale,
        seed=args.projection_seed,
    )

    original_distances = pairwise_hyperbolic_distances(points, scale=args.scale)
    projected_distances = pairwise_hyperbolic_distances(
        projected, scale=args.scale
    )
    pair_indices = np.triu_indices(args.n_points, k=1)
    original_pairs = original_distances[pair_indices]
    projected_pairs = projected_distances[pair_indices]
    distortion = np.abs(projected_pairs - original_pairs) / original_pairs

    radius_error = np.max(
        np.abs(
            hyperbolic_radii(points, scale=args.scale)
            - hyperbolic_radii(projected, scale=args.scale)
        )
    )

    print("RAP Quickstart")
    print(
        f"Points: {args.n_points} | Spatial dimensions: "
        f"{args.ambient_dim} -> {args.target_dim}"
    )
    print(f"Curvature: {-1.0 / args.scale**2:g}")
    print(f"Maximum radius error: {radius_error:.3e}")
    print(f"Exact pairs evaluated: {len(distortion):,}")
    print("Relative distance distortion:")
    print(f"  mean = {np.mean(distortion):.4f}")
    print(f"  p95  = {np.quantile(distortion, 0.95):.4f}")
    print(f"  p99  = {np.quantile(distortion, 0.99):.4f}")
    print(f"  max  = {np.max(distortion):.4f}")


if __name__ == "__main__":
    main()
