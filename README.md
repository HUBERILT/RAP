# RAP: Radial-Angular Random Projection for Hyperbolic Representations

**Jiahong Liu, Wenhao Yu, Yifan Li, Yanyu Chen, Rex Ying, Menglin Yang, and Irwin King**

This repository provides a compact, dependency-light implementation of
**radial-angular projection (RAP)** for hyperbolic embeddings in the Lorentz
(hyperboloid) model.

RAP reduces the angular dimension with a Gaussian random projection and then
renormalizes the projected direction. The hyperbolic radius of every point is
kept unchanged.

## Quick start

```bash
git clone https://github.com/HUBERILT/RAP.git
cd RAP
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
```

Example output:

```text
RAP Quickstart
Points: 256 | Spatial dimensions: 128 -> 32
Curvature: -1
Maximum radius error: 2.220e-16
Exact pairs evaluated: 32,640
Relative distance distortion:
  mean = 0.0335
  p95  = 0.0980
  p99  = 0.1481
  max  = 0.3560
```

Try a different configuration:

```bash
python demo.py --n-points 128 --ambient-dim 64 --target-dim 16
```

## Use RAP in your code

Input points use the Lorentz-model convention
`(time, spatial coordinates...)` and have shape `(n, ambient_dim + 1)`.

```python
from rap import rap_project

projected_points = rap_project(
    points,
    target_dim=32,
    scale=1.0,  # curvature = -1 / scale**2
    seed=2_000_003,
)
```

The returned array has shape `(n, target_dim + 1)`. `demo.py` includes
synthetic data generation and an exact all-pairs evaluation for a compact,
reproducible example.

## Files

- `rap.py`: RAP plus lightweight Lorentz-model utilities.
- `demo.py`: runnable synthetic example and evaluation.
- `requirements.txt`: NumPy dependency.
