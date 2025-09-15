# diffusion/__init__.py

"""
Conditional Diffusion (class-conditioned DDPM) package.

What this package provides
--------------------------
- :func:`build_diffusion_model` (in :mod:`diffusion.models`):
    Builds a conditional diffusion U-Net that predicts noise ε_θ(x_t, t, y).
    The model is compiled (MSE on noise) and ready for training or weight loading.

- :class:`DiffusionPipeline` (in :mod:`diffusion.pipeline`):
    High-level orchestration for training, checkpointing, and synthesizing
    per-class samples to disk. Synthesis writes files under
    ``ARTIFACTS/synthetic`` using a consistent contract:

      * ``gen_class_{k}.npy``     – float32 images in [0, 1], shape (N_k, H, W, C)
      * ``labels_class_{k}.npy``  – int32 labels (all = k), shape (N_k,)

    This contract is consumed by the shared evaluator so results are comparable
    across your other model repos.

- :func:`save_grid_from_model` (optional, in :mod:`diffusion.sample`):
    Convenience helper that renders a quick PNG grid (one sample per class)
    for logs and sanity checks.

Typical usage
-------------
>>> from diffusion import build_diffusion_model, DiffusionPipeline
>>> model = build_diffusion_model(img_shape=(40, 40, 1), num_classes=9)
>>> pipe = DiffusionPipeline(cfg)
>>> model = pipe.train(x_train, y_train, x_val, y_val)
>>> x_s, y_s = pipe.synthesize(model)  # also persists per-class .npy files

The heavy lifting lives in the submodules; this package file exposes a tidy API.
"""

from __future__ import annotations

# Public API re-exports
from .models import build_diffusion_model
from .pipeline import DiffusionPipeline

# Optional helper (import softly to avoid hard dependency during tests/CLI)
try:
    from .sample import save_grid_from_model  # type: ignore
except Exception:  # pragma: no cover - helper is optional
    save_grid_from_model = None  # type: ignore

__all__ = [
    "build_diffusion_model",
    "DiffusionPipeline",
    "save_grid_from_model",
]

# Simple package version (override via CI/build if desired)
__version__ = "0.1.0"
