# common/interfaces.py

"""
Shared type aliases and lightweight protocol interfaces used across the
individual model projects (GAN, VAE, Autoregressive, Diffusion, …).

Why this file exists
--------------------
Each project exposes a small, *consistent* surface so orchestration scripts
(app/main.py) and common tooling (evaluation, aggregation) can be reused
without copy/paste. This module keeps that contract explicit and documented.

You can import these helpers from any project:

    from common.interfaces import (
        NDArrayF, OneHot, LogCallback,
        Pipeline, SummaryJSON,
        is_one_hot, assert_image_batch, assert_one_hot,
        latest_weights_in, WEIGHTS_SUFFIX,
    )

Guiding principles
------------------
- **No heavy dependencies**: only stdlib + NumPy + TensorFlow typing.
- **Non-invasive**: projects aren’t forced to inherit concrete base classes;
  they can just *conform* to Protocols.
- **Safe & clear errors**: helper assertions fail with actionable messages.

Notes on shapes & ranges
------------------------
- Images are `(N, H, W, C)` float32 in **[0, 1]** unless a project explicitly
  converts to `[-1, 1]` for training. Keep this convention at the boundaries
  (IO, evaluation).
- Labels are one-hot `(N, K)` float32. Use :func:`is_one_hot` to validate.

Checkpoints
-----------
Keras 3 recommends the suffix `.weights.h5` for weight-only checkpoints.
Use :data:`WEIGHTS_SUFFIX` to avoid typos. The helper :func:`latest_weights_in`
can be used to find the newest/best file given a set of preferred names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Tuple, TypedDict, TypeVar

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
NDArrayF = np.ndarray  # float32 arrays (images, metrics)
OneHot = np.ndarray    # float32 one-hot label arrays (N, K)

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Logging callback signature (used by pipelines to stream metrics)
# -----------------------------------------------------------------------------
# cb(epoch: int, train_loss: float, val_loss: Optional[float]) -> None
LogCallback = Callable[[int, float, Optional[float]], None]


# -----------------------------------------------------------------------------
# Minimal pipeline Protocol all projects adhere to
# -----------------------------------------------------------------------------
class Pipeline(Protocol):
    """
    A training/synthesis pipeline should provide:

    - Attributes:
        cfg: Dict[str, Any]
        ckpt_dir: Path
        synth_dir: Path
        log_cb: Optional[LogCallback]

    - Methods:
        train(x_train, y_train, x_val?, y_val?) -> tf.keras.Model
        synthesize(model: Optional[tf.keras.Model] = None)
            -> Tuple[NDArrayF, OneHot]
    """

    cfg: Dict[str, Any]
    ckpt_dir: Path
    synth_dir: Path
    log_cb: Optional[LogCallback]

    def train(
        self,
        x_train: NDArrayF,
        y_train: OneHot,
        x_val: Optional[NDArrayF] = None,
        y_val: Optional[OneHot] = None,
    ) -> tf.keras.Model: ...

    def synthesize(
        self,
        model: Optional[tf.keras.Model] = None,
    ) -> Tuple[NDArrayF, OneHot]: ...


# -----------------------------------------------------------------------------
# Tiny dataclasses for clarity in function signatures (optional)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainSplit:
    x: NDArrayF
    y: OneHot


@dataclass(frozen=True)
class EvalSplits:
    train: TrainSplit
    val: TrainSplit
    test: TrainSplit


# -----------------------------------------------------------------------------
# JSON summary (evaluation) envelope — intentionally loose-typed
# -----------------------------------------------------------------------------
class SummaryJSON(TypedDict, total=False):
    """
    Minimal schema used by the evaluator & aggregation tooling. Kept generic
    so projects can add fields without breaking type-checkers.
    """
    model: str
    seed: int
    images: Dict[str, int]
    generative: Dict[str, Any]
    utility_real_only: Dict[str, Any]
    utility_real_plus_synth: Dict[str, Any]
    deltas_RS_minus_R: Dict[str, Any]


# -----------------------------------------------------------------------------
# Shape / validity helpers
# -----------------------------------------------------------------------------
def is_one_hot(y: np.ndarray, *, num_classes: Optional[int] = None) -> bool:
    """
    Return True if `y` looks like a proper one-hot array of shape (N, K).
    - Each row sums (approximately) to 1.
    - All entries are in [0, 1].
    - If `num_classes` is given, checks K == num_classes.
    """
    if y.ndim != 2:
        return False
    if num_classes is not None and y.shape[1] != num_classes:
        return False
    # numerical tolerance for floats
    row_sums = y.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        return False
    if np.min(y) < -1e-6 or np.max(y) > 1.0 + 1e-6:
        return False
    return True


def assert_image_batch(x: np.ndarray, *, H: int, W: int, C: int, range01: bool = True) -> None:
    """
    Assert that `x` has shape (N, H, W, C) and, if `range01` is True, that it’s
    inside [0, 1] (with a small tolerance).
    """
    if x.ndim != 4 or x.shape[1:] != (H, W, C):
        raise ValueError(f"Expected images of shape (N,{H},{W},{C}), got {x.shape}.")
    if range01:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmin < -1e-3 or xmax > 1.0 + 1e-3:
            raise ValueError(
                f"Expected pixel range [0,1], got min={xmin:.4f}, max={xmax:.4f}. "
                "Ensure you normalized/reshaped data correctly."
            )


def assert_one_hot(y: np.ndarray, *, K: int) -> None:
    """Raise a clear error if labels are not one-hot with K classes."""
    if not is_one_hot(y, num_classes=K):
        raise ValueError(
            f"Labels must be one-hot with K={K}. Got shape {y.shape} and row sums "
            f"in [{y.sum(axis=1).min():.3f}, {y.sum(axis=1).max():.3f}]."
        )


# -----------------------------------------------------------------------------
# Checkpoint helpers (agnostic of model type)
# -----------------------------------------------------------------------------
WEIGHTS_SUFFIX = ".weights.h5"  # Keras 3-recommended suffix for weight-only saves


def latest_weights_in(
    directory: Path,
    *,
    prefer: Iterable[str] = (),
    pattern: str = f"*{WEIGHTS_SUFFIX}",
) -> Optional[Path]:
    """
    Return the most suitable weights file in `directory`.

    Selection strategy
    ------------------
    1) If `prefer` contains filenames that exist, return the first match.
       e.g., prefer=("G_best.weights.h5", "AR_best.weights.h5", "D_best.weights.h5")
    2) Otherwise, return the most recently modified file matching `pattern`.
    3) If none found, return None.

    Examples
    --------
    >>> latest = latest_weights_in(Path("artifacts/checkpoints"),
    ...                            prefer=("AR_best.weights.h5", "AR_last.weights.h5"))
    """
    directory = Path(directory)
    # 1) Preferred names
    for name in prefer:
        p = directory / name
        if p.exists():
            return p

    # 2) Newest matching file
    matches = sorted(directory.glob(pattern))
    if matches:
        return max(matches, key=lambda p: p.stat().st_mtime)

    return None
