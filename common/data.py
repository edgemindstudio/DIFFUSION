# common/data.py

"""
Common data utilities shared across projects (GAN, VAE, Autoregressive, Diffusion).

This module standardizes how we:
- Load the malware dataset from .npy files
- Normalize & reshape images to (H, W, C)
- One-hot encode labels
- Create tf.data pipelines for efficient training/eval

It is intentionally small, dependency-light, and model-agnostic.

File contract (under DATA_DIR)
------------------------------
- train_data.npy      (N_train, H*W*C) or (N_train, H, W, C)
- train_labels.npy    (N_train,) integer class IDs
- test_data.npy       (N_test,  H*W*C) or (N_test,  H, W, C)
- test_labels.npy     (N_test,)  integer class IDs

Returned splits
---------------
load_dataset_npy(...) returns 6 NumPy arrays:
  (x_train, y_train, x_val, y_val, x_test, y_test)
Images are float32 in [0, 1] and HWC-shaped. Labels are one-hot.

Notes
-----
- Models that expect [-1, 1] (e.g., tanh decoders) can call `to_minus1_1(x01)`
  after loading; for display or metrics that require [0, 1], keep as-is.
- If your .npy images are already in [0, 1], we preserve that range (only clip).
- For reproducible experiments, consider setting seeds in your app/main.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Encoding / Scaling helpers
# -----------------------------------------------------------------------------
def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Ensure labels are one-hot encoded (float32).
    Accepts int class IDs of shape (N,) or already one-hot shape (N, K).
    """
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32", copy=False)
    return tf.keras.utils.to_categorical(y.astype("int32"), num_classes=num_classes).astype("float32")


def to_01_hwc(x: np.ndarray, img_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert input array to float32 in [0, 1] and reshape to HWC.
    - If max(x) > 1.5, assume 0..255 and scale by 255.
    - Otherwise, clip into [0, 1] but keep as-is.
    """
    H, W, C = img_shape
    x = np.asarray(x, dtype="float32")
    if x.size == 0:
        return x.reshape((-1, H, W, C))
    # Detect 0..255 data
    if np.nanmax(x) > 1.5:
        x = x / 255.0
    x = x.reshape((-1, H, W, C))
    return np.clip(x, 0.0, 1.0)


def to_minus1_1(x01: np.ndarray) -> np.ndarray:
    """Map [0, 1] -> [-1, 1]."""
    return (np.asarray(x01, dtype="float32") - 0.5) * 2.0


def to_0_1(xm11: np.ndarray) -> np.ndarray:
    """Map [-1, 1] -> [0, 1]."""
    return (np.asarray(xm11, dtype="float32") * 0.5) + 0.5


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def _require_files(data_dir: Path) -> None:
    """Raise a clear error if expected .npy files are missing."""
    required = [
        "train_data.npy", "train_labels.npy",
        "test_data.npy",  "test_labels.npy",
    ]
    missing = [f for f in required if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing dataset files in {data_dir}: {', '.join(missing)}. "
            "Expected train/test *_data.npy and *_labels.npy."
        )


def load_dataset_npy(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from .npy files and return (train, val, test) splits.

    Parameters
    ----------
    data_dir : Path
        Folder containing the four .npy files (see module docstring).
    img_shape : (H, W, C)
        Target image shape. We reshape data to this HWC shape.
    num_classes : int
        Number of classes for one-hot encoding.
    val_fraction : float
        Fraction of the provided *test* split to use as validation;
        the remainder is returned as the final test split.

    Returns
    -------
    x_train, y_train, x_val, y_val, x_test, y_test : np.ndarray
        - x_* : float32 images in [0, 1], shape (N, H, W, C)
        - y_* : float32 one-hot labels, shape (N, num_classes)
    """
    data_dir = Path(data_dir)
    _require_files(data_dir)

    H, W, C = img_shape

    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test  = np.load(data_dir / "test_data.npy")
    y_test  = np.load(data_dir / "test_labels.npy")

    x_train = to_01_hwc(x_train, img_shape)
    x_test  = to_01_hwc(x_test,  img_shape)

    y_train = one_hot(y_train, num_classes)
    y_test  = one_hot(y_test,  num_classes)

    # Deterministic contiguous split (first val_fraction goes to val)
    n_val = int(len(x_test) * float(val_fraction))
    x_val, y_val = x_test[:n_val], y_test[:n_val]
    x_test_
