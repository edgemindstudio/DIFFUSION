# diffusion/sample.py

"""
Lightweight sampling & preview utilities for the Conditional Diffusion model.

This module is intentionally dependency-lite and safe to import in CLI scripts.
It provides:
  • :func:`save_grid_from_model` – generate a quick PNG grid (one image per class)
  • :func:`sample_batch`         – generate a batch of conditional samples (NumPy)

Both functions assume you already have a compiled & weight-loaded Keras model
built via :func:`diffusion.models.build_diffusion_model`, whose input signature is:
    model([x_t, y_onehot, t_vec]) -> ε̂ (predicted noise)

Conventions
-----------
- Images are channels-last (H, W, C).
- The model outputs images in [-1, 1] during reverse diffusion; we rescale to [0, 1]
  for saving/returning.
- Labels are one-hot vectors of length `num_classes`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------
# Scheduling utilities
# ---------------------------------------------------------------------
def _linear_alpha_hat_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray:
    """
    Create a simple linear beta schedule then return ᾱ_t = ∏_{s<=t} (1 - β_s).

    Parameters
    ----------
    T : int
        Number of diffusion steps.
    beta_start, beta_end : float
        Start/end values for the linear β schedule.

    Returns
    -------
    np.ndarray of shape (T,), dtype float32 with values in (0, 1].
    """
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_hat = np.cumprod(alphas, axis=0)
    return alpha_hat.astype("float32")


# ---------------------------------------------------------------------
# Core reverse sampler
# ---------------------------------------------------------------------
def _reverse_diffuse(
    model: tf.keras.Model,
    *,
    y_onehot: np.ndarray,
    img_shape: Tuple[int, int, int],
    T: int,
    alpha_hat: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Perform a DDPM-style reverse diffusion pass given class-condition labels.

    This uses a compact update in terms of ᾱ_t (alpha_hat), consistent with the
    pipeline’s training and intended for small grayscale images.

    Parameters
    ----------
    model : tf.keras.Model
        Keras model taking [x_t, y_onehot, t_vec] and predicting noise ε̂.
    y_onehot : np.ndarray
        One-hot class labels, shape (B, num_classes), float32.
    img_shape : (H, W, C)
        Output image shape.
    T : int
        Number of reverse steps.
    alpha_hat : Optional[np.ndarray]
        Precomputed ᾱ schedule (shape (T,)). If None, a default linear schedule
        is created.
    seed : Optional[int]
        RNG seed for reproducible sampling.

    Returns
    -------
    np.ndarray
        Samples in [-1, 1], shape (B, H, W, C), dtype float32.
    """
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_seed(seed)

    H, W, C = img_shape
    B = int(y_onehot.shape[0])

    if alpha_hat is None:
        alpha_hat = _linear_alpha_hat_schedule(T)
    alpha_hat_tf = tf.constant(alpha_hat, dtype=tf.float32)

    # Start from Gaussian noise
    x = tf.random.normal((B, H, W, C))

    y = tf.convert_to_tensor(y_onehot, dtype=tf.float32)

    # Reverse process: t = T - 1 ... 0
    for t in reversed(range(T)):
        t_vec = tf.fill([B], tf.cast(t, tf.int32))
        # Predict ε at (x_t, t, y)
        eps_pred = model([x, y, t_vec], training=False)

        # Gather ᾱ_t and compute update
        a = alpha_hat_tf[t]                 # scalar tensor
        a = tf.reshape(a, (1, 1, 1, 1))
        one_minus_a = 1.0 - a

        # Add noise except at t = 0
        noise = tf.random.normal(tf.shape(x)) if t > 0 else 0.0

        # Simple DDPM-like update using ᾱ_t
        # x_{t-1} ≈ (x_t - (1-ᾱ_t)/sqrt(1-ᾱ_t) * ε̂) / sqrt(ᾱ_t) + sqrt(1-ᾱ_t) * z
        x = (x - (one_minus_a / tf.sqrt(one_minus_a)) * eps_pred) / tf.sqrt(a) + tf.sqrt(one_minus_a) * noise

    return x.numpy().astype("float32")  # in [-1, 1]


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def sample_batch(
    model: tf.keras.Model,
    *,
    num_samples: int,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    T: int = 200,
    alpha_hat: Optional[np.ndarray] = None,
    class_ids: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a batch of conditional samples.

    Parameters
    ----------
    model : tf.keras.Model
        Weight-loaded diffusion model.
    num_samples : int
        Number of images to generate.
    num_classes : int
        Number of classes (for one-hot).
    img_shape : (H, W, C)
        Output image shape.
    T : int, default 200
        Number of reverse steps (smaller than training T is fine for previews).
    alpha_hat : Optional[np.ndarray]
        ᾱ schedule of shape (T,). If None, build a linear schedule.
    class_ids : Optional[np.ndarray]
        Integer class ids of shape (num_samples,). If None, draw uniformly.
    seed : Optional[int]
        Seed for RNG.

    Returns
    -------
    (x, y_onehot)
        x: float32 in [0, 1], shape (N, H, W, C)
        y_onehot: float32 one-hot labels, shape (N, num_classes)
    """
    if class_ids is None:
        class_ids = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int32)
    y_onehot = tf.keras.utils.to_categorical(class_ids, num_classes=num_classes).astype("float32")

    # Reverse diffusion to get images in [-1, 1]
    x_m11 = _reverse_diffuse(
        model,
        y_onehot=y_onehot,
        img_shape=img_shape,
        T=T,
        alpha_hat=alpha_hat,
        seed=seed,
    )
    # Rescale to [0, 1]
    x_01 = np.clip((x_m11 + 1.0) / 2.0, 0.0, 1.0).astype("float32")
    return x_01, y_onehot


def save_grid_from_model(
    model: tf.keras.Model,
    *,
    num_classes: int,
    img_shape: Tuple[int, int, int],
    path: Path,
    T: int = 200,
    alpha_hat: Optional[np.ndarray] = None,
    dpi: int = 200,
    titles: bool = True,
    seed: Optional[int] = None,
) -> Path:
    """
    Generate ONE sample per class and save a horizontal grid PNG.

    Parameters
    ----------
    model : tf.keras.Model
        Weight-loaded diffusion model.
    num_classes : int
        Number of classes.
    img_shape : (H, W, C)
        Output image shape (C=1 assumed for malware grayscale).
    path : Path
        Where to write the PNG.
    T : int, default 200
        Number of reverse steps for sampling (lower -> faster preview).
    alpha_hat : Optional[np.ndarray]
        ᾱ schedule; if None, a linear schedule is used.
    dpi : int
        Matplotlib DPI for saved figure.
    titles : bool
        If True, annotate each tile with class id.
    seed : Optional[int]
        RNG seed for reproducibility.

    Returns
    -------
    Path
        The saved figure path.
    """
    import matplotlib.pyplot as plt  # local import to keep CLI deps light

    # One sample for each class, ordered 0..K-1
    class_ids = np.arange(num_classes, dtype=np.int32)
    y_onehot = tf.keras.utils.to_categorical(class_ids, num_classes=num_classes).astype("float32")

    x_01, _ = sample_batch(
        model,
        num_samples=num_classes,
        num_classes=num_classes,
        img_shape=img_shape,
        T=T,
        alpha_hat=alpha_hat,
        class_ids=class_ids,
        seed=seed,
    )

    H, W, C = img_shape
    n = num_classes
    fig_w = max(1.2 * n, 6.0)
    fig_h = 1.6
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]

    for i in range(n):
        ax = axes[i]
        img = x_01[i]
        if C == 1:
            ax.imshow(img[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.squeeze(img))
        ax.set_axis_off()
        if titles:
            ax.set_title(f"C{i}", fontsize=9)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


__all__ = ["sample_batch", "save_grid_from_model"]
