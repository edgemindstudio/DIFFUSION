# diffusion/train.py
"""
Training utilities for the Conditional Diffusion model.

This module keeps the lower-level training logic (noise schedule, train/eval
steps, early stopping, periodic checkpoints) separate from the model
definition and pipeline orchestration.

What this file provides
-----------------------
- build_alpha_hat(...):     Create ᾱ_t schedule (cumulative product of α_t).
- train_step(...):          Single gradient step (tf.function) for ε-prediction.
- eval_step(...):           Forward-only loss for validation (tf.function).
- train_diffusion(...):     Full training loop with early stopping & checkpoints.

Conventions
-----------
- Input images are expected in [0, 1] and shaped (N, H, W, C).
- Labels are one-hot of shape (N, num_classes).
- The diffusion model follows the signature:
      model([x_t, y_onehot, t_vec]) -> ε̂ (predicted noise)
- Checkpoint filenames use Keras 3–friendly names: *.weights.h5
  (Legacy *.h5 files are also written for convenience.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


# =============================================================================
# Noise schedule (ᾱ_t)
# =============================================================================
def build_alpha_hat(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule: str = "linear",
) -> np.ndarray:
    """
    Build ᾱ (alpha-hat) schedule used by DDPM-style training.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    beta_start, beta_end : float
        Range for β_t.
    schedule : {"linear"}
        Future hook for cosine/exp schedules if needed.

    Returns
    -------
    np.ndarray
        ᾱ_t array of shape (T,), dtype float32 in (0, 1].
    """
    if schedule != "linear":
        raise ValueError(f"Unsupported schedule '{schedule}' (only 'linear' is implemented)")
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_hat = np.cumprod(alphas, axis=0)
    return alpha_hat.astype("float32")


# =============================================================================
# Training primitives (tf.function)
# =============================================================================
mse_loss = tf.keras.losses.MeanSquaredError()


@tf.function
def train_step(
    model: tf.keras.Model,
    x0: tf.Tensor,                # clean image batch in [0,1], shape (B,H,W,C)
    y_onehot: tf.Tensor,          # one-hot labels, shape (B,K)
    t_vec: tf.Tensor,             # integer timesteps, shape (B,)
    alpha_hat_tf: tf.Tensor,      # ᾱ_t as tf.float32 tensor, shape (T,)
    optimizer: tf.keras.optimizers.Optimizer,
) -> tf.Tensor:
    """
    One optimization step for ε-prediction objective.

    Loss = MSE(ε, ε̂(x_t, t, y)), where
      x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * ε
    """
    noise = tf.random.normal(shape=tf.shape(x0))

    sqrt_a = tf.sqrt(tf.gather(alpha_hat_tf, t_vec))                    # (B,)
    sqrt_oma = tf.sqrt(1.0 - tf.gather(alpha_hat_tf, t_vec))            # (B,)
    sqrt_a = tf.reshape(sqrt_a, (-1, 1, 1, 1))
    sqrt_oma = tf.reshape(sqrt_oma, (-1, 1, 1, 1))

    x_t = sqrt_a * x0 + sqrt_oma * noise

    with tf.GradientTape() as tape:
        eps_pred = model([x_t, y_onehot, t_vec], training=True)         # ε̂
        loss = mse_loss(noise, eps_pred)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


@tf.function
def eval_step(
    model: tf.keras.Model,
    x0: tf.Tensor,
    y_onehot: tf.Tensor,
    t_vec: tf.Tensor,
    alpha_hat_tf: tf.Tensor,
) -> tf.Tensor:
    """Forward-only loss (no gradient) for validation."""
    noise = tf.random.normal(shape=tf.shape(x0))
    sqrt_a = tf.sqrt(tf.gather(alpha_hat_tf, t_vec))
    sqrt_oma = tf.sqrt(1.0 - tf.gather(alpha_hat_tf, t_vec))
    sqrt_a = tf.reshape(sqrt_a, (-1, 1, 1, 1))
    sqrt_oma = tf.reshape(sqrt_oma, (-1, 1, 1, 1))
    x_t = sqrt_a * x0 + sqrt_oma * noise
    eps_pred = model([x_t, y_onehot, t_vec], training=False)
    return mse_loss(noise, eps_pred)


# =============================================================================
# Utilities
# =============================================================================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _as_float(x) -> float:
    try:
        return float(x)
    except Exception:
        arr = tf.convert_to_tensor(x)
        return float(arr.numpy().reshape(-1)[0])


def _make_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        # Large buffer for decent shuffling on medium-scale datasets
        ds = ds.shuffle(buffer_size=min(len(x), 10_000))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# Main training loop
# =============================================================================
def train_diffusion(
    *,
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    beta_1: float = 0.9,
    alpha_hat: Optional[np.ndarray] = None,
    T: int = 1000,
    patience: int = 10,
    ckpt_dir: Path = Path("artifacts/diffusion/checkpoints"),
    log_cb = None,  # Optional callable(epoch:int, train_loss:float, val_loss:Optional[float])
) -> Dict[str, float]:
    """
    Train the diffusion model with ε-prediction objective, early stopping,
    and periodic checkpoints.

    Parameters
    ----------
    model : tf.keras.Model
        Diffusion model with signature model([x_t, y_onehot, t_vec]) -> ε̂.
    x_train, y_train : np.ndarray
        Training images in [0,1] and one-hot labels.
    x_val, y_val : Optional[np.ndarray]
        Validation split (optional but recommended for early stopping).
    epochs, batch_size : int
        Training schedule.
    learning_rate, beta_1 : float
        Optimizer parameters (Adam).
    alpha_hat : Optional[np.ndarray]
        ᾱ_t schedule; if None, one is created with T steps.
    T : int
        Number of diffusion steps (used if alpha_hat is None, and for sampling t).
    patience : int
        Early stopping patience (epochs).
    ckpt_dir : Path
        Where to write checkpoints.
    log_cb : Optional[callable]
        If provided, called each epoch with (epoch, train_loss, val_loss).

    Returns
    -------
    Dict[str, float]
        Final metrics: {"train_loss": ..., "val_loss": ..., "best_val": ...}
    """
    _ensure_dir(ckpt_dir)

    if alpha_hat is None:
        alpha_hat = build_alpha_hat(T)
    alpha_hat_tf = tf.constant(alpha_hat, dtype=tf.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)

    train_ds = _make_batches(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_ds = None
    if x_val is not None and y_val is not None:
        val_ds = _make_batches(x_val, y_val, batch_size=batch_size, shuffle=False)

    best_val = np.inf
    patience_ctr = 0

    # Compact helper to sample random t per batch size
    def _sample_t(n: tf.Tensor) -> tf.Tensor:
        # use dynamic length to be tf.function-safe
        T_dyn = tf.shape(alpha_hat_tf)[0]
        return tf.random.uniform(shape=(n,), minval=0, maxval=T_dyn, dtype=tf.int32)

    for epoch in range(1, epochs + 1):
        # --------- Training ---------
        train_losses = []
        for xb, yb in train_ds:
            bsz = tf.shape(xb)[0]
            t_vec = _sample_t(bsz)
            loss = train_step(model, xb, yb, t_vec, alpha_hat_tf, optimizer)
            train_losses.append(loss)

        train_loss = _as_float(tf.reduce_mean(train_losses))

        # --------- Validation ---------
        val_loss = None
        if val_ds is not None:
            val_losses = []
            for xb, yb in val_ds:
                bsz = tf.shape(xb)[0]
                t_vec = _sample_t(bsz)
                vloss = eval_step(model, xb, yb, t_vec, alpha_hat_tf)
                val_losses.append(vloss)
            val_loss = _as_float(tf.reduce_mean(val_losses))

        # --------- Logging / TensorBoard hook ---------
        if log_cb is not None:
            log_cb(epoch, train_loss, val_loss)

        # --------- Checkpoints ---------
        # Periodic epoch snapshot (and at epoch 1 for easy inspection)
        if epoch == 1 or epoch % 25 == 0:
            model.save_weights(str(ckpt_dir / f"DDPM_epoch_{epoch:04d}.weights.h5"))
        # Always update the "last" checkpoint
        model.save_weights(str(ckpt_dir / "DDPM_last.weights.h5"))

        # Legacy convenience files for older scripts
        model.save_weights(str(ckpt_dir / "ddpm_last.h5"))

        # Best model tracking (prefer validation if available)
        score_for_early_stop = val_loss if val_loss is not None else train_loss
        if score_for_early_stop < best_val:
            best_val = score_for_early_stop
            patience_ctr = 0
            model.save_weights(str(ckpt_dir / "DDPM_best.weights.h5"))
            model.save_weights(str(ckpt_dir / "ddpm_best.h5"))  # legacy
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break  # early stop

    return {
        "train_loss": float(train_loss),
        "val_loss": float(val_loss) if val_loss is not None else float("nan"),
        "best_val": float(best_val),
    }


__all__ = [
    "build_alpha_hat",
    "train_step",
    "eval_step",
    "train_diffusion",
]
