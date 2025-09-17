# diffusion/models.py

"""
Model builders for the Conditional Diffusion (class-conditioned DDPM) project.

What this module provides
-------------------------
- SinusoidalTimeEmbedding: Keras layer mapping scalar timesteps -> embeddings.
- build_diffusion_model(...): compact UNet-like noise predictor εθ(x_t, t, y).
  The returned model is **compiled** (MSE on noise, Adam) and ready for training
  or weight loading.

Conventions
-----------
- Images use channels-last format (H, W, C).
- Inputs during training are *noisy* images x_t in the same scale as real data
  (e.g., [-1, 1] or [0, 1]); the model predicts the additive Gaussian noise.
- Labels are one-hot vectors of length `num_classes`.
- Filenames for `save_weights()` should end with `.weights.h5` (Keras 3 style).

Notes
-----
Architecture is intentionally compact for CPU / Apple Silicon, but includes:
  * sinusoidal timestep embedding
  * class conditioning
  * UNet-style down/up path with skip connections

Tune `base_filters`, `depth`, and `time_emb_dim` for larger models.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


# ---------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------
class SinusoidalTimeEmbedding(layers.Layer):
    """
    Sinusoidal timestep embedding as in Transformer/DDPM literature.

    Input:   t  – int32/float32 tensor of shape (B,) or (B, 1)
    Output:  emb – float32 tensor of shape (B, dim)

    Impl details:
    - Produce an even-sized embedding (2 * half_dim) via sin/cos pairs.
    - If `dim` is odd, project to exact `dim` with a Dense layer.
    - Robust when dim is very small (avoids divide-by-zero).
    """
    def __init__(self, dim: int, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.half_dim = max(1, self.dim // 2)
        self.proj = None
        if self.dim != 2 * self.half_dim:
            self.proj = layers.Dense(self.dim, name="time_proj")

    def call(self, t):
        # Flatten to (B,)
        t = tf.reshape(t, (-1,))
        t = tf.cast(t, tf.float32)

        # Frequencies: exp(-log(10000) * i / (half_dim-1)), i=0..half_dim-1
        # Guard denominator to avoid division by zero when half_dim == 1
        denom = tf.cast(tf.maximum(self.half_dim - 1, 1), tf.float32)
        freqs = tf.exp(
            tf.range(self.half_dim, dtype=tf.float32)
            * -(tf.math.log(10000.0) / denom)
        )  # (half_dim,)

        # Outer product -> (B, half_dim)
        args = tf.expand_dims(t, -1) * tf.expand_dims(freqs, 0)
        emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)  # (B, 2*half_dim)

        if self.proj is not None:
            emb = self.proj(emb)
        return emb

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim})
        return cfg


# ---------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------
def _conv_block(x, filters: int, name: str):
    """Two Conv2D + LayerNorm + Swish blocks."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name + "_conv1")(x)
    x = layers.LayerNormalization(name=name + "_ln1")(x)
    x = layers.Activation("swish", name=name + "_act1")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=name + "_conv2")(x)
    x = layers.LayerNormalization(name=name + "_ln2")(x)
    x = layers.Activation("swish", name=name + "_act2")(x)
    return x


def _broadcast_to_spatial(emb, h: int, w: int):
    """
    (B, D) -> (B, H, W, D) via expand + tile, wrapped in Lambdas for KerasTensors.
    """
    x = layers.Lambda(lambda e: tf.expand_dims(tf.expand_dims(e, 1), 1), name="cond_expand")(emb)  # (B,1,1,D)
    x = layers.Lambda(lambda e: tf.tile(e, [1, h, w, 1]), name="cond_tile")(x)                     # (B,H,W,D)
    return x


# ---------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------
def build_diffusion_model(
    *,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    base_filters: int = 64,
    depth: int = 2,
    time_emb_dim: int = 128,
    learning_rate: float = 2e-4,
    beta_1: float = 0.9,
) -> tf.keras.Model:
    """
    Build a compact conditional UNet that predicts noise ε from (x_t, t, y).

    Inputs:
      noisy_image: (H, W, C) float32
      class_label: (num_classes,) one-hot float32
      timestep:    () int32
    Output:
      predicted_noise: (H, W, C) float32 (no activation)
    """
    assert depth >= 1, "depth must be >= 1"
    H, W, C = img_shape
    stride_total = 2 ** depth
    if (H % stride_total) != 0 or (W % stride_total) != 0:
        raise ValueError(
            f"img_shape spatial dims must be divisible by 2**depth ({stride_total}). "
            f"Got H={H}, W={W}, depth={depth}."
        )

    # ---------------- Inputs ----------------
    noisy_in = layers.Input(shape=img_shape, name="noisy_image")
    y_in     = layers.Input(shape=(num_classes,), name="class_label")
    t_in     = layers.Input(shape=(), dtype=tf.int32, name="timestep")

    # ---------------- Embeddings ----------------
    t_emb = SinusoidalTimeEmbedding(time_emb_dim, name="time_embed")(t_in)
    t_emb = layers.Dense(time_emb_dim, activation="swish", name="time_mlp1")(t_emb)
    t_emb = layers.Dense(time_emb_dim, activation="swish", name="time_mlp2")(t_emb)

    y_emb = layers.Dense(time_emb_dim, activation="swish", name="label_mlp1")(y_in)
    y_emb = layers.Dense(time_emb_dim, activation="swish", name="label_mlp2")(y_emb)

    cond = layers.Concatenate(name="cond_concat")([t_emb, y_emb])           # (B, 2*D)
    cond = layers.Dense(time_emb_dim, activation="swish", name="cond_proj")(cond)
    cond_spatial = _broadcast_to_spatial(cond, H, W)                        # (B, H, W, D)

    # ---------------- UNet backbone ----------------
    x = layers.Concatenate(name="input_concat")([noisy_in, cond_spatial])   # (B, H, W, C + D)

    # Down path: store pre-downsample skips; downsample at EVERY level
    skips = []
    filters = base_filters
    for d in range(depth):
        x = _conv_block(x, filters, name=f"down{d}")
        skips.append(x)  # pre-pool features at this resolution
        x = layers.Conv2D(filters, 3, strides=2, padding="same", name=f"down{d}_ds")(x)  # H/2 each level
        filters *= 2

    # Bottleneck (smallest spatial size)
    x = _conv_block(x, filters, name="bottleneck")

    # Up path: for each saved skip, upsample once, concat, then convs
    for d, skip in enumerate(reversed(skips)):
        filters //= 2
        x = layers.UpSampling2D(size=2, interpolation="nearest", name=f"up{d}_us")(x)
        x = layers.Concatenate(name=f"up{d}_skip")([x, skip])
        x = _conv_block(x, filters, name=f"up{d}")

    # Output head: predict noise (no activation)
    out = layers.Conv2D(C, 1, padding="same", name="pred_noise")(x)

    model = models.Model(
        inputs=[noisy_in, y_in, t_in],
        outputs=out,
        name="ConditionalDiffusion_UNet"
    )

    # Compile for MSE-on-noise objective (standard DDPM target)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
        loss="mse",
    )
    return model


__all__ = ["SinusoidalTimeEmbedding", "build_diffusion_model"]
