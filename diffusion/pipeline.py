# diffusion/pipeline.py

"""
Training + synthesis pipeline for the Conditional Diffusion model (class-conditioned DDPM).

Why this exists
---------------
A small, production-friendly wrapper that:
  • builds a compact UNet-like noise predictor εθ(x_t, t, y)
  • trains with the standard DDPM MSE objective using tf.data
  • writes Keras 3-style checkpoints (*.weights.h5)
  • synthesizes class-balanced samples via a numerically robust reverse loop

It mirrors your GAN/VAE/AR repos so downstream evaluation & aggregation can be identical.

Artifacts written by `.synthesize(...)`
---------------------------------------
ARTIFACTS/diffusion/synthetic/
  gen_class_{k}.npy        -> float32 images in [0, 1], shape (Nk, H, W, C)
  labels_class_{k}.npy     -> int labels (k), shape (Nk,)
  x_synth.npy, y_synth.npy -> convenience concatenations

Conventions
-----------
- Images are channels-last (H, W, C) with values in [0, 1].
- Labels are one-hot, shape (N, num_classes).
- Checkpoint names: DIF_best.weights.h5, DIF_last.weights.h5, DIF_epoch_xxxx.weights.h5
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import tensorflow as tf

from diffusion.models import build_diffusion_model


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _as_float(x) -> float:
    """Convert scalars / 0-D tensors / numpy arrays to a Python float."""
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        return float(arr.reshape(-1)[0])


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _linear_alpha_hat_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray:
    """
    Linear beta schedule used in the original DDPM:
      beta_t linearly increases; α_t = 1 - beta_t; ᾱ_t = ∏_{s<=t} α_s
    Returns ᾱ (alpha_hat) as a numpy array of length T.
    """
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_hat = np.cumprod(alphas, axis=0)
    return alpha_hat


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
class DiffusionPipeline:
    """
    Orchestrates training and synthesis for a class-conditioned diffusion model.
    """

    DEFAULTS = {
        "IMG_SHAPE": (40, 40, 1),
        "NUM_CLASSES": 9,
        "EPOCHS": 200,
        "BATCH_SIZE": 128,
        "LR": 2e-4,
        "BETA_1": 0.9,
        "BASE_FILTERS": 64,
        "DEPTH": 2,
        "TIME_EMB_DIM": 128,
        "T": 1000,                 # diffusion steps
        "BETA_START": 1e-4,        # schedule lower bound (training + sampling)
        "BETA_END": 2e-2,          # schedule upper bound (training + sampling)
        "LOG_EVERY": 25,           # save periodic epoch checkpoints
        "PATIENCE": 10,            # early stopping
        "SAMPLES_PER_CLASS": 1000,
        "ARTIFACTS": {
            "checkpoints": "artifacts/diffusion/checkpoints",
            "synthetic": "artifacts/diffusion/synthetic",
        },
    }

    def __init__(self, cfg: Dict):
        self.cfg = cfg or {}
        d = self.DEFAULTS

        # Hyperparameters / shapes
        self.img_shape: Tuple[int, int, int] = tuple(self.cfg.get("IMG_SHAPE", d["IMG_SHAPE"]))
        self.num_classes: int = int(self.cfg.get("NUM_CLASSES", d["NUM_CLASSES"]))
        self.epochs: int = int(self.cfg.get("EPOCHS", d["EPOCHS"]))
        self.batch_size: int = int(self.cfg.get("BATCH_SIZE", d["BATCH_SIZE"]))
        self.lr: float = float(self.cfg.get("LR", d["LR"]))
        self.beta_1: float = float(self.cfg.get("BETA_1", d["BETA_1"]))
        self.base_filters: int = int(self.cfg.get("BASE_FILTERS", d["BASE_FILTERS"]))
        self.depth: int = int(self.cfg.get("DEPTH", d["DEPTH"]))
        self.time_emb_dim: int = int(self.cfg.get("TIME_EMB_DIM", d["TIME_EMB_DIM"]))
        self.T: int = int(self.cfg.get("T", d["T"]))
        self.beta_start: float = float(self.cfg.get("BETA_START", d["BETA_START"]))
        self.beta_end: float = float(self.cfg.get("BETA_END", d["BETA_END"]))
        self.log_every: int = int(self.cfg.get("LOG_EVERY", d["LOG_EVERY"]))
        self.patience: int = int(self.cfg.get("PATIENCE", d["PATIENCE"]))
        self.samples_per_class: int = int(self.cfg.get("SAMPLES_PER_CLASS", d["SAMPLES_PER_CLASS"]))

        # Artifacts
        arts = self.cfg.get("ARTIFACTS", d["ARTIFACTS"])
        self.ckpt_dir = Path(arts.get("checkpoints", d["ARTIFACTS"]["checkpoints"]))
        self.synth_dir = Path(arts.get("synthetic", d["ARTIFACTS"]["synthetic"]))
        _ensure_dir(self.ckpt_dir)
        _ensure_dir(self.synth_dir)

        # Optional logging callback: cb(epoch:int, train_loss, val_loss)
        self.log_cb = self.cfg.get("LOG_CB", None)

        # Training diffusion schedule (ᾱ_t) and its terms (used for forward noising)
        self.alpha_hat = _linear_alpha_hat_schedule(self.T, self.beta_start, self.beta_end)
        self.alpha_hat_tf = tf.constant(self.alpha_hat, dtype=tf.float32)
        self.sqrt_alpha_hat_tf = tf.sqrt(self.alpha_hat_tf)
        self.sqrt_one_minus_alpha_hat_tf = tf.sqrt(1.0 - self.alpha_hat_tf)

        # Build a fresh compiled model
        self.model: tf.keras.Model = build_diffusion_model(
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            base_filters=self.base_filters,
            depth=self.depth,
            time_emb_dim=self.time_emb_dim,
            learning_rate=self.lr,       # NOTE: named param matches builder
            beta_1=self.beta_1,
        )

    # ----------------------- tf.data ---------------------------------
    def _make_dataset(self, x: np.ndarray, y: np.ndarray, shuffle: bool) -> tf.data.Dataset:
        """
        Create a dataset that yields ((x_t, y_onehot, t), noise) pairs where:
          x_t   = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε,  ε ~ N(0, I)
          t     ~ Uniform{0..T-1}
        Target = ε (standard DDPM MSE objective).
        """
        AUTOTUNE = tf.data.AUTOTUNE
        T_const = tf.constant(self.T, dtype=tf.int32)

        def _map(x0, y1h):
            b = tf.shape(x0)[0]
            # Random time step per sample
            t = tf.random.uniform(shape=(b,), minval=0, maxval=T_const, dtype=tf.int32)
            # Gaussian noise
            eps = tf.random.normal(shape=tf.shape(x0))
            # Gather schedule terms
            sa = tf.gather(self.sqrt_alpha_hat_tf, t)               # (B,)
            soma = tf.gather(self.sqrt_one_minus_alpha_hat_tf, t)   # (B,)
            # Broadcast to image dims
            sa = tf.reshape(sa, (-1, 1, 1, 1))
            soma = tf.reshape(soma, (-1, 1, 1, 1))
            # Noisy input x_t and target noise
            x_t = sa * x0 + soma * eps
            return (x_t, y1h, t), eps

        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(x), 8192), reshuffle_each_iteration=True)
        ds = ds.batch(self.batch_size, drop_remainder=False)
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
        return ds

    # ----------------------- Training -----------------------
    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> tf.keras.Model:
        """
        Fit the diffusion model by predicting injected Gaussian noise.

        Args
        ----
        x_train : float32 (N, H, W, C), values in [0, 1]
        y_train : float32 (N, num_classes), one-hot
        x_val, y_val : optional validation sets

        Returns
        -------
        The trained Keras model (and saves checkpoints along the way).
        """
        H, W, C = self.img_shape
        assert x_train.shape[1:] == (H, W, C), "x_train shape mismatch"
        assert y_train.shape[1] == self.num_classes, "y_train must be one-hot"

        train_ds = self._make_dataset(x_train, y_train, shuffle=True)
        val_ds = self._make_dataset(x_val, y_val, shuffle=False) if x_val is not None and y_val is not None else None

        # Keras callbacks
        callbacks: List[tf.keras.callbacks.Callback] = []

        # Early stopping / checkpointing — monitor val_loss if available, else loss
        monitor_metric = "val_loss" if val_ds is not None else "loss"

        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric, patience=self.patience, restore_best_weights=True
            )
        )
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.ckpt_dir / "DIF_best.weights.h5"),
                monitor=monitor_metric,
                save_best_only=True,
                save_weights_only=True,
            )
        )

        # Periodic manual checkpoint + external logging callback
        class _PeriodicSaver(tf.keras.callbacks.Callback):
            def __init__(self, outer, log_every: int):
                super().__init__()
                self.outer = outer
                self.log_every = max(1, int(log_every))

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                e = epoch + 1
                if self.outer.log_cb is not None:
                    tr = logs.get("loss")
                    vl = logs.get("val_loss")
                    if tr is not None:
                        self.outer.log_cb(e, _as_float(tr), _as_float(vl) if vl is not None else None)
                if e % self.log_every == 0 or e == 1:
                    path = self.outer.ckpt_dir / f"DIF_epoch_{e:04d}.weights.h5"
                    self.model.save_weights(str(path))

        callbacks.append(_PeriodicSaver(self, self.log_every))

        # Fit
        self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=0,  # external logger handles printing
        )

        # Final checkpoint
        self.model.save_weights(str(self.ckpt_dir / "DIF_last.weights.h5"))
        return self.model

    # ----------------------- Synthesis -----------------------
    def synthesize(self, model: Optional[tf.keras.Model] = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a class-balanced synthetic dataset via a robust DDPM reverse loop.

        If `model` is None, a fresh model is built and the latest available
        checkpoint is loaded from `self.ckpt_dir`.

        Returns
        -------
        x_synth : float32, shape (N_total, H, W, C), values in [0, 1]
        y_synth : float32, shape (N_total, num_classes), one-hot labels
        """
        if model is None:
            model = build_diffusion_model(
                img_shape=self.img_shape,
                num_classes=self.num_classes,
                base_filters=self.base_filters,
                depth=self.depth,
                time_emb_dim=self.time_emb_dim,
                learning_rate=self.lr,
                beta_1=self.beta_1,
            )
            ckpt = self._latest_checkpoint()
            if ckpt is not None:
                model.load_weights(str(ckpt))

        # ---- Sample all classes in one batch for efficiency ----
        x_s, y_s = self._ddpm_reverse_sample_balanced(
            model=model,
            img_shape=self.img_shape,
            num_classes=self.num_classes,
            timesteps=self.T,
            samples_per_class=self.samples_per_class,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )

        # ---- Sanitize (finite + clamp to [0,1]) before saving ----
        if x_s.size == 0:
            print("[warn] Sampler returned empty batch.")
            return x_s, y_s

        finite_mask = np.isfinite(x_s).all(axis=(1, 2, 3))
        if not finite_mask.any():
            print("[warn] Sampler produced only non-finite samples.")
            # Return empty so main/evaluator will run REAL-only
            return np.empty((0, *self.img_shape), dtype="float32"), np.empty((0, self.num_classes), dtype="float32")

        dropped = int((~finite_mask).sum())
        if dropped > 0:
            print(f"[warn] Dropping {dropped} non-finite samples before saving.")
        x_s = x_s[finite_mask]
        y_s = y_s[finite_mask]

        x_s = np.clip(x_s, 0.0, 1.0).astype("float32")

        # ---- Persist per-class dumps (contract used by evaluator) ----
        self.synth_dir.mkdir(parents=True, exist_ok=True)
        labels_int = np.argmax(y_s, axis=1).astype(int)
        for k in range(self.num_classes):
            cls_mask = labels_int == k
            cls_imgs = x_s[cls_mask]
            np.save(self.synth_dir / f"gen_class_{k}.npy", cls_imgs)
            np.save(self.synth_dir / f"labels_class_{k}.npy", np.full((cls_imgs.shape[0],), k, dtype=np.int32))

        # Convenience combined dumps
        np.save(self.synth_dir / "x_synth.npy", x_s)
        np.save(self.synth_dir / "y_synth.npy", y_s)

        print(f"[synthesize] {x_s.shape[0]} samples ({self.samples_per_class} per class requested) -> {self.synth_dir}")
        return x_s, y_s

    # ----------------------- Internals -----------------------
    def _latest_checkpoint(self) -> Optional[Path]:
        """
        Choose a checkpoint to load for synthesis:
          prefer DIF_best.weights.h5, then DIF_last.weights.h5,
          else the newest DIF_epoch_*.weights.h5 (or legacy *.h5).
        """
        order = [
            self.ckpt_dir / "DIF_best.weights.h5",
            self.ckpt_dir / "DIF_last.weights.h5",
        ]
        epoch_ckpts = sorted(self.ckpt_dir.glob("DIF_epoch_*.weights.h5"))
        if epoch_ckpts:
            order.append(max(epoch_ckpts, key=lambda p: p.stat().st_mtime))

        for p in order:
            if p.exists():
                return p

        legacy = sorted(self.ckpt_dir.glob("DIF_epoch_*.h5"))
        return max(legacy, key=lambda p: p.stat().st_mtime) if legacy else None

    # --- Numerically robust DDPM reverse sampler (balanced classes) ---
    def _ddpm_reverse_sample_balanced(
        self,
        *,
        model: tf.keras.Model,
        img_shape: Tuple[int, int, int],
        num_classes: int,
        timesteps: int,
        samples_per_class: int,
        beta_start: float,
        beta_end: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate class-balanced samples using a safe DDPM reverse loop with variance.

        Returns
        -------
        x_synth : float32 in [0,1], shape (N, H, W, C)
        y_onehot: float32 one-hot, shape (N, num_classes)
        """
        H, W, C = img_shape
        n_per = int(samples_per_class)
        n_total = n_per * num_classes
        eps_guard = 1e-6

        # Linear beta schedule + derived terms
        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
        alphas = 1.0 - betas
        alpha_bars = np.cumprod(alphas)

        # Start from Gaussian noise (Tensor)
        x = tf.random.normal((n_total, H, W, C), dtype=tf.float32)

        # Balanced labels -> Tensor one-hot (avoid NumPy here!)
        labels_id = np.repeat(np.arange(num_classes), n_per)
        y_onehot = tf.one_hot(labels_id, depth=num_classes, dtype=tf.float32)  # (N, K)

        # Reverse diffusion x_T -> x_0
        for t in reversed(range(timesteps)):
            # Tensor timestep vector
            t_vec = tf.fill([n_total], tf.cast(t, tf.int32))

            # Predict noise epsilon (all inputs are Tensors)
            eps_pred = model([x, y_onehot, t_vec], training=False)

            # Per-step α_t and cumulative ᾱ_t as Tensors
            alpha_t = tf.convert_to_tensor(alphas[t], dtype=tf.float32)           # scalar
            alpha_bar_t = tf.convert_to_tensor(alpha_bars[t], dtype=tf.float32)   # scalar
            alpha_bar_tm1 = tf.convert_to_tensor(alpha_bars[t - 1] if t > 0 else 1.0, dtype=tf.float32)
            beta_t = tf.convert_to_tensor(betas[t], dtype=tf.float32)

            # Mean update (Ho et al., 2020), with eps-guards
            coef1 = tf.math.rsqrt(tf.maximum(alpha_t, eps_guard))
            coef2 = (1.0 - alpha_t) / tf.sqrt(tf.maximum(1.0 - alpha_bar_t, eps_guard))
            x = coef1 * (x - coef2 * eps_pred)

            # Variance (add noise except at t = 0)
            var_t = ((1.0 - alpha_bar_tm1) / tf.maximum(1.0 - alpha_bar_t, eps_guard)) * beta_t
            sigma_t = tf.sqrt(tf.maximum(var_t, 0.0))
            if t > 0:
                x = x + sigma_t * tf.random.normal(tf.shape(x), dtype=tf.float32)

            # Guard against inf/nan mid-trajectory
            x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))

        # Final clamp for safety to [0,1]
        x = tf.clip_by_value(x, 0.0, 1.0)

        # Convert to NumPy for saving/returning
        return x.numpy().astype("float32"), y_onehot.numpy().astype("float32")


__all__ = ["DiffusionPipeline"]
