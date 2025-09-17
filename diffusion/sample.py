# diffusion/sample.py

"""
Lightweight sampling & preview utilities for the Conditional Diffusion model.

This module is intentionally dependency-lite and safe to import in CLI scripts.
It provides:
  • save_grid_from_model(model, ...) – quick PNG grid (one image per class)
  • sample_batch(model, ...)         – batch of conditional samples (NumPy)
  • synth(cfg, output_root, seed)    – unified-CLI entrypoint (PNGs + manifest)

Assumes a compiled & weight-loaded Keras model built via
diffusion.models.build_diffusion_model with signature:
    model([x_t, y_onehot, t_vec]) -> eps_hat  (predicted noise)

Conventions
-----------
- Images are channels-last (H, W, C).
- Reverse diffusion produces images in [-1, 1]; we rescale to [0, 1] for saving.
- Labels are one-hot vectors of length `num_classes`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import tensorflow as tf

from PIL import Image

# local
from diffusion.models import build_diffusion_model


# ---------------------------------------------------------------------
# Small config helpers
# ---------------------------------------------------------------------
def _cfg_get(cfg: dict, dotted: str, default=None):
    cur = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------
# Scheduling utilities
# ---------------------------------------------------------------------
def _linear_alpha_hat_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> np.ndarray:
    """
    Linear beta schedule; returns ᾱ_t = ∏_{s<=t} (1 - β_s) for t=0..T-1.
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

    Returns images in [-1, 1], shape (B,H,W,C), float32.
    """
    if seed is not None:
        np.random.seed(int(seed))
        tf.random.set_seed(int(seed))

    H, W, C = img_shape
    B = int(y_onehot.shape[0])

    if alpha_hat is None:
        alpha_hat = _linear_alpha_hat_schedule(T)
    alpha_hat_tf = tf.constant(alpha_hat, dtype=tf.float32)

    # Start from Gaussian noise
    x = tf.random.normal((B, H, W, C))
    y = tf.convert_to_tensor(y_onehot, dtype=tf.float32)

    # Reverse process: t = T-1 ... 0
    for t in reversed(range(T)):
        t_vec = tf.fill([B], tf.cast(t, tf.int32))
        eps_pred = model([x, y, t_vec], training=False)  # ε̂(x_t, t, y)

        a = tf.reshape(alpha_hat_tf[t], (1, 1, 1, 1))  # ᾱ_t
        one_minus_a = 1.0 - a

        # Add noise except at t = 0
        noise = tf.random.normal(tf.shape(x)) if t > 0 else 0.0

        # Simple update using ᾱ_t (preview-friendly)
        x = (x - (one_minus_a / tf.sqrt(one_minus_a)) * eps_pred) / tf.sqrt(a) + tf.sqrt(one_minus_a) * noise

    return x.numpy().astype("float32")  # [-1, 1]


# ---------------------------------------------------------------------
# Public sampling helpers
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

    Returns:
      x_01: float32 in [0, 1], shape (N,H,W,C)
      y_onehot: float32, shape (N,num_classes)
    """
    if class_ids is None:
        class_ids = np.random.randint(0, num_classes, size=(num_samples,), dtype=np.int32)
    else:
        class_ids = np.asarray(class_ids, dtype=np.int32).reshape(-1)
        num_samples = int(class_ids.shape[0])

    y_onehot = tf.keras.utils.to_categorical(class_ids, num_classes=num_classes).astype("float32")

    # Reverse diffusion in [-1,1]
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
    """
    import matplotlib.pyplot as plt  # local import to keep CLI deps light
    H, W, C = img_shape

    class_ids = np.arange(num_classes, dtype=np.int32)
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


# ---------------------------------------------------------------------
# Orchestrator-facing helpers (PNG writing, checkpoint loading)
# ---------------------------------------------------------------------
def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img01 * 255.0), 0, 255).astype(np.uint8)


def _save_png(img01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = img01
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
        mode = "L"
    elif x.ndim == 3 and x.shape[-1] == 3:
        mode = "RGB"
    else:
        x = x.squeeze()
        mode = "L"
    Image.fromarray(_to_uint8(x), mode=mode).save(out_path)


def _load_diffusion_from_checkpoints(
    ckpt_dir: Path | str,
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
    Build diffusion model and try to load weights from {best→last}. If unavailable
    or incompatible, proceed with random weights (emit a warning).
    """
    from diffusion.models import build_diffusion_model

    ckpt_dir = Path(ckpt_dir)
    H, W, C = img_shape

    model = build_diffusion_model(
        img_shape=img_shape,
        num_classes=num_classes,
        base_filters=base_filters,
        depth=depth,
        time_emb_dim=time_emb_dim,
        learning_rate=learning_rate,
        beta_1=beta_1,
    )

    # Build variables (Keras 3-friendly)
    dummy_x = tf.zeros((1, H, W, C), dtype=tf.float32)
    dummy_y = tf.one_hot([0], depth=num_classes, dtype=tf.float32)
    dummy_t = tf.zeros((1,), dtype=tf.int32)
    _ = model([dummy_x, dummy_y, dummy_t], training=False)

    best = ckpt_dir / "DDPM_best.weights.h5"
    last = ckpt_dir / "DDPM_last.weights.h5"
    to_load = best if best.exists() else last

    if not to_load.exists():
        print(f"[ckpt][warn] no DDPM checkpoint in {ckpt_dir}; using random weights.")
        return model

    try:
        model.load_weights(str(to_load))
        print(f"[ckpt] Loaded {to_load.name}")
    except Exception as e:
        print(f"[ckpt][warn] failed to load {to_load.name}: {e}\n→ continuing with random weights.")
    return model


# ---------------------------------------------------------------------
# Unified CLI entrypoint (used by app/main.py)
# ---------------------------------------------------------------------
def synth(cfg: dict, output_root: str, seed: int = 42) -> Dict:
    """
    Generate S PNGs per class into {output_root}/{class}/{seed}/... and return a manifest.

    Config keys (with fallbacks):
      IMG_SHAPE, NUM_CLASSES, DIFFUSION_STEPS (default 200)
      DIFF.base_filters, DIFF.depth, DIFF.time_dim (or diffusion.* variants)
      LR, BETA_1
      paths.artifacts (default "artifacts")
      ARTIFACTS.diffusion_checkpoints (default: <artifacts>/diffusion/checkpoints)
    """
    # Shapes & hyperparams
    H, W, C = tuple(_cfg_get(cfg, "IMG_SHAPE", _cfg_get(cfg, "img.shape", (40, 40, 1))))
    K = int(_cfg_get(cfg, "NUM_CLASSES", _cfg_get(cfg, "num_classes", 9)))
    S = int(_cfg_get(cfg, "SAMPLES_PER_CLASS", _cfg_get(cfg, "samples_per_class", 25)))
    T = int(_cfg_get(cfg, "DIFFUSION_STEPS", _cfg_get(cfg, "diffusion.steps", 200)))

    base_filters = int(_cfg_get(cfg, "DIFF.base_filters", _cfg_get(cfg, "diffusion.base_filters", 64)))
    depth        = int(_cfg_get(cfg, "DIFF.depth",        _cfg_get(cfg, "diffusion.depth", 2)))
    time_dim     = int(_cfg_get(cfg, "DIFF.time_dim",     _cfg_get(cfg, "diffusion.time_emb_dim", 128)))

    LR     = float(_cfg_get(cfg, "LR", _cfg_get(cfg, "diffusion.lr", 2e-4)))
    BETA_1 = float(_cfg_get(cfg, "BETA_1", _cfg_get(cfg, "diffusion.beta_1", 0.9)))

    artifacts_root = Path(_cfg_get(cfg, "paths.artifacts", "artifacts"))
    ckpt_dir = Path(_cfg_get(cfg, "ARTIFACTS.diffusion_checkpoints",
                             artifacts_root / "diffusion" / "checkpoints"))

    # Seeding
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))

    # Build & load model (or proceed with random weights)
    model = _load_diffusion_from_checkpoints(
        ckpt_dir,
        img_shape=(H, W, C),
        num_classes=K,
        base_filters=base_filters,
        depth=depth,
        time_emb_dim=time_dim,
        learning_rate=LR,
        beta_1=BETA_1,
    )

    out_root = Path(output_root)
    per_class_counts: Dict[str, int] = {str(k): 0 for k in range(K)}
    paths: List[Dict] = []

    # Optional precomputed schedule for speed
    alpha_hat = _linear_alpha_hat_schedule(T)

    # Generate per class
    for k in range(K):
        class_ids = np.full((S,), k, dtype=np.int32)
        y_onehot = tf.keras.utils.to_categorical(class_ids, num_classes=K).astype("float32")

        x01, _ = sample_batch(
            model,
            num_samples=S,
            num_classes=K,
            img_shape=(H, W, C),
            T=T,
            alpha_hat=alpha_hat,
            class_ids=class_ids,
            seed=seed,
        )

        cls_dir = out_root / str(k) / str(seed)
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j in range(S):
            p = cls_dir / f"diff_{j:05d}.png"
            _save_png(x01[j], p)
            paths.append({"path": str(p), "label": int(k)})
        per_class_counts[str(k)] = int(S)

    manifest = {
        "dataset": _cfg_get(cfg, "data.root", _cfg_get(cfg, "DATA_DIR", "data")),
        "seed": int(seed),
        "per_class_counts": per_class_counts,
        "paths": paths,
    }
    return manifest


__all__ = ["sample_batch", "save_grid_from_model", "synth"]
