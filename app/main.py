# app/main.py
# =============================================================================
# Conditional Diffusion (class-conditioned DDPM) — production pipeline entry point
#
# Commands
# --------
#   python -m app.main train         # train diffusion model, save checkpoints
#   python -m app.main synth         # load best/latest checkpoint and synthesize per-class samples
#   python -m app.main eval          # standardized evaluation with/without synthetic
#   python -m app.main all           # train -> synth -> eval
#
# Evaluation (Phase 2)
# --------------------
# - Uses gcs_core.val_common.compute_all_metrics(...) to compute metrics.
# - Tries gcs_core.val_common.write_summary_with_gcs_core(...) with multiple
#   signatures. If all fail (API drift), falls back to a local writer that emits:
#     • runs/console.txt
#     • runs/summary.jsonl
#     • artifacts/diffusion/summaries/ConditionalDiffusion_eval_summary_seed{SEED}.json
#
# Conventions
# -----------
# - Images are float32, NHWC in [0, 1].
# - Labels are one-hot (N, K) float32.
# - File structure & outputs mirror other repos (GAN, VAE, AR, MAF, RBM, GMM, RBM).
# =============================================================================

from __future__ import annotations

# --- Repo-local imports setup -------------------------------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

import argparse
import json
from typing import Dict, Tuple, Optional, Any, Mapping, Union

import numpy as np
import tensorflow as tf
import yaml

# ------------------------------------------------------------------------------
# Project pipeline
# ------------------------------------------------------------------------------
from diffusion.pipeline import DiffusionPipeline  # type: ignore

# ------------------------------------------------------------------------------
# Frozen core helpers (shared & versioned)
# ------------------------------------------------------------------------------
from gcs_core.synth_loader import resolve_synth_dir, load_synth_any
from gcs_core.val_common import compute_all_metrics, write_summary_with_gcs_core


# =============================================================================
# GPU niceties (safe on CPU-only machines)
# =============================================================================
def _enable_gpu_mem_growth() -> None:
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int = 42) -> None:
    """Deterministic NumPy + TF RNG."""
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_yaml(path: Path) -> Dict:
    """Load a YAML file into a Python dict."""
    with path.open("r") as f:
        return yaml.safe_load(f)


def ensure_dirs(cfg: Dict) -> None:
    """Create artifact directories defined in the config if they do not exist."""
    arts = cfg.get("ARTIFACTS", {})
    for key in ("checkpoints", "synthetic", "summaries", "tensorboard"):
        p = arts.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """Ensure labels are one-hot (shape [N, K], float32)."""
    if y.ndim == 2 and y.shape[1] == num_classes:
        return y.astype("float32")
    return tf.keras.utils.to_categorical(y.astype(int), num_classes=num_classes).astype("float32")


def load_dataset_npy(
    data_dir: Path,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    val_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from four .npy files:
      train_data.npy, train_labels.npy, test_data.npy, test_labels.npy

    Returns images in [0,1] (NHWC), labels as one-hot, and splits the provided
    test set into (val, test) using val_fraction.
    """
    H, W, C = img_shape

    x_train = np.load(data_dir / "train_data.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    x_test  = np.load(data_dir / "test_data.npy")
    y_test  = np.load(data_dir / "test_labels.npy")

    def to_01_hwc(x: np.ndarray) -> np.ndarray:
        x = x.astype("float32")
        if x.max() > 1.5:  # 0..255 → 0..1
            x = x / 255.0
        x = x.reshape((-1, H, W, C))
        return np.clip(x, 0.0, 1.0)

    x_train01 = to_01_hwc(x_train)
    x_test01  = to_01_hwc(x_test)

    y_train1h = one_hot(y_train, num_classes)
    y_test1h  = one_hot(y_test,  num_classes)

    # Split provided test into validation and final test
    n_val = int(len(x_test01) * val_fraction)
    x_val01, y_val1h = x_test01[:n_val], y_test1h[:n_val]
    x_test01, y_test1h = x_test01[n_val:], y_test1h[n_val:]

    return x_train01, y_train1h, x_val01, y_val1h, x_test01, y_test1h


# =============================================================================
# TensorBoard-friendly logging callback
# =============================================================================
def make_log_cb(tboard_dir: Optional[Path]):
    """Return cb(epoch, train_loss, val_loss). Also writes TensorBoard scalars."""
    writer = tf.summary.create_file_writer(str(tboard_dir)) if tboard_dir else None

    def cb(epoch: int, train_loss: float, val_loss: Optional[float]):
        msg = f"[epoch {epoch:05d}] train={train_loss:.4f}" + (f" | val={val_loss:.4f}" if val_loss is not None else "")
        print(msg)
        if writer:
            with writer.as_default():
                tf.summary.scalar("loss/train_total", train_loss, step=epoch)
                if val_loss is not None:
                    tf.summary.scalar("loss/val_total", val_loss, step=epoch)
                writer.flush()
    return cb


# =============================================================================
# Minimal reverse-diffusion preview (robust local sampler for a sanity grid)
# =============================================================================
def _alpha_terms_linear(T: int, beta_start: float, beta_end: float):
    """Return betas, alphas, and cumulative alpha_bars for a linear schedule."""
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars


def _robust_reverse_preview_grid(
    model: tf.keras.Model,
    img_shape: Tuple[int, int, int],
    num_classes: int,
    timesteps: int,
    beta_start: float,
    beta_end: float,
) -> np.ndarray:
    """
    Generate one image per class via a numerically-stable DDPM reverse loop.
    Assumes model takes [x_t, y_onehot, t_vec] and predicts noise ε.
    Outputs array of shape [num_classes, H, W, C] in [0,1].
    """
    H, W, C = img_shape
    n = num_classes

    betas, alphas, alpha_bars = _alpha_terms_linear(timesteps, beta_start, beta_end)
    eps_guard = 1e-6

    # Start from Gaussian noise
    x = tf.random.normal((n, H, W, C), dtype=tf.float32)
    labels = tf.one_hot(tf.range(num_classes), depth=num_classes, dtype=tf.float32)

    for t in reversed(range(timesteps)):
        t_vec = tf.fill([n], tf.cast(t, tf.int32))
        eps_pred = model([x, labels, t_vec], training=False)

        alpha_t = tf.convert_to_tensor(alphas[t], dtype=tf.float32)
        alpha_bar_t = tf.convert_to_tensor(alpha_bars[t], dtype=tf.float32)
        alpha_bar_tm1 = tf.convert_to_tensor(alpha_bars[t - 1] if t > 0 else 1.0, dtype=tf.float32)
        beta_t = tf.convert_to_tensor(betas[t], dtype=tf.float32)

        # Mean update (Ho et al., 2020) with numerical guards
        coef1 = tf.math.rsqrt(tf.maximum(alpha_t, eps_guard))
        coef2 = (1.0 - alpha_t) / tf.sqrt(tf.maximum(1.0 - alpha_bar_t, eps_guard))
        x = coef1 * (x - coef2 * eps_pred)

        # Posterior variance (noise except at t=0)
        var_t = ((1.0 - alpha_bar_tm1) / tf.maximum(1.0 - alpha_bar_t, eps_guard)) * beta_t
        sigma_t = tf.sqrt(tf.maximum(var_t, 0.0))
        if t > 0:
            x = x + sigma_t * tf.random.normal(tf.shape(x), dtype=tf.float32)

        # Guard against inf/nan mid-trajectory
        x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))

    x = tf.clip_by_value(x, 0.0, 1.0)
    return x.numpy().astype("float32")


def _save_preview_grid_png(arr: np.ndarray, path: Path) -> None:
    """Save a horizontal grid (one image per class) as PNG."""
    import matplotlib.pyplot as plt  # lazy import
    n = arr.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 1.6))
    if n == 1:
        axes = [axes]
    for i in range(n):
        img = arr[i]
        if img.shape[-1] == 1:
            axes[i].imshow(img[..., 0], cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axes[i].imshow(np.clip(img, 0.0, 1.0))
        axes[i].set_axis_off()
        axes[i].set_title(f"C{i}", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
# Evaluation helpers: mapping, deltas, console & local writer
# =============================================================================
def _map_util_names(util_block: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Normalize utility metric names to a stable schema.
    Accepts either 'bal_acc' or 'balanced_accuracy', etc.
    """
    if not util_block:
        return {}
    bal = util_block.get("balanced_accuracy", util_block.get("bal_acc"))
    return {
        "accuracy":               util_block.get("accuracy"),
        "macro_f1":               util_block.get("macro_f1"),
        "balanced_accuracy":      bal,
        "macro_auprc":            util_block.get("macro_auprc"),
        "recall_at_1pct_fpr":     util_block.get("recall_at_1pct_fpr"),
        "ece":                    util_block.get("ece"),
        "brier":                  util_block.get("brier"),
        "per_class":              util_block.get("per_class"),
    }


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    return None if (a is None or b is None) else float(a - b)


def _build_phase2_record(
    *,
    model_name: str,
    seed: int,
    images_counts: Mapping[str, Optional[int]],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Construct Phase-2 aggregator record from metrics and counts."""
    util_R  = _map_util_names(metrics.get("real_only"))
    util_RS = _map_util_names(metrics.get("real_plus_synth"))

    deltas = {
        "accuracy":           _delta(util_RS.get("accuracy"),          util_R.get("accuracy")),
        "macro_f1":           _delta(util_RS.get("macro_f1"),          util_R.get("macro_f1")),
        "balanced_accuracy":  _delta(util_RS.get("balanced_accuracy"), util_R.get("balanced_accuracy")),
        "macro_auprc":        _delta(util_RS.get("macro_auprc"),       util_R.get("macro_auprc")),
        "recall_at_1pct_fpr": _delta(util_RS.get("recall_at_1pct_fpr"),util_R.get("recall_at_1pct_fpr")),
        "ece":                _delta(util_RS.get("ece"),               util_R.get("ece")),
        "brier":              _delta(util_RS.get("brier"),             util_R.get("brier")),
    }

    generative = {
        "fid":          metrics.get("fid_macro"),
        "fid_macro":    metrics.get("fid_macro"),
        "cfid_macro":   metrics.get("cfid_macro"),
        "js":           metrics.get("js"),
        "kl":           metrics.get("kl"),
        "diversity":    metrics.get("diversity"),
        # extras preserved if present:
        "cfid_per_class": metrics.get("cfid_per_class"),
        "fid_domain":     metrics.get("fid_domain"),
    }

    return {
        "model": model_name,
        "seed":  int(seed),
        "images": {
            "train_real": int(images_counts.get("train_real") or 0),
            "val_real":   int(images_counts.get("val_real") or 0),
            "test_real":  int(images_counts.get("test_real") or 0),
            "synthetic":  (int(images_counts["synthetic"]) if images_counts.get("synthetic") is not None else None),
        },
        "generative": generative,
        "utility_real_only": util_R,
        "utility_real_plus_synth": util_RS,
        "deltas_RS_minus_R": deltas,
    }


def _write_console_block(record: Dict[str, Any]) -> str:
    """Format a concise console block and return it."""
    gen = record.get("generative", {})
    util_R  = record.get("utility_real_only", {})
    util_RS = record.get("utility_real_plus_synth", {})
    counts  = record.get("images", {})
    lines = [
        f"Model: {record.get('model')}   Seed: {record.get('seed')}",
        f"Counts → train:{counts.get('train_real')}  "
        f"val:{counts.get('val_real')}  "
        f"test:{counts.get('test_real')}  "
        f"synth:{counts.get('synthetic')}",
        f"Generative → FID(macro): {gen.get('fid_macro')}  cFID(macro): {gen.get('cfid_macro')}  "
        f"JS: {gen.get('js')}  KL: {gen.get('kl')}  Div: {gen.get('diversity')}",
        f"Utility R   → acc: {util_R.get('accuracy')}  bal_acc: {util_R.get('balanced_accuracy')}  "
        f"macro_f1: {util_R.get('macro_f1')}",
        f"Utility R+S → acc: {util_RS.get('accuracy')}  bal_acc: {util_RS.get('balanced_accuracy')}  "
        f"macro_f1: {util_RS.get('macro_f1')}",
    ]
    return "\n".join(lines) + "\n"


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def _save_console(path: Path, block: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(block)


# -----------------------------------------------------------------------------
# Writer shim (tries gcs_core variants; falls back to local writer)
# -----------------------------------------------------------------------------
PathLike = Union[str, Path]

def _build_real_dirs(data_dir: Path) -> Dict[str, str]:
    """Stable 'real_dirs' mapping used by newer writers."""
    return {
        "train": str(data_dir / "train_data.npy"),
        "val":   f"{data_dir}/(split of test_data.npy)",
        "test":  f"{data_dir}/(split of test_data.npy)",
    }


def _ensure_images_block(record: Dict[str, Any], images_counts: Mapping[str, Optional[int]]) -> None:
    """Ensure 'images' counts exist even if the core writer omitted them."""
    record.setdefault("images", {})
    for k, v in images_counts.items():
        if record["images"].get(k) is None:
            record["images"][k] = v


def _try_core_writer(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: str,
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str,
    output_json_path: PathLike,
    output_console_path: PathLike,
) -> Optional[Dict[str, Any]]:
    """
    Try multiple signatures of gcs_core.write_summary_with_gcs_core.
    Returns the record on success; None if every attempt fails.
    """
    base_kwargs = dict(
        model_name=model_name,
        seed=seed,
        fid_cap_per_class=fid_cap_per_class,
        output_json=str(output_json_path),
        output_console=str(output_console_path),
        metrics=dict(metrics),
        notes=notes,
    )
    real_dirs = _build_real_dirs(data_dir)

    attempts = [
        # Newest: real_dirs + images_counts + synth_dir
        dict(base_kwargs, real_dirs=real_dirs, images_counts=dict(images_counts), synth_dir=synth_dir),
        # Mid: real_dirs + synth_dir (no images_counts)
        dict(base_kwargs, real_dirs=real_dirs, synth_dir=synth_dir),
        # Old: synth_dir only
        dict(base_kwargs, synth_dir=synth_dir),
    ]

    for kw in attempts:
        try:
            rec = write_summary_with_gcs_core(**kw)
            _ensure_images_block(rec, images_counts)
            return rec
        except TypeError:
            # Signature mismatch – try the next layout
            continue
        except Exception:
            # Any other internal failure – try next layout
            continue
    return None


def _local_write_summary(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: str,
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str,
    output_json_path: PathLike,
    output_console_path: PathLike,
) -> Dict[str, Any]:
    """
    Build the Phase-2 record locally and write console + JSONL outputs.
    Used when gcs_core writer signatures don't match (API drift).
    """
    record = _build_phase2_record(
        model_name=model_name,
        seed=seed,
        images_counts=images_counts,
        metrics=metrics,
    )
    # Include a few extras for traceability:
    record.setdefault("meta", {})
    record["meta"].update({
        "notes": notes,
        "fid_cap_per_class": int(fid_cap_per_class),
        "synth_dir": synth_dir,
        "real_dirs": _build_real_dirs(data_dir),
    })

    console_block = _write_console_block(record)
    _save_console(Path(output_console_path), console_block)
    _append_jsonl(Path(output_json_path), record)
    return record


def _write_phase2_summary(
    *,
    model_name: str,
    seed: int,
    data_dir: Path,
    synth_dir: Optional[str],
    fid_cap_per_class: int,
    metrics: Mapping[str, Any],
    images_counts: Mapping[str, Optional[int]],
    notes: str = "",
    output_json_path: PathLike = "runs/summary.jsonl",
    output_console_path: PathLike = "runs/console.txt",
) -> Dict[str, Any]:
    """
    Version-agnostic writer:
      1) Try multiple gcs_core signatures.
      2) If all fail, write locally with a schema-compatible record.
    """
    sdir = synth_dir or ""
    # 1) Try core writer in descending order of modernity
    rec = _try_core_writer(
        model_name=model_name,
        seed=seed,
        data_dir=data_dir,
        synth_dir=sdir,
        fid_cap_per_class=fid_cap_per_class,
        metrics=metrics,
        images_counts=images_counts,
        notes=notes,
        output_json_path=output_json_path,
        output_console_path=output_console_path,
    )
    if rec is not None:
        return rec

    # 2) Fallback to local writer
    return _local_write_summary(
        model_name=model_name,
        seed=seed,
        data_dir=data_dir,
        synth_dir=sdir,
        fid_cap_per_class=fid_cap_per_class,
        metrics=metrics,
        images_counts=images_counts,
        notes=notes,
        output_json_path=output_json_path,
        output_console_path=output_console_path,
    )


# =============================================================================
# Orchestration
# =============================================================================
def run_train(cfg: Dict) -> None:
    """Train diffusion model and save a small preview grid."""
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])

    x_train01, y_train, x_val01, y_val, _, _ = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    pipe = DiffusionPipeline(cfg | {"LOG_CB": make_log_cb(Path(cfg["ARTIFACTS"]["tensorboard"]))})
    model = pipe.train(x_train=x_train01, y_train=y_train, x_val=x_val01, y_val=y_val)

    # Save a small grid preview (robust local sampler)
    preview_path = Path(cfg["ARTIFACTS"]["summaries"]) / "train_preview.png"
    T = int(cfg.get("TIMESTEPS", 1000))
    beta_start = float(cfg.get("BETA_START", 1e-4))
    beta_end = float(cfg.get("BETA_END", 2e-2))
    try:
        grid = _robust_reverse_preview_grid(
            model,
            img_shape=img_shape,
            num_classes=num_classes,
            timesteps=T,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        _save_preview_grid_png(grid, preview_path)
        print(f"Saved preview grid to {preview_path}")
    except Exception as e:
        # Preview is best-effort; keep training output usable even if it fails.
        print(f"[preview] WARN: could not generate preview grid -> {e}")


def run_synth(cfg: Dict) -> None:
    """Synthesize per-class dataset using the pipeline (loads best/latest checkpoint)."""
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)

    pipe = DiffusionPipeline(cfg)
    x_s, y_s = pipe.synthesize()  # pipeline handles checkpoint selection & sampling
    synth_dir = Path(cfg["ARTIFACTS"]["synthetic"])
    print(f"Synthesized: {x_s.shape[0]} samples (saved under {synth_dir}).")


def run_eval(cfg: Dict, include_synth: bool) -> None:
    """
    Run standardized evaluation:
      - Generative quality (FID/cFID/JS/KL/Diversity) on VAL vs SYNTH.
      - Downstream utility on REAL test with the fixed small CNN.
      - Writes:
          * runs/console.txt
          * runs/summary.jsonl
          * ARTIFACTS/summaries/*.json
    """
    set_seed(cfg.get("SEED", 42))
    ensure_dirs(cfg)
    Path("runs").mkdir(exist_ok=True)

    data_dir    = Path(cfg["DATA_DIR"])
    img_shape   = tuple(cfg["IMG_SHAPE"])
    num_classes = int(cfg["NUM_CLASSES"])

    # --- Load REAL data (all in [0,1]) ---
    x_train01, y_train, x_val01, y_val, x_test01, y_test = load_dataset_npy(
        data_dir, img_shape, num_classes, val_fraction=cfg.get("VAL_FRACTION", 0.5)
    )

    # --- SYNTH (optional) -----------------------------------------------------
    x_s, y_s = (None, None)
    synth_dir_str = ""
    if include_synth:
        repo_root = Path(__file__).resolve().parents[1]
        synth_dir = resolve_synth_dir(cfg, repo_root)
        synth_dir_str = str(synth_dir)
        try:
            x_s, y_s = load_synth_any(synth_dir, num_classes)
            # sanitize
            if x_s is not None and y_s is not None:
                x_s = np.asarray(x_s, dtype=np.float32)
                y_s = np.asarray(y_s, dtype=np.float32)
                if x_s.ndim == 3:
                    H, W, C = img_shape
                    x_s = x_s.reshape((-1, H, W, C))
            if isinstance(x_s, np.ndarray):
                print(f"[eval] Using synthetic from {synth_dir} (N={len(x_s)})")
            else:
                print(f"[eval] WARN: no usable synthetic under {synth_dir}; proceeding REAL-only.")
        except Exception as e:
            print(f"[eval] WARN: could not load synthetic -> {e}. Proceeding REAL-only.")
            x_s, y_s = None, None

    # --- Compute metrics ------------------------------------------------------
    metrics = compute_all_metrics(
        img_shape=img_shape,
        x_train_real=x_train01, y_train_real=y_train,
        x_val_real=x_val01,     y_val_real=y_val,
        x_test_real=x_test01,   y_test_real=y_test,
        x_synth=x_s,            y_synth=y_s,
        fid_cap_per_class=int(cfg.get("FID_CAP", 200)),
        seed=int(cfg.get("SEED", 42)),
        domain_embed_fn=None,
        epochs=int(cfg.get("EVAL_EPOCHS", 20)),
    )

    # --- Writer shim: core (multiple signatures) -> local fallback -----------
    images_counts = {
        "train_real": len(x_train01),
        "val_real":   len(x_val01),
        "test_real":  len(x_test01),
        "synthetic":  (len(x_s) if isinstance(x_s, np.ndarray) else None),
    }

    record = _write_phase2_summary(
        model_name="ConditionalDiffusion",
        seed=int(cfg.get("SEED", 42)),
        data_dir=data_dir,
        synth_dir=synth_dir_str or str(Path(cfg["ARTIFACTS"]["synthetic"])),
        fid_cap_per_class=int(cfg.get("FID_CAP", 200)),
        metrics=metrics,
        images_counts=images_counts,
        notes="phase2-real",
        output_json_path="runs/summary.jsonl",
        output_console_path="runs/console.txt",
    )

    # --- Pretty JSON copy under artifacts/summaries ---------------------------
    out_path = Path(cfg["ARTIFACTS"]["summaries"]) / f"ConditionalDiffusion_eval_summary_seed{int(cfg.get('SEED', 0))}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(record, f, indent=2)
    print(f"Saved evaluation summary to {out_path}")


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional Diffusion pipeline runner")
    p.add_argument("command", choices=["train", "synth", "eval", "all"], help="Which step to run")
    p.add_argument("--config", default="config.yaml", help="Path to YAML config")
    p.add_argument("--no-synth", action="store_true", help="(for eval/all) skip synthetic data in evaluation")
    return p.parse_args()


def main() -> None:
    _enable_gpu_mem_growth()

    args = parse_args()
    cfg = load_yaml(Path(args.config))

    # Sensible defaults (non-destructive; keep parity with other repos)
    cfg.setdefault("SEED", 42)
    cfg.setdefault("VAL_FRACTION", 0.5)
    cfg.setdefault("FID_CAP", 200)
    cfg.setdefault("TIMESTEPS", 1000)
    cfg.setdefault("BETA_START", 1e-4)
    cfg.setdefault("BETA_END", 2e-2)
    cfg.setdefault("EVAL_EPOCHS", 20)              # evaluator CNN epochs
    cfg.setdefault("IMG_SHAPE", [40, 40, 1])
    cfg.setdefault("NUM_CLASSES", 9)
    cfg.setdefault("ARTIFACTS", {})
    cfg["ARTIFACTS"].setdefault("checkpoints", "artifacts/diffusion/checkpoints")
    cfg["ARTIFACTS"].setdefault("synthetic",   "artifacts/diffusion/synthetic")
    cfg["ARTIFACTS"].setdefault("summaries",   "artifacts/diffusion/summaries")
    cfg["ARTIFACTS"].setdefault("tensorboard", "artifacts/tensorboard")

    print(f"[config] Using {Path(args.config).resolve()}")
    print(f"Synth outputs -> {Path(cfg['ARTIFACTS']['synthetic']).resolve()}")

    if args.command == "train":
        run_train(cfg)
    elif args.command == "synth":
        run_synth(cfg)
    elif args.command == "eval":
        run_eval(cfg, include_synth=not args.no_synth)
    elif args.command == "all":
        run_train(cfg)
        run_synth(cfg)
        run_eval(cfg, include_synth=not args.no_synth)


if __name__ == "__main__":
    main()
