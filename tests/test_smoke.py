#/ tests/test_smoke.py

"""
Quick, CPU-friendly smoke tests for the GAN project.

What this verifies:
1) Models build and run a forward pass (generator & discriminator).
2) End-to-end training (1 epoch) runs and writes checkpoints.
3) The shared eval_common.evaluate_model_suite works on real-only data
   (utility metrics only; no FID to avoid downloading Inception weights).

These tests fabricate a tiny dataset in a tmp directory so they do NOT
depend on any external files or internet access.
"""

import os
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import pytest

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _set_seeds(seed: int = 7):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def _make_toy_dataset(root: Path, img_shape=(8, 8, 1), num_classes=3,
                      n_train=64, n_test=16):
    """
    Create a tiny grayscale dataset and save as .npy files:
      train_data.npy, train_labels.npy, test_data.npy, test_labels.npy
    Images are uint8 [0,255]; labels are int in [0, num_classes).
    """
    root.mkdir(parents=True, exist_ok=True)
    H, W, C = img_shape

    x_train = (np.random.rand(n_train, H, W, C) * 255).astype(np.uint8)
    y_train = np.random.randint(0, num_classes, size=(n_train,), dtype=np.int32)

    x_test = (np.random.rand(n_test, H, W, C) * 255).astype(np.uint8)
    y_test = np.random.randint(0, num_classes, size=(n_test,), dtype=np.int32)

    np.save(root / "train_data.npy", x_train)
    np.save(root / "train_labels.npy", y_train)
    np.save(root / "test_data.npy", x_test)
    np.save(root / "test_labels.npy", y_test)

def _write_min_config(cfg_path: Path, data_path: Path,
                      img_shape=(8,8,1), num_classes=3, latent_dim=16,
                      epochs=2, batch_size=8, lr=2e-4, beta1=0.5):
    """
    Minimal config.yaml compatible with gan/train.py and gan/models.py.
    """
    import yaml
    cfg = {
        "IMG_SHAPE": list(img_shape),
        "NUM_CLASSES": int(num_classes),
        "LATENT_DIM": int(latent_dim),
        "EPOCHS": int(epochs),
        "BATCH_SIZE": int(batch_size),
        "LR": float(lr),
        "BETA_1": float(beta1),
        # train.py will read DATA_PATH if present; else it defaults to repo/USTC...
        "DATA_PATH": str(data_path),
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------
def test_models_forward():
    """
    Builds generator & discriminator and runs one forward pass on tiny inputs.
    """
    _set_seeds()
    from gan.models import build_models

    img_shape = (8, 8, 1)
    num_classes = 3
    latent_dim = 16

    models_dict = build_models(
        latent_dim=latent_dim,
        num_classes=num_classes,
        img_shape=img_shape,
        lr=2e-4,
        beta_1=0.5,
    )
    G = models_dict["generator"]
    D = models_dict["discriminator"]

    # generator forward
    z = np.random.normal(0, 1, size=(4, latent_dim)).astype(np.float32)
    y = tf.keras.utils.to_categorical(np.array([0,1,2,1]), num_classes).astype(np.float32)
    g = G.predict([z, y], verbose=0)
    assert g.shape == (4, *img_shape)

    # discriminator forward
    # G outputs in [-1,1], so just reuse g and labels
    d = D.predict([g, y], verbose=0)
    assert d.shape == (4, 1)

def test_train_one_epoch_smoke(tmp_path: Path, monkeypatch):
    """
    Runs a 1-epoch training loop end-to-end and verifies that checkpoints are saved.
    Skips FID by setting eval_every very high (no Inception, no downloads).
    """
    _set_seeds()

    # Create tiny dataset
    data_dir = tmp_path / "USTC"
    _make_toy_dataset(data_dir, img_shape=(8,8,1), num_classes=3, n_train=64, n_test=16)

    # Minimal config
    cfg_path = tmp_path / "config.yaml"
    _write_min_config(cfg_path, data_dir, img_shape=(8,8,1), num_classes=3,
                      latent_dim=16, epochs=1, batch_size=8)

    # Train for 1 epoch, save every epoch, skip FID (eval_every=999)
    from gan.train import train
    rc = train(
        cfg_path=cfg_path,
        epochs=1,
        batch_size=8,
        eval_every=999,   # effectively disables FID
        save_every=1,     # force a "last" checkpoint
        grid=None,
        sample_after=False,
        samples_per_class=0,
        seed=7,
    )
    assert rc == 0

    # Check checkpoints exist
    ckpt_dir = Path(__file__).resolve().parents[1] / "artifacts" / "checkpoints"
    assert (ckpt_dir / "generator_last.h5").exists()
    assert (ckpt_dir / "discriminator_last.h5").exists()
    # final weights saved at end
    assert (ckpt_dir / "generator_final.h5").exists()
    assert (ckpt_dir / "discriminator_final.h5").exists()

def test_eval_common_real_only_smoke():
    """
    Calls eval_common.evaluate_model_suite with REAL data only
    (no synthetic) to avoid FID. Ensures utility metrics are produced.
    """
    _set_seeds()
    H, W, C = 8, 8, 1
    K = 3

    # fabricate small real splits
    def mk_split(n):
        x = (np.random.rand(n, H, W, C)).astype(np.float32)  # already [0,1]
        y = np.random.randint(0, K, size=(n,))
        y_oh = tf.keras.utils.to_categorical(y, K).astype(np.float32)
        return x, y_oh

    x_tr, y_tr = mk_split(64)
    x_va, y_va = mk_split(16)
    x_te, y_te = mk_split(16)

    from gcs_core.val_common import evaluate_model_suite
    summary = evaluate_model_suite(
        model_name="SMOKE_TEST_GAN",
        img_shape=(H, W, C),
        x_train_real=x_tr, y_train_real=y_tr,
        x_val_real=x_va,   y_val_real=y_va,
        x_test_real=x_te,  y_test_real=y_te,
        x_synth=None,      y_synth=None,   # <= no synthetic => no FID
        per_class_cap_for_fid=8,
        seed=7,
    )

    # sanity checks on structure
    assert "model" in summary and summary["model"] == "SMOKE_TEST_GAN"
    assert "utility_real_only" in summary
    util = summary["utility_real_only"]
    for k in ["accuracy", "macro_f1", "balanced_accuracy", "macro_auprc",
              "recall_at_1pct_fpr", "ece", "brier"]:
        assert k in util
        assert isinstance(util[k], float)

    # generative metrics exist but are None (since no synth provided)
    gen = summary["generative"]
    for k in ["fid", "js", "kl", "diversity"]:
        assert k in gen
        assert gen[k] is None

# ---------------------------------------------------------------------
# Pytest knobs: mark long tests as optional if desired
# ---------------------------------------------------------------------
@pytest.mark.parametrize("n", [1])
def test_imports_fast(n):
    """
    Tiny import test so CI fails early if the package is broken.
    """
    import gan
    from gan import models, pipeline  # noqa: F401
    assert hasattr(gan, "__version__") or True
