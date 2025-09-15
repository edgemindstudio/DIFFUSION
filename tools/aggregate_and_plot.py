# tools/aggregate_and_plot.py

#!/usr/bin/env python3
"""
Aggregate evaluation summaries from multiple runs (and models) and produce
publication-quality plots for synthetic-data comparative studies.

Outputs
-------
PNG figures into --outdir:
  Utility (↑):            accuracy, macro_f1, balanced_accuracy, macro_auprc, recall_at_1pct_fpr
  Calibration (↓):        ece, brier
  Per-class ΔF1:          (Real+Synth - Real) bar plot per model
  Generative (↓/↑):       cfid_macro bars, cfid_per_class heatmap, JS (↓), KL (↓), Diversity (↑)

Inputs
------
Each JSON file is an output of `evaluate_model_suite` and must contain:
  - "model": str
  - "utility_real_only":      dict with metrics and "per_class" sub-dict (precision/recall/f1/support)
  - "utility_real_plus_synth": same keys as above
  - "generative":             dict with "cfid_macro", "cfid_per_class", "js", "kl", "diversity"
Non-conforming files are ignored.

Usage
-----
# Scan all subtrees for summaries
python tools/aggregate_and_plot.py \
  --glob "artifacts/**/summaries/*.json" \
  --outdir reports

# Multiple sources (pass --glob multiple times)
python tools/aggregate_and_plot.py \
  --glob "artifacts/gan/summaries/*.json" \
  --glob "../GMM/artifacts/gmm/summaries/*.json" \
  --outdir reports
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# Styling (matplotlib only; no seaborn)
# -----------------------------
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
    }
)


# -----------------------------
# I/O + aggregation
# -----------------------------
def load_summaries(globs: List[str]) -> List[Dict]:
    """Load all JSON summaries matched by the provided glob patterns."""
    files: List[str] = []
    for g in globs:
        files.extend(glob.glob(g, recursive=True))
    files = sorted(set(files))

    out: List[Dict] = []
    for fp in files:
        try:
            with open(fp, "r") as f:
                s = json.load(f)
        except Exception:
            continue
        if not isinstance(s, dict):
            continue
        # minimal schema guard
        if "model" not in s:
            continue
        if "utility_real_only" not in s or "utility_real_plus_synth" not in s:
            continue
        out.append(s)
    return out


def group_by_model(summaries: List[Dict]) -> Dict[str, List[Dict]]:
    """Group summary dicts by their 'model' field."""
    by_model: Dict[str, List[Dict]] = defaultdict(list)
    for s in summaries:
        by_model[str(s.get("model", "UNKNOWN"))].append(s)
    # Return as a regular dict with sorted keys for stable plot ordering
    return {k: by_model[k] for k in sorted(by_model.keys())}


# -----------------------------
# Stats helpers
# -----------------------------
def mean_ci95(values: List[float]) -> Tuple[float, float]:
    """
    Compute mean and 95% CI half-width using normal approximation.
    For n<=1, CI half-width is 0.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    m = float(np.mean(arr))
    if arr.size <= 1:
        return m, 0.0
    sd = float(np.std(arr, ddof=1))
    hw = 1.96 * sd / math.sqrt(arr.size)
    return m, hw


# -----------------------------
# Plot helpers
# -----------------------------
def _bar_two_groups(ax, labels, means_R, ci_R, means_RS, ci_RS, title, ylabel):
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, means_R, width, yerr=ci_R, capsize=3, label="Real only")
    ax.bar(x + width / 2, means_RS, width, yerr=ci_RS, capsize=3, label="Real + Synth")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def _save(fig, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.png")
    plt.close(fig)


# -----------------------------
# Figure set 1: Utility (↑)
# -----------------------------
UTILITY_KEYS = [
    ("accuracy", "Accuracy (↑)"),
    ("macro_f1", "Macro F1 (↑)"),
    ("balanced_accuracy", "Balanced Acc (↑)"),
    ("macro_auprc", "Macro AUPRC (↑)"),
    ("recall_at_1pct_fpr", "R@1%FPR (↑)"),
]


def plot_utility_bars(by_model: Dict[str, List[Dict]], outdir: Path):
    """Paired bars (Real vs Real+Synth) with mean ±95% CI for each model."""
    for key, pretty in UTILITY_KEYS:
        labels: List[str] = []
        means_R: List[float] = []
        means_RS: List[float] = []
        ci_R: List[float] = []
        ci_RS: List[float] = []

        for model in by_model:
            runs = by_model[model]
            vals_R = [r["utility_real_only"].get(key) for r in runs]
            vals_RS = [r["utility_real_plus_synth"].get(key) for r in runs]
            vals_R = [v for v in vals_R if v is not None]
            vals_RS = [v for v in vals_RS if v is not None]
            if not vals_R or not vals_RS:
                continue
            mR, hR = mean_ci95(vals_R)
            mS, hS = mean_ci95(vals_RS)
            labels.append(model)
            means_R.append(mR)
            ci_R.append(hR)
            means_RS.append(mS)
            ci_RS.append(hS)

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(labels) + 2), 4.0))
        _bar_two_groups(ax, labels, means_R, ci_R, means_RS, ci_RS, f"Utility: {pretty}", pretty)
        _save(fig, outdir, f"utility_{key}")


# -----------------------------
# Figure set 2: Calibration (↓)
# -----------------------------
def plot_calibration_bars(by_model: Dict[str, List[Dict]], outdir: Path):
    for key, pretty in [("ece", "ECE (↓)"), ("brier", "Brier (↓)")]:
        labels: List[str] = []
        means_R: List[float] = []
        means_RS: List[float] = []
        ci_R: List[float] = []
        ci_RS: List[float] = []

        for model in by_model:
            runs = by_model[model]
            vals_R = [r["utility_real_only"].get(key) for r in runs]
            vals_RS = [r["utility_real_plus_synth"].get(key) for r in runs]
            vals_R = [v for v in vals_R if v is not None]
            vals_RS = [v for v in vals_RS if v is not None]
            if not vals_R or not vals_RS:
                continue
            mR, hR = mean_ci95(vals_R)
            mS, hS = mean_ci95(vals_RS)
            labels.append(model)
            means_R.append(mR)
            ci_R.append(hR)
            means_RS.append(mS)
            ci_RS.append(hS)

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(labels) + 2), 4.0))
        _bar_two_groups(ax, labels, means_R, ci_R, means_RS, ci_RS, f"Calibration: {pretty}", pretty)
        _save(fig, outdir, f"calibration_{key}")


# -----------------------------
# Figure set 3: Per-class ΔF1 (RS − R)
# -----------------------------
def plot_per_class_delta_f1(by_model: Dict[str, List[Dict]], outdir: Path):
    """
    For each model, compute per-class ΔF1 = F1(Real+Synth) − F1(Real).
    Bars show seed-mean ±95% CI (if multiple seeds available).
    """
    for model in by_model:
        runs = by_model[model]
        deltas: List[np.ndarray] = []

        for r in runs:
            pc_R = (r.get("utility_real_only") or {}).get("per_class") or {}
            pc_RS = (r.get("utility_real_plus_synth") or {}).get("per_class") or {}
            f1_R = pc_R.get("f1")
            f1_RS = pc_RS.get("f1")
            if not (isinstance(f1_R, list) and isinstance(f1_RS, list)):
                continue
            a = np.asarray(f1_R, dtype=float)
            b = np.asarray(f1_RS, dtype=float)
            if a.shape != b.shape:
                continue
            deltas.append(b - a)

        if not deltas:
            continue

        D = np.stack(deltas, axis=0)  # (seeds, classes)
        mean_delta = np.mean(D, axis=0)
        if D.shape[0] > 1:
            hw = 1.96 * np.std(D, axis=0, ddof=1) / np.sqrt(D.shape[0])
        else:
            hw = np.zeros_like(mean_delta)

        cls_labels = [f"C{k}" for k in range(len(mean_delta))]
        x = np.arange(len(cls_labels))
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cls_labels) + 2), 4.0))
        ax.bar(x, mean_delta, yerr=hw, capsize=3)
        ax.axhline(0.0, color="k", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(cls_labels)
        ax.set_ylabel("Δ F1 (RS − R)")
        ax.set_title(f"Per-class ΔF1: {model}")
        _save(fig, outdir, f"delta_f1_per_class_{model}")


# -----------------------------
# Figure set 4: cFID heatmap + macro bars
# -----------------------------
def plot_cfid(by_model: Dict[str, List[Dict]], outdir: Path):
    # Macro bars first (works even without per-class arrays)
    macro_labels: List[str] = []
    macro_vals: List[float] = []
    macro_ci: List[float] = []

    for model in by_model:
        runs = by_model[model]
        vals = [(r.get("generative") or {}).get("cfid_macro") for r in runs]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        m, h = mean_ci95(vals)
        macro_labels.append(model)
        macro_vals.append(m)
        macro_ci.append(h)

    if macro_labels:
        fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(macro_labels) + 2), 4.0))
        x = np.arange(len(macro_labels))
        ax.bar(x, macro_vals, yerr=macro_ci, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(macro_labels, rotation=20, ha="right")
        ax.set_ylabel("cFID (macro) ↓")
        ax.set_title("Class-conditional FID (macro)")
        _save(fig, outdir, "cfid_macro_bars")

    # Per-class heatmap (requires cfid_per_class arrays)
    model_names: List[str] = []
    per_model_means: List[np.ndarray] = []
    max_classes = 0

    for model in by_model:
        runs = by_model[model]
        arrays = []
        for r in runs:
            arr = (r.get("generative") or {}).get("cfid_per_class")
            if isinstance(arr, list) and arr:
                arrays.append(np.array(arr, dtype=float))
        if not arrays:
            continue
        max_len = max(len(a) for a in arrays)
        # pad with NaNs for unequal class counts, then average
        padded = [np.pad(a, (0, max_len - len(a)), constant_values=np.nan) for a in arrays]
        mean_arr = np.nanmean(np.stack(padded, axis=0), axis=0)
        model_names.append(model)
        per_model_means.append(mean_arr)
        max_classes = max(max_classes, max_len)

    if per_model_means:
        # Normalize shapes
        M = np.stack(
            [
                a if len(a) == max_classes else np.pad(a, (0, max_classes - len(a)), constant_values=np.nan)
                for a in per_model_means
            ],
            axis=1,
        )  # shape: (classes, models)
        classes = [f"C{k}" for k in range(max_classes)]
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(model_names) + 3), max(6, 0.4 * max_classes + 2)))
        im = ax.imshow(M, aspect="auto", interpolation="nearest")
        ax.set_yticks(np.arange(max_classes))
        ax.set_yticklabels(classes)
        ax.set_xticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.set_title("cFID per class (lower is better)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        _save(fig, outdir, "cfid_per_class_heatmap")


# -----------------------------
# Figure set 5: Generative diagnostics (optional)
# -----------------------------
def plot_generative_diagnostics(by_model: Dict[str, List[Dict]], outdir: Path):
    for key, pretty in [("js", "JS divergence (↓)"), ("kl", "KL divergence (↓)"), ("diversity", "Diversity (↑)")]:
        labels: List[str] = []
        means: List[float] = []
        cis: List[float] = []

        for model in by_model:
            runs = by_model[model]
            vals = [(r.get("generative") or {}).get(key) for r in runs]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            m, h = mean_ci95(vals)
            labels.append(model)
            means.append(m)
            cis.append(h)

        if not labels:
            continue

        fig, ax = plt.subplots(figsize=(max(4.5, 1.2 * len(labels) + 2), 4.0))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=cis, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(pretty)
        ax.set_title(pretty)
        _save(fig, outdir, f"gen_{key}_bars")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate eval summaries and produce plots.")
    ap.add_argument(
        "--glob",
        action="append",
        required=True,
        help="Glob pattern(s) to JSON summaries. Use multiple --glob to combine sources.",
    )
    ap.add_argument("--outdir", default="reports", help="Directory to write PNG plots.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    summaries = load_summaries(args.glob)
    if not summaries:
        print("No summaries found. Check your --glob patterns.")
        return 1

    by_model = group_by_model(summaries)

    # Generate all figures
    plot_utility_bars(by_model, outdir)
    plot_calibration_bars(by_model, outdir)
    plot_per_class_delta_f1(by_model, outdir)
    plot_cfid(by_model, outdir)
    plot_generative_diagnostics(by_model, outdir)

    print(f"Wrote plots to {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
