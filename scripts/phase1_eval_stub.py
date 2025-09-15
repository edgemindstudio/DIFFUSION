# scripts/phase1_eval_stub.py

from __future__ import annotations
import argparse
from pathlib import Path
from gcs_core.val_common import evaluate_model_suite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)  # e.g., diffusion
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-dir", default="USTC-TFC2016_malware/train")
    p.add_argument("--val-dir",   default="USTC-TFC2016_malware/val")
    p.add_argument("--test-dir",  default="USTC-TFC2016_malware/test")
    p.add_argument("--synth-dir", default="artifacts/synthetic")
    p.add_argument("--fid-cap", type=int, default=200)
    p.add_argument("--json", default="runs/summary.jsonl")
    p.add_argument("--console", default="runs/console.txt")
    args = p.parse_args()

    Path(args.synth_dir).mkdir(parents=True, exist_ok=True)
    Path("runs").mkdir(exist_ok=True)

    # Phase 1: write a valid console block + one-line JSON (stub metrics)
    evaluate_model_suite(
        model_name=args.model,
        seed=args.seed,
        real_dirs={"train": args.train_dir, "val": args.val_dir, "test": args.test_dir},
        synth_dir=args.synth_dir,
        fid_cap_per_class=args.fid_cap,
        evaluator="small_cnn_v1",
        output_json=args.json,
        output_console=args.console,
        metrics=None,               # Phase 2 will pass REAL metrics here
        notes="phase1-stub"
    )
    print("Wrote:", args.console, "and", args.json)

if __name__ == "__main__":
    main()
