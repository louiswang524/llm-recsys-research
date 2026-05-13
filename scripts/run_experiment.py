#!/usr/bin/env python3
"""End-to-end pipeline: download → preprocess → vocab → train → evaluate.

Usage:
  python scripts/run_experiment.py                          # defaults
  python scripts/run_experiment.py --dataset amazon_beauty --stage cpt+sft
  python scripts/run_experiment.py --skip-download --skip-preprocess
  python scripts/run_experiment.py --item-tokenizer semantic_id --loss lm_engagement

Colab example:
  !python scripts/run_experiment.py --movielens ml-1m --model qwen2_7b_lora
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {desc}")
    print(f"{'=' * 60}")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n[FAILED] {desc} exited with code {result.returncode}")
        sys.exit(result.returncode)


def hydra_args(overrides: dict) -> list[str]:
    return [f"{k}={v}" for k, v in overrides.items() if v is not None]


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end LLM-RecSys experiment pipeline")

    # Dataset
    parser.add_argument("--dataset", default="movielens_1m",
                        choices=["movielens_1m", "movielens_20m", "amazon_beauty"],
                        help="Dataset config name")
    parser.add_argument("--movielens", default="ml-1m", choices=["ml-1m", "ml-20m"],
                        help="MovieLens version to download")
    parser.add_argument("--amazon-category", default="",
                        help="Amazon category (e.g. All_Beauty) if using Amazon dataset")

    # Model
    parser.add_argument("--model", default="qwen2_7b_lora",
                        choices=["qwen2_7b_lora", "llama3_8b_lora"])
    parser.add_argument("--item-tokenizer", default="text", choices=["text", "semantic_id"])

    # Training
    parser.add_argument("--stage", default="sft",
                        choices=["cpt", "sft", "cpt+sft"],
                        help="Training stage. 'cpt+sft' runs CPT then SFT sequentially.")
    parser.add_argument("--loss", default="lm_only",
                        choices=["lm_only", "lm_engagement", "lm_mtp"])
    parser.add_argument("--verbalization", default="rating_history",
                        choices=["rating_history", "minimal"])

    # Pipeline control
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--checkpoint", default=None,
                        help="Skip training and evaluate an existing checkpoint directly")

    # Extra Hydra overrides passed through
    parser.add_argument("--extra", nargs="*", default=[],
                        help="Extra Hydra overrides, e.g. --extra model.lora.r=8 use_wandb=true")

    args = parser.parse_args()

    scripts = Path(__file__).parent
    python = sys.executable

    shared_overrides = hydra_args({
        "data": args.dataset,
        "model": args.model,
        f"model/item_tokenizer": args.item_tokenizer,
        "loss": args.loss,
        f"data.verbalization_template": args.verbalization,
    }) + (args.extra or [])

    # ── 1. Download ──────────────────────────────────────────────────────────
    if not args.skip_download and args.checkpoint is None:
        dl_args = [python, str(scripts / "01_download_data.py")]
        if "movielens" in args.dataset:
            dl_args += ["--movielens", args.movielens]
        if args.amazon_category or args.dataset.startswith("amazon"):
            category = args.amazon_category or "All_Beauty"
            dl_args += ["--amazon-category", category]
        run(dl_args, "Step 1 / 5 — Download data")

    # ── 2. Preprocess ────────────────────────────────────────────────────────
    if not args.skip_preprocess and args.checkpoint is None:
        run(
            [python, str(scripts / "02_preprocess.py")] + shared_overrides,
            "Step 2 / 5 — Preprocess & split",
        )

    # ── 3. Build item vocab ──────────────────────────────────────────────────
    if args.checkpoint is None:
        run(
            [python, str(scripts / "03_build_item_vocab.py")] + shared_overrides,
            f"Step 3 / 5 — Build item vocab ({args.item_tokenizer})",
        )

    # ── 4. Train ─────────────────────────────────────────────────────────────
    checkpoint_path = args.checkpoint
    if not args.skip_train and checkpoint_path is None:
        if args.stage == "cpt+sft":
            # CPT stage
            run(
                [python, str(scripts / "04_train.py"), "training=cpt"] + shared_overrides,
                "Step 4a / 5 — Train (CPT)",
            )
            # Find the CPT output dir and pass it to SFT
            import glob, os
            cpt_runs = sorted(glob.glob("outputs/*/final"), key=os.path.getmtime)
            if not cpt_runs:
                print("[ERROR] CPT output not found. Check outputs/ directory.")
                sys.exit(1)
            cpt_checkpoint = cpt_runs[-1]
            print(f"\n  CPT checkpoint: {cpt_checkpoint}")
            run(
                [python, str(scripts / "04_train.py"), "training=sft",
                 f"training.resume_from_checkpoint={cpt_checkpoint}"] + shared_overrides,
                "Step 4b / 5 — Train (SFT on CPT checkpoint)",
            )
        else:
            run(
                [python, str(scripts / "04_train.py"), f"training={args.stage}"] + shared_overrides,
                f"Step 4 / 5 — Train ({args.stage})",
            )

        # Find latest checkpoint
        import glob, os
        runs = sorted(glob.glob("outputs/*/final"), key=os.path.getmtime)
        if runs:
            checkpoint_path = runs[-1]
            print(f"\n  Checkpoint: {checkpoint_path}")

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    if not args.skip_eval:
        if checkpoint_path is None:
            print("[WARN] No checkpoint found — skipping evaluation.")
        else:
            run(
                [python, str(scripts / "05_evaluate.py"),
                 f"checkpoint={checkpoint_path}"] + shared_overrides,
                "Step 5 / 5 — Evaluate",
            )

    print(f"\n{'=' * 60}")
    print("  Experiment complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
