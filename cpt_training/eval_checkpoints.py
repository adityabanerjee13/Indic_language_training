"""
Run an Axolotl `evaluate` config against every saved checkpoint of a training
run (plus the final model), instead of just the one `base_model` path in the
YAML. Produces one combined CSV of eval loss vs. checkpoint step.

Why this is needed: `axolotl evaluate <config>.yml` only ever evaluates the
single `base_model:` path baked into the YAML. To see how held-out loss
changes over the course of training (e.g. to check for overfitting), the same
eval must be re-run once per `checkpoint-N` directory under the training
run's `output_dir`.

How it works:
  - Reads `base_model:` out of the given eval YAML — that path IS the
    training run's output_dir (root = final model). Checkpoints are its
    `checkpoint-*` subdirectories.
  - For each one (root + every checkpoint-N, sorted by step), writes a temp
    copy of the YAML with `base_model:`/`output_dir:`/`wandb_name:` swapped
    in, then runs `axolotl evaluate <temp config>`.
    (CLI overrides like `--base_model=...` do NOT work here: Axolotl's
    `evaluate` command only forwards *declared* CLI options through
    `build_command`; anything else falls into `ctx.args` and gets inserted
    before `-m axolotl.cli.evaluate` in the `accelerate launch` command
    line, which then fails to parse it as a launcher flag.)
  - `dataset_prepared_path` is left untouched, so the tokenized eval set is
    cached once and reused across every checkpoint instead of re-tokenizing
    each run.
  - After each run, reads `eval_summary.csv` (written by Axolotl's own
    `evaluate()` to `output_dir`) and pulls the "loss" row's "validation"
    column.
  - Writes a combined `checkpoint_eval_summary.csv` next to the base config,
    plus prints a step -> eval_loss table sorted by step.

Usage:
    python eval_checkpoints.py --config qwen2.5_0.5b_cpt_mix_1to2_eval_web.yml
    python eval_checkpoints.py --config <cfg>.yml --gpu 0 --stride 2
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the Axolotl evaluate YAML (its base_model: is the "
             "training run's output_dir; checkpoint-* subdirs are found there).",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="Index of the single GPU to evaluate on (sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Evaluate every Nth checkpoint only (final model is always included).",
    )
    parser.add_argument(
        "--skip-final", action="store_true",
        help="Skip evaluating the root/final model, only checkpoint-* dirs.",
    )
    return parser.parse_args()


def find_checkpoints(train_dir: Path, stride: int, skip_final: bool):
    ckpts = sorted(
        train_dir.glob("checkpoint-*"),
        key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)),
    )
    ckpts = ckpts[::stride]

    entries = []  # (label, step_for_sorting, path)
    if not skip_final and (train_dir / "config.json").exists():
        entries.append(("final", 10**9, train_dir))
    for p in ckpts:
        step = int(re.search(r"(\d+)$", p.name).group(1))
        entries.append((p.name, step, p))
    entries.sort(key=lambda e: e[1])
    return entries


def read_eval_loss(output_dir: Path):
    summary = output_dir / "eval_summary.csv"
    if not summary.exists():
        print(f"  WARNING: {summary} not found — run may have failed.")
        return None
    with summary.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("metric") == "loss":
                val = row.get("validation")
                return float(val) if val not in (None, "") else None
    print(f"  WARNING: no 'loss' row in {summary}.")
    return None


def main() -> None:
    args = parse_args()
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_dir = Path(cfg["base_model"])
    if not train_dir.exists():
        raise SystemExit(f"base_model dir from config not found: {train_dir}")

    entries = find_checkpoints(train_dir, args.stride, args.skip_final)
    if not entries:
        raise SystemExit(f"No checkpoint-* dirs (or final model) found under {train_dir}")

    print(f"Found {len(entries)} model(s) to evaluate under {train_dir}:")
    for label, _step, path in entries:
        print(f"  {label}: {path}")

    eval_root = train_dir / "eval-per-checkpoint"
    axolotl_bin = shutil.which("axolotl")

    results = []
    for label, step, path in entries:
        out_dir = eval_root / label
        print(f"\n=== Evaluating {label} ({path}) -> {out_dir} ===")

        env_overrides = [
            f"--base_model={path}",
            f"--output_dir={out_dir}",
            f"--wandb_name={cfg.get('wandb_name', 'eval')}-{label}",
        ]

        if axolotl_bin:
            cmd = [axolotl_bin, "evaluate", str(args.config)] + env_overrides
        else:
            cmd = [sys.executable, "-m", "axolotl.cli.evaluate", str(args.config)] + env_overrides

        import os
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["TOKENIZERS_PARALLELISM"] = "false"

        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  WARNING: evaluate exited {result.returncode} for {label}; skipping.")
            results.append((label, step, None))
            continue

        loss = read_eval_loss(out_dir)
        print(f"  {label}: eval_loss = {loss}")
        results.append((label, step, loss))

    summary_path = eval_root / "checkpoint_eval_summary.csv"
    eval_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "step", "eval_loss"])
        for label, step, loss in results:
            writer.writerow([label, step, loss])

    print(f"\nWrote combined summary: {summary_path}")
    print("\nstep\teval_loss\tcheckpoint")
    for label, step, loss in sorted(results, key=lambda r: r[1]):
        print(f"{step}\t{loss}\t{label}")


if __name__ == "__main__":
    main()
