"""
Launch full-parameter Continued Pre-Training (CPT) of Qwen2.5-0.5B on Indic
text with Axolotl, pinned to a single GPU.

The actual training config lives in qwen2.5_0.5b_cpt_full.yml next to this
file — edit that YAML (base model, dataset repo(s), hyperparameters) rather
than this script. This script just pins CUDA_VISIBLE_DEVICES to one device
and invokes `axolotl train`.

Usage:
    python train.py
    python train.py --config qwen2.5_0.5b_cpt_full.yml --gpu 0
    python train.py --resume-from-checkpoint outputs/qwen2.5-0.5b-indic-cpt-full/checkpoint-1500

Requires:
    pip install "axolotl[flash-attn]"
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "qwen2.5_0.5b_cpt_full.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Path to the Axolotl YAML config (default: {DEFAULT_CONFIG.name})",
    )
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="Index of the single GPU to train on (sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--resume-from-checkpoint", type=str, default=None,
        help="Path to a checkpoint dir to resume training from.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def check_single_gpu(gpu_index: int) -> None:
    try:
        import torch
    except ImportError:
        print("WARNING: torch not importable in this environment; skipping GPU check.")
        return

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to torch — CPT requires a GPU.")

    count = torch.cuda.device_count()
    if gpu_index >= count:
        raise SystemExit(f"Requested --gpu {gpu_index} but only {count} GPU(s) visible.")

    name = torch.cuda.get_device_name(gpu_index)
    print(f"Using GPU {gpu_index}: {name}")


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    check_single_gpu(args.gpu)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    axolotl_bin = shutil.which("axolotl")
    if axolotl_bin:
        cmd = [axolotl_bin, "train", str(args.config)]
    else:
        # Fall back to the module entrypoint if the console script isn't on PATH.
        cmd = [sys.executable, "-m", "axolotl.cli.train", str(args.config)]

    if args.resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", args.resume_from_checkpoint]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
