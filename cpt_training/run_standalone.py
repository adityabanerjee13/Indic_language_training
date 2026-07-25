"""
Self-contained launcher: writes the Axolotl CPT config to the current working
directory, then runs `axolotl train` on it, pinned to a single GPU.

Unlike train.py (which reads qwen2.5_0.5b_cpt_full.yml from next to itself),
this script has the config embedded as a string, so it's a single file you
can copy onto a rented GPU box (RunPod/Vast.ai/etc.) without also copying the
YAML. Editing the config means editing CONFIG_YAML below.

Usage:
    python run_standalone.py
    python run_standalone.py --gpu 0
    python run_standalone.py --resume-from-checkpoint outputs/qwen2.5-0.5b-indic-cpt-full/checkpoint-1500

Requires:
    pip install "axolotl[flash-attn]"
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONFIG_YAML = """\
# ==============================================================================
# Axolotl config — full-parameter Continued Pre-Training (CPT) of Qwen2.5-0.5B
# on Indic-language raw text, single GPU.
# ==============================================================================

base_model: Qwen/Qwen2.5-0.5B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: false

# --- Full-parameter fine-tuning: no LoRA/QLoRA adapter, no quantization ---
adapter:
load_in_8bit: false
load_in_4bit: false

# --- Data -------------------------------------------------------------------
datasets:
  - path: adityabanerjee13/hi-cpt-test-sample   # Hindi-only smoke-test sample (25k lines)
    type: completion
    field: text
    split: train
  # Add more language repos once ready to move past the smoke test, e.g.:
  # - path: your-hf-username/indic-cpt-corpus-ta
  #   type: completion
  #   field: text
  #   split: train

dataset_prepared_path: ./last_run_prepared
val_set_size: 0.001
output_dir: ./outputs/qwen2.5-0.5b-indic-cpt-full

# --- Sequence packing ---------------------------------------------------
sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true
eval_sample_packing: false

# --- Optimization -------------------------------------------------------
gradient_accumulation_steps: 8
micro_batch_size: 4
num_epochs: 1
optimizer: adamw_torch_fused
lr_scheduler: cosine
learning_rate: 2e-5
warmup_ratio: 0.03
weight_decay: 0.01
max_grad_norm: 1.0

train_on_inputs: true
group_by_length: false

# --- Precision / memory (single consumer/cloud GPU) ----------------------
bf16: auto
fp16:
tf32: true
gradient_checkpointing: true
flash_attention: true

# No `deepspeed:` / `fsdp:` keys — single-GPU run, plain accelerate process.

# --- Logging / checkpoints -----------------------------------------------
logging_steps: 10
save_strategy: steps
save_steps: 500
save_total_limit: 3
evals_per_epoch: 4

wandb_project: indic-cpt
wandb_entity: models-na9841
wandb_watch: gradients
wandb_name: qwen2.5-0.5b-indic-cpt-full
wandb_log_model: "false"

# --- Hub push --------------------------------------------------------------
# Pushes the final trained model to the Hugging Face Hub at the end of
# training. Requires HF_TOKEN (write access) set in the environment.
hub_model_id: adityabanerjee13/qwen2.5-0.5b-indic-cpt
hub_strategy: end

special_tokens:
"""

CONFIG_FILENAME = "qwen2.5_0.5b_cpt_full.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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

    print(f"Using GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")


def main() -> None:
    args = parse_args()

    config_path = Path.cwd() / CONFIG_FILENAME
    config_path.write_text(CONFIG_YAML, encoding="utf-8")
    print(f"Wrote config: {config_path}")

    check_single_gpu(args.gpu)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    axolotl_bin = shutil.which("axolotl")
    if axolotl_bin:
        cmd = [axolotl_bin, "train", str(config_path)]
    else:
        cmd = [sys.executable, "-m", "axolotl.cli.train", str(config_path)]

    if args.resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", args.resume_from_checkpoint]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
