"""
Upload an already-trained model folder to the HF Hub.

Use this when Axolotl finished training and SAVED the model, but its own
`push_to_hub` crashed at model-card creation, e.g.:
    TypeError: create_model_card() got an unexpected keyword argument 'dataset_tags'
(an Axolotl <-> transformers/TRL version mismatch). The weights are already on
disk in output_dir; this just pushes those files to the Hub directly via
huggingface_hub.upload_folder -- no retraining, no trainer model-card call.

It uploads only the final model + tokenizer files and ignores optimizer state,
intermediate checkpoint-*/ dirs, wandb logs, and the tokenized dataset cache.

Auth: set HF_TOKEN in the environment (or `huggingface-cli login`), or pass --token.

Usage:
    python upload_to_hf.py                                   # uses the defaults below
    python upload_to_hf.py --model-dir ./outputs/qwen2.5-0.5b-dpo-IT-8xh100 \
                           --repo-id adityabanerjee13/qwen2.5-0.5b-dpo-IT-8xh100
    python upload_to_hf.py --checkpoint last                 # push the latest checkpoint-* instead of root
    python upload_to_hf.py --public
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

DEFAULT_MODEL_DIR = "./outputs/qwen2.5-0.5b-dpo-IT-8xh100"
DEFAULT_REPO_ID = "adityabanerjee13/qwen2.5-0.5b-dpo-IT-8xh100"

# Files that make a loadable model repo; everything else (optimizer, rng, etc.) is skipped.
IGNORE = [
    "checkpoint-*",          # intermediate checkpoints (fnmatch '*' spans '/', so this
                             # also excludes their contents)
    "global_step*",          # deepspeed/fsdp step dirs
    "*.pt", "*.bin.tmp",
    "optimizer*", "scheduler*", "rng_state*", "trainer_state.json",
    "wandb*", "runs*",
    "last_run_prepared*",    # tokenized dataset cache
    "*.lock", "*.log",
]

# What a valid HF model folder should contain (used to auto-resolve the dir).
MODEL_MARKERS = ("config.json",)
WEIGHT_MARKERS = ("model.safetensors", "model.safetensors.index.json",
                  "pytorch_model.bin", "pytorch_model.bin.index.json")


def looks_like_model(d):
    files = set(os.listdir(d)) if os.path.isdir(d) else set()
    return any(m in files for m in MODEL_MARKERS) and any(w in files for w in WEIGHT_MARKERS)


def latest_checkpoint(root):
    if not os.path.isdir(root):
        return None
    cks = [os.path.join(root, d) for d in os.listdir(root)
           if d.startswith("checkpoint-") and os.path.isdir(os.path.join(root, d))]
    cks = [c for c in cks if looks_like_model(c)]
    if not cks:
        return None
    # checkpoint-<step> -> pick the highest step
    return max(cks, key=lambda c: int(c.rsplit("-", 1)[-1]) if c.rsplit("-", 1)[-1].isdigit() else -1)


def resolve_model_dir(model_dir, checkpoint):
    if checkpoint == "last":
        ck = latest_checkpoint(model_dir)
        if ck is None:
            sys.exit(f"[error] no valid checkpoint-* under {model_dir}")
        return ck
    if checkpoint:  # an explicit checkpoint-XXXX name or path
        cand = checkpoint if os.path.isabs(checkpoint) else os.path.join(model_dir, checkpoint)
        if not looks_like_model(cand):
            sys.exit(f"[error] {cand} does not look like a model folder")
        return cand
    # default: the output_dir root (Axolotl saves the final model there)
    if looks_like_model(model_dir):
        return model_dir
    ck = latest_checkpoint(model_dir)
    if ck:
        print(f"[warn] no model at {model_dir} root; falling back to latest checkpoint {ck}")
        return ck
    sys.exit(f"[error] no model files (config.json + weights) found in {model_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Training output_dir (or a model folder).")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Target HF repo id.")
    ap.add_argument("--checkpoint", default=None,
                    help="'last' for newest checkpoint-*, or a specific checkpoint-XXXX; default = output_dir root.")
    ap.add_argument("--public", action="store_true", help="Create the repo public (default: private).")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (else uses HF_TOKEN / cached login).")
    ap.add_argument("--message", default="Upload DPO model (bypassing broken trainer push_to_hub)")
    args = ap.parse_args()

    src = resolve_model_dir(os.path.abspath(args.model_dir), args.checkpoint)
    print(f"[upload] source folder : {src}")
    print(f"[upload] target repo   : {args.repo_id}  (private={not args.public})")
    files = [f for f in sorted(os.listdir(src)) if os.path.isfile(os.path.join(src, f))]
    print(f"[upload] files present : {', '.join(files)}")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(
        folder_path=src,
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=IGNORE,
        commit_message=args.message,
    )
    print(f"\n[done] https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
