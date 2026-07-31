"""
Unified STEP 2 of the DPO triplet pipeline.

Generate `rejected` responses for ALL inputs in ONE process (the model is loaded
a single time and reused), writing (prompt, chosen, rejected) triplets to each
input's own output file:

    data/indicalign_toxic_pairs.jsonl      -> data/toxic_dpo_triplets.jsonl          (format: pairs)
    data/sft_source/indic_sampled.jsonl    -> data/sft_source/indic_dpo_triplets.jsonl (format: sft)
    data/sft_source/tulu_sampled.jsonl     -> data/sft_source/tulu_dpo_triplets.jsonl  (format: sft)

Two input schemas are supported:
  * "pairs": record already has {"prompt":[...], "chosen":[...]}   (IndicAlign-Toxic).
  * "sft"  : record has {"messages":[...]}; prompt = everything up to and
             including the last user turn, chosen = the final assistant turn.

For every record, `prompt` is rendered with the model's chat template and a
response is sampled -> that becomes `rejected`, giving the DPO contrast
(gold/refusal `chosen` vs. model sample `rejected`).

Output = HF / TRL DPOTrainer "conversational" preference format:
    {"prompt":[...], "chosen":[{assistant}], "rejected":[{assistant}], language?, source?}

RESUMABLE: each output is appended to. On start a job counts the triplets already
written (dropping any partial trailing line from an interrupted/killed run), skips
that many input records, and continues. Pass --overwrite to start every job fresh.

Requirements: pip install torch transformers huggingface_hub tqdm
Usage:
    python generate_all_rejected.py --model adityabanerjee13/qwen2.5-0.5b-sft-IT
    python generate_all_rejected.py --jobs toxic indic --batch-size 32
    python generate_all_rejected.py --device cuda
"""
import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dotenv():
    """Populate os.environ from a gitignored .env (repo root or this folder).

    Secrets (HF_TOKEN, WANDB_API_KEY) must never be hardcoded in this file --
    GitHub push protection blocks the push, and the token leaks to anyone with
    read access. Export them in your shell or drop them in `.env` instead.
    Existing environment values always win.
    """
    for candidate in (os.path.join(HERE, ".env"),
                      os.path.join(os.path.dirname(HERE), ".env")):
        if not os.path.exists(candidate):
            continue
        with io.open(candidate, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


load_dotenv()

# name -> (input rel path, output rel path, format)
JOBS = {
    "toxic":     ("data/indicalign_toxic_pairs.jsonl",       "data/toxic_dpo_triplets.jsonl",                "pairs"),
    "indic":     ("data/sft_source/indic_sampled.jsonl",     "data/sft_source/indic_dpo_triplets.jsonl",     "sft"),
    "tulu":      ("data/sft_source/tulu_sampled.jsonl",      "data/sft_source/tulu_dpo_triplets.jsonl",      "sft"),
    "indic_val": ("data/sft_source/indic_val_sampled.jsonl", "data/sft_source/indic_val_dpo_triplets.jsonl", "sft"),
    "tulu_val":  ("data/sft_source/tulu_val_sampled.jsonl",  "data/sft_source/tulu_val_dpo_triplets.jsonl",  "sft"),
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pick_device(requested):
    import torch
    if requested and requested != "auto":
        return requested
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def seed_everything(seed):
    """Local reimplementation of transformers.set_seed. Avoids importing
    transformers.trainer_utils, which pulls in peft/bitsandbytes/triton and can
    fail on mismatched envs (e.g. `No module named 'triton.ops'`)."""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_completed(out_path):
    """How many valid triplets already exist in `out_path`. Rewrites the file to
    drop any blank/partial trailing line (from an interrupted run) so a resumed
    append stays clean. The run then skips this many input records and appends."""
    if not os.path.exists(out_path):
        return 0
    good = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
            except Exception:
                break  # first corrupt/partial line -> stop; drop everything after
            good.append(s)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in good:
            f.write(s + "\n")
    return len(good)


def as_messages(seq):
    """Normalize a list of turns into clean [{role, content}] dicts."""
    return [{"role": m["role"], "content": m["content"]} for m in seq]


def split_prompt_chosen(messages):
    """messages -> (prompt_msgs, chosen_msgs). Requires the last turn to be assistant."""
    msgs = as_messages(messages)
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        return None
    return msgs[:-1], [msgs[-1]]


def build_prompt_text(tokenizer, prompt_msgs):
    """Render the prompt with the model's chat template, or a plain fallback."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    parts = [f"{m['role'].capitalize()}: {m['content']}" for m in prompt_msgs]
    parts.append("Assistant:")
    return "\n".join(parts)


def load_records(in_path, fmt):
    """Return (items, skipped) where each item = (meta, prompt_msgs, chosen_msgs)."""
    items, skipped = [], 0
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            r = json.loads(s)
            if fmt == "pairs":
                prompt_msgs, chosen_msgs = r.get("prompt"), r.get("chosen")
                if not prompt_msgs or not chosen_msgs:
                    skipped += 1
                    continue
                prompt_msgs, chosen_msgs = as_messages(prompt_msgs), as_messages(chosen_msgs)
            else:  # "sft"
                sc = split_prompt_chosen(r.get("messages", []))
                if sc is None:
                    skipped += 1
                    continue
                prompt_msgs, chosen_msgs = sc
            meta = {"language": r.get("language"), "source": r.get("source")}
            items.append((meta, prompt_msgs, chosen_msgs))
    return items, skipped


# --------------------------------------------------------------------------- #
# per-job generation
# --------------------------------------------------------------------------- #
def run_job(name, in_path, out_path, fmt, tokenizer, model, device, torch, tqdm, args):
    items, skipped = load_records(in_path, fmt)
    if args.limit:
        items = items[: args.limit]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Resume: skip records already present in the output (unless --overwrite).
    if args.overwrite and os.path.exists(out_path):
        os.remove(out_path)
    done = count_completed(out_path)
    pending = items[done:]
    print(f"\n=== {name}: {len(items)} records ({skipped} skipped), "
          f"{done} already done, {len(pending)} to generate "
          f"-> {os.path.relpath(out_path, HERE)} ===")
    if not pending:
        print(f"[{name}] already complete; nothing to do.")
        return 0

    n_written = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(pending), args.batch_size), desc=name, unit="batch"):
            batch = pending[start:start + args.batch_size]
            texts = [build_prompt_text(tokenizer, pm) for (_, pm, _) in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = out[:, enc["input_ids"].shape[1]:]  # left-padded -> uniform slice point
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for (meta, prompt_msgs, chosen_msgs), rej in zip(batch, decoded):
                triplet = {
                    "language": meta.get("language"),
                    "source": meta.get("source"),
                    "prompt": prompt_msgs,
                    "chosen": chosen_msgs,
                    "rejected": [{"role": "assistant", "content": rej.strip()}],
                }
                out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                out_f.flush()  # flush every record so a kill -9 never loses generated work
                n_written += 1
    print(f"[{name}] wrote {n_written} new triplets ({done + n_written} total) -> {out_path}")
    return n_written


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="adityabanerjee13/qwen2.5-0.5b-sft-IT",
                    help="HF model path for the rejected generator.")
    ap.add_argument("--jobs", nargs="+", choices=list(JOBS), default=list(JOBS),
                    help="Which jobs to run (default: all three, in order).")
    ap.add_argument("--limit", type=int, default=None, help="Per-job cap on records (for testing).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Start each job fresh instead of resuming from existing output.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="auto", help="auto|cuda|xpu|cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # validate inputs up front
    todo = []
    for name in args.jobs:
        in_rel, out_rel, fmt = JOBS[name]
        in_path = os.path.join(HERE, in_rel)
        if not os.path.exists(in_path):
            print(f"[skip] {name}: input not found -> {in_rel}")
            continue
        todo.append((name, in_path, os.path.join(HERE, out_rel), fmt))
    if not todo:
        print("[abort] no valid inputs to process.")
        return

    # ---- load the model ONCE, reuse across jobs ----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm
    seed_everything(args.seed)

    device = pick_device(args.device)
    print(f"[load] model={args.model}  device={device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        # Old `tokenizers` can't parse a tokenizer.json saved by a newer stack
        # ("data did not match any variant of untagged enum ModelWrapper ...").
        # Retry with the slow tokenizer (needs vocab.json + merges.txt in the repo).
        print(f"  fast tokenizer failed ({e}); retrying with use_fast=False. "
              f"If this also fails, upgrade: pip install -U transformers tokenizers")
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched causal generation
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    try:
        model.to(device)
    except Exception as e:
        print(f"  WARNING: could not move model to {device} ({e!r}); falling back to cpu")
        device = "cpu"
        model.to(device)
    model.eval()

    total = 0
    for name, in_path, out_path, fmt in todo:
        total += run_job(name, in_path, out_path, fmt, tokenizer, model, device, torch, tqdm, args)

    print(f"\n[all done] wrote {total} triplets across {len(todo)} job(s).")


if __name__ == "__main__":
    main()
