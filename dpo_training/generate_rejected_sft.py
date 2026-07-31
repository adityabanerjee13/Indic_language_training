"""
STEP 2 (SFT-source prep for DPO): sample `rejected` responses for the sampled
SFT subsets, forming (prompt, chosen, rejected) DPO triplets.

For each SFT record (a `messages` chat):
    prompt   = all turns up to and including the last user turn
    chosen   = the final assistant turn (the SFT gold response)
    rejected = a response sampled from the base model (default Qwen/Qwen2.5-0.5B)

This is the OFF-POLICY "gold-vs-base" construction. Note (see EXPERIMENT_PLAN):
the base model is far from the SFT policy, so negatives may be easy / rely on
script or length cues — pair this with an on-policy variant for comparison.

Output = HF / TRL DPOTrainer "conversational" preference format:
    {"prompt":[...], "chosen":[{"role":"assistant",...}], "rejected":[{"role":"assistant",...}]}
plus language/source metadata (ignored by trainers).

Requirements: transformers, torch, huggingface_hub
Usage (smoke test):
    python generate_rejected_sft.py \
        --input data/sft_source/indic_sampled_4k_per_lang.jsonl \
        --output data/sft_source/indic_dpo_triplets_sample.jsonl --limit 8
Full run:
    python generate_rejected_sft.py --input data/sft_source/indic_sampled_4k_per_lang.jsonl \
        --output data/sft_source/indic_dpo_triplets.jsonl --batch-size 16
    python generate_rejected_sft.py --input data/sft_source/tulu_sampled_20k.jsonl \
        --output data/sft_source/tulu_dpo_triplets.jsonl --batch-size 16
"""

import argparse
import io
import json
import os
import sys


def count_completed(out_path):
    """Return how many valid triplets already exist in `out_path`, rewriting the
    file to drop any blank/partial trailing line so appends stay clean.
    A run can then skip that many input records and resume by appending."""
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
                break  # first corrupt/partial line -> stop; everything after is dropped
            good.append(s)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in good:
            f.write(s + "\n")
    return len(good)


def pick_device(requested):
    import torch
    if requested and requested != "auto":
        return requested
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def split_prompt_chosen(messages):
    """messages -> (prompt_msgs, chosen_msgs). Requires last turn to be assistant."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
    if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
        return None
    return msgs[:-1], [msgs[-1]]


def build_prompt_text(tokenizer, prompt_msgs):
    """Render prompt with the model's chat template, or a plain fallback for base models."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    parts = [f"{m['role'].capitalize()}: {m['content']}" for m in prompt_msgs]
    parts.append("Assistant:")
    return "\n".join(parts)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Sampled SFT subset jsonl (records with `messages`).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                    help="HF model path for the rejected generator. Default: Qwen/Qwen2.5-0.5B.")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N records (for testing).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Start fresh (ignore/replace any existing output). Default: resume + append.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--device", default="auto", help="auto|cuda|xpu|cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    in_path = args.input if os.path.isabs(args.input) else os.path.join(here, args.input)
    out_path = args.output if os.path.isabs(args.output) else os.path.join(here, args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- load records, split into prompt/chosen (prompt_text built later) ----
    items = []  # (record_meta, prompt_msgs, chosen_msgs)
    skipped = 0
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sc = split_prompt_chosen(r.get("messages", []))
            if sc is None:
                skipped += 1
                continue
            prompt_msgs, chosen_msgs = sc
            items.append((r, prompt_msgs, chosen_msgs))
    if args.limit:
        items = items[: args.limit]

    # Resume: skip records already present in the output (unless --overwrite).
    if args.overwrite and os.path.exists(out_path):
        os.remove(out_path)
    done = count_completed(out_path)
    if done:
        print(f"[resume] {done} triplets already in {os.path.basename(out_path)}; skipping those.")
    items = items[done:]
    if not items:
        print(f"[done] nothing to do — output already has all {done} triplets ({skipped} skipped).")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(args.seed)

    device = pick_device(args.device)
    print(f"[load] model={args.model}  device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched causal generation
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    try:
        model.to(device)
    except Exception as e:
        print(f"  WARNING: could not move to {device} ({e!r}); using cpu")
        device = "cpu"
        model.to(device)
    model.eval()

    from tqdm import tqdm
    print(f"[gen] {len(items)} remaining records ({skipped} skipped); batch_size={args.batch_size}")

    n_written = 0
    with open(out_path, "a", encoding="utf-8") as out_f:
        for start in tqdm(range(0, len(items), args.batch_size),
                          desc="rejected", unit="batch"):
            batch = items[start:start + args.batch_size]
            texts = [build_prompt_text(tokenizer, prompt_msgs) for (_, prompt_msgs, _) in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
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
            for (rec, prompt_msgs, chosen_msgs), rej in zip(batch, decoded):
                triplet = {
                    "language": rec.get("language"),
                    "source": rec.get("source"),
                    "prompt": prompt_msgs,
                    "chosen": chosen_msgs,
                    "rejected": [{"role": "assistant", "content": rej.strip()}],
                }
                out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
                n_written += 1
            out_f.flush()  # persist each batch so a kill leaves a clean resume point

    print(f"\n[done] wrote {n_written} new triplets ({done + n_written} total) -> {out_path}")


if __name__ == "__main__":
    main()
