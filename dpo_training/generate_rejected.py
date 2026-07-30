"""
STEP 2 of the DPO triplet pipeline.

Read the (prompt, chosen) pairs produced by download_indicalign_toxic.py, and
for each prompt generate a response from a base model (given by an HF model
path). That generated response becomes the `rejected` completion, producing
(prompt, chosen, rejected) triplets.

Rationale: `chosen` is the human-aligned refusal; `rejected` is whatever the
base model produces (often a less-safe or lower-quality completion). DPO then
teaches the policy to prefer the refusal over the base model's answer.

Output format = HF / TRL DPOTrainer "conversational" preference format:
    {
      "prompt":   [{"role": "user", "content": "..."}],
      "chosen":   [{"role": "assistant", "content": "..."}],
      "rejected": [{"role": "assistant", "content": "..."}]
    }
(plus "language" / "source" metadata columns, which trainers ignore).
This loads directly via `datasets.load_dataset("json", ...)` into DPOTrainer.

The default base model is Qwen/Qwen2.5-0.5B (a non-safety-tuned base model, which
is more likely to actually comply with a harmful prompt -> a genuinely unsafe
`rejected`, giving the classic safe-vs-unsafe DPO contrast). Override with --model.

Requirements: pip install transformers torch huggingface_hub
Usage (test on a few samples, uses default model):
    python generate_rejected.py \
        --input data/indicalign_toxic_pairs.jsonl \
        --output data/dpo_triplets_sample.jsonl --limit 5
Full run (override model if desired):
    python generate_rejected.py --model <HF_MODEL_PATH> \
        --input data/indicalign_toxic_pairs.jsonl \
        --output data/dpo_triplets.jsonl
"""

import argparse
import io
import json
import os
import sys


def pick_device(requested):
    import torch
    if requested and requested != "auto":
        return requested
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def build_prompt_text(tokenizer, prompt_msgs):
    """Render prompt messages with the model's chat template if it has one,
    else fall back to a simple concatenation (works for base models too)."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
    # Fallback for base models without a chat template
    parts = []
    for m in prompt_msgs:
        parts.append(f"{m['role'].capitalize()}: {m['content']}")
    parts.append("Assistant:")
    return "\n".join(parts)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B",
                    help="HF model path/id for the base model (rejected generator). Default: Qwen/Qwen2.5-0.5B.")
    ap.add_argument("--input", default="data/indicalign_toxic_pairs.jsonl")
    ap.add_argument("--output", default="data/dpo_triplets.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N pairs (for testing).")
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

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    set_seed(args.seed)

    device = pick_device(args.device)
    print(f"[load] model={args.model}  device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    try:
        model.to(device)
    except Exception as e:
        print(f"  WARNING: could not move model to {device} ({e!r}); falling back to cpu")
        device = "cpu"
        model.to(device)
    model.eval()

    # Load pairs
    pairs = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[gen] generating rejected responses for {len(pairs)} pairs ...")

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for i, pair in enumerate(pairs):
            prompt_msgs = pair["prompt"]
            prompt_text = build_prompt_text(tokenizer, prompt_msgs)
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_tokens = out[0][inputs["input_ids"].shape[1]:]
            rejected_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            triplet = {
                "language": pair.get("language"),
                "source": pair.get("source"),
                "prompt": prompt_msgs,
                "chosen": pair["chosen"],
                "rejected": [{"role": "assistant", "content": rejected_text}],
            }
            out_f.write(json.dumps(triplet, ensure_ascii=False) + "\n")
            n_written += 1
            print(f"  [{i+1}/{len(pairs)}] lang={pair.get('language')} "
                  f"chosen_len={len(pair['chosen'][0]['content'])} rejected_len={len(rejected_text)}")

    print(f"\n[done] wrote {n_written} DPO triplets -> {out_path}")


if __name__ == "__main__":
    main()
