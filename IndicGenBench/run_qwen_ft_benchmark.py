"""
Run the fine-tuned CPT model (adityabanerjee13/qwen2.5-0.5b-indic-cpt, or any
other local/HF checkpoint) over the 4 IndicGenBench tasks, using direct HF
`transformers` inference instead of Ollama.

This mirrors run_qwen_benchmark.py (same tasks, prompt templates, sampled
examples, and metrics.py scoring) but swaps the inference backend: rather
than calling a local Ollama server (which needs the model converted to GGUF
and registered), this loads the Hugging Face checkpoint directly with
`AutoModelForCausalLM`/`AutoTokenizer` and runs `.generate()` greedily.
Reusing the same task logic keeps scores directly comparable to whatever
baseline you already have under qwen_benchmark_results/.

Note: the checkpoint here is a *base* (non-instruct) CPT model, so it is
being evaluated zero-shot on raw completion prompts exactly like the Ollama
baseline was (Ollama's /api/generate also bypasses any chat template) - no
chat formatting is applied on either side.

Requires: transformers, torch, sacrebleu (see requirements.txt)

Usage:
    python run_qwen_ft_benchmark.py
    python run_qwen_ft_benchmark.py --model adityabanerjee13/qwen2.5-0.5b-indic-cpt
    python run_qwen_ft_benchmark.py --num-examples 50 --tasks xquad_in flores_in
    python run_qwen_ft_benchmark.py --langs hi bn ta --device cpu
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_qwen_benchmark import (  # noqa: E402
    CROSSSUM_TEMPLATE,
    FLORES_ENXX_TEMPLATE,
    FLORES_XXEN_TEMPLATE,
    LANG_NAMES,
    LANGS,
    MAX_ARTICLE_CHARS,
    MAX_CONTEXT_CHARS,
    ROOT,
    XORQA_TEMPLATE,
    XQUAD_TEMPLATE,
    load_examples,
)
from metrics import run_benchmark  # noqa: E402

DEFAULT_MODEL = "adityabanerjee13/qwen2.5-0.5b-cpt-mix-1to4"
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "qwen_ft_benchmark_results")
DEFAULT_BATCH = 128  # prompts per forward pass (higher = faster, more memory)

def _load_tokenizer(model_id: str):
    """Load the tokenizer, self-healing a corrupted tokenizer_config.json.

    Some checkpoints (incl. adityabanerjee13/qwen2.5-0.5b-indic-cpt) were saved
    with a buggy transformers that serialized `extra_special_tokens` as a list
    and dumped runtime-only keys. transformers>=5 calls `.keys()` on that field
    and crashes. tokenizer.json itself is intact, so we sanitize the config in a
    local copy and load from there.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id)
    except AttributeError as exc:
        if "keys" not in str(exc):
            raise
        print("  tokenizer_config.json is malformed; sanitizing and retrying...")

    import json as _json

    from huggingface_hub import snapshot_download

    # Junk keys that a broken save_pretrained leaked into the config.
    junk_keys = {"backend", "is_local", "local_files_only"}

    if os.path.isdir(model_id):
        src = model_id
    else:
        src = snapshot_download(
            model_id,
            allow_patterns=[
                "tokenizer*.json",
                "tokenizer.model",
                "vocab.json",
                "merges.txt",
                "special_tokens_map.json",
                "chat_template.jinja",
                "*.jinja",
            ],
        )

    fixed_dir = os.path.join(
        tempfile.gettempdir(), "qwen_ft_tokenizer_" + os.path.basename(model_id.rstrip("/"))
    )
    os.makedirs(fixed_dir, exist_ok=True)
    for fname in os.listdir(src):
        shutil.copy2(os.path.join(src, fname), os.path.join(fixed_dir, fname))

    cfg_path = os.path.join(fixed_dir, "tokenizer_config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = _json.load(fh)

    est = cfg.get("extra_special_tokens")
    if isinstance(est, list):
        cfg.pop("extra_special_tokens", None)
    for key in junk_keys:
        cfg.pop(key, None)

    with open(cfg_path, "w", encoding="utf-8") as fh:
        _json.dump(cfg, fh, ensure_ascii=False, indent=2)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(fixed_dir)


def load_model(model_id: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM

    print(f"Downloading/loading {model_id} ...")
    tokenizer = _load_tokenizer(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation needs left padding so that, within a batch, every
    # sequence's real tokens end flush against the generated continuation.
    tokenizer.padding_side = "left"

    # Pass dtype as a *string*, not a torch.dtype object: transformers 5.14.1
    # stores the value on the config and then json-serializes it when logging the
    # config repr; a torch.dtype object is not JSON-serializable and crashes.
    torch_dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device} ({dtype}).")
    return model, tokenizer


def generate(model, tokenizer, device: str, prompt: str, max_new_tokens: int) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_batch(model, tokenizer, device: str, prompts: list, max_new_tokens: int,
                   batch_size: int) -> list:
    """Greedily generate continuations for many prompts, batch_size at a time.

    Relies on the tokenizer's left padding (set in load_model): with left
    padding every sequence in a chunk shares the same padded input length, so
    the newly generated tokens are the same trailing slice for the whole batch.
    """
    import torch

    outputs = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        inputs = tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=3072, padding=True
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[:, inputs["input_ids"].shape[-1]:]
        outputs.extend(
            text.strip()
            for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        )
    return outputs


def run_xquad(lang: str, n: int, seed: int, model, tokenizer, device: str, batch_size: int) -> list:
    path = os.path.join(ROOT, "xquad_in", f"xquad_{lang}_test.json")
    if not os.path.exists(path):
        return []
    examples = list(load_examples(path, n, seed))
    prompts = [
        XQUAD_TEMPLATE.format(
            lang=LANG_NAMES[lang], context=ex["context"][:MAX_CONTEXT_CHARS], question=ex["question"]
        )
        for ex in examples
    ]
    predictions = generate_batch(model, tokenizer, device, prompts, 30, batch_size)
    records = []
    for ex, prompt, prediction in zip(examples, prompts, predictions):
        references = [a["text"] for a in ex["answers"]]
        records.append({
            "id": ex.get("id"), "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": references,
        })
    return records


def run_xorqa(lang: str, n: int, seed: int, model, tokenizer, device: str, batch_size: int) -> list:
    path = os.path.join(ROOT, "xorqa_in", f"xorqa_{lang}_test.json")
    if not os.path.exists(path):
        return []
    examples = list(load_examples(path, n, seed))
    prompts = [
        XORQA_TEMPLATE.format(
            lang=LANG_NAMES[lang], context=ex["context"][:MAX_CONTEXT_CHARS], question=ex["question"]
        )
        for ex in examples
    ]
    predictions = generate_batch(model, tokenizer, device, prompts, 30, batch_size)
    records = []
    for ex, prompt, prediction in zip(examples, prompts, predictions):
        gold = ex.get("translated_answers") or ex["answers"]
        references = [a["text"] for a in gold]
        records.append({
            "id": None, "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": references,
        })
    return records


def run_crosssum(lang: str, n: int, seed: int, model, tokenizer, device: str, batch_size: int) -> list:
    path = os.path.join(ROOT, "crosssum_in", f"crosssum_english-{lang}_test.json")
    if not os.path.exists(path):
        return []
    examples = list(load_examples(path, n, seed))
    prompts = [
        CROSSSUM_TEMPLATE.format(lang=LANG_NAMES[lang], text=ex["text"][:MAX_ARTICLE_CHARS])
        for ex in examples
    ]
    predictions = generate_batch(model, tokenizer, device, prompts, 100, batch_size)
    records = []
    for ex, prompt, prediction in zip(examples, prompts, predictions):
        records.append({
            "id": None, "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": [ex["summary"]],
        })
    return records


def run_flores(lang: str, n: int, seed: int, model, tokenizer, device: str, batch_size: int) -> list:
    items = []  # (direction, ex, prompt)
    for direction, filename, template in [
        ("enxx", f"flores_en_{lang}_test.json", FLORES_ENXX_TEMPLATE),
        ("xxen", f"flores_{lang}_en_test.json", FLORES_XXEN_TEMPLATE),
    ]:
        path = os.path.join(ROOT, "flores_in", filename)
        if not os.path.exists(path):
            continue
        for ex in load_examples(path, n, seed):
            prompt = template.format(lang=LANG_NAMES[lang], source=ex["source"])
            items.append((direction, ex, prompt))
    if not items:
        return []
    predictions = generate_batch(
        model, tokenizer, device, [prompt for _, _, prompt in items], 200, batch_size
    )
    records = []
    for (direction, ex, prompt), prediction in zip(items, predictions):
        records.append({
            "id": None, "lang": lang, "direction": direction, "prompt": prompt,
            "prediction": prediction, "references": [ex["target"]],
        })
    return records


TASK_RUNNERS = {
    "xquad_in": run_xquad,
    "xorqa_in": run_xorqa,
    "crosssum_in": run_crosssum,
    "flores_in": run_flores,
}


def free_accelerator_memory(model, device: str) -> None:
    """Move the model off the accelerator and release its cached device memory.

    Dropping Python references alone isn't enough: the XPU/CUDA caching
    allocator keeps freed blocks reserved, so the GPU still shows the memory as
    used until we explicitly empty the cache. Called from a finally block so it
    runs on success, error, or interrupt.
    """
    import gc

    import torch

    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    print(f"Released model from {device} memory.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path of the fine-tuned checkpoint.")
    parser.add_argument("--device", default=None, help="cuda / xpu / cpu (default: cuda > xpu > cpu, whichever is available)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Prompts generated together per forward pass (higher = faster, more memory).")
    parser.add_argument("--num-examples", type=int, default=20, help="Sampled examples per task/language (per direction for flores)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="*", default=list(TASK_RUNNERS.keys()), choices=list(TASK_RUNNERS.keys()))
    parser.add_argument("--langs", nargs="*", default=LANGS, choices=LANGS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    import torch

    def pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"

    device = args.device or pick_device()
    if device == "cpu" and args.dtype != "fp32":
        print("WARNING: running on CPU with non-fp32 dtype can be slow/unsupported; consider --dtype fp32.")

    model, tokenizer = load_model(args.model, device, args.dtype)

    try:
        # Save each run into its own subfolder so runs don't overwrite each other:
        # <output-dir>/<model-name>_<timestamp>/
        run_name = f"{os.path.basename(args.model.rstrip('/'))}_{time.strftime('%Y%m%d-%H%M%S')}"
        output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to {output_dir}")

        all_scores = {}
        for task in args.tasks:
            runner = TASK_RUNNERS[task]
            task_records = []
            for lang in args.langs:
                print(f"[{task}] {lang} ...", flush=True)
                start = time.time()
                try:
                    records = runner(lang, args.num_examples, args.seed, model, tokenizer, device, args.batch_size)
                except Exception as e:
                    print(f"    ERROR loading/running {lang}: {e!r}", flush=True)
                    continue
                task_records.extend(records)
                print(f"    {len(records)} examples in {time.time() - start:.1f}s", flush=True)

            out_path = os.path.join(output_dir, f"{task}_predictions.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(task_records, f, ensure_ascii=False, indent=2)

            scores = run_benchmark(task, task_records)
            all_scores[task] = scores
            print(f"\n=== {task} scores ===")
            print(json.dumps(scores, ensure_ascii=False, indent=2))
            print()

        scores_path = os.path.join(output_dir, "scores_summary.json")
        with open(scores_path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, ensure_ascii=False, indent=2)
        print(f"All scores saved to {scores_path}")
    finally:
        # Free the model from accelerator (XPU/CUDA) memory whether we finished
        # cleanly, hit an error, or were interrupted.
        free_accelerator_memory(model, device)
        del model, tokenizer


if __name__ == "__main__":
    main()
