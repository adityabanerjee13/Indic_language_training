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
import sys
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

DEFAULT_MODEL = "adityabanerjee13/qwen2.5-0.5b-indic-cpt"
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "qwen_ft_benchmark_results")


def load_model(model_id: str, device: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading/loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
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


def run_xquad(lang: str, n: int, seed: int, model, tokenizer, device: str) -> list:
    path = os.path.join(ROOT, "xquad_in", f"xquad_{lang}_test.json")
    if not os.path.exists(path):
        return []
    records = []
    for ex in load_examples(path, n, seed):
        prompt = XQUAD_TEMPLATE.format(
            lang=LANG_NAMES[lang], context=ex["context"][:MAX_CONTEXT_CHARS], question=ex["question"]
        )
        prediction = generate(model, tokenizer, device, prompt, max_new_tokens=30)
        references = [a["text"] for a in ex["answers"]]
        records.append({
            "id": ex.get("id"), "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": references,
        })
    return records


def run_xorqa(lang: str, n: int, seed: int, model, tokenizer, device: str) -> list:
    path = os.path.join(ROOT, "xorqa_in", f"xorqa_{lang}_test.json")
    if not os.path.exists(path):
        return []
    records = []
    for ex in load_examples(path, n, seed):
        prompt = XORQA_TEMPLATE.format(
            lang=LANG_NAMES[lang], context=ex["context"][:MAX_CONTEXT_CHARS], question=ex["question"]
        )
        prediction = generate(model, tokenizer, device, prompt, max_new_tokens=30)
        gold = ex.get("translated_answers") or ex["answers"]
        references = [a["text"] for a in gold]
        records.append({
            "id": None, "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": references,
        })
    return records


def run_crosssum(lang: str, n: int, seed: int, model, tokenizer, device: str) -> list:
    path = os.path.join(ROOT, "crosssum_in", f"crosssum_english-{lang}_test.json")
    if not os.path.exists(path):
        return []
    records = []
    for ex in load_examples(path, n, seed):
        prompt = CROSSSUM_TEMPLATE.format(lang=LANG_NAMES[lang], text=ex["text"][:MAX_ARTICLE_CHARS])
        prediction = generate(model, tokenizer, device, prompt, max_new_tokens=100)
        records.append({
            "id": None, "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": [ex["summary"]],
        })
    return records


def run_flores(lang: str, n: int, seed: int, model, tokenizer, device: str) -> list:
    records = []
    for direction, filename, template in [
        ("enxx", f"flores_en_{lang}_test.json", FLORES_ENXX_TEMPLATE),
        ("xxen", f"flores_{lang}_en_test.json", FLORES_XXEN_TEMPLATE),
    ]:
        path = os.path.join(ROOT, "flores_in", filename)
        if not os.path.exists(path):
            continue
        for ex in load_examples(path, n, seed):
            prompt = template.format(lang=LANG_NAMES[lang], source=ex["source"])
            prediction = generate(model, tokenizer, device, prompt, max_new_tokens=200)
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


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path of the fine-tuned checkpoint.")
    parser.add_argument("--device", default=None, help="cuda / xpu / cpu (default: cuda > xpu > cpu, whichever is available)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
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

    os.makedirs(args.output_dir, exist_ok=True)

    all_scores = {}
    for task in args.tasks:
        runner = TASK_RUNNERS[task]
        task_records = []
        for lang in args.langs:
            print(f"[{task}] {lang} ...", flush=True)
            start = time.time()
            try:
                records = runner(lang, args.num_examples, args.seed, model, tokenizer, device)
            except Exception as e:
                print(f"    ERROR loading/running {lang}: {e!r}", flush=True)
                continue
            task_records.extend(records)
            print(f"    {len(records)} examples in {time.time() - start:.1f}s", flush=True)

        out_path = os.path.join(args.output_dir, f"{task}_predictions.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(task_records, f, ensure_ascii=False, indent=2)

        scores = run_benchmark(task, task_records)
        all_scores[task] = scores
        print(f"\n=== {task} scores ===")
        print(json.dumps(scores, ensure_ascii=False, indent=2))
        print()

    scores_path = os.path.join(args.output_dir, "scores_summary.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    print(f"All scores saved to {scores_path}")


if __name__ == "__main__":
    main()
