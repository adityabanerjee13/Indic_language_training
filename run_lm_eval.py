"""
Run MMLU (5-shot) and IFEval over the CPT/SFT/DPO model family with
lm-evaluation-harness, in the prompt envelope each checkpoint expects.

Why a wrapper rather than raw `lm_eval` invocations:

  * Envelope per model. Base/CPT checkpoints are completion models and are
    scored on raw prompts; the SFT/DPO checkpoints were trained on text
    rendered by `tokenizer.apply_chat_template` (data/cap_and_rebalance.py:74),
    so they get `--apply_chat_template` (plus `--fewshot_as_multiturn` on MMLU,
    so the 5 shots become real conversation turns instead of one glued-together
    user message). Getting this wrong reads as a model regression rather than a
    harness mistake, so it is encoded here per model instead of retyped.

  * IFEval generation budget -- and the reason this uses the Python API rather
    than the `lm_eval` CLI. The ifeval task decodes until EOS, and a *base*
    model never emits one on those prompts, so it runs the full budget every
    time; on this box (Intel Arc iGPU, ~205 ms per decode step at batch 16)
    2048 tokens is ~1 h per batch-of-16. Capping the budget is therefore the
    difference between an overnight run and a week.

    The cap cannot be set from the CLI. `--gen_kwargs max_gen_toks=512` is
    accepted and logged, but Qwen's own generation_config.json ships
    `max_new_tokens=2048`, and in transformers max_new_tokens beats the
    max_length that max_gen_toks computes -- so generation silently runs to
    2048 anyway. Passing `max_new_tokens=512` alongside does not help either:
    lm-eval 0.4.12 de-duplicates "multiple max token args" and keeps only
    max_gen_toks, so max_new_tokens never reaches generate(). The only reliable
    fix is to clear the model's generation_config field in-process, which is
    what patch_generation_config() below does.

    The cap applies identically to every model, so cross-model comparison
    holds; absolute IFEval numbers are not comparable to published results that
    decode to 2048.

  * Windows console encoding. lm-eval's final `make_table` prints a non-cp1252
    arrow glyph in its header and dies with UnicodeEncodeError on a Windows
    console -- *after* writing results.json, so the scores survive but the run
    looks failed. Going through the API and printing our own JSON avoids it.
    (Keep this file's own text ASCII for the same reason: argparse echoes this
    docstring for --help, and a stray non-cp1252 character makes --help crash.)

Usage:
    python run_lm_eval.py                      # full sweep, both tasks
    python run_lm_eval.py --tasks ifeval       # one task
    python run_lm_eval.py --models sft-IT dpo-IT
    python run_lm_eval.py --limit 8            # smoke test
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(ROOT, "lm_eval_results")

# short name -> (hf repo id, envelope). "chat" means the prompt is wrapped in
# the checkpoint's own chat template; "completion" means raw text.
MODELS = {
    "base":          ("Qwen/Qwen2.5-0.5B",                              "completion"),
    "cpt-mix-1to2":  ("adityabanerjee13/qwen2.5-0.5b-cpt-mix-1to2",     "completion"),
    "indic-cpt":     ("adityabanerjee13/qwen2.5-0.5b-indic-cpt",        "completion"),
    "sft-I":         ("adityabanerjee13/qwen2.5-0.5b-sft-I",            "chat"),
    "sft-IT":        ("adityabanerjee13/qwen2.5-0.5b-sft-IT",           "chat"),
    "dpo-IT":        ("adityabanerjee13/qwen2.5-0.5b-dpo-IT",           "chat"),
    "dpo-IT_non_align": ("adityabanerjee13/qwen2.5-0.5b-dpo-IT_non_align", "chat"),
    "sarvam-1":      ("sarvamai/sarvam-1",                              "completion"),
}

MMLU_FEWSHOT = 5          # Open LLM Leaderboard convention, not lm-eval's 0-shot default
IFEVAL_MAX_GEN_TOKS = 512  # see module docstring

# The two tasks are limited by opposite things, so they get different batches.
#
#   mmlu   - loglikelihood scoring materializes a (batch, seq, 151936) logit
#            tensor in fp32. A 5-shot prompt is ~1400 tokens, so one sequence
#            alone is 4 x 1400 x 151936 x 4 B = 0.85 GiB: batch 32 asks for
#            ~27 GiB and batch 8 for ~6.7 GiB, both of which OOM this 16 GiB
#            card. Batch 4 fits with headroom. Raising sequence length or shot
#            count means lowering this again.
#   ifeval - generation only ever holds logits for the newest token, so memory
#            is trivial and the batch is bounded by decode throughput instead.
DEFAULT_BATCH_SIZES = {"mmlu": 4, "ifeval": 16}


def patch_generation_config(lm, max_gen_toks: int) -> None:
    """Stop the checkpoint's own generation_config from overriding our budget.

    Qwen2.5 repos ship `max_new_tokens: 2048` in generation_config.json. In
    transformers max_new_tokens wins over the max_length lm-eval derives from
    max_gen_toks, so the budget silently reverts to 2048. Clearing the field
    hands control back to max_gen_toks. Also clear max_length, which would
    otherwise cap prompt+completion instead of just the completion.
    """
    gen_cfg = lm.model.generation_config
    before = gen_cfg.max_new_tokens
    gen_cfg.max_new_tokens = None
    gen_cfg.max_length = None
    print(f"  generation_config.max_new_tokens {before} -> None (budget is max_gen_toks={max_gen_toks})")


def load_model_to_device(repo: str, device: str):
    """Materialize the weights on CPU, then move them to the accelerator.

    Letting HFLM pass a target device straight to from_pretrained sends
    transformers down its caching_allocator_warmup path, which reserves ONE
    contiguous block the size of the whole model. This iGPU refuses a single
    4.71 GiB allocation even with 14.4 GiB free, so ~2B models (sarvam-1) die
    at load with a misleading "XPU out of memory". Building on CPU and calling
    .to() afterwards allocates per tensor and sidesteps the limit -- the same
    thing IndicGenBench/run_qwen_ft_benchmark.py:137 does.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16")
    model.to(device)
    model.eval()
    print(f"  loaded {repo} on {device} (bf16)", flush=True)
    return model


def result_path(model_key: str, task: str, args) -> str:
    _, envelope = MODELS[model_key]
    return os.path.join(args.output_dir, f"{model_key}__{envelope}__{task}", "results.json")


def run_one(model_key: str, task: str, args) -> dict:
    """Evaluate one model on one task, returning lm-eval's results dict.

    Runs in a dedicated child process (see run_one_subprocess): the XPU
    allocator does not reliably hand a 0.5-2B model's weights back when the
    HFLM that owns them goes out of scope, so evaluating several models in one
    process fills the 16 GiB card and every load after the first dies with
    UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY. Process exit always releases it.
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    repo, envelope = MODELS[model_key]
    is_chat = envelope == "chat"
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[task]

    lm = HFLM(
        pretrained=load_model_to_device(repo, args.device),
        tokenizer=repo,
        batch_size=batch_size,
    )

    kwargs = {}
    if task == "mmlu":
        kwargs["num_fewshot"] = args.num_fewshot
        # Lay the shots out as real conversation turns rather than one glued
        # user message. Only valid alongside apply_chat_template.
        if is_chat and args.num_fewshot > 0:
            kwargs["fewshot_as_multiturn"] = True
    if task == "ifeval":
        kwargs["gen_kwargs"] = f"max_gen_toks={args.max_gen_toks}"
        patch_generation_config(lm, args.max_gen_toks)

    return lm_eval.simple_evaluate(
        model=lm,
        tasks=[task],
        apply_chat_template=is_chat,
        limit=args.limit,
        log_samples=args.log_samples,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        **kwargs,
    )


def write_results(model_key: str, task: str, args, results: dict) -> str:
    out_dir = os.path.dirname(result_path(model_key, task, args))
    os.makedirs(out_dir, exist_ok=True)
    samples = results.pop("samples", None)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    if samples is not None:
        with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as fh:
            json.dump(samples, fh, ensure_ascii=False, indent=2, default=str)
    return out_dir


def run_one_subprocess(model_key: str, task: str, args) -> int:
    """Re-invoke this script with --single so the run gets its own process."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--single", model_key, task,
        "--device", args.device,
        "--num-fewshot", str(args.num_fewshot),
        "--max-gen-toks", str(args.max_gen_toks),
        "--seed", str(args.seed),
        "--output-dir", args.output_dir,
    ]
    if args.batch_size:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.log_samples:
        cmd += ["--log-samples"]

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, env=env).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--tasks", nargs="*", default=["mmlu", "ifeval"], choices=["mmlu", "ifeval"])
    # NOTE: "xpu:0", not "xpu". lm-eval 0.4.12 validates --device against a
    # device_list that holds bare "cuda"/"cpu"/"mps" but only *indexed* xpu
    # entries (lm_eval/models/huggingface.py:276-283). A bare "xpu" therefore
    # fails the membership test, logs "Device not specified", and silently
    # falls back to CPU -- ~10x slower, with nothing that looks like an error.
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                        help=f"Override the per-task default ({DEFAULT_BATCH_SIZES}).")
    parser.add_argument("--num-fewshot", dest="num_fewshot", type=int, default=MMLU_FEWSHOT)
    parser.add_argument("--max-gen-toks", dest="max_gen_toks", type=int, default=IFEVAL_MAX_GEN_TOKS)
    parser.add_argument("--limit", type=int, default=None, help="Docs per task; smoke tests only.")
    parser.add_argument("--log-samples", dest="log_samples", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even where a results.json already exists (default: skip, so an interrupted sweep resumes).")
    parser.add_argument("--single", nargs=2, metavar=("MODEL", "TASK"), default=None,
                        help="Internal: evaluate exactly one model/task in this process, then exit.")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(args.output_dir, exist_ok=True)

    # Child mode: one run, one process, then exit so the XPU is released.
    if args.single:
        model_key, task = args.single
        results = run_one(model_key, task, args)
        out_dir = write_results(model_key, task, args, results)
        print(f"  wrote {out_dir}", flush=True)
        print(json.dumps(results["results"], indent=2, default=str), flush=True)
        return

    failures = []
    for model_key in args.models:
        for task in args.tasks:
            _, envelope = MODELS[model_key]
            label = f"{model_key} [{envelope}] / {task}"

            if not args.overwrite and os.path.exists(result_path(model_key, task, args)):
                print(f"\nSKIP (already have results): {label}", flush=True)
                continue

            print(f"\n{'=' * 70}\n{label}\n{'=' * 70}", flush=True)
            start = time.time()
            returncode = run_one_subprocess(model_key, task, args)
            elapsed = (time.time() - start) / 60

            if returncode == 0:
                print(f"ok in {elapsed:.1f} min", flush=True)
            else:
                print(f"FAILED after {elapsed:.1f} min (exit {returncode})", flush=True)
                failures.append(label)

    print("\nSWEEP COMPLETE" + (f" with {len(failures)} failure(s): {failures}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
