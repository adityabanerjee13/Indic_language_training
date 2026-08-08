"""
Sweep every checkpoint over every prompt in prompts/ at several temperatures,
and write one JSON with all the completions.

Same envelope as infer.py: each prompt is fed raw, with no chat template. That
is the right treatment for base/CPT checkpoints and a deliberate off-
distribution probe for the SFT/DPO ones -- their scores here are not evidence
of an instruction-following regression, only of what they do without their
envelope.

Sampling is pure temperature: no top_p, no top_k, so the temperature is the
only thing varying and nothing truncates the tail behind it. The seed is fixed
and re-applied before every generation, so a given (model, prompt, temperature)
is reproducible and differences across temperature are not seed noise.

Why one child process per model, and one warmup generation inside it:

  * Process per model. The XPU allocator does not reliably hand a model's
    weights back when the owning object goes out of scope, so loading several
    checkpoints in one process fills the 16 GiB card and every load after the
    first dies with UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY. Process exit always
    releases it. Same reason run_lm_eval.py:130 shells out.

  * Warmup. The first *sampled* generate() in a process costs ~90 s on this
    iGPU while every one after it costs ~0.2 s -- a one-time compile of the
    sampling path, not a per-token cost, and it lands on whichever sampled call
    happens to run first regardless of temperature/top_p/top_k. Left alone it
    would be billed to whatever the first (prompt, temperature) cell happens to
    be and would make that cell look pathologically slow. warm_up_sampling()
    pays it explicitly, up front, so the recorded timings are comparable.

Results land in prompts/sweep_results.json. Per-model partials are kept in
prompts/_partial/ so an interrupted sweep resumes instead of restarting; pass
--overwrite to force a re-run.

Keep this file's text ASCII -- argparse echoes it for --help through cp1252.

Usage:
    python run_sweep.py
    python run_sweep.py --models base indic-cpt --max-new-tokens 80
    python run_sweep.py --temperatures 0.1 0.3 0.5 --overwrite
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import types

import infer

ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(ROOT, "prompts")
PARTIAL_DIR = os.path.join(PROMPT_DIR, "_partial")
RESULTS_PATH = os.path.join(PROMPT_DIR, "sweep_results.json")

# Endpoints of the requested range plus its midpoint.
DEFAULT_TEMPERATURES = [0.1, 0.35, 0.6]
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_SEED = 42


def load_prompts(prompt_dir: str) -> dict:
    """Every .txt in `prompt_dir`, keyed by filename stem, read byte for byte.

    Non-recursive on purpose: prompts/ondist/ and any other subfolder is its own
    prompt set, so adding one does not silently widen a sweep over prompts/.
    """
    prompts = {}
    for path in sorted(glob.glob(os.path.join(prompt_dir, "*.txt"))):
        with open(path, encoding="utf-8") as fh:
            prompts[os.path.splitext(os.path.basename(path))[0]] = fh.read()
    if not prompts:
        sys.exit(f"No .txt prompts found in {prompt_dir}")
    return prompts


def warm_up_sampling(model, tokenizer, device: str) -> float:
    """Pay the one-time sampling-path compile so it is not billed to a result."""
    import torch

    inputs = tokenizer("warmup", return_tensors="pt", add_special_tokens=False).to(device)
    start = time.time()
    with torch.no_grad():
        model.generate(
            **inputs, max_new_tokens=2, do_sample=True, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - start
    infer.log(f"  sampling warmup: {elapsed:.1f}s (one-time)")
    return elapsed


def run_model(model_key: str, args) -> list:
    """Every (prompt, temperature) cell for one checkpoint, in this process."""
    repo = infer.resolve_model(model_key)
    model, tokenizer = infer.load_model(repo, args.device, args.dtype)
    warm_up_sampling(model, tokenizer, args.device)

    prompts = load_prompts(args.prompt_dir)
    rows = []
    for prompt_name, prompt_text in prompts.items():
        for temperature in args.temperatures:
            # infer.generate reads its knobs off an argparse-style namespace;
            # building one keeps generation semantics (no special tokens, no
            # stripping, seed re-applied per call) identical to the CLI.
            gen_args = types.SimpleNamespace(
                max_new_tokens=args.max_new_tokens,
                max_prompt_tokens=3072,
                temperature=temperature,
                top_p=None,
                top_k=None,
                seed=args.seed,
                stream=False,
            )
            result = infer.generate(model, tokenizer, args.device, prompt_text, gen_args)
            infer.log(f"  {prompt_name} @ T={temperature}: {result['new_tokens']} tokens "
                      f"in {result['seconds']}s")
            rows.append({
                "model": model_key,
                "repo": repo,
                "prompt": prompt_name,
                "temperature": temperature,
                **result,
            })
    return rows


def partial_path(model_key: str, args) -> str:
    return os.path.join(args.partial_dir, f"{model_key}.json")


def merge_results(model_keys: list, args) -> dict:
    rows = []
    missing = []
    for model_key in model_keys:
        path = partial_path(model_key, args)
        if not os.path.exists(path):
            missing.append(model_key)
            continue
        with open(path, encoding="utf-8") as fh:
            rows.extend(json.load(fh))

    return {
        "metadata": {
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device,
            "dtype": args.dtype,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperatures": args.temperatures,
            "decoding": "pure temperature sampling (do_sample=True, no top_p, no top_k)",
            "envelope": "raw completion - no chat template applied to any model",
            "envelope_caveat": (
                "sft-*/dpo-* checkpoints were trained on chat-templated text; feeding them "
                "a bare prompt runs them off-distribution on purpose, so weak output here is "
                "not an instruction-following regression"
            ),
            "models_missing": missing,
            "prompt_dir": os.path.relpath(args.prompt_dir, ROOT).replace("\\", "/"),
        },
        "prompts": load_prompts(args.prompt_dir),
        "results": rows,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--models", nargs="*", default=list(infer.MODEL_ALIASES),
                        choices=list(infer.MODEL_ALIASES))
    parser.add_argument("--temperatures", nargs="*", type=float, default=DEFAULT_TEMPERATURES)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # Prompt set, results file and partial dir move together: a run over a
    # different prompt set must not resume from another run's partials, since
    # those hold completions for prompts that are not in this set.
    parser.add_argument("--prompt-dir", dest="prompt_dir", default=PROMPT_DIR,
                        help="Directory of .txt prompts, non-recursive (default: prompts/).")
    parser.add_argument("--output", default=RESULTS_PATH, help="Results JSON path.")
    parser.add_argument("--partial-dir", dest="partial_dir", default=PARTIAL_DIR,
                        help="Where per-model partials live; must be unique per prompt set.")
    parser.add_argument("--device", default=None, help="Default: cuda > xpu > cpu.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run models that already have a partial (default: resume).")
    parser.add_argument("--single", default=None,
                        help="Internal: run exactly this model in this process, then exit.")
    args = parser.parse_args()

    if args.device is None:
        import torch

        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            args.device = "xpu"
        else:
            args.device = "cpu"

    os.makedirs(args.partial_dir, exist_ok=True)

    # Child mode: one model, one process, then exit so the XPU is released.
    if args.single:
        rows = run_model(args.single, args)
        with open(partial_path(args.single, args), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
        return

    failures = []
    for model_key in args.models:
        if not args.overwrite and os.path.exists(partial_path(model_key, args)):
            print(f"SKIP {model_key} (already have a partial)", flush=True)
            continue

        print(f"\n{'=' * 60}\n{model_key}\n{'=' * 60}", flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--single", model_key,
            "--device", args.device,
            "--dtype", args.dtype,
            "--seed", str(args.seed),
            "--max-new-tokens", str(args.max_new_tokens),
            "--prompt-dir", args.prompt_dir,
            "--partial-dir", args.partial_dir,
            "--temperatures", *[str(t) for t in args.temperatures],
        ]
        start = time.time()
        returncode = subprocess.run(cmd, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).returncode
        if returncode == 0:
            print(f"ok in {(time.time() - start) / 60:.1f} min", flush=True)
        else:
            print(f"FAILED (exit {returncode})", flush=True)
            failures.append(model_key)

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(merge_results(args.models, args), fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {args.output}")
    if failures:
        print(f"{len(failures)} model(s) failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
