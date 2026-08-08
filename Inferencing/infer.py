"""
Raw-completion inference: give a model name and a prompt, get the continuation.

The prompt is fed to the model exactly as given -- no chat template, no system
message, no added instruction wrapper. This is the *completion* envelope, the
one the base and CPT checkpoints were trained in (Qwen2.5-0.5B, the *-cpt-*
models, sarvam-1), and it is what you want when inspecting what a checkpoint
actually learned rather than how well it follows an instruction format. Feeding
a bare prompt to an SFT/DPO checkpoint runs it off-distribution on purpose;
run_qwen_ft_benchmark.py --chat is the tool for the wrapped envelope.

Only the completion reaches stdout. Load messages, device and timing go to
stderr, so `python infer.py ... > out.txt` captures the continuation alone.

Two consequences of the raw envelope worth knowing before reading the output:

  * A completion model has no reason to stop. It never emits EOS on a raw
    prompt, so generation runs the full --max-new-tokens budget every time and
    typically ends mid-sentence. That is the budget, not a failure.
  * Trailing whitespace in the prompt changes the tokenization and therefore
    the continuation. --prompt keeps whatever you pass verbatim; use
    --prompt-file or --stdin when the exact bytes matter (a shell will eat
    trailing spaces).

Decoding is greedy by default, so repeated runs of one prompt agree. Pass
--temperature (and optionally --top-p / --top-k) to sample instead -- but note
that sampling is drastically slower on the Arc iGPU here: measured ~0.3 s per
token greedy against ~16 s per token sampled on Qwen2.5-0.5B. The individual
sampling ops (sort, softmax, multinomial, topk over the 151936-wide vocab) all
benchmark under 1 ms on the XPU in both fp32 and bf16, so the cost is somewhere
else in the sampling path and is not yet diagnosed. Greedy is the default
partly for this reason.

One prompt per invocation, deliberately: this is a probe, not a benchmark
runner. The batched-generation path lives in IndicGenBench/.

Keep this file's own text ASCII: argparse echoes this docstring for --help, and
on a Windows console that write happens through cp1252, so a single Devanagari
character in here makes --help die with UnicodeEncodeError. (Indic *prompts* are
fine -- main() switches the streams to UTF-8 before anything is generated.)

Usage:
    python infer.py --model base --prompt "The capital of India is"
    python infer.py --model indic-cpt --prompt "<Hindi prompt>" --max-new-tokens 128
    python infer.py --model adityabanerjee13/qwen2.5-0.5b-indic-cpt --stdin < prompt.txt
    python infer.py --model sarvam-1 --prompt "..." --temperature 0.8 --seed 0 --stream
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

# Short names for the checkpoints in this project, so the common case is
# `--model indic-cpt` instead of the full repo id. Anything not in here is
# passed to transformers untouched, so a raw HF repo id or a local checkpoint
# directory works just as well. Kept in step with run_lm_eval.py:62.
MODEL_ALIASES = {
    "base":              "Qwen/Qwen2.5-0.5B",
    "cpt-mix-1to2":      "adityabanerjee13/qwen2.5-0.5b-cpt-mix-1to2",
    "indic-cpt":         "adityabanerjee13/qwen2.5-0.5b-indic-cpt",
    "sft-I":             "adityabanerjee13/qwen2.5-0.5b-sft-I",
    "sft-IT":            "adityabanerjee13/qwen2.5-0.5b-sft-IT",
    "dpo-IT":            "adityabanerjee13/qwen2.5-0.5b-dpo-IT",
    "dpo-IT_non_align":  "adityabanerjee13/qwen2.5-0.5b-dpo-IT_non_align",
    "sarvam-1":          "sarvamai/sarvam-1",
}
DEFAULT_MAX_NEW_TOKENS = 256


def log(message: str) -> None:
    """Progress goes to stderr so stdout stays exactly the completion."""
    print(message, file=sys.stderr, flush=True)


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def load_tokenizer(model_id: str):
    """Load the tokenizer, self-healing a corrupted tokenizer_config.json.

    Some checkpoints here (incl. adityabanerjee13/qwen2.5-0.5b-indic-cpt) were
    saved by a buggy transformers that serialized `extra_special_tokens` as a
    list and leaked runtime-only keys. transformers>=5 calls `.keys()` on that
    field and crashes. tokenizer.json itself is fine, so sanitize the config in
    a local copy and load from there. Same fix as
    IndicGenBench/run_qwen_ft_benchmark.py:78.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_id)
    except AttributeError as exc:
        if "keys" not in str(exc):
            raise
        log("  tokenizer_config.json is malformed; sanitizing and retrying...")

    from huggingface_hub import snapshot_download

    if os.path.isdir(model_id):
        src = model_id
    else:
        src = snapshot_download(
            model_id,
            allow_patterns=[
                "tokenizer*.json", "tokenizer.model", "vocab.json", "merges.txt",
                "special_tokens_map.json", "*.jinja",
            ],
        )

    fixed_dir = os.path.join(
        tempfile.gettempdir(), "infer_tokenizer_" + os.path.basename(model_id.rstrip("/\\"))
    )
    os.makedirs(fixed_dir, exist_ok=True)
    for fname in os.listdir(src):
        shutil.copy2(os.path.join(src, fname), os.path.join(fixed_dir, fname))

    cfg_path = os.path.join(fixed_dir, "tokenizer_config.json")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if isinstance(cfg.get("extra_special_tokens"), list):
        cfg.pop("extra_special_tokens", None)
    for key in ("backend", "is_local", "local_files_only"):
        cfg.pop(key, None)
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)

    return AutoTokenizer.from_pretrained(fixed_dir)


def load_model(model_id: str, device: str, dtype: str):
    """Materialize the weights on CPU, then move them to the accelerator.

    Never hand a device to from_pretrained: that takes transformers down its
    caching_allocator_warmup path, which reserves ONE contiguous block the size
    of the whole model. The Arc iGPU refuses a single 4.71 GiB allocation even
    with 14.4 GiB free, so ~2B checkpoints (sarvam-1) die at load with a
    misleading "XPU out of memory". Building on CPU and calling .to() allocates
    per tensor and sidesteps the limit.

    dtype is passed as a *string*: transformers 5.14.1 stores it on the config
    and json-serializes that when logging the config repr, which a torch.dtype
    object is not serializable for.
    """
    from transformers import AutoModelForCausalLM

    log(f"Loading {model_id} ...")
    tokenizer = load_tokenizer(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)
    model.to(device)
    model.eval()

    # Qwen2.5 repos ship sampling defaults (do_sample=True with a temperature,
    # top_p and top_k) plus max_new_tokens=2048 in generation_config.json.
    # Every one of those is decided per call below, and leaving them set makes
    # transformers warn about sampling params being ignored on a greedy run.
    # (The lm-eval trap where generation_config beats the requested budget --
    # see run_lm_eval.py:89 -- does not apply here: an explicit max_new_tokens
    # kwarg wins over the config. Clearing it is belt and braces.)
    gen_cfg = model.generation_config
    gen_cfg.do_sample = False
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.max_new_tokens = None
    gen_cfg.max_length = None

    log(f"Loaded on {device} ({dtype}).")
    return model, tokenizer


def read_prompt(args) -> str:
    """The prompt, byte for byte, from whichever source was chosen.

    Deliberately not stripped: for a completion model a trailing space or
    newline is part of the input and changes the continuation.
    """
    if args.stdin:
        return sys.stdin.read()
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as fh:
            return fh.read()
    return args.prompt


def generate(model, tokenizer, device: str, prompt: str, args) -> dict:
    """Continue `prompt`, returning the completion plus what it cost.

    Returns a dict rather than a bare string so callers that log runs
    (run_sweep.py) get the token counts and timing without re-deriving them.
    """
    import torch

    # add_special_tokens=False keeps the prompt literally the prompt. Qwen2.5
    # adds nothing here anyway, but a tokenizer with a BOS would otherwise
    # prepend a token the caller never wrote.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_prompt_tokens,
    ).to(device)

    prompt_tokens = inputs["input_ids"].shape[-1]
    if prompt_tokens == args.max_prompt_tokens:
        log(f"WARNING: prompt hit the {args.max_prompt_tokens}-token cap and was truncated "
            f"(raise --max-prompt-tokens).")

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature is None:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=args.temperature)
        if args.top_p is not None:
            gen_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            gen_kwargs["top_k"] = args.top_k
        torch.manual_seed(args.seed)

    streamer = None
    if args.stream:
        from transformers import TextStreamer

        # skip_prompt keeps stdout to the completion alone, matching the
        # non-streaming path. skip_special_tokens likewise.
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

    start = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - start

    new_tokens = output_ids[0][prompt_tokens:]
    print("f:", tokenizer.decode(new_tokens, skip_special_tokens=True))
    log(f"{prompt_tokens} prompt tokens -> {len(new_tokens)} new tokens "
        f"in {elapsed:.1f}s ({len(new_tokens) / max(elapsed, 1e-9):.1f} tok/s)")

    return {
        # Not stripped, for the same reason the prompt is not: the leading
        # space of a continuation is part of it.
        "completion": tokenizer.decode(new_tokens, skip_special_tokens=True),
        "prompt_tokens": int(prompt_tokens),
        "new_tokens": int(len(new_tokens)),
        "seconds": round(elapsed, 2),
    }


def main() -> None:
    # Before parse_args, not after: --help is printed from inside parse_args, so
    # a later reconfigure would come too late to save it. Indic script does not
    # survive the Windows console's cp1252 default, and without this a perfectly
    # good completion dies in a UnicodeEncodeError at print time.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True,
                        help="Alias (" + ", ".join(MODEL_ALIASES) + "), HF repo id, or local path.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="Prompt text, used exactly as given.")
    source.add_argument("--prompt-file", dest="prompt_file", help="Read the prompt from a file (UTF-8).")
    source.add_argument("--stdin", action="store_true", help="Read the prompt from stdin.")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int,
                        default=DEFAULT_MAX_NEW_TOKENS,
                        help=f"Completion budget; a completion model runs all of it (default {DEFAULT_MAX_NEW_TOKENS}).")
    parser.add_argument("--max-prompt-tokens", dest="max_prompt_tokens", type=int, default=3072,
                        help="Truncate the prompt above this length (default 3072).")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Enable sampling at this temperature (default: greedy).")
    parser.add_argument("--top-p", dest="top_p", type=float, default=None, help="Only with --temperature.")
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="Only with --temperature.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed; greedy decoding ignores it.")
    parser.add_argument("--device", default=None,
                        help="cuda / xpu / cpu (default: cuda > xpu > cpu, whichever is available).")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--stream", action="store_true",
                        help="Print tokens as they are decoded instead of waiting for the full completion.")
    args = parser.parse_args()

    if args.temperature is None and (args.top_p is not None or args.top_k is not None):
        parser.error("--top-p/--top-k only apply to sampling; pass --temperature too.")

    import torch

    def pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"

    device = args.device or pick_device()
    if device == "cpu" and args.dtype != "fp32":
        log("WARNING: CPU with a non-fp32 dtype is slow and sometimes unsupported; consider --dtype fp32.")

    model, tokenizer = load_model(resolve_model(args.model), device, args.dtype)
    result = generate(model, tokenizer, device, read_prompt(args), args)

    # --stream already wrote the completion to stdout token by token.
    if not args.stream:
        sys.stdout.write(result["completion"])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
