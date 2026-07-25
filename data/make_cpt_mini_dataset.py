"""
Build a small multi-language CPT "mini" dataset: ~1M Qwen2.5 tokens per
language, sampled from data/CPT/<lang>_..._200Mtok_slice.txt, then push it
to the Hugging Face Hub as a single dataset repo (columns: text, lang).

Token counts are computed with the actual base-model tokenizer (not a
byte-based estimate) since Indic-script tokenization efficiency varies a lot
depending on how well each script is covered by Qwen2.5's vocab (see prior
discussion: byte-level BPE falls back to long token runs for
under-represented scripts, so "1M tokens" of Hindi and "1M tokens" of Tamil
can be very different amounts of text).

Usage:
    # Build the local sample only (no upload):
    python make_cpt_mini_dataset.py

    # Build and push to the Hub:
    python make_cpt_mini_dataset.py --repo-id adityabanerjee13/indic-cpt-mini --push

    # Different budget per language:
    python make_cpt_mini_dataset.py --tokens-per-lang 500000 --repo-id ... --push
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CPT_DIR = SCRIPT_DIR / "CPT"
DEFAULT_OUTPUT_DIR = CPT_DIR / "CPT_mini_dataset"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B"

# lang code -> source filename (matches data/download_cpt_data.py output naming)
LANGUAGE_FILES = {
    "bn": "bn_Bengali_200Mtok_slice.txt",
    "gu": "gu_Gujarati_200Mtok_slice.txt",
    "hi": "hi_Hindi_200Mtok_slice.txt",
    "kn": "kn_Kannada_200Mtok_slice.txt",
    "ml": "ml_Malayalam_200Mtok_slice.txt",
    "mr": "mr_Marathi_200Mtok_slice.txt",
    "or": "or_Odia_200Mtok_slice.txt",
    "pa": "pa_Punjabi_200Mtok_slice.txt",
    "ta": "ta_Tamil_200Mtok_slice.txt",
    "te": "te_Telugu_200Mtok_slice.txt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpt-dir", type=Path, default=CPT_DIR,
                         help="Directory containing the per-language *_200Mtok_slice.txt files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="Where to write the local JSONL dataset.")
    parser.add_argument("--tokens-per-lang", type=int, default=1_000_000,
                         help="Target token budget per language (default: 1,000,000).")
    parser.add_argument("--base-model", type=str, default=DEFAULT_BASE_MODEL,
                         help="Tokenizer used to count tokens; should match the CPT training base model.")
    parser.add_argument("--repo-id", type=str, default=None,
                         help="HF dataset repo id to push to, e.g. adityabanerjee13/indic-cpt-mini.")
    parser.add_argument("--private", action="store_true",
                         help="Create the HF dataset repo as private.")
    parser.add_argument("--push", action="store_true",
                         help="Actually push to the Hub (requires --repo-id and HF_TOKEN).")
    return parser.parse_args()


def sample_language(path: Path, tokenizer, target_tokens: int, lang: str) -> list[dict]:
    rows = []
    total_tokens = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            n_tokens = len(tokenizer.encode(line, add_special_tokens=False))
            if n_tokens == 0:
                continue
            rows.append({"text": line, "lang": lang})
            total_tokens += n_tokens
            if total_tokens >= target_tokens:
                break
    print(f"  [{lang}] {len(rows):,} lines, {total_tokens:,} tokens")
    return rows


def main() -> None:
    args = parse_args()

    if args.push and not args.repo_id:
        raise SystemExit("--push requires --repo-id (e.g. adityabanerjee13/indic-cpt-mini)")

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    all_rows: list[dict] = []
    for lang, filename in LANGUAGE_FILES.items():
        path = args.cpt_dir / filename
        if not path.exists():
            print(f"  [{lang}] SKIP - file not found: {path}")
            continue
        print(f"Sampling {lang} from {path.name} ...")
        all_rows.extend(sample_language(path, tokenizer, args.tokens_per_lang, lang))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "indic_cpt_mini.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote local dataset: {out_path} ({len(all_rows):,} rows, {out_path.stat().st_size:,} bytes)")

    if args.push:
        from datasets import Dataset

        dataset = Dataset.from_list(all_rows)
        print(f"Pushing {len(dataset)} rows to hub as '{args.repo_id}' (private={args.private}) ...")
        dataset.push_to_hub(args.repo_id, private=args.private)
        print(f"Done: https://huggingface.co/datasets/{args.repo_id}")
    else:
        print("Skipped upload (pass --push --repo-id <repo> to publish to the Hub).")


if __name__ == "__main__":
    main()
