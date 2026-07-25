"""
Build a small Hindi-only CPT test sample from data/CPT/hi_Hindi_200Mtok_slice.txt
and (optionally) push it to the Hugging Face Hub as a dataset repo.

The full Hindi slice is ~1.4GB - too large to sanity-check the CPT pipeline
(tokenization, packing, config wiring) quickly. This script takes the first
N non-empty lines instead, writes them locally as JSONL with a `text` column
(matching `field: text` in qwen2.5_0.5b_cpt_full.yml), and can push that as
a HF dataset repo.

Usage:
    # Just build the local sample (no upload):
    python make_cpt_test_sample.py

    # Build and push to the Hub (requires `huggingface-cli login` first):
    python make_cpt_test_sample.py --repo-id your-hf-username/hi-cpt-test-sample --push

    # Bigger/smaller sample:
    python make_cpt_test_sample.py --num-lines 20000 --repo-id ... --push
"""

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "CPT" / "hi_Hindi_200Mtok_slice.txt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "CPT" / "CPT_test_sample"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                         help="Source Hindi CPT text file (one doc/line).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help="Where to write the local JSONL sample.")
    parser.add_argument("--num-lines", type=int, default=25000,
                         help="Number of non-empty lines to sample (default: 25000).")
    parser.add_argument("--repo-id", type=str, default=None,
                         help="HF dataset repo id to push to, e.g. your-hf-username/hi-cpt-test-sample.")
    parser.add_argument("--private", action="store_true",
                         help="Create the HF dataset repo as private.")
    parser.add_argument("--push", action="store_true",
                         help="Actually push to the Hub (requires --repo-id and HF login).")
    return parser.parse_args()


def read_sample_lines(input_path: Path, num_lines: int) -> list[str]:
    lines = []
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            lines.append(line)
            if len(lines) >= num_lines:
                break
    return lines


def main() -> None:
    args = parse_args()

    if args.push and not args.repo_id:
        raise SystemExit("--push requires --repo-id (e.g. your-hf-username/hi-cpt-test-sample)")

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    print(f"Reading up to {args.num_lines} lines from {args.input} ...")
    lines = read_sample_lines(args.input, args.num_lines)
    print(f"Collected {len(lines)} non-empty lines.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "hi_test_sample.jsonl"

    import json
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
    print(f"Wrote local sample: {out_path} ({out_path.stat().st_size:,} bytes)")

    if args.push:
        from datasets import Dataset

        dataset = Dataset.from_list([{"text": line} for line in lines])
        print(f"Pushing {len(dataset)} rows to hub as '{args.repo_id}' (private={args.private}) ...")
        dataset.push_to_hub(args.repo_id, private=args.private)
        print(f"Done: https://huggingface.co/datasets/{args.repo_id}")
    else:
        print("Skipped upload (pass --push --repo-id <repo> to publish to the Hub).")


if __name__ == "__main__":
    main()
