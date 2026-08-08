"""
Pull the Hindi slice of a CPT mini dataset out of the Hub and write it locally.

The published mini corpora (adityabanerjee13/indic-cpt-mini-train and friends)
interleave all ten Sarvam-1 languages in one split, tagged by a `lang` column
(see data/make_cpt_mini_dataset.py, which builds them). The axolotl configs in
this folder consume the whole thing -- `path: adityabanerjee13/indic-cpt-mini-
train`, `type: completion`, `field: text`. This script keeps only lang == "hi"
and writes it out, for single-language ablations where the other nine languages
are the confound you are trying to remove.

Output is JSONL with the source columns unchanged (`text`, `lang`), so it drops
straight into a config as a local dataset:

    datasets:
      - path: cpt_training/data/indic_cpt_mini_train_hi.jsonl
        ds_type: json
        type: completion
        field: text
        split: train

Note *.jsonl is gitignored at the repo root, so the extracted data stays out of
version control -- the Hub repo remains the source of truth and this script the
reproducible path back to it.

Usage:
    python download_hindi_cpt_data.py
    python download_hindi_cpt_data.py --dataset adityabanerjee13/indic-cpt-mini-val
    python download_hindi_cpt_data.py --lang ta --output data/ta.jsonl
"""

import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = "adityabanerjee13/indic-cpt-mini-train"
DEFAULT_LANG = "hi"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF dataset repo id.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--lang", default=DEFAULT_LANG, help="Value of the `lang` column to keep.")
    parser.add_argument("--output", default=None,
                        help="Output .jsonl path (default: data/<dataset>_<lang>.jsonl next to this script).")
    args = parser.parse_args()

    if args.output is None:
        stem = args.dataset.split("/")[-1].replace("-", "_")
        args.output = os.path.join(SCRIPT_DIR, "data", f"{stem}_{args.lang}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    from datasets import load_dataset

    print(f"Loading {args.dataset} [{args.split}] ...")
    dataset = load_dataset(args.dataset, split=args.split)

    if "lang" not in dataset.column_names:
        sys.exit(f"No `lang` column in {args.dataset}; columns are {dataset.column_names}")

    before = len(dataset)
    subset = dataset.filter(lambda row: row["lang"] == args.lang)
    if len(subset) == 0:
        present = sorted(set(dataset["lang"]))
        sys.exit(f"No rows with lang == {args.lang!r}. Present: {present}")

    with open(args.output, "w", encoding="utf-8") as fh:
        for row in subset:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    chars = sum(len(text) for text in subset["text"])
    print(f"  kept {len(subset):,} of {before:,} rows (lang == {args.lang!r}), {chars:,} characters")
    print(f"Wrote {args.output} ({os.path.getsize(args.output):,} bytes)")


if __name__ == "__main__":
    main()
