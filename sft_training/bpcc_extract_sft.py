"""
Extract translation SFT data from ai4bharat/BPCC for the 10 CPT target
languages ONLY, and emit ready-to-train instruction/response JSONL.

BPCC is a machine-translation parallel corpus: every row is an (English, Indic)
bitext pair. This script downloads only the requested per-language TSV files
from the chosen subsets (so no other language is ever touched), then wraps each
pair into a translation instruction example — in both directions, with rotating
instruction templates so the model learns the task rather than one phrasing.

See BPCC_SFT_Assessment.md for full dataset analysis.

Requirements:
    pip install huggingface_hub
    A Hugging Face token with gated-repo read access (BPCC is gated):
      huggingface-cli login   (or set HF_TOKEN)

Usage examples:
    # Human-quality subsets, both directions, all 10 languages:
    python bpcc_extract_sft.py --tier human

    # Just the best single subset, forward direction only:
    python bpcc_extract_sft.py --subsets bpcc-seed-latest --directions en2indic

    # A subset of languages:
    python bpcc_extract_sft.py --tier human --langs hi ta bn
"""

import argparse
import csv
import json
import os
import random
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/BPCC"

# Your 10 CPT target languages ONLY. short_code -> (BPCC file code, English name).
# No other BPCC language is ever downloaded or emitted.
TARGET_LANGS = {
    "bn": ("ben_Beng", "Bengali"),
    "gu": ("guj_Gujr", "Gujarati"),
    "hi": ("hin_Deva", "Hindi"),
    "kn": ("kan_Knda", "Kannada"),
    "ml": ("mal_Mlym", "Malayalam"),
    "mr": ("mar_Deva", "Marathi"),
    "or": ("ory_Orya", "Odia"),
    "pa": ("pan_Guru", "Punjabi"),
    "ta": ("tam_Taml", "Tamil"),
    "te": ("tel_Telu", "Telugu"),
}

# Which subsets exist per tier. (The script skips any language file that a
# given subset does not provide — e.g. MASSIVE only has 6 of the 10.)
TIER_SUBSETS = {
    "human": ["bpcc-seed-latest", "wiki", "daily", "massive", "ilci"],
    "seed": ["bpcc-seed-latest"],
    "mined": ["samanantar_v0.3_filtered", "nllb_filtered", "comparable"],
    "all": ["bpcc-seed-latest", "wiki", "daily", "massive", "ilci",
            "samanantar_v0.3_filtered", "nllb_filtered", "comparable"],
}

# Languages actually present in each subset (from the repo file tree). Files not
# listed here are silently skipped so no request 404s.
SUBSET_LANG_AVAILABILITY = {
    "massive": {"bn", "hi", "kn", "ml", "ta", "te"},          # missing gu, mr, or, pa
    "ilci":    {"bn", "gu", "hi", "kn", "ml", "mr", "pa", "ta", "te"},  # missing or
    # all others: assume all 10 present
}

# Rotating instruction templates. {L} is filled with the English language name.
TEMPLATES_EN2INDIC = [
    "Translate the following English text into {L}.",
    "Convert this English sentence to {L}.",
    "Render the following into {L}:",
    "What is the following English sentence in {L}?",
    "Please provide the {L} translation of the text below.",
]
TEMPLATES_INDIC2EN = [
    "Translate the following {L} text into English.",
    "Convert this {L} sentence to English.",
    "Render the following {L} text into English:",
    "What does the following {L} sentence mean in English?",
    "Please provide the English translation of the {L} text below.",
]


def read_bitext_rows(path):
    """Yield (src_english, tgt_indic) from a BPCC tsv, keyed by header name so
    the differing column ORDER across subsets is handled correctly."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src = (row.get("src") or "").strip()
            tgt = (row.get("tgt") or "").strip()
            if src and tgt:
                yield src, tgt


def make_example(instruction, input_text, output_text):
    return {"instruction": instruction, "input": input_text, "output": output_text}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=list(TIER_SUBSETS), default="human",
                        help="Preset group of subsets (default: human).")
    parser.add_argument("--subsets", nargs="*", default=None,
                        help="Explicit subset list; overrides --tier.")
    parser.add_argument("--langs", nargs="*", default=list(TARGET_LANGS),
                        help="Subset of the 10 target langs (default: all).")
    parser.add_argument("--directions", choices=["both", "en2indic", "indic2en"], default="both")
    parser.add_argument("--max-per-lang-subset", type=int, default=None,
                        help="Cap pairs read per (language, subset) — useful to keep SFT size sane.")
    parser.add_argument("--output-dir", default="bpcc_sft_out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    subsets = args.subsets or TIER_SUBSETS[args.tier]
    langs = [l for l in args.langs if l in TARGET_LANGS]
    if not langs:
        sys.exit("No valid target languages selected.")

    os.makedirs(args.output_dir, exist_ok=True)
    counts = {}

    for short in langs:
        file_code, lang_name = TARGET_LANGS[short]
        out_path = os.path.join(args.output_dir, f"{short}_{lang_name}_sft.jsonl")
        n_written = 0

        with open(out_path, "w", encoding="utf-8") as out_f:
            for subset in subsets:
                avail = SUBSET_LANG_AVAILABILITY.get(subset)
                if avail is not None and short not in avail:
                    continue  # this subset doesn't cover this language

                rel = f"{subset}/{file_code}.tsv"
                try:
                    local = hf_hub_download(REPO_ID, rel, repo_type="dataset",
                                            local_dir=args.output_dir + "/_raw")
                except Exception as e:
                    print(f"  skip {rel}: {e!r}")
                    continue

                read = 0
                for src, tgt in read_bitext_rows(local):
                    if args.max_per_lang_subset and read >= args.max_per_lang_subset:
                        break
                    read += 1

                    if args.directions in ("both", "en2indic"):
                        instr = random.choice(TEMPLATES_EN2INDIC).format(L=lang_name)
                        out_f.write(json.dumps(make_example(instr, src, tgt), ensure_ascii=False) + "\n")
                        n_written += 1
                    if args.directions in ("both", "indic2en"):
                        instr = random.choice(TEMPLATES_INDIC2EN).format(L=lang_name)
                        out_f.write(json.dumps(make_example(instr, tgt, src), ensure_ascii=False) + "\n")
                        n_written += 1

                print(f"  {short} <- {subset}: read {read} pairs")

        counts[short] = n_written
        print(f"[{short}/{lang_name}] wrote {n_written} SFT examples -> {out_path}\n")

    print("=" * 60)
    print("DONE. SFT examples per language:")
    for short in langs:
        print(f"  {short}: {counts.get(short, 0)}")
    print(f"Total: {sum(counts.values())}")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
