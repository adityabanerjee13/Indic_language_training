"""
STEP 1 (SFT-source prep for DPO): download the SFT-train datasets from HF and
create sampled subsets to later turn into (prompt, chosen, rejected) triplets.

Downloads (private, requires an HF token with read access):
    adityabanerjee13/indic-sft-mini-train
    adityabanerjee13/tulu-sft-mini-train

Saves into dpo_training/data/sft_source/ :
    indic_sft_train_full.jsonl        full download (kept locally)
    tulu_sft_train_full.jsonl         full download (kept locally)
    indic_sampled_4k_per_lang.jsonl   4,000 per language  -> 40,000 records
    tulu_sampled_20k.jsonl            20,000 records

Each record keeps the SFT chat shape: {language?, source, task?, messages:[...]}.
STEP 2 (generate_rejected_sft.py) consumes the two *sampled* files.

Requirements: huggingface_hub
Usage:
    python download_and_sample_sft.py
    python download_and_sample_sft.py --indic-per-lang 4000 --tulu-n 20000
"""

import argparse
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict

from huggingface_hub import hf_hub_download

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sft_source")
SEED = 42

SOURCES = {
    "indic": ("adityabanerjee13/indic-sft-mini-train", "indic-sft-mini-train.jsonl"),
    "tulu":  ("adityabanerjee13/tulu-sft-mini-train",  "tulu-sft-mini-train.jsonl"),
}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indic-per-lang", type=int, default=4000, help="Samples per language for Indic.")
    ap.add_argument("--tulu-n", type=int, default=20000, help="Total samples for Tulu.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    # ---- download both full train sets locally ----
    local_full = {}
    for name, (repo, fname) in SOURCES.items():
        print(f"[download] {repo}/{fname} ...")
        cached = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
        rows = load_jsonl(cached)
        full_path = os.path.join(OUT_DIR, f"{name}_sft_train_full.jsonl")
        write_jsonl(rows, full_path)
        local_full[name] = rows
        print(f"  {name}: {len(rows)} records -> {os.path.relpath(full_path, OUT_DIR)}")

    # ---- Indic: 4k per language ----
    indic_rows = local_full["indic"]
    by_lang = defaultdict(list)
    for r in indic_rows:
        by_lang[r.get("language", "?")].append(r)
    indic_sample = []
    print("\n[sample] indic: {} per language".format(args.indic_per_lang))
    for lang in sorted(by_lang):
        pool = by_lang[lang]
        take = min(args.indic_per_lang, len(pool))
        if take < args.indic_per_lang:
            print(f"  WARNING: {lang} has only {len(pool)} (< {args.indic_per_lang})")
        picked = rng.sample(pool, take)
        indic_sample.extend(picked)
        print(f"  {lang}: {take}")
    rng.shuffle(indic_sample)
    indic_out = os.path.join(OUT_DIR, "indic_sampled_4k_per_lang.jsonl")
    write_jsonl(indic_sample, indic_out)

    # ---- Tulu: 20k total ----
    tulu_rows = local_full["tulu"]
    take = min(args.tulu_n, len(tulu_rows))
    if take < args.tulu_n:
        print(f"\nWARNING: tulu has only {len(tulu_rows)} (< {args.tulu_n})")
    tulu_sample = rng.sample(tulu_rows, take)
    rng.shuffle(tulu_sample)
    tulu_out = os.path.join(OUT_DIR, "tulu_sampled_20k.jsonl")
    write_jsonl(tulu_sample, tulu_out)

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "indic_full": len(indic_rows),
        "tulu_full": len(tulu_rows),
        "indic_sampled": len(indic_sample),
        "indic_per_language": dict(sorted(Counter(r.get("language", "?") for r in indic_sample).items())),
        "tulu_sampled": len(tulu_sample),
        "output_dir": OUT_DIR,
    }, ensure_ascii=False, indent=2))
    print("\nSaved 4 files: 2 full downloads + 2 sampled subsets.")
    print("Next: python generate_rejected_sft.py --input data/sft_source/indic_sampled_4k_per_lang.jsonl ...")


if __name__ == "__main__":
    main()
