"""
STEP 1 (SFT-source prep for DPO): download the SFT train/val datasets from HF and
create sampled subsets to later turn into (prompt, chosen, rejected) triplets.

Downloads (private, requires an HF token with read access):
    adityabanerjee13/indic-sft-mini-{train,val}
    adityabanerjee13/tulu-sft-mini-{train,val}

Saves into dpo_training/data/sft_source/ :
    TRAIN:
      indic_sft_train_full.jsonl   full download (kept locally)
      tulu_sft_train_full.jsonl    full download (kept locally)
      indic_sampled.jsonl          `--indic-per-lang` per language
      tulu_sampled.jsonl           `--tulu-frac` * (Indic sampled rows)
    VAL:
      indic_sft_val_full.jsonl     full download (kept locally)
      tulu_sft_val_full.jsonl      full download (kept locally)
      indic_val_sampled.jsonl      `--indic-val-per-lang` per language
      tulu_val_sampled.jsonl       `--tulu-frac` * (Indic val sampled rows)

Ratio: Tulu is sized to 60% of the sampled Indic rows in each split, matching the
SFT mix from data/cap_and_rebalance.py. The val per-language default (70) mirrors
the train sampling fraction (625/1725 ~= 36%) applied to the 192/lang val set, so
the DPO val set stays proportional to the DPO train set.

Each record keeps the SFT chat shape: {language?, source, task?, messages:[...]}.
STEP 2 (generate_all_rejected.py) consumes the *sampled* files.

Requirements: huggingface_hub
Usage:
    python download_and_sample_sft.py                        # train+val at defaults
    python download_and_sample_sft.py --splits val           # only the val subsets
    python download_and_sample_sft.py --indic-per-lang 1725 --indic-val-per-lang 192
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

# split -> {who: (repo, filename)}
SOURCES = {
    "train": {
        "indic": ("adityabanerjee13/indic-sft-mini-train", "indic-sft-mini-train.jsonl"),
        "tulu":  ("adityabanerjee13/tulu-sft-mini-train",  "tulu-sft-mini-train.jsonl"),
    },
    "val": {
        "indic": ("adityabanerjee13/indic-sft-mini-val", "indic-sft-mini-val.jsonl"),
        "tulu":  ("adityabanerjee13/tulu-sft-mini-val",  "tulu-sft-mini-val.jsonl"),
    },
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


def sampled_name(split, who):
    # keep the original train names (indic_sampled.jsonl / tulu_sampled.jsonl)
    return f"{who}_sampled.jsonl" if split == "train" else f"{who}_val_sampled.jsonl"


def sample_split(split, indic_per_lang, tulu_frac, rng):
    """Download the split's indic+tulu sets, write full copies + sampled subsets."""
    print(f"\n########## SPLIT: {split} ##########")
    local_full = {}
    for who, (repo, fname) in SOURCES[split].items():
        print(f"[download] {repo}/{fname} ...")
        cached = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset")
        rows = load_jsonl(cached)
        full_path = os.path.join(OUT_DIR, f"{who}_sft_{split}_full.jsonl")
        write_jsonl(rows, full_path)
        local_full[who] = rows
        print(f"  {who}: {len(rows)} records -> {os.path.relpath(full_path, OUT_DIR)}")

    # ---- Indic: `indic_per_lang` per language ----
    by_lang = defaultdict(list)
    for r in local_full["indic"]:
        by_lang[r.get("language", "?")].append(r)
    indic_sample = []
    print(f"\n[sample] indic ({split}): {indic_per_lang} per language")
    for lang in sorted(by_lang):
        pool = by_lang[lang]
        take = min(indic_per_lang, len(pool))
        if take < indic_per_lang:
            print(f"  WARNING: {lang} has only {len(pool)} (< {indic_per_lang})")
        indic_sample.extend(rng.sample(pool, take))
        print(f"  {lang}: {take}")
    rng.shuffle(indic_sample)
    write_jsonl(indic_sample, os.path.join(OUT_DIR, sampled_name(split, "indic")))

    # ---- Tulu: `tulu_frac` of the sampled Indic rows (default 60%) ----
    tulu_rows = local_full["tulu"]
    tulu_target = round(tulu_frac * len(indic_sample))
    take = min(tulu_target, len(tulu_rows))
    if take < tulu_target:
        print(f"\nWARNING: tulu ({split}) has only {len(tulu_rows)} (< target {tulu_target})")
    tulu_sample = rng.sample(tulu_rows, take)
    rng.shuffle(tulu_sample)
    write_jsonl(tulu_sample, os.path.join(OUT_DIR, sampled_name(split, "tulu")))

    return {
        "split": split,
        "indic_full": len(local_full["indic"]),
        "tulu_full": len(tulu_rows),
        "indic_sampled": len(indic_sample),
        "indic_per_language": dict(sorted(Counter(r.get("language", "?") for r in indic_sample).items())),
        "tulu_sampled": len(tulu_sample),
        "files": [sampled_name(split, "indic"), sampled_name(split, "tulu")],
    }


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", choices=["train", "val"], default=["train", "val"],
                    help="Which splits to sample (default: both).")
    ap.add_argument("--indic-per-lang", type=int, default=625,
                    help="TRAIN Indic samples per language (default 625 -> 6,250).")
    ap.add_argument("--indic-val-per-lang", type=int, default=70,
                    help="VAL Indic samples per language (default 70 -> 700).")
    ap.add_argument("--tulu-frac", type=float, default=0.60,
                    help="Tulu rows as a fraction of sampled Indic rows in each split.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    reports = []
    for split in args.splits:
        per_lang = args.indic_per_lang if split == "train" else args.indic_val_per_lang
        reports.append(sample_split(split, per_lang, args.tulu_frac, rng))

    print("\n=== SUMMARY ===")
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print(f"\noutput_dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
