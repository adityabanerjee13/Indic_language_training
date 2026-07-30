"""
Refactor the SFT datasets to target sizes, split each into train(0.9)/val(0.1),
and upload all splits to the Hugging Face Hub as {name}-sft-mini-{train,val}.

Targets:
    indic -> 100,000 examples   (source: SFT/indicalign_sft_100k.jsonl)
    tulu  ->  80,000 examples   (source: tulu_sft/tulu_half.jsonl, subsampled)

Splitting: shuffle (fixed seed), take 10% as val, 90% as train.
Uploading: one HF dataset repo per split, private:
    adityabanerjee13/indic-sft-mini-train  (90,000)
    adityabanerjee13/indic-sft-mini-val    (10,000)
    adityabanerjee13/tulu-sft-mini-train   (72,000)
    adityabanerjee13/tulu-sft-mini-val     ( 8,000)

Requirements: huggingface_hub (logged in with write token).
Usage:
    python refactor_split_upload.py                 # cap + split + upload all
    python refactor_split_upload.py --no-upload     # just write local split files
"""

import argparse
import io
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HF_USER = "adityabanerjee13"
VAL_FRAC = 0.1
SEED = 42

# name -> (source jsonl path, target example count)
DATASETS = {
    "indic": (os.path.join(HERE, "SFT", "indicalign_sft_100k.jsonl"), 100_000),
    "tulu":  (os.path.join(HERE, "tulu_sft", "tulu_half.jsonl"),        80_000),
}

OUT_DIR = os.path.join(HERE, "sft_splits")


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


def make_card(name, split, n):
    return (
        f"---\nlicense: cc-by-4.0\npretty_name: {name}-sft-mini-{split}\n"
        f"size_categories: [10K<n<100K]\ntags: [sft, {name}]\n---\n\n"
        f"# {name}-sft-mini-{split}\n\n"
        f"{split.capitalize()} split ({n} examples) of the refactored **{name}** SFT dataset "
        f"(90/10 train/val). Chat-format `messages` records.\n"
    )


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-upload", action="store_true", help="Only write local split files, skip HF upload.")
    ap.add_argument("--private", action="store_true", default=True)
    args = ap.parse_args()

    rng = random.Random(SEED)
    api = None
    if not args.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()

    summary = {}
    for name, (path, target) in DATASETS.items():
        print(f"\n=== {name} ===")
        if not os.path.exists(path):
            sys.exit(f"ERROR: source not found: {path}\n"
                     f"(For indic, build it first: python SFT/build_sft_indicalign.py "
                     f"--target-records 100000 --output SFT/indicalign_sft_100k.jsonl)")
        rows = load_jsonl(path)
        print(f"loaded {len(rows)} records from {os.path.relpath(path, HERE)}")
        if len(rows) < target:
            sys.exit(f"ERROR: {name} has {len(rows)} < target {target}; cannot upsample.")

        rng.shuffle(rows)
        rows = rows[:target]                       # cap (subsample) to exact target
        n_val = round(target * VAL_FRAC)
        val, train = rows[:n_val], rows[n_val:]
        print(f"refactored to {target} -> train {len(train)} / val {len(val)}")

        for split, data in (("train", train), ("val", val)):
            local = os.path.join(OUT_DIR, f"{name}-sft-mini-{split}.jsonl")
            write_jsonl(data, local)
            print(f"  wrote {local} ({len(data)})")
            if api is not None:
                repo = f"{HF_USER}/{name}-sft-mini-{split}"
                print(f"  uploading -> {repo} (private={args.private}) ...")
                api.create_repo(repo_id=repo, repo_type="dataset", private=args.private, exist_ok=True)
                api.upload_file(path_or_fileobj=local, path_in_repo=f"{name}-sft-mini-{split}.jsonl",
                                repo_id=repo, repo_type="dataset")
                card = os.path.join(OUT_DIR, f"_README_{name}_{split}.md")
                with open(card, "w", encoding="utf-8") as cf:
                    cf.write(make_card(name, split, len(data)))
                api.upload_file(path_or_fileobj=card, path_in_repo="README.md",
                                repo_id=repo, repo_type="dataset")
                print(f"    done: https://huggingface.co/datasets/{repo}")

        summary[name] = {"target": target, "train": len(train), "val": len(val)}

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
