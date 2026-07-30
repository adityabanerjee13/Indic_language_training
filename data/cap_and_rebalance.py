"""
Cap the Indic SFT train split to <= 50M tokens (Qwen2.5-0.5B tokenizer) using
EQUAL RECORDS PER LANGUAGE, rebuild a matching val split (preserving the ~90/10
train:val ratio), and size Tulu to 60% of the Indic row counts in each split.
Optionally overwrite the four HF dataset repos.

Token counting: render chat template -> tokenize (exact, same method throughout).

Sources:
    Indic train  : dpo_training/data/sft_source/indic_sft_train_full.jsonl (local, 90k)
    Indic val    : hf adityabanerjee13/indic-sft-mini-val
    Tulu  train  : dpo_training/data/sft_source/tulu_sft_train_full.jsonl (local, 72k)
    Tulu  val    : hf adityabanerjee13/tulu-sft-mini-val

Outputs (data/capped/) and (with upload) overwrite:
    indic-sft-mini-train / -val , tulu-sft-mini-train / -val

Usage:
    python cap_and_rebalance.py --no-upload      # build + report only
    python cap_and_rebalance.py                  # build + overwrite the 4 HF repos
"""

import argparse
import io
import json
import os
import random
import sys
from bisect import bisect_right
from collections import defaultdict, Counter

from huggingface_hub import hf_hub_download

TOKENIZER = "Qwen/Qwen2.5-0.5B"
HF_USER = "adityabanerjee13"
HERE = os.path.dirname(os.path.abspath(__file__))
DPO_SRC = os.path.join(HERE, "..", "dpo_training", "data", "sft_source")
OUT_DIR = os.path.join(HERE, "capped")
SEED = 42

INDIC_TRAIN_LOCAL = os.path.join(DPO_SRC, "indic_sft_train_full.jsonl")
TULU_TRAIN_LOCAL = os.path.join(DPO_SRC, "tulu_sft_train_full.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-tokens", type=int, default=50_000_000)
    ap.add_argument("--val-ratio", type=float, default=1/9, help="val_rows = train_rows * ratio (1/9 keeps 90/10).")
    ap.add_argument("--tulu-frac", type=float, default=0.60)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def ntok(messages):
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return len(tok(text, add_special_tokens=False)["input_ids"])

    # ---------- load sources ----------
    print("[load] indic train (local), tulu train (local); downloading val splits ...")
    indic_train = load_jsonl(INDIC_TRAIN_LOCAL)
    tulu_train = load_jsonl(TULU_TRAIN_LOCAL)
    indic_val_all = load_jsonl(hf_hub_download(f"{HF_USER}/indic-sft-mini-val", "indic-sft-mini-val.jsonl", repo_type="dataset"))
    tulu_val_all = load_jsonl(hf_hub_download(f"{HF_USER}/tulu-sft-mini-val", "tulu-sft-mini-val.jsonl", repo_type="dataset"))

    # ---------- Indic TRAIN: equal records/lang, <= target tokens ----------
    print("[tokenize] indic train (per-record) ...")
    by_lang = defaultdict(list)
    for r in indic_train:
        by_lang[r.get("language", "?")].append(r)
    langs = sorted(by_lang)
    # shuffle each language, compute token prefix sums
    prefix = {}
    for lang in langs:
        rng.shuffle(by_lang[lang])
        toks = [ntok(r["messages"]) for r in by_lang[lang]]
        ps = [0]
        for t in toks:
            ps.append(ps[-1] + t)
        prefix[lang] = ps

    min_len = min(len(prefix[l]) - 1 for l in langs)

    def total_at(R):
        return sum(prefix[l][min(R, len(prefix[l]) - 1)] for l in langs)

    # largest R with total <= target
    lo, hi, R = 1, min_len, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if total_at(mid) <= args.target_tokens:
            R = mid
            lo = mid + 1
        else:
            hi = mid - 1
    train_total_tokens = total_at(R)
    indic_train_cap = []
    train_lang_tok = {}
    for lang in langs:
        picked = by_lang[lang][:R]
        indic_train_cap.extend(picked)
        train_lang_tok[lang] = prefix[lang][R]
    rng.shuffle(indic_train_cap)
    print(f"[indic train] R={R}/lang -> {len(indic_train_cap)} rows, {train_total_tokens:,} tokens (cap {args.target_tokens:,})")

    # ---------- Indic VAL: equal records/lang at 90/10 ratio ----------
    val_R = max(1, round(R * args.val_ratio))
    by_lang_val = defaultdict(list)
    for r in indic_val_all:
        by_lang_val[r.get("language", "?")].append(r)
    indic_val_cap = []
    for lang in langs:
        pool = by_lang_val.get(lang, [])
        take = min(val_R, len(pool))
        indic_val_cap.extend(rng.sample(pool, take))
    rng.shuffle(indic_val_cap)
    print(f"[indic val] {val_R}/lang -> {len(indic_val_cap)} rows")

    # ---------- Tulu: 60% of indic row counts per split ----------
    tulu_train_n = round(args.tulu_frac * len(indic_train_cap))
    tulu_val_n = round(args.tulu_frac * len(indic_val_cap))
    tulu_train_cap = rng.sample(tulu_train, min(tulu_train_n, len(tulu_train)))
    tulu_val_cap = rng.sample(tulu_val_all, min(tulu_val_n, len(tulu_val_all)))
    print(f"[tulu] train {len(tulu_train_cap)} (target {tulu_train_n}), val {len(tulu_val_cap)} (target {tulu_val_n})")

    # ---------- write locally ----------
    outputs = {
        "indic-sft-mini-train": indic_train_cap,
        "indic-sft-mini-val": indic_val_cap,
        "tulu-sft-mini-train": tulu_train_cap,
        "tulu-sft-mini-val": tulu_val_cap,
    }
    for name, rows in outputs.items():
        write_jsonl(rows, os.path.join(OUT_DIR, f"{name}.jsonl"))

    report = {
        "tokenizer": TOKENIZER,
        "indic_train": {"rows": len(indic_train_cap), "records_per_lang": R,
                        "total_tokens": train_total_tokens,
                        "tokens_per_lang": train_lang_tok},
        "indic_val": {"rows": len(indic_val_cap), "records_per_lang": val_R},
        "tulu_train": {"rows": len(tulu_train_cap)},
        "tulu_val": {"rows": len(tulu_val_cap)},
    }
    with open(os.path.join(OUT_DIR, "cap_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== REPORT ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # ---------- upload (overwrite) ----------
    if not args.no_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        for name, rows in outputs.items():
            repo = f"{HF_USER}/{name}"
            local = os.path.join(OUT_DIR, f"{name}.jsonl")
            print(f"[upload] overwriting {repo} ({len(rows)} rows) ...")
            api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=local, path_in_repo=f"{name}.jsonl",
                            repo_id=repo, repo_type="dataset")
        print("[upload] done — 4 repos updated.")
    else:
        print("\n(--no-upload) Local build only; HF repos untouched.")


if __name__ == "__main__":
    main()
