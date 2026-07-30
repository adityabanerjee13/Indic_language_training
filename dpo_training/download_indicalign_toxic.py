"""
STEP 1 of the DPO triplet pipeline.

Download IndicAlign-Toxic (HHRLHF-T + Toxic-Matrix) for the 10 target Indic
languages and extract (prompt, chosen) pairs, where:
    prompt  = the harmful/toxic user prompt
    chosen  = the human-aligned REFUSAL response (the good answer)

The `rejected` side is added later by generate_rejected.py, which samples a
response from a base model. Together they form (prompt, chosen, rejected)
DPO triplets that teach the model to prefer refusals over unsafe completions.

Both toxic sub-datasets are "Layout A" (per-language columns; each cell is an
array of [user, assistant] turns). We stream only the 10 native-script columns
via HfFileSystem range reads -> no full-file downloads.

Output: data/indicalign_toxic_pairs.jsonl
    {"language","source","prompt":[msg...],"chosen":[msg...]}

Requirements: pip install huggingface_hub pyarrow pandas
Usage:
    python download_indicalign_toxic.py                     # default 500 rows/source
    python download_indicalign_toxic.py --rows-per-source 1000
"""

import argparse
import io
import json
import os
import sys

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

REPO = "datasets/ai4bharat/indic-align"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_FILE = os.path.join(OUT_DIR, "indicalign_toxic_pairs.jsonl")

LANG_COLS = {
    "bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda",
    "ml": "mal_Mlym", "mr": "mar_Deva", "or": "ory_Orya", "pa": "pan_Guru",
    "ta": "tam_Taml", "te": "tel_Telu",
}

# (relative path, source_label)
SUBSETS = [
    ("indicalign-toxic/hhrlhf/hh-rlhf.parquet", "hhrlhf_t"),
    ("indicalign-toxic/toxicmatrix/toxic_prompts_sarvam.parquet", "toxic_matrix"),
]


def to_messages(cell):
    """Normalize an IndicAlign cell (array of turns) into a chat messages list."""
    if cell is None:
        return None
    try:
        turns = list(cell)
    except TypeError:
        return None
    if len(turns) == 0:
        return None
    msgs = []
    is_nested = all((not isinstance(t, str)) and hasattr(t, "__len__") and len(t) == 2 for t in turns)
    if is_nested:
        for t in turns:
            u, a = str(t[0]).strip(), str(t[1]).strip()
            if u:
                msgs.append({"role": "user", "content": u})
            if a:
                msgs.append({"role": "assistant", "content": a})
    else:
        flat = []
        for t in turns:
            if isinstance(t, str) or not hasattr(t, "__len__"):
                flat.append(str(t).strip())
            else:
                flat.extend(str(x).strip() for x in t)
        for i, txt in enumerate(flat):
            if txt:
                msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": txt})
    if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[-1]["role"] != "assistant":
        return None
    return msgs


def split_prompt_chosen(msgs):
    """Split a full exchange into (prompt_messages, chosen_messages).
    prompt = everything up to and including the last user turn;
    chosen = the final assistant turn."""
    if msgs[-1]["role"] != "assistant":
        return None
    return msgs[:-1], [msgs[-1]]


def collect_pairs(fs, rel, source, rows_per_source, cols):
    pairs, scanned = [], 0
    with fs.open(f"{REPO}/{rel}") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=512, columns=cols):
            df = batch.to_pandas()
            for ridx in range(len(df)):
                scanned += 1
                row = df.iloc[ridx]
                for short, col in LANG_COLS.items():
                    if col not in df.columns:
                        continue
                    msgs = to_messages(row[col])
                    if msgs is None:
                        continue
                    split = split_prompt_chosen(msgs)
                    if split is None:
                        continue
                    prompt_msgs, chosen_msgs = split
                    pairs.append({
                        "language": short,
                        "source": f"indicalign/{source}",
                        "prompt": prompt_msgs,
                        "chosen": chosen_msgs,
                    })
                if scanned >= rows_per_source:
                    break
            if scanned >= rows_per_source:
                break
    return pairs, scanned


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows-per-source", type=int, default=500,
                    help="Source rows to read per sub-dataset (each row -> up to 10 pairs).")
    ap.add_argument("--output", default=OUT_FILE)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    fs = HfFileSystem()
    cols = list(LANG_COLS.values())

    from collections import Counter
    by_lang, by_source = Counter(), Counter()
    all_pairs = []

    for rel, source in SUBSETS:
        print(f"[fetch] {source}: reading up to {args.rows_per_source} rows from {rel.split('/')[-1]} ...")
        pairs, scanned = collect_pairs(fs, rel, source, args.rows_per_source, cols)
        for p in pairs:
            by_lang[p["language"]] += 1
            by_source[source] += 1
        all_pairs.extend(pairs)
        print(f"  {source}: +{len(pairs)} pairs (scanned {scanned} rows) -> total {len(all_pairs)}")

    with open(args.output, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print("\n=== SUMMARY ===")
    print(json.dumps({
        "total_pairs": len(all_pairs),
        "by_language": dict(sorted(by_lang.items())),
        "by_source": dict(by_source),
    }, ensure_ascii=False, indent=2))
    print(f"\nWrote pairs -> {args.output}")
    print("Next: python generate_rejected.py --model <HF_MODEL_PATH> --input", os.path.basename(args.output))


if __name__ == "__main__":
    main()
