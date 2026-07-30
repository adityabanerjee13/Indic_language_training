"""
Build an 80K-example SFT dataset (8,000 per language) from ai4bharat/indic-align
(IndicAlign), covering the 10 target Indic languages, and write it into this folder.
Safety sources (HHRLHF-T, Toxic-Matrix) are excluded.

Design (see ../../CURATION_PLAN_SFT_DPO.md and ../../IndicAlign_SFT_Assessment.md):
- Uses the 8 "Layout A" sub-datasets (per-language columns, n-way parallel).
  Each source ROW yields one record per target language, so N rows -> N*10 records.
- Excludes IndoWordNet (Layout B, grouped-by-language, dictionary-terse; per the
  assessment it should be capped/optional, and balanced extraction would require
  scanning GBs). Excludes Anudesh (Layout B, English-heavy).
- Streams only the needed columns from each parquet via HfFileSystem + pyarrow
  range reads, taking only the first N rows per source -> no full-file downloads.

Total: 15,000 source rows x 10 languages = 150,000 SFT records, balanced 15K/language.

Requirements: pip install huggingface_hub pyarrow pandas
Usage: python build_sft_indicalign.py
"""

import io
import json
import os
import random
import sys

from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq

REPO = "datasets/ai4bharat/indic-align"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(OUT_DIR, "indicalign_sft_80k.jsonl")
STATS_FILE = os.path.join(OUT_DIR, "stats.json")
SEED = 42

# short code -> native-script column name in the parquet
LANG_COLS = {
    "bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva", "kn": "kan_Knda",
    "ml": "mal_Mlym", "mr": "mar_Deva", "or": "ory_Orya", "pa": "pan_Guru",
    "ta": "tam_Taml", "te": "tel_Telu",
}

# (relative path, rows_to_take, source_label, task_label)
# rows sum to 8,000 -> x10 languages = 80,000 records.
# Safety sources (HHRLHF-T, Toxic-Matrix) intentionally excluded.
SUBSETS = [
    ("indicalign-instruct/indicsharellama/indic_sharellama.parquet", 1500, "indic_sharellama", "instruction"),
    ("indicalign-instruct/dolly/Dolly.parquet",                      1500, "dolly_t",          "instruction"),
    ("indicalign-instruct/oasst/oasst.parquet",                      1250, "openassistant_t",  "chat"),
    ("indicalign-instruct/wikihow/wiki_how.parquet",                 1250, "wikihow",          "howto"),
    ("indicalign-instruct/wiki_conv/wiki_conv.parquet",              1500, "wiki_conv",        "chat"),
    ("indicalign-instruct/wiki_chat/wiki_chat_0000_of_0031.parquet", 1000, "wiki_chat",        "chat"),
]


def to_messages(cell):
    """Normalize an IndicAlign cell (array of turns) into a chat messages list.
    Handles both layouts: array of [user, assistant] pairs, and flat alternating
    lists of strings. Returns None if it can't form a valid user-first exchange."""
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


def collect_valid_units(fs, rel, target, cols, max_rows_factor=6):
    """Stream a parquet file and collect `target` fully-valid row units.

    A "unit" is a source row for which ALL 10 target-language cells produce a
    valid messages list. Requiring all-10 guarantees perfect per-language balance
    and lets us hit the exact target despite structurally-invalid rows. Reads
    until `target` valid units are found (or the file/cap is exhausted)."""
    units, scanned = [], 0
    cap = target * max_rows_factor
    with fs.open(f"{REPO}/{rel}") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1024, columns=cols):
            df = batch.to_pandas()
            for ridx in range(len(df)):
                scanned += 1
                row = df.iloc[ridx]
                per_lang = {}
                ok = True
                for short, col in LANG_COLS.items():
                    msgs = to_messages(row[col]) if col in df.columns else None
                    if msgs is None:
                        ok = False
                        break
                    per_lang[short] = msgs
                if ok:
                    nt = row["num_turns"] if "num_turns" in df.columns else None
                    nt = int(nt) if nt == nt and nt is not None else None
                    units.append((per_lang, nt))
                if len(units) >= target or scanned >= cap:
                    break
            if len(units) >= target or scanned >= cap:
                break
    return units, scanned


def scale_subsets(subsets, target_records):
    """Scale the per-source row counts proportionally so that
    sum(rows) * 10 languages == target_records exactly."""
    if target_records % 10 != 0:
        raise ValueError("target_records must be divisible by 10 (10 languages).")
    total_rows = target_records // 10
    base_total = sum(r for _, r, _, _ in subsets)
    scaled = []
    for rel, rows, source, task in subsets:
        scaled.append([rel, max(1, round(rows * total_rows / base_total)), source, task])
    # fix rounding drift by adjusting the largest source
    drift = total_rows - sum(s[1] for s in scaled)
    if drift != 0:
        i = max(range(len(scaled)), key=lambda k: scaled[k][1])
        scaled[i][1] += drift
    return [tuple(s) for s in scaled]


def main():
    import argparse
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-records", type=int, default=80000,
                    help="Total records to build (must be divisible by 10). Default 80000.")
    ap.add_argument("--output", default=OUT_FILE, help="Output JSONL path.")
    args = ap.parse_args()

    random.seed(SEED)
    fs = HfFileSystem()
    needed_cols = ["num_turns"] + list(LANG_COLS.values())
    subsets = scale_subsets(SUBSETS, args.target_records)
    out_file = args.output
    print(f"[config] target_records={args.target_records} -> "
          f"{args.target_records // 10} rows/source-total; output={out_file}")

    records = []
    from collections import Counter
    by_lang, by_source, by_task = Counter(), Counter(), Counter()

    for rel, target_rows, source, task in subsets:
        print(f"[fetch] {source}: collecting {target_rows} valid rows from {rel.split('/')[-1]} ...")
        units, scanned = collect_valid_units(fs, rel, target_rows, needed_cols)
        if len(units) < target_rows:
            print(f"  WARNING: only {len(units)}/{target_rows} valid rows found (scanned {scanned})")
        for per_lang, nt in units:
            for short, msgs in per_lang.items():
                records.append({
                    "language": short,
                    "source": f"indicalign/{source}",
                    "task": task,
                    "num_turns": nt if nt is not None else (len(msgs) // 2),
                    "messages": msgs,
                })
                by_lang[short] += 1
                by_source[source] += 1
                by_task[task] += 1
        print(f"  {source}: +{len(units)} rows (scanned {scanned}) -> running total {len(records)} records")

    random.shuffle(records)

    print(f"\n[write] {len(records)} records -> {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "total_records": len(records),
        "by_language": dict(sorted(by_lang.items())),
        "by_source": dict(by_source),
        "by_task": dict(by_task),
    }
    stats_file = os.path.splitext(out_file)[0] + "_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote: {out_file}")
    print(f"Stats: {stats_file}")


if __name__ == "__main__":
    main()
