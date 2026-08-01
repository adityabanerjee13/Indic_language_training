"""
Downsample the IndicAlign-Toxic DPO triplets (train + val) to a fraction of their
size, keeping the language x source balance intact.

Why stratify: the toxic files are near-uniform over 10 languages x 2 sources
(~1470 per cell in train). A naive random 50% drifts those cells by a few percent
each; sampling within each (language, source) cell keeps every cell at exactly
half, so the safety signal stays balanced across languages.

Why you'd want this: toxic is 29,400 of the 39,400 training triplets (75% of the
mix). Halving it gives 14,700 toxic / 6,250 indic / 3,750 tulu -- still safety-
heavy, but no longer swamping the instruction-following preferences.

Selection is deterministic: same --seed and same input file always drop the same
records, and the kept records stay in their original file order.

Usage:
    # non-destructive: writes data/toxic_dpo_triplets_{train,val}_50pct.jsonl
    python downsample_toxic.py

    # overwrite the files the config already points at (originals -> *.bak)
    python downsample_toxic.py --in-place

    # other fractions / different files
    python downsample_toxic.py --fraction 0.25
    python downsample_toxic.py --files data/indic_dpo_train.jsonl --in-place
"""
import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_FILES = [
    "data/toxic_dpo_triplets_train.jsonl",
    "data/toxic_dpo_triplets_val.jsonl",
]


def read_records(path):
    """Return [(lineno, raw_line, parsed)] for every non-blank line."""
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            out.append((i, s, json.loads(s)))
    return out


def stratified_keep(records, fraction, seed):
    """Indices to keep, stratified over (language, source).

    Within each cell we take round(n * fraction) records, chosen by shuffling the
    cell's indices with a per-cell seed so the result is reproducible and does not
    depend on how the cells happen to be interleaved in the file.
    """
    cells = defaultdict(list)
    for idx, (_, _, rec) in enumerate(records):
        cells[(rec.get("language"), rec.get("source"))].append(idx)

    keep = []
    per_cell = {}
    for cell, idxs in sorted(cells.items(), key=lambda kv: str(kv[0])):
        rng = random.Random(f"{seed}|{cell}")
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        n_keep = round(len(idxs) * fraction)
        keep.extend(shuffled[:n_keep])
        per_cell[cell] = (len(idxs), n_keep)
    keep.sort()  # preserve original file order
    return keep, per_cell


def process(path, fraction, seed, in_place, suffix, show_cells):
    records = read_records(path)
    if not records:
        print(f"  !! {path} is empty, skipping")
        return

    keep_idx, per_cell = stratified_keep(records, fraction, seed)
    keep_set = set(keep_idx)

    if in_place:
        out_path = path
        bak = path + ".bak"
        if os.path.exists(bak):
            raise SystemExit(
                f"refusing to overwrite an existing backup: {bak}\n"
                f"  move or delete it first, so the pre-downsample original is not lost"
            )
        shutil.copy2(path, bak)
    else:
        root, ext = os.path.splitext(path)
        out_path = f"{root}{suffix}{ext}"
        bak = None

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for idx, (_, raw, _) in enumerate(records):
            if idx in keep_set:
                f.write(raw + "\n")
    os.replace(tmp, out_path)

    before, after = len(records), len(keep_idx)
    print(f"  {os.path.relpath(path, HERE)}")
    print(f"     {before} -> {after} records ({after / before:.1%}), "
          f"{before - after} dropped")
    print(f"     wrote {os.path.relpath(out_path, HERE)}"
          + (f"  (original -> {os.path.basename(bak)})" if bak else ""))

    if show_cells:
        langs = Counter()
        srcs = Counter()
        for (lang, src), (n, k) in per_cell.items():
            langs[lang] += k
            srcs[src] += k
        print(f"     kept by language: {dict(sorted(langs.items(), key=lambda kv: str(kv[0])))}")
        print(f"     kept by source  : {dict(sorted(srcs.items(), key=lambda kv: str(kv[0])))}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                    help="jsonl files to downsample (paths relative to this script)")
    ap.add_argument("--fraction", type=float, default=0.3,
                    help="fraction of each (language, source) cell to KEEP (default 0.5)")
    ap.add_argument("--seed", default=42, help="selection seed; same seed = same records")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the inputs, saving the originals to *.bak")
    ap.add_argument("--suffix", default=None,
                    help="suffix for non-in-place output (default derived from --fraction)")
    ap.add_argument("--no-cells", action="store_true", help="skip the per-cell breakdown")
    args = ap.parse_args()

    if not 0 < args.fraction <= 1:
        ap.error("--fraction must be in (0, 1]")
    suffix = args.suffix if args.suffix is not None else f"_{round(args.fraction * 100)}pct"

    print(f"keeping {args.fraction:.0%} of each (language, source) cell, seed={args.seed}\n")
    for rel in args.files:
        path = rel if os.path.isabs(rel) else os.path.join(HERE, rel)
        if not os.path.exists(path):
            print(f"  !! not found: {rel}")
            continue
        process(path, args.fraction, args.seed, args.in_place, suffix, not args.no_cells)

    if not args.in_place:
        print(f"\nNothing was overwritten. To use these, either rerun with --in-place,"
              f"\nor point the `path:` entries in qwen2.5_0.5b_dpo_IT.yml at the"
              f"\n*{suffix}.jsonl files.")
    print("\nValidate before training:  python check_jsonl.py --config qwen2.5_0.5b_dpo_IT.yml")


if __name__ == "__main__":
    main()
