"""
Validate (and optionally repair) the local .jsonl datasets referenced by an
Axolotl config, BEFORE handing them to `datasets.load_dataset`.

Why this exists
---------------
HF `datasets` reads local JSONL through pyarrow in blocks. A single malformed
line makes the whole load die with an opaque error like:

    pyarrow.lib.ArrowInvalid: JSON parse error:
        Missing a closing quotation mark in string. in row 2853

Two traps in that message:
  * the row index is relative to the ~10MB *batch* arrow was parsing, NOT the
    file, so "row 2853" is not line 2853;
  * the `ValueError: Trailing data` above it in the traceback is just datasets'
    pandas fallback trying to read JSONL as one JSON document -- ignore it.

"Missing a closing quotation mark" specifically means the input ended in the
middle of a string, i.e. a TRUNCATED line -- typically the last line of a file
whose upload or generation run was interrupted.

Usage
-----
    # report only
    python check_jsonl.py data/*.jsonl
    python check_jsonl.py --config qwen2.5_0.5b_dpo_IT.yml

    # rewrite each file with bad lines dropped (originals kept as *.bak)
    python check_jsonl.py --config qwen2.5_0.5b_dpo_IT.yml --fix

Also checks the DPO schema (prompt/chosen/rejected as message lists), since a
structurally-valid line with the wrong shape fails later and just as opaquely.
"""
import argparse
import json
import os
import re
import sys

REQUIRED_DPO_FIELDS = ("prompt", "chosen", "rejected")


def paths_from_config(cfg_path):
    """Pull local .jsonl paths out of an Axolotl yml without needing pyyaml.

    Matches `path: something.jsonl` under datasets:/test_datasets:. Paths are
    resolved relative to the config's directory, which is how Axolotl resolves
    them when you launch from that folder.
    """
    base = os.path.dirname(os.path.abspath(cfg_path))
    out = []
    with open(cfg_path, encoding="utf-8") as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            m = re.match(r"\s*-?\s*path:\s*(\S+\.jsonl)\s*$", line)
            if m:
                p = m.group(1)
                out.append(p if os.path.isabs(p) else os.path.join(base, p))
    # de-dupe, preserve order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def schema_problem(obj):
    """Return a description of a DPO-schema violation, or None if the record is fine."""
    if not isinstance(obj, dict):
        return f"top-level JSON is {type(obj).__name__}, expected object"
    for field in REQUIRED_DPO_FIELDS:
        if field not in obj:
            return f"missing field {field!r}"
        val = obj[field]
        if not isinstance(val, list) or not val:
            return f"{field!r} is {type(val).__name__}, expected non-empty list of messages"
        for i, msg in enumerate(val):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return f"{field}[{i}] is not a {{role, content}} object"
            if not isinstance(msg["content"], str):
                return f"{field}[{i}].content is {type(msg['content']).__name__}, expected str"
    return None


def check_file(path, check_schema=True):
    """Return (n_records, [(lineno, reason, raw_line)]) for one file."""
    bad = []
    records = 0
    # newline="" so a stray \r inside a record is NOT silently turned into a
    # line break -- we want to see the file exactly as arrow will.
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for n, raw in enumerate(f.read().split("\n"), 1):
            s = raw.strip()
            if not s:
                continue  # trailing newline at EOF, or a blank separator line
            records += 1
            try:
                obj = json.loads(s)
            except Exception as e:
                bad.append((n, f"{type(e).__name__}: {e}", raw))
                continue
            if check_schema:
                problem = schema_problem(obj)
                if problem:
                    bad.append((n, f"schema: {problem}", raw))
    return records, bad


def fix_file(path, bad_linenos):
    """Rewrite `path` without the bad lines. Original is moved to path + '.bak'."""
    bak = path + ".bak"
    tmp = path + ".tmp"
    drop = set(bad_linenos)
    kept = 0
    with open(path, encoding="utf-8", errors="replace", newline="") as src, \
         open(tmp, "w", encoding="utf-8", newline="\n") as dst:
        for i, raw in enumerate(src.read().split("\n"), 1):
            s = raw.strip()
            if not s or i in drop:
                continue
            dst.write(s + "\n")
            kept += 1
    os.replace(path, bak)
    os.replace(tmp, path)
    return kept, bak


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="jsonl files to check")
    ap.add_argument("--config", help="Axolotl yml; check every local .jsonl it references")
    ap.add_argument("--fix", action="store_true",
                    help="rewrite each file with bad lines dropped (backup at *.bak)")
    ap.add_argument("--no-schema", action="store_true",
                    help="only check JSON validity, not the DPO prompt/chosen/rejected shape")
    ap.add_argument("--show", type=int, default=5, help="how many bad lines to print per file")
    args = ap.parse_args()

    files = list(args.files)
    if args.config:
        files += paths_from_config(args.config)
    if not files:
        ap.error("give some .jsonl files or --config")

    total_bad = 0
    for path in files:
        if not os.path.exists(path):
            print(f"MISSING  {path}")
            total_bad += 1
            continue
        size = os.path.getsize(path)
        n, bad = check_file(path, check_schema=not args.no_schema)
        status = "OK  " if not bad else "BAD "
        print(f"{status} {path}\n       {n} records, {size:,} bytes, {len(bad)} bad")
        for lineno, reason, raw in bad[:args.show]:
            print(f"       line {lineno}: {reason}")
            print(f"         len={len(raw)} head={raw[:100]!r}")
            print(f"         tail={raw[-100:]!r}")
        if len(bad) > args.show:
            print(f"       ... and {len(bad) - args.show} more")
        if bad and args.fix:
            kept, bak = fix_file(path, [b[0] for b in bad])
            print(f"       FIXED -> kept {kept} lines; original at {bak}")
        total_bad += len(bad)

    print(f"\n{'all clean' if not total_bad else f'{total_bad} problem(s) found'}")
    return 1 if total_bad and not args.fix else 0


if __name__ == "__main__":
    sys.exit(main())
