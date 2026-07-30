"""
Exactly count the token budget of the Indic SFT dataset with the Qwen2.5-0.5B
tokenizer, then stream the Tulu-3 SFT mixture and keep exactly enough examples
to reach HALF that token count (measured with the same tokenizer).

Token counting (identical method for both datasets, no approximation):
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False)
    n    = len(tokenizer(text, add_special_tokens=False)["input_ids"])
(We render the chat template to text then tokenize, because
 apply_chat_template(tokenize=True) is unreliable in this transformers version.)

Outputs (this folder):
    indic_token_report.json   exact Indic token totals + per-language
    tulu_half.jsonl           kept Tulu examples (messages/id/source)
    tulu_token_report.json    exact Tulu totals + target

Requirements: transformers, datasets, torch
Usage: python count_indic_and_fetch_tulu.py
"""

import io
import json
import os
import sys

TOKENIZER = "Qwen/Qwen2.5-0.5B"
INDIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SFT", "indicalign_sft_80k.jsonl")
TULU_REPO = "allenai/tulu-3-sft-mixture"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    from transformers import AutoTokenizer

    print(f"[load] tokenizer = {TOKENIZER}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def render(messages):
        # normalize to plain role/content dicts the template accepts
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

    def count_texts(texts):
        """Batch-tokenize rendered texts; return list of token counts."""
        enc = tok(texts, add_special_tokens=False)
        return [len(ids) for ids in enc["input_ids"]]

    # ---------------- Pass 1: exact Indic token count ----------------
    print(f"[pass 1] counting Indic SFT tokens from {os.path.basename(INDIC_PATH)} ...")
    from collections import Counter
    indic_total = 0
    by_lang = Counter()
    n_records = 0
    batch_texts, batch_langs = [], []
    BATCH = 512

    def flush(batch_texts, batch_langs):
        nonlocal_total = 0
        counts = count_texts(batch_texts)
        for c, lang in zip(counts, batch_langs):
            by_lang[lang] += c
            nonlocal_total += c
        return nonlocal_total

    with open(os.path.abspath(INDIC_PATH), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            batch_texts.append(render(r["messages"]))
            batch_langs.append(r.get("language", "?"))
            n_records += 1
            if len(batch_texts) >= BATCH:
                indic_total += flush(batch_texts, batch_langs)
                batch_texts, batch_langs = [], []
                if n_records % 10240 == 0:
                    print(f"    ...{n_records} records, {indic_total:,} tokens so far")
    if batch_texts:
        indic_total += flush(batch_texts, batch_langs)

    indic_report = {
        "tokenizer": TOKENIZER,
        "indic_records": n_records,
        "indic_total_tokens": indic_total,
        "indic_avg_tokens_per_record": indic_total / max(n_records, 1),
        "by_language_tokens": dict(sorted(by_lang.items())),
    }
    with open(os.path.join(OUT_DIR, "indic_token_report.json"), "w", encoding="utf-8") as f:
        json.dump(indic_report, f, ensure_ascii=False, indent=2)
    print(f"[pass 1] EXACT Indic total = {indic_total:,} tokens over {n_records:,} records "
          f"(avg {indic_total/max(n_records,1):.1f})")

    target = indic_total // 2
    print(f"[target] half of Indic = {target:,} tokens -> collect this many Tulu tokens")

    # ---------------- Pass 2: stream Tulu until half reached ----------------
    print(f"[pass 2] streaming {TULU_REPO} (train), tokenizing with same tokenizer ...")
    from datasets import load_dataset
    ds = load_dataset(TULU_REPO, split="train", streaming=True)

    out_path = os.path.join(OUT_DIR, "tulu_half.jsonl")
    tulu_tokens = 0
    tulu_kept = 0
    tulu_skipped = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for ex in ds:
            msgs = ex.get("messages")
            if not msgs:
                tulu_skipped += 1
                continue
            try:
                text = render(msgs)
                n = len(tok(text, add_special_tokens=False)["input_ids"])
            except Exception:
                tulu_skipped += 1
                continue
            rec = {"id": ex.get("id"), "source": ex.get("source"), "messages": msgs}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tulu_tokens += n
            tulu_kept += 1
            if tulu_kept % 2000 == 0:
                print(f"    ...kept {tulu_kept} Tulu examples, {tulu_tokens:,}/{target:,} tokens")
            if tulu_tokens >= target:
                break

    tulu_report = {
        "tokenizer": TOKENIZER,
        "tulu_repo": TULU_REPO,
        "target_tokens_half_of_indic": target,
        "tulu_kept_examples": tulu_kept,
        "tulu_total_tokens": tulu_tokens,
        "tulu_skipped": tulu_skipped,
        "achieved_ratio_tulu_over_indic": tulu_tokens / max(indic_total, 1),
    }
    with open(os.path.join(OUT_DIR, "tulu_token_report.json"), "w", encoding="utf-8") as f:
        json.dump(tulu_report, f, ensure_ascii=False, indent=2)

    print("\n=== FINAL ===")
    print(json.dumps({**indic_report, **tulu_report,
                      "by_language_tokens": "(see indic_token_report.json)"},
                     ensure_ascii=False, indent=2))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
