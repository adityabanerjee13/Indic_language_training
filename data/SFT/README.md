---
license: cc-by-4.0
language:
- bn
- gu
- hi
- kn
- ml
- mr
- or
- pa
- ta
- te
task_categories:
- text-generation
pretty_name: Indic SFT Mini
size_categories:
- 10K<n<100K
tags:
- indic
- instruction-tuning
- sft
- multilingual
---

# Indic SFT Mini

A compact, balanced supervised fine-tuning (SFT) dataset for **10 Indic languages**, derived from
[ai4bharat/indic-align](https://huggingface.co/datasets/ai4bharat/indic-align) (IndicAlign).

- **80,000 examples** — exactly **8,000 per language**.
- Languages: Bengali (bn), Gujarati (gu), Hindi (hi), Kannada (kn), Malayalam (ml), Marathi (mr), Odia (or), Punjabi (pa), Tamil (ta), Telugu (te).
- Native-script only. Chat-format, ready for TRL / SFT trainers.

## Composition

Drawn from 6 IndicAlign instruction sub-datasets (safety/toxic sources intentionally excluded):

| Source | Records | Task |
|---|---|---|
| Indic-ShareLlama | 15,000 | instruction (open QA) |
| Dolly-T | 15,000 | instruction (task-diverse) |
| Wiki-Conv | 15,000 | chat (short multi-turn) |
| OpenAssistant-T | 12,500 | chat (multi-turn) |
| WikiHow | 12,500 | how-to |
| Wiki-Chat | 10,000 | chat (long conversations) |

By task: chat 37,500 · instruction 30,000 · how-to 12,500.

## Record schema

```json
{
  "language": "hi",
  "source": "indicalign/dolly_t",
  "task": "instruction",
  "num_turns": 1,
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Multi-turn conversations (Wiki-Conv, Wiki-Chat, OpenAssistant-T) preserve the full turn sequence.

## Curation

- Streamed the 10 native-script columns from each IndicAlign parquet; kept only rows where **all 10
  languages** parse to a valid `user → … → assistant` exchange (guarantees exact per-language balance).
- IndoWordNet (dictionary-terse, grouped-by-language) and the safety sub-datasets (HHRLHF-T, Toxic-Matrix) were excluded.
- Native-script consistency verified (<0.3% script mismatch on sampling).

## Provenance & license

Derived from IndicAlign (CC-BY-4.0), which was built with responses from Llama-2-70B-Chat and translated
into Indic languages via IndicTrans2. This subset inherits that license and those quality characteristics.
Please cite IndicLLMSuite (Khan et al., 2024, arXiv:2403.06350) and IndicTrans2 (Gala et al., 2023).

Build script: `build_sft_indicalign.py` (included in this repo).
