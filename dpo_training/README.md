# DPO Triplet Pipeline (IndicAlign-Toxic)

Two-step pipeline to build `(prompt, chosen, rejected)` preference triplets for DPO,
covering the 10 target Indic languages, in **HF / TRL DPOTrainer conversational format**.

```
IndicAlign-Toxic ──[step 1]──▶ (prompt, chosen) pairs ──[step 2: base model]──▶ (prompt, chosen, rejected) triplets
   HHRLHF-T                       chosen = refusal            rejected = base model's generation
   Toxic-Matrix
```

## Step 1 — download pairs
```bash
python download_indicalign_toxic.py --rows-per-source 300
```
- Streams HHRLHF-T + Toxic-Matrix (10 native-script columns) from `ai4bharat/indic-align`.
- Emits `data/indicalign_toxic_pairs.jsonl`: `{language, source, prompt:[user], chosen:[assistant refusal]}`.
- `--rows-per-source N` controls volume (each source row → up to 10 pairs, one per language).

## Step 2 — generate rejected responses (any HF model path)
```bash
# Uses the default base model Qwen/Qwen2.5-0.5B; override with --model
python generate_rejected.py \
    --input data/indicalign_toxic_pairs.jsonl \
    --output data/dpo_triplets.jsonl \
    [--model <HF_MODEL_PATH>] [--limit N] [--max-new-tokens 256] [--device auto|cuda|xpu|cpu]
```
- Loads the base model with `transformers`, renders the prompt via the model's chat template
  (falls back to a plain format for base models with no template), samples a response = `rejected`.
- Emits triplets in HF conversational DPO format:
  ```json
  {"prompt":[{"role":"user","content":"..."}],
   "chosen":[{"role":"assistant","content":"..."}],
   "rejected":[{"role":"assistant","content":"..."}]}
  ```
  (+ `language`/`source` metadata columns, ignored by trainers.)
- Loads directly via `datasets.load_dataset("json", data_files=...)` into `trl.DPOTrainer`.

## ⚠️ Choice of base model matters
- An **aligned/safety-tuned** model (e.g. Qwen2.5-*-Instruct) will often *also refuse* the harmful
  prompt, so `rejected` becomes a (usually shorter / English) refusal. DPO then learns "prefer the
  detailed target-language refusal over a curt/English one" — a **language-consistency / refusal-quality**
  signal, not a safe-vs-unsafe one.
- To get the classic **safe (chosen) vs unsafe (rejected)** contrast, generate `rejected` from a
  **base / non-safety-tuned** model that will actually comply with the harmful prompt.
- The pipeline is model-agnostic — pass whichever HF model path fits your DPO objective.

## Files
- `download_indicalign_toxic.py` — Step 1
- `generate_rejected.py` — Step 2
- `data/indicalign_toxic_pairs.jsonl` — pairs (Step 1 output)
- `data/dpo_triplets_sample.jsonl` — tested 5-sample triplet output (Step 2)
