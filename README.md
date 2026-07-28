# Indic Language Training of **Qwen2.5-0.5B** on 10 Indian languages

Languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia,
Punjabi, Tamil, Telugu (the 10 Sarvam-1 / IndicCorp v2 languages).


## Continued Pre-Training (CPT)

### 1. Data (`data/`)


### 2. CPT training (`cpt_training/`)
Axolotl configs for full-parameter next-token training (`type: completion`) on
raw Indic text, single-GPU, with sequence packing.

| Config | What it trains |
|---|---|
| `qwen2.5_0.5b_cpt_full_test_sample.yml` | quick end-to-end smoke test |
| `qwen2.5_0.5b_cpt_full.yml` | full-parameter, Indic only |
| `qwen2.5_0.5b_cpt_frozen.yml` | only first/last 3 layers + embeddings/norm trainable |
| `qwen2.5_0.5b_cpt_mix_1to4.yml` | Indic : FineWeb replay at ratio 4:1 experiments |
| `qwen2.5_0.5b_cpt_mix_1to2.yml` | Indic : FineWeb replay at ratio 2:1 experiments |
| `qwen2.5_0.5b_cpt_mix_1to1.yml` | Indic : FineWeb replay at ratio 1:1 experiments |

- **`multi_eval_plugin.py`** — Axolotl plugin that logs a **separate** eval loss
  per validation source (`eval_indic_cpt_mini_val_loss` vs
  `eval_fineweb_cpt_val_loss`) instead of one blended number.
- **`train.py`** — launcher: pins one GPU and invokes `axolotl train <config>`.
- **`artifacts/`** — training loss/perplexity and Indic/FineWeb validation curves.

### 3. Evaluation

#### 3.1 Model validation losses

Five CPT configs were tracked in Weights & Biases — `indic-cpt-full` (full-param,
Indic only), `indic-cpt-frozen` (only first/last 3 layers + embeddings/norm),
and the three replay mixes `cpt-mix-1to1 / 1to2 / 1to4` (Indic : FineWeb-English).
The `multi_eval_plugin` logs held-out loss **separately** on an Indic set and a
FineWeb (English) set, so Indic learning and English forgetting can be read off
independently.

| Training loss | Training perplexity |
|---|---|
| ![train/loss](cpt_training/artifacts/trainloss.png) | ![train/ppl](cpt_training/artifacts/trainppl.png) |

| Held-out Indic loss | Held-out FineWeb (English) loss |
|---|---|
| ![eval indic](cpt_training/artifacts/validationIndic.png) | ![eval fineweb](cpt_training/artifacts/validationFineWeb.png) |

#### 3.2 Benchmark scores  (`IndicGenBench/`)

Runs the CPT'd model (`run_qwen_ft_benchmark.py`) over four IndicGenBench tasks and scores it per the benchmark's protocol:

| Task | Type | Metric |
|---|---|---|
| XQuAD-IN | in-language extractive QA | SQuAD Token-F1 |
| XORQA-IN | cross-lingual open-retrieval QA | SQuAD Token-F1 |
| CrossSum-IN | cross-lingual summarization | chrF |
| Flores-IN | translation (en→xx and xx→en) | chrF |

##### Results

Scored with `run_qwen_ft_benchmark.py` (direct HF `transformers` inference on an
Intel Arc XPU): 20 examples/language, seed 42, greedy decoding, macro-averaged
across the 10 languages. Higher is better throughout.

Models: **Base** = `Qwen2.5-0.5B` (no CPT) · **Indic-CPT** =
`qwen2.5-0.5b-indic-cpt` (Indic only) · **Mix-1to1 / 1to2 / 1to4** =
`qwen2.5-0.5b-cpt-mix-*` (Indic + FineWeb English replay at 1:1, 1:2, 1:4
Indic:English ratios — larger denominator = more English replay). **Sarvam-1†** =
`sarvamai/sarvam-1`, a ~2B Indic-native model included as a stronger external
reference (a rough ceiling), **not** a 0.5B CPT variant.

| Task | Metric | Base | Mix-1to1 | Mix-1to2 | Mix-1to4 | Indic-CPT | Best 0.5B | Sarvam-1† |
|---|---|---:|---:|---:|---:|---:|:--:|---:|
| XQuAD-IN (QA) | Token-F1 ×100 | 4.81 | 2.05 | 4.73 | 5.07 | **5.19** | Indic-CPT | 8.79 |
| XORQA-IN (QA) | Token-F1 ×100 | 1.84 | 1.06 | 1.72 | **2.32** | 2.01 | Mix-1to4 | 6.67 |
| CrossSum-IN (summarization) | chrF | 1.59 | 0.43 | 2.58 | **3.15** | 2.56 | Mix-1to4 | 1.60 |
| Flores-IN en→xx (into Indic) | chrF | 7.38 | 7.57 | **8.64** | 8.49 | 8.46 | Mix-1to2 | 18.65 |
| Flores-IN xx→en (into English) | chrF | **14.23** | 6.51 | 12.74 | 11.26 | 7.17 | Base | 31.73 |

† The **Best 0.5B** column names the winner *among the Qwen2.5-0.5B variants only*;
Sarvam-1 (~2B) is listed separately as an external reference, so bolding and the
"best" label are kept within the same-size CPT experiments.
