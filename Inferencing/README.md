# Inferencing

Direct inference against any checkpoint in this project: give a model name and a
prompt, get the continuation. Two entry points —

| Script | Use |
|---|---|
| `infer.py` | one model, one prompt, one completion, printed to stdout |
| `run_sweep.py` | every model × every prompt × every temperature → one JSON |

Results published in the project README come from `run_sweep.py`; the raw JSON
lives in `prompts/*_T0.3_results.json`.

## Run it

Use `C:\Python313\python.exe`, which has `torch 2.13.0+xpu` and reaches the Arc
iGPU. The conda env `ladder-of-descent-py311` is CPU-only and ships a broken
torchvision that crashes `from_pretrained`; `conda deactivate` first if it is
active. Confirm with the startup line `Loaded on xpu (bf16).`

```bash
# single prompt
C:\Python313\python.exe infer.py --model base --prompt "The capital of India is"
C:\Python313\python.exe infer.py --model indic-cpt --prompt-file prompts/ondist/01_news_cultural.txt

# a whole grid
C:\Python313\python.exe run_sweep.py --models base cpt-mix-1to2 sft-IT dpo-IT \
    --temperatures 0.3 --prompt-dir prompts/ondist \
    --output prompts/ondist_T0.3_results.json --partial-dir prompts/_partial_ondist
```

`--model` takes a short alias (`base`, `cpt-mix-1to2`, `indic-cpt`, `sft-I`,
`sft-IT`, `dpo-IT`, `dpo-IT_non_align`, `sarvam-1`), a Hugging Face repo id, or a
local checkpoint directory.

---

# How inference actually works here

Two probes have been run with these scripts — **prompt completion** and
**question answering**. The single most important thing to understand is that
they are *the same code path*. Only the prompt text differs. Nothing in this
directory applies a chat template.

## The shared path

Both cases go through exactly these steps, in `infer.py`:

**1. Load the prompt byte for byte.** `read_prompt()` does not strip. For a
completion model a trailing space or newline is part of the input and changes
the continuation, so `--prompt-file` / `--stdin` are the reliable way in when
exact bytes matter (a shell eats trailing spaces).

**2. Tokenize with `add_special_tokens=False`.** The prompt stays literally the
prompt. Qwen2.5 adds nothing here anyway, but a tokenizer with a BOS would
otherwise prepend a token the caller never wrote.

**3. Generate.** Greedy by default; `--temperature` switches to sampling, with
the seed re-applied before every call so a given (model, prompt, temperature) is
reproducible. `eos_token_id` is the tokenizer's EOS and **nothing else**.

**4. Decode the new tokens only**, `skip_special_tokens=True`, and do not strip
— the leading space of a continuation is part of it.

That is the whole pipeline. There is no system message, no instruction wrapper,
no `<|im_start|>`, no BOS, no post-processing.

## Case 1 — prompt completion

**The prompt is a fragment of a document, cut mid-sentence.** The model's job is
to carry on writing the same document.

```
इंदौर। शहर के स्थानीय कला केंद्र में गुरुवार शाम को लोक नृत्य की प्रस्तुति दी गई।
कार्यक्रम में विभिन्न विद्यालयों के छात्र-छात्राओं ने हिस्सा लिया। आयोजकों ने बताया कि
                                                                                    ↑
                                                                    generation starts here
```

What reaches the model's forward pass is that string and nothing else. The first
generated token is whatever most plausibly follows `कि` in a Hindi news report.

This is the **correct, on-distribution** envelope for `base`, the `*-cpt-*`
checkpoints and `sarvam-1`: they were trained with `type: completion` on raw
IndicCorp text (see `cpt_training/*.yml`), which is exactly this. It is also the
envelope that matches the Ollama `/api/generate` baseline, which likewise
bypasses any template.

Two consequences worth internalising before reading output:

- **The model has no reason to stop.** A completion model does not emit EOS on a
  raw prompt, so generation runs the full `--max-new-tokens` budget and ends
  wherever that lands — usually mid-word, sometimes mid-UTF-8-character, which
  decodes as a trailing `�`. That is the budget, not a failure.
- **Prompt design is the experiment.** Cutting at `आयोजकों ने बताया कि`
  ("the organisers said that…") constrains the continuation to reported speech;
  cutting at `अभियान के तहत जिले की` invites the statistics pattern. What you cut
  determines what you are testing.

## Case 2 — question answering

**The prompt is a complete question.** Nothing else changes:

```
पल्स पोलियो अभियान के तहत किस आयु वर्ग के बच्चों को दवा पिलाई जाती है, ...?
                                                                          ↑
                                                          generation starts here
```

The model receives that question **as bare text** — not as a user turn. There is
no `<|im_start|>user … <|im_end|><|im_start|>assistant` around it, and
`<|im_end|>` is *not* registered as a stop token.

This is the subtle part, and it is why the two cases behave so differently
despite sharing a code path:

| Checkpoint | What this input looks like to it |
|---|---|
| `base`, `*-cpt-*` | a document that happens to open with a question — so it continues the *document*, typically by echoing the question and carrying on |
| `sft-IT`, `dpo-*` | a malformed conversation: the tokens that mark "your turn to answer" are missing |

So Case 2 is **not** a chat-completion test. It is a probe of how much assistant
behaviour is baked into the weights strongly enough to fire without the
scaffolding. (Empirically: quite a lot — both instruction-tuned checkpoints open
with `निश्चित रूप से,` anyway. See the project README.)

### What a real chat envelope would add

Not implemented here, deliberately. For contrast, `run_qwen_ft_benchmark.py
--chat` does it, and needs two changes this directory does not make:

1. **Wrap the prompt**:
   `tokenizer.apply_chat_template([{"role": "user", ...}], add_generation_prompt=True)`,
   the same rendering the SFT/DPO training text was built with
   (`data/cap_and_rebalance.py:74`).
2. **Add `<|im_end|>` as a stop token.** Qwen2.5's `eos_token` is
   `<|endoftext|>`, but an SFT-trained assistant turn ends at `<|im_end|>`.
   Without it the model sails past its answer and starts hallucinating the next
   `user` turn, which then gets decoded into the output.

Getting either wrong reads as a model regression rather than a harness mistake.
Benchmark numbers in the project README were produced *in* each checkpoint's own
envelope for exactly this reason; the probes here were not, and are labelled
accordingly.

---

## Decoding, and one hardware trap

Greedy by default, so repeated runs agree. `--temperature` enables sampling;
`--top-p` / `--top-k` apply only alongside it; `--seed` makes a sampled run
reproducible.

**The first sampled generation in a process costs ~90 s on this iGPU.** Every
sampled call after it costs ~0.2 s. It is a one-time compile of the XPU sampling
path, not a per-token cost, and it lands on whichever sampled call runs first
regardless of temperature or top-p/top-k. Measured on Qwen2.5-0.5B:

| | first sampled call | subsequent |
|---|---:|---:|
| 8 tokens | 89.3 s | 0.2 s |

Left alone this gets billed to whatever prompt happens to run first, making that
one cell look pathologically slow. `run_sweep.py` calls `warm_up_sampling()`
right after loading each model to pay it explicitly, so recorded timings are
comparable. If you script your own loop, do the same — or you will conclude
sampling is unusably slow, which it is not.

## `run_sweep.py` mechanics

- **One child process per model.** The XPU allocator does not reliably release a
  model's weights when the owning object goes out of scope, so loading several
  checkpoints in one process fills the 16 GiB card and every load after the
  first dies with `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`. Process exit always
  releases it — the same reason `run_lm_eval.py` shells out.
- **Prompt set, output file and partial dir move together.** `--prompt-dir`
  globs `*.txt` non-recursively, so `prompts/ondist/` and
  `prompts/ondist_questions/` are separate sets rather than one growing pile.
  Pass all three flags together: resuming from another run's partials would mix
  in completions for prompts that are not in the current set.
- **Resumable.** Per-model partials are written as each model finishes all its
  cells; re-running skips models that already have one. `--overwrite` forces a
  re-run.

## Other flags

| flag | default | notes |
|---|---|---|
| `--max-new-tokens` | 256 (`infer.py`) / 128 (`run_sweep.py`) | completion models spend all of it |
| `--max-prompt-tokens` | 3072 | prompt truncated above this, with a warning |
| `--device` | `cuda` > `xpu` > `cpu` | whichever is available |
| `--dtype` | `bf16` | use `fp32` on CPU |
| `--stream` | off | print tokens as they decode (`infer.py` only) |

## Layout

```
Inferencing/
├── infer.py                          single prompt → single completion
├── run_sweep.py                      models × prompts × temperatures → JSON
└── prompts/
    ├── ondist/                       Probe A: document fragments, cut mid-sentence
    ├── ondist_questions/             Probe B: complete questions
    ├── ondist_T0.3_results.json      Probe A results
    ├── ondist_questions_T0.3_results.json   Probe B results
    └── _partial*/                    per-model partials (resume state)
```

Each result JSON holds run metadata (device, dtype, seed, temperatures, token
budget, decoding, envelope caveat), the prompts verbatim, and one row per cell
with `model`, `repo`, `prompt`, `temperature`, `completion`, `prompt_tokens`,
`new_tokens` and `seconds`.
