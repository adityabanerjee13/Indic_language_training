"""
Axolotl plugin: report a SEPARATE eval_loss per `test_datasets` entry for SFT
(chat) runs, all in the same training run.

This is the SFT counterpart of cpt_training/multi_eval_plugin.py. Axolotl merges
every `test_datasets` entry into a single eval set and logs one blended
`eval_loss`. The HuggingFace Trainer, however, natively supports a *dict* of
named eval datasets — when `eval_dataset` is a dict it evaluates each one and
logs `eval_<name>_loss` (plus runtime/samples metrics) for every entry at each
eval step.

Difference vs. the CPT plugin: SFT eval data is chat `messages`, not a flat
`text` field. Each conversation is rendered through the model's chat template,
and (unless `train_on_inputs: true`) every token that is not part of an assistant
turn is masked to -100 so the per-source eval_loss is measured the same way as
the masked SFT training loss.

Enable it by adding to the YAML:

    plugins:
      - multi_eval_plugin.MultiEvalPlugin

The module must be importable — run via `python train.py --config <cfg>` (which
puts this folder on PYTHONPATH), or `PYTHONPATH=. axolotl train <cfg>` from here.

Each `test_datasets` entry should use `type: chat_template` with
`field_messages` (default "messages"); a `type: completion` entry is still
supported and handled the CPT way (labels = input ids) for mixed configs.
"""

import re

from axolotl.integrations.base import BasePlugin
from datasets import load_dataset


def _short_name(path: str) -> str:
    """Turn a repo id into a safe metric-name suffix, e.g.
    'adityabanerjee13/indic-sft-mini' -> 'indic_sft_mini'."""
    base = path.rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_") or "eval"


def _encode(tokenizer, messages, add_generation_prompt, max_length=None):
    """Render `messages` through the chat template and return a FLAT list of
    token ids.

    `return_dict=True` is used deliberately: passing tokenizer kwargs (e.g.
    truncation/max_length) to `apply_chat_template` can make it return a full
    BatchEncoding rather than a bare list, so we always extract `input_ids`
    ourselves to guarantee a plain `list[int]` (never a dict — which the
    collator's `tokenizer.pad` rejects).
    """
    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
        truncation=max_length is not None,
        max_length=max_length,
    )
    return list(enc["input_ids"])


def _mask_non_assistant(tokenizer, messages, input_ids):
    """Return labels where only assistant-turn tokens carry loss (rest = -100).

    Recomputes each assistant turn's token span by rendering the conversation
    prefix (with the assistant generation prompt) and the prefix+turn, then
    unmasking the tokens in between. Bounded by len(input_ids) so it stays
    aligned even when the full render was truncated to max_length.
    """
    labels = [-100] * len(input_ids)
    n = len(input_ids)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix = _encode(tokenizer, messages[:i], add_generation_prompt=True)
        full = _encode(tokenizer, messages[: i + 1], add_generation_prompt=False)
        start = min(len(prefix), n)
        end = min(len(full), n)
        for j in range(start, end):
            labels[j] = input_ids[j]
    return labels


class MultiEvalPlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        test_datasets = cfg.get("test_datasets") or []
        if len(test_datasets) < 2:
            # Nothing to split apart — leave Axolotl's single eval set as-is.
            return []

        # transformers >=4.46 renamed `tokenizer` -> `processing_class`.
        tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
        seq_len = cfg.sequence_len
        # SFT masks the prompt (train on assistant turns only); CPT does not.
        train_on_inputs = bool(cfg.get("train_on_inputs", False))

        def build(entry):
            etype = entry.get("type", "chat_template")
            ds = load_dataset(
                entry["path"],
                name=entry.get("name"),
                data_files=entry.get("data_files"),
                split=entry.get("split", "train"),
            )

            if etype == "chat_template":
                field_messages = entry.get("field_messages", "messages")

                def render(example):
                    messages = example[field_messages]
                    input_ids = _encode(
                        tokenizer,
                        messages,
                        add_generation_prompt=False,
                        max_length=seq_len,
                    )
                    if train_on_inputs:
                        labels = list(input_ids)
                    else:
                        labels = _mask_non_assistant(
                            tokenizer, messages, input_ids
                        )
                    return {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "labels": labels,
                    }

                return ds.map(render, remove_columns=ds.column_names)

            # Fallback: CPT / completion entry (flat text, unmasked LM loss).
            field = entry.get("field", "text")

            def tokenize(batch):
                enc = tokenizer(
                    batch[field], truncation=True, max_length=seq_len
                )
                enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
                return enc

            return ds.map(
                tokenize, batched=True, remove_columns=ds.column_names
            )

        eval_sets = {}
        for entry in test_datasets:
            name = _short_name(entry["path"])
            # Disambiguate if two entries share a basename.
            if name in eval_sets:
                name = f"{name}_{len(eval_sets)}"
            eval_sets[name] = build(entry)

        trainer.eval_dataset = eval_sets
        print(
            f"[multi_eval_plugin] per-source eval enabled for: "
            f"{', '.join(eval_sets)} (metrics: "
            f"{', '.join(f'eval_{n}_loss' for n in eval_sets)})"
        )
        return []
