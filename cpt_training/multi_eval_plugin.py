"""
Axolotl plugin: report a SEPARATE eval_loss per `test_datasets` entry, all in
the same training run.

Axolotl normally merges every `test_datasets` entry into a single eval set, so
its periodic eval logs only one blended `eval_loss`. The HuggingFace Trainer,
however, natively supports a *dict* of named eval datasets — when `eval_dataset`
is a dict it evaluates each one and logs `eval_<name>_loss` (plus runtime/samples
metrics) for every entry at each eval step.

This plugin hooks `add_callbacks_post_trainer` (called with the already-built
Trainer), re-prepares each `test_datasets` entry on its own, and installs them as
that dict. The merged eval set Axolotl built is simply replaced. Eval must still
be enabled in the YAML (keep `test_datasets` + `evals_per_epoch`/`eval_steps`) so
Axolotl turns on the Trainer's eval loop in the first place — this plugin only
swaps in the per-source datasets.

Enable it by adding to the YAML:

    plugins:
      - multi_eval_plugin.MultiEvalPlugin

The module must be importable — run via `python train.py --config <cfg>` (which
puts this folder on PYTHONPATH), or `PYTHONPATH=. axolotl train <cfg>` from here.

Note: each entry is loaded exactly the way its `path`/`name`/`split` specify,
i.e. the same download path Axolotl uses for the training slice — if a source is
too large to pull for training, it's equally impractical here; swap it for a
smaller dataset rather than special-casing eval.
"""

import re

from axolotl.integrations.base import BasePlugin
from datasets import load_dataset


def _short_name(path: str) -> str:
    """Turn a repo id into a safe metric-name suffix, e.g.
    'adityabanerjee13/indic-cpt-mini' -> 'indic_cpt_mini'."""
    base = path.rstrip("/").split("/")[-1]
    return re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_") or "eval"


class MultiEvalPlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        test_datasets = cfg.get("test_datasets") or []
        if len(test_datasets) < 2:
            # Nothing to split apart — leave Axolotl's single eval set as-is.
            return []

        # transformers >=4.46 renamed `tokenizer` -> `processing_class`.
        tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer
        seq_len = cfg.sequence_len

        def build(entry):
            field = entry.get("field", "text")
            ds = load_dataset(
                entry["path"],
                name=entry.get("name"),
                data_files=entry.get("data_files"),
                split=entry.get("split", "train"),
            )

            def tokenize(batch):
                enc = tokenizer(
                    batch[field], truncation=True, max_length=seq_len
                )
                # CPT / completion: labels are the input ids (next-token loss).
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
