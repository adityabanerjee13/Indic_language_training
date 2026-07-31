"""
Axolotl plugin: report SEPARATE eval metrics per `test_datasets` entry for DPO
runs, all in the same training run.

This is the DPO counterpart of sft_training/multi_eval_plugin.py. Axolotl merges
every `test_datasets` entry into ONE eval set and logs a single blended
`eval_loss`. The HuggingFace Trainer, however, natively supports a *dict* of
named eval datasets: when `eval_dataset` is a dict it evaluates each one and logs
`eval_<name>_<metric>` for every entry at each eval step. For DPO that means, per
source, you get:

    eval_<name>_loss
    eval_<name>_rewards/chosen        eval_<name>_rewards/rejected
    eval_<name>_rewards/accuracies    eval_<name>_rewards/margins
    eval_<name>_logps/chosen          eval_<name>_logps/rejected

Difference vs. the SFT plugin: DPO eval data is preference triplets
{prompt, chosen, rejected}, not chat `messages` with masked labels. Rather than
tokenize by hand, we delegate to the DPO trainer's OWN dataset preparation
(`_prepare_dataset`) so each per-source eval set is processed EXACTLY like the
trainer's train/eval data (chat-template rendering, prompt extraction, and the
DPO tokenize_row). This guarantees format-consistency across TRL versions.

Enable it by adding to the YAML:

    plugins:
      - multi_eval_plugin.MultiEvalPlugin

The module must be importable — run via `python -m axolotl.cli.train <cfg>` from
this folder, or `PYTHONPATH=. accelerate launch -m axolotl.cli.train <cfg>`.

Each `test_datasets` entry is the same conversational preference format the DPO
`datasets` use: columns prompt / chosen / rejected (remapped from field_prompt /
field_chosen / field_rejected if those are set).
"""

import re

from axolotl.integrations.base import BasePlugin
from datasets import load_dataset


def _short_name(path: str) -> str:
    """Turn a path/repo id into a safe metric-name suffix, e.g.
    'data/sft_source/indic_val_dpo_triplets.jsonl' -> 'indic_val_dpo_triplets'."""
    base = path.rstrip("/").split("/")[-1]
    base = re.sub(r"\.(jsonl|json)$", "", base)
    return re.sub(r"[^0-9a-zA-Z]+", "_", base).strip("_") or "eval"


def _load_entry(entry):
    """Load a `test_datasets` entry as a raw HF Dataset, honoring local json
    files (ds_type: json / *.jsonl) as well as Hub repos."""
    path = entry["path"]
    split = entry.get("split", "train")
    data_files = entry.get("data_files")
    ds_type = entry.get("ds_type")
    is_json = ds_type == "json" or (
        data_files is None and str(path).endswith((".json", ".jsonl"))
    )
    if is_json:
        return load_dataset("json", data_files=(data_files or path), split=split)
    return load_dataset(path, name=entry.get("name"), data_files=data_files, split=split)


def _standardize_columns(ds, entry):
    """Rename field_prompt/field_chosen/field_rejected -> prompt/chosen/rejected
    and drop everything else (e.g. language/source metadata)."""
    wanted = {
        "prompt": entry.get("field_prompt", "prompt"),
        "chosen": entry.get("field_chosen", "chosen"),
        "rejected": entry.get("field_rejected", "rejected"),
    }
    for std, src in wanted.items():
        if src != std and src in ds.column_names and std not in ds.column_names:
            ds = ds.rename_column(src, std)
    keep = [c for c in ("prompt", "chosen", "rejected") if c in ds.column_names]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def _prepare_for_dpo(trainer, tokenizer, ds, name):
    """Run `ds` through the DPO trainer's own dataset preparation so it matches
    the format DPOTrainer's collator expects. `_prepare_dataset` is the same
    method the trainer uses internally for its train/eval datasets; its signature
    has shifted across TRL versions, so try the known forms in order."""
    fn = getattr(trainer, "_prepare_dataset", None)
    if fn is None:
        raise RuntimeError(
            "trainer has no _prepare_dataset — this plugin requires a DPO "
            "(TRL) trainer. Use the SFT multi_eval_plugin for SFT runs."
        )
    for call in (
        lambda: fn(ds, tokenizer, trainer.args, name),   # modern TRL
        lambda: fn(ds, tokenizer, trainer.args),         # no dataset_name
        lambda: fn(ds, tokenizer),
        lambda: fn(ds),
    ):
        try:
            return call()
        except TypeError:
            continue
    # Last attempt: let the real error surface instead of hiding it.
    return fn(ds, tokenizer, trainer.args, name)


class MultiEvalPlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        test_datasets = cfg.get("test_datasets") or []
        if len(test_datasets) < 2:
            # Nothing to split apart — leave Axolotl's single eval set as-is.
            return []

        # transformers >=4.46 renamed `tokenizer` -> `processing_class`.
        tokenizer = getattr(trainer, "processing_class", None) or trainer.tokenizer

        eval_sets = {}
        for entry in test_datasets:
            name = _short_name(entry["path"])
            if name in eval_sets:  # disambiguate shared basenames
                name = f"{name}_{len(eval_sets)}"
            ds = _standardize_columns(_load_entry(entry), entry)
            eval_sets[name] = _prepare_for_dpo(trainer, tokenizer, ds, name)

        trainer.eval_dataset = eval_sets
        print(
            f"[multi_eval_plugin] per-source DPO eval enabled for: "
            f"{', '.join(eval_sets)} "
            f"(metrics: {', '.join(f'eval_{n}_loss' for n in eval_sets)}, "
            f"plus per-source rewards/accuracies & margins)"
        )
        return []
