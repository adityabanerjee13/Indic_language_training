"""
Scoring utilities for IndicGenBench.

Per the benchmark's own evaluation protocol:
  - CrossSum-IN and Flores-IN (cross-lingual summarization / translation)
    are scored with Character-F1 (chrF, Popovic 2015), since token-level
    metrics like ROUGE/BLEU are unreliable for low-resource languages.
  - XQuAD-IN and XORQA-IN (QA tasks) are scored with SQuAD-style Token-F1,
    to stay consistent with existing QA literature.
  - Flores-IN is reported in both translation directions: en->xx (enxx)
    and xx->en (xxen).

Every function here takes already-produced (prediction, reference) pairs -
dataset loading and model inference happen elsewhere.

Requires: sacrebleu (pip install sacrebleu)
"""

import re
import string
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Union

from sacrebleu.metrics import CHRF

Reference = Union[str, Sequence[str]]


# --------------------------------------------------------------------------
# SQuAD-style Token-F1 - XQuAD-IN, XORQA-IN
# --------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """SQuAD normalization: lowercase, strip articles/punctuation/extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between a single prediction and a single ground-truth answer."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1_score(prediction: str, ground_truths: Reference) -> float:
    """Max Token-F1 over one or more acceptable ground-truth answers (SQuAD-style)."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return max(token_f1(prediction, gt) for gt in ground_truths)


def evaluate_token_f1(examples: Iterable[dict]) -> Dict[str, float]:
    """
    Aggregate Token-F1 for XQuAD-IN / XORQA-IN.

    examples: iterable of {"lang": str, "prediction": str, "references": str | list[str]}
    Returns per-language average Token-F1 plus a macro-averaged "overall".
    """
    by_lang: Dict[str, List[float]] = defaultdict(list)
    for ex in examples:
        by_lang[ex["lang"]].append(token_f1_score(ex["prediction"], ex["references"]))

    results = {lang: sum(scores) / len(scores) for lang, scores in by_lang.items()}
    results["overall"] = sum(results.values()) / len(results)
    return results


# --------------------------------------------------------------------------
# Character-F1 / chrF (Popovic, 2015) - CrossSum-IN, Flores-IN
# --------------------------------------------------------------------------

def chrf_score(prediction: str, references: Reference, word_order: int = 0) -> float:
    """
    Sentence-level chrF for a single prediction.

    word_order=0 reproduces the original chrF metric (Popovic, 2015); pass
    word_order=2 if you specifically want chrF++ instead.
    """
    if isinstance(references, str):
        references = [references]
    return CHRF(word_order=word_order).sentence_score(prediction, references).score


def evaluate_chrf(examples: Iterable[dict], word_order: int = 0) -> Dict[str, float]:
    """
    Aggregate chrF for CrossSum-IN (and single-direction Flores-IN slices).

    examples: iterable of {"lang": str, "prediction": str, "references": str | list[str]}
    Scores each language corpus-level (chrF is defined over a corpus, not
    averaged per-sentence), then macro-averages languages into "overall".
    """
    hyps_by_lang: Dict[str, List[str]] = defaultdict(list)
    refs_by_lang: Dict[str, List[List[str]]] = defaultdict(list)

    for ex in examples:
        refs = ex["references"]
        if isinstance(refs, str):
            refs = [refs]
        hyps_by_lang[ex["lang"]].append(ex["prediction"])
        refs_by_lang[ex["lang"]].append(refs)

    metric = CHRF(word_order=word_order)
    results = {}
    for lang, hyps in hyps_by_lang.items():
        refs = refs_by_lang[lang]
        # sacrebleu expects references transposed to [ref_slot][sentence];
        # pad missing alternate references by repeating the first one.
        num_refs = max(len(r) for r in refs)
        padded = [r + [r[0]] * (num_refs - len(r)) for r in refs]
        transposed = [list(slot) for slot in zip(*padded)]
        results[lang] = metric.corpus_score(hyps, transposed).score

    results["overall"] = sum(results.values()) / len(results)
    return results


def evaluate_flores_chrf(examples: Iterable[dict], word_order: int = 0) -> Dict[str, Dict[str, float]]:
    """
    Aggregate chrF for Flores-IN, reported separately per translation direction.

    examples: iterable of {"lang": str, "direction": "enxx" | "xxen",
                            "prediction": str, "references": str | list[str]}
    Returns {"enxx": {lang: score, ..., "overall": score}, "xxen": {...}}
    """
    by_direction: Dict[str, List[dict]] = defaultdict(list)
    for ex in examples:
        by_direction[ex["direction"]].append(ex)

    return {
        direction: evaluate_chrf(direction_examples, word_order=word_order)
        for direction, direction_examples in by_direction.items()
    }


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------

TASK_METRICS = {
    "xquad_in": evaluate_token_f1,
    "xorqa_in": evaluate_token_f1,
    "crosssum_in": evaluate_chrf,
    "flores_in": evaluate_flores_chrf,
}


def run_benchmark(task: str, examples: Iterable[dict]) -> Dict:
    """Score `examples` with the metric IndicGenBench specifies for `task`."""
    if task not in TASK_METRICS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {list(TASK_METRICS)}")
    return TASK_METRICS[task](examples)
