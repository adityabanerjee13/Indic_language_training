"""
Run qwen2.5:0.5b (via a local Ollama server) over the 4 IndicGenBench tasks
and score it with metrics.py.

Tasks and their single-shot prompt templates (each template is the whole
prompt sent to /api/generate - Ollama's generate endpoint has no separate
system/user roles, so instructions + input are combined into one string):

  xquad_in    - in-language extractive QA. Context and question are both in
                the target language; the model must copy the answer span.
  xorqa_in    - cross-lingual open-retrieval QA. Context is English, the
                question is in the target language, and the answer must be
                produced in the target language (scored against
                translated_answers).
  crosssum_in - cross-lingual summarization. English article in, one-sentence
                summary in the target language out.
  flores_in   - translation, evaluated in both directions (enxx and xxen).

Only the 10 languages under data/CPT are benchmarked, since those are the
languages the rest of this project cares about. Each task/language pulls a
random sample (default 20, seeded) from its *_test.json split rather than
the full split (some splits run 500-1200 examples/language; with 10
languages x 4 tasks x 2 flores directions that's 50x the per-language count
in raw model calls, which is impractical to run in full locally).

Requires: an Ollama server running locally with qwen2.5:0.5b pulled.

Usage:
    python run_qwen_benchmark.py
    python run_qwen_benchmark.py --num-examples 50 --tasks xquad_in flores_in
    python run_qwen_benchmark.py --langs hi bn ta
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ollama_inference_test import query_ollama  # noqa: E402

from metrics import run_benchmark  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

LANG_NAMES = {
    "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi",
    "ta": "Tamil", "te": "Telugu",
}
LANGS = list(LANG_NAMES.keys())

# Single combined instruction+input prompt per task - no separate system role.
XQUAD_TEMPLATE = (
    "Read the passage below, written in {lang}, and answer the question that "
    "follows it, also written in {lang}. Copy the exact short answer phrase "
    "from the passage. Output only that answer phrase in {lang} - no "
    "explanation, no translation, nothing else.\n\n"
    "Passage: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

XORQA_TEMPLATE = (
    "Read the English passage below, then answer the question that follows "
    "it. The question is written in {lang}. Output only a short answer "
    "written in {lang} - do not answer in English, do not explain.\n\n"
    "Passage: {context}\n\n"
    "Question ({lang}): {question}\n\n"
    "Answer ({lang}):"
)

CROSSSUM_TEMPLATE = (
    "Summarize the English news article below in exactly one sentence "
    "written in {lang}. Output only that one-sentence {lang} summary - no "
    "preamble, no English.\n\n"
    "Article: {text}\n\n"
    "One-sentence summary ({lang}):"
)

FLORES_ENXX_TEMPLATE = (
    "Translate the following English sentence into {lang}. Output only the "
    "{lang} translation - no explanation, no English.\n\n"
    "English: {source}\n\n"
    "{lang} translation:"
)

FLORES_XXEN_TEMPLATE = (
    "Translate the following {lang} sentence into English. Output only the "
    "English translation - no explanation.\n\n"
    "{lang}: {source}\n\n"
    "English translation:"
)

# Contexts/articles are truncated to keep per-call latency reasonable on a
# local 0.5B model; short-answer QA passages rarely need this, summarization
# articles often do.
MAX_CONTEXT_CHARS = 2000
MAX_ARTICLE_CHARS = 3000


def load_examples(path: str, n: int, seed: int) -> list:
    with open(path, encoding="utf-8") as f:
        examples = json.load(f)["examples"]
    if len(examples) <= n:
        return examples
    return random.Random(seed).sample(examples, n)


def run_xquad(lang: str, n: int, seed: int, model: str) -> list:
    path = os.path.join(ROOT, "xquad_in", f"xquad_{lang}_test.json")
    if not os.path.exists(path):
        return []
    records = []
    for ex in load_examples(path, n, seed):
        prompt = XQUAD_TEMPLATE.format(
            lang=LANG_NAMES[lang], context=ex["context"][:MAX_CONTEXT_CHARS], question=ex["question"]
        )
        data = query_ollama(model, prompt, num_predict=30)
        prediction = data.get("response", "").strip()
        references = [a["text"] for a in ex["answers"]]
        records.append({
            "id": ex.get("id"), "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": references,
        })
    return records


def run_xorqa(lang: str, n: int, seed: int, model: str) -> list:
    path = os.path.join(ROOT, "xorqa_in", f"xorqa_{lang}_test.json")
    if not os.path.exists(path):
        return []
    records = []
    for ex in load_examples(path, n, seed):
        prompt = XORQA_TEMPLATE.format(
            lang=LANG_NAMES[lang], context=ex["context"][:MAX_CONTEXT_CHARS], question=ex["question"]
        )
        data = query_ollama(model, prompt, num_predict=30)
        prediction = data.get("response", "").strip()
        gold = ex.get("translated_answers") or ex["answers"]
        references = [a["text"] for a in gold]
        records.append({
            "id": None, "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": references,
        })
    return records


def run_crosssum(lang: str, n: int, seed: int, model: str) -> list:
    path = os.path.join(ROOT, "crosssum_in", f"crosssum_english-{lang}_test.json")
    if not os.path.exists(path):
        return []
    records = []
    for ex in load_examples(path, n, seed):
        prompt = CROSSSUM_TEMPLATE.format(lang=LANG_NAMES[lang], text=ex["text"][:MAX_ARTICLE_CHARS])
        data = query_ollama(model, prompt, num_predict=100)
        prediction = data.get("response", "").strip()
        records.append({
            "id": None, "lang": lang, "prompt": prompt,
            "prediction": prediction, "references": [ex["summary"]],
        })
    return records


def run_flores(lang: str, n: int, seed: int, model: str) -> list:
    records = []
    for direction, filename, template in [
        ("enxx", f"flores_en_{lang}_test.json", FLORES_ENXX_TEMPLATE),
        ("xxen", f"flores_{lang}_en_test.json", FLORES_XXEN_TEMPLATE),
    ]:
        path = os.path.join(ROOT, "flores_in", filename)
        if not os.path.exists(path):
            continue
        for ex in load_examples(path, n, seed):
            prompt = template.format(lang=LANG_NAMES[lang], source=ex["source"])
            data = query_ollama(model, prompt, num_predict=200)
            prediction = data.get("response", "").strip()
            records.append({
                "id": None, "lang": lang, "direction": direction, "prompt": prompt,
                "prediction": prediction, "references": [ex["target"]],
            })
    return records


TASK_RUNNERS = {
    "xquad_in": run_xquad,
    "xorqa_in": run_xorqa,
    "crosssum_in": run_crosssum,
    "flores_in": run_flores,
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--num-examples", type=int, default=20, help="Sampled examples per task/language (per direction for flores)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="*", default=list(TASK_RUNNERS.keys()), choices=list(TASK_RUNNERS.keys()))
    parser.add_argument("--langs", nargs="*", default=LANGS, choices=LANGS)
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "qwen_benchmark_results"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_scores = {}
    for task in args.tasks:
        runner = TASK_RUNNERS[task]
        task_records = []
        for lang in args.langs:
            print(f"[{task}] {lang} ...", flush=True)
            start = time.time()
            try:
                records = runner(lang, args.num_examples, args.seed, args.model)
            except Exception as e:
                print(f"    ERROR loading/running {lang}: {e!r}", flush=True)
                continue
            task_records.extend(records)
            print(f"    {len(records)} examples in {time.time() - start:.1f}s", flush=True)

        out_path = os.path.join(args.output_dir, f"{task}_predictions.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(task_records, f, ensure_ascii=False, indent=2)

        scores = run_benchmark(task, task_records)
        all_scores[task] = scores
        print(f"\n=== {task} scores ===")
        print(json.dumps(scores, ensure_ascii=False, indent=2))
        print()

    scores_path = os.path.join(args.output_dir, "scores_summary.json")
    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)
    print(f"All scores saved to {scores_path}")


if __name__ == "__main__":
    main()
