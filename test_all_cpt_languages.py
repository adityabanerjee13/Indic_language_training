"""
Run every model in ollama_inference_test.py against every language present in
data/CPT, and report which models actually generate in-script text (vs.
falling back to English or another script) per language.

Requires: an Ollama server running locally with all MODELS pulled (see
ollama_inference_test.py).

Usage:
    python test_all_cpt_languages.py
    python test_all_cpt_languages.py --output results.json
"""

import argparse
import json
import re
import sys
import time

from ollama_inference_test import MODELS, PROMPTS, query_ollama

# Unicode block per language script (matches the languages under data/CPT).
SCRIPT_RANGES = {
    "hi": r"ऀ-ॿ",  # Devanagari
    "mr": r"ऀ-ॿ",  # Devanagari
    "bn": r"ঀ-৿",  # Bengali
    "gu": r"઀-૿",  # Gujarati
    "pa": r"਀-੿",  # Gurmukhi
    "or": r"଀-୿",  # Odia
    "ta": r"஀-௿",  # Tamil
    "te": r"ఀ-౿",  # Telugu
    "kn": r"ಀ-೿",  # Kannada
    "ml": r"ഀ-ൿ",  # Malayalam
}

# CPT filenames -> language code, for reference/labeling only.
CPT_LANGS = {
    "bn": "Bengali", "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada",
    "ml": "Malayalam", "mr": "Marathi", "or": "Odia", "pa": "Punjabi",
    "ta": "Tamil", "te": "Telugu",
}


def script_share(text: str, lang: str) -> float:
    """Fraction of non-whitespace chars in `text` that fall in `lang`'s script block."""
    stripped = re.sub(r"\s+", "", text)
    if not stripped:
        return 0.0
    pattern = re.compile(f"[{SCRIPT_RANGES[lang]}]")
    in_script = len(pattern.findall(stripped))
    return in_script / len(stripped)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=str, default="cpt_lang_results.json")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--langs", nargs="*", default=None, choices=list(CPT_LANGS.keys()))
    args = parser.parse_args()

    models = args.models or MODELS
    langs = args.langs or list(CPT_LANGS.keys())

    results = []
    for lang in langs:
        prompt = PROMPTS[lang]
        for model in models:
            print(f"[{lang}] {model} ...", flush=True)
            try:
                data = query_ollama(model, prompt)
                response = data.get("response", "").strip()
                share = script_share(response, lang)
                results.append({
                    "lang": lang, "model": model, "prompt": prompt,
                    "response": response, "script_share": share,
                    "elapsed_s": data["_elapsed_s"],
                })
                print(f"    script_share={share:.2f}  resp={response[:60]!r}", flush=True)
            except Exception as e:
                results.append({"lang": lang, "model": model, "prompt": prompt, "error": repr(e)})
                print(f"    ERROR: {e!r}", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(results)} results to {args.output}")

    # Summary matrix: rows=models, cols=langs, cell = script_share (or "ERR")
    print("\n" + "=" * 100)
    print(f"{'model':45s}" + "".join(f"{l:>7s}" for l in langs))
    for model in models:
        row = f"{model:45.45s}"
        for lang in langs:
            match = next((r for r in results if r["model"] == model and r["lang"] == lang), None)
            if match is None or "error" in match:
                row += f"{'ERR':>7s}"
            else:
                row += f"{match['script_share']*100:6.0f}%"
        print(row)


if __name__ == "__main__":
    main()
