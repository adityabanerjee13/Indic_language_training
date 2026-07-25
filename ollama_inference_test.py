"""
Load the 9 candidate base models via Ollama and run a test inference prompt
against each, to sanity-check that all models pull and respond correctly.

Requires: an Ollama server running locally (default http://localhost:11434)
and the models already pulled (`ollama pull <tag>` for each entry below,
or `ollama pull hf.co/<repo>[:quant]` for the two Hugging Face GGUF models).

Usage:
    python ollama_inference_test.py
    python ollama_inference_test.py --prompt "Custom prompt here"
    python ollama_inference_test.py --lang hi     # use the built-in Hindi prompt
"""

import argparse
import json
import sys
import time
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "gemma2:2b",
    "tinyllama",
    "stablelm2",
    "smollm:1.7b",
    "llama3.2:1b",
    "hf.co/MuffinMich/MiniCPM-2B-dpo-bf16-Q4_K_M-GGUF",
    "hf.co/bartowski/sarvam-1-GGUF:Q4_K_M",
]

PROMPTS = {
    "en": "What is the capital of France? Answer in one short sentence.",
    "hi": "भारत की राजधानी क्या है? एक वाक्य में उत्तर दें।",
    "bn": "ভারতের রাজধানী কী? একটি বাক্যে উত্তর দিন।",
    "gu": "ભારતની રાજધાની શું છે? એક વાક્યમાં જવાબ આપો.",
    "kn": "ಭಾರತದ ರಾಜಧಾನಿ ಯಾವುದು? ಒಂದು ವಾಕ್ಯದಲ್ಲಿ ಉತ್ತರಿಸಿ.",
    "ml": "ഇന്ത്യയുടെ തലസ്ഥാനം ഏതാണ്? ഒരു വാക്യത്തിൽ ഉത്തരം നൽകുക.",
    "mr": "भारताची राजधानी काय आहे? एका वाक्यात उत्तर द्या.",
    "or": "ଭାରତର ରାଜଧାନୀ କ'ଣ? ଗୋଟିଏ ବାକ୍ୟରେ ଉତ୍ତର ଦିଅନ୍ତୁ।",
    "pa": "ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ ਕੀ ਹੈ? ਇੱਕ ਵਾਕ ਵਿੱਚ ਜਵਾਬ ਦਿਓ।",
    "ta": "இந்தியாவின் தலைநகரம் என்ன? ஒரு வாக்கியத்தில் பதிலளிக்கவும்.",
    "te": "భారతదేశ రాజధాని ఏమిటి? ఒక వాక్యంలో సమాధానం ఇవ్వండి.",
}


def query_ollama(model: str, prompt: str, num_predict: int = 80, timeout: int = 120) -> dict:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": num_predict},
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    data["_elapsed_s"] = time.time() - start
    return data


def main() -> None:
    # Force UTF-8 stdout so non-Latin scripts (Devanagari, etc.) print
    # correctly regardless of the terminal's default codepage.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt to use for all models")
    parser.add_argument("--lang", choices=list(PROMPTS.keys()), default="en", help="Use a built-in prompt (en/hi)")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model tags to test (default: all)")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results as JSON")
    args = parser.parse_args()

    prompt = args.prompt or PROMPTS[args.lang]
    models = args.models or MODELS

    results = []
    for model in models:
        print("=" * 60)
        print(f"MODEL: {model}")
        print("=" * 60)
        try:
            data = query_ollama(model, prompt)
            response_text = data.get("response", "").strip()
            print(f"RESPONSE: {response_text}")
            print(f"(took {data['_elapsed_s']:.1f}s, eval_count={data.get('eval_count', '?')})")
            results.append({"model": model, "prompt": prompt, "response": response_text,
                             "elapsed_s": data["_elapsed_s"], "eval_count": data.get("eval_count")})
        except Exception as e:
            print(f"ERROR: {e!r}")
            results.append({"model": model, "prompt": prompt, "error": repr(e)})
        print()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
