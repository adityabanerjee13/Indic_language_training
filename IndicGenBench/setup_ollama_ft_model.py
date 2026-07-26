"""
Register the fine-tuned CPT checkpoint (adityabanerjee13/qwen2.5-0.5b-indic-cpt)
with a local Ollama server, so it can be benchmarked with the *existing*
run_qwen_benchmark.py exactly like the baseline qwen2.5:0.5b model was.

run_qwen_benchmark.py already takes --model as a plain Ollama tag and calls
query_ollama() with it - no new benchmark-running code is needed. This
script only handles the missing step: getting a plain HF safetensors
checkpoint (not GGUF) into Ollama.

Two ways Ollama can pick up a HF repo, tried in order:
  1. Direct pull (`ollama pull hf.co/<repo>`) - works out of the box on
     recent Ollama versions for common architectures (Qwen2 included), no
     manual conversion needed.
  2. Local Modelfile (`ollama create <name> -f Modelfile` with
     `FROM <local-dir>`) pointing at a `snapshot_download`-ed local copy of
     the repo - fallback if (1) fails (e.g. older Ollama version without
     direct Safetensors import support).

Requires: Ollama installed and its daemon running locally; huggingface_hub
for the fallback path.

Usage:
    python setup_ollama_ft_model.py
    python setup_ollama_ft_model.py --repo-id adityabanerjee13/qwen2.5-0.5b-indic-cpt --name qwen2.5-0.5b-indic-cpt

Then run the benchmark with the existing script, e.g.:
    python run_qwen_benchmark.py --model <tag printed below> --output-dir qwen_ft_ollama_benchmark_results
"""

import argparse
import subprocess
from pathlib import Path

DEFAULT_REPO_ID = "adityabanerjee13/qwen2.5-0.5b-indic-cpt"
DEFAULT_MODEL_NAME = "qwen2.5-0.5b-indic-cpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF repo id of the fine-tuned checkpoint.")
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME, help="Local Ollama model tag to register under (fallback path only).")
    parser.add_argument("--local-dir", default=None, help="Where to download the HF snapshot for the fallback path (default: ./.ollama_import_<name>).")
    return parser.parse_args()


def ollama_model_exists(tag: str) -> bool:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return tag in result.stdout


def try_direct_pull(repo_id: str) -> bool:
    print(f"Trying direct pull: ollama pull hf.co/{repo_id}")
    result = subprocess.run(["ollama", "pull", f"hf.co/{repo_id}"])
    return result.returncode == 0


def patch_rope_config(local_dir: Path) -> None:
    """
    Work around an Ollama 0.32.3 GGUF-conversion bug: recent `transformers`
    (5.x) versions write rope settings nested under `rope_parameters`
    (`{"rope_theta": ..., "rope_type": ...}`) instead of the older flat
    top-level `rope_theta` key. Ollama's converter only reads the old flat
    key, silently falls back to a wrong default, and produces a corrupted
    model that generates pure garbage (e.g. "@@@@@@@@...") regardless of
    prompt or language. Writing the flat key back in fixes it - confirmed
    by testing generation before/after on this exact checkpoint.
    """
    import json

    config_path = local_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if "rope_theta" in config:
        return  # already flat (older transformers save, or already patched)

    rope_params = config.get("rope_parameters") or {}
    config["rope_theta"] = rope_params.get("rope_theta", 1000000.0)
    config["rope_scaling"] = None

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Patched {config_path} with flat rope_theta={config['rope_theta']} (Ollama conversion workaround).")


def create_from_local_snapshot(repo_id: str, name: str, local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} to {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))

    patch_rope_config(local_dir)

    modelfile_path = local_dir / "Modelfile"
    modelfile_path.write_text(f"FROM {local_dir}\n", encoding="utf-8")

    print(f"Creating Ollama model '{name}' from {local_dir} ...")
    subprocess.run(["ollama", "create", name, "-f", str(modelfile_path)], check=True)


def main() -> None:
    args = parse_args()

    if ollama_model_exists(args.name):
        print(f"'{args.name}' is already registered in Ollama - nothing to do.")
        print(f"\nRun: python run_qwen_benchmark.py --model {args.name} --output-dir qwen_ft_ollama_benchmark_results")
        return

    direct_tag = f"hf.co/{args.repo_id}"
    if ollama_model_exists(direct_tag):
        print(f"'{direct_tag}' is already registered in Ollama - nothing to do.")
        print(f"\nRun: python run_qwen_benchmark.py --model {direct_tag} --output-dir qwen_ft_ollama_benchmark_results")
        return

    if try_direct_pull(args.repo_id):
        print(f"\nPulled successfully as '{direct_tag}'.")
        print(f"\nRun: python run_qwen_benchmark.py --model {direct_tag} --output-dir qwen_ft_ollama_benchmark_results")
        return

    print("\nDirect pull failed (older Ollama version, or no direct Safetensors import support).")
    print("Falling back to local download + `ollama create` ...")
    local_dir = Path(args.local_dir) if args.local_dir else Path.cwd() / f".ollama_import_{args.name}"
    local_dir.mkdir(parents=True, exist_ok=True)
    create_from_local_snapshot(args.repo_id, args.name, local_dir)

    print(f"\nCreated successfully as '{args.name}'.")
    print(f"\nRun: python run_qwen_benchmark.py --model {args.name} --output-dir qwen_ft_ollama_benchmark_results")


if __name__ == "__main__":
    main()
