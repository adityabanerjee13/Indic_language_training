"""
Download the IndicGenBench evaluation suite from Hugging Face.

Pulls 4 Google IndicGenBench datasets, each into its own subfolder, to give
a spread of eval tasks (QA, cross-lingual QA, summarization, translation)
for measuring whether Indic CPT/fine-tuning actually moved the needle.

Output:
  IndicGenBench/xquad_in/      - XQuAD-in (extractive QA)
  IndicGenBench/crosssum_in/   - CrossSum-in (cross-lingual summarization)
  IndicGenBench/xorqa_in/      - XOR-QA-in (cross-lingual open-retrieval QA)
  IndicGenBench/flores_in/     - Flores-in (translation)

Requires: huggingface_hub (pip install huggingface_hub)
"""

import os

from huggingface_hub import snapshot_download

OUTPUT_ROOT = os.path.dirname(os.path.abspath(__file__))

# (HF repo_id, local subfolder name)
DATASETS = [
    ("google/IndicGenBench_xquad_in", "xquad_in"),
    ("google/IndicGenBench_crosssum_in", "crosssum_in"),
    ("google/IndicGenBench_xorqa_in", "xorqa_in"),
    ("google/IndicGenBench_flores_in", "flores_in"),
]


def main() -> None:
    for repo_id, folder in DATASETS:
        local_dir = os.path.join(OUTPUT_ROOT, folder)
        print(f"Downloading {repo_id} -> {folder}")
        path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir)
        print(f"  Done: {path}")

    print("\nDone. Datasets written under:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
