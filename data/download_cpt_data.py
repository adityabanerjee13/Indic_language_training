"""
Download a budget-sized CPT corpus from AI4Bharat's IndicCorp v2.

Pulls a fixed-size byte slice (via HTTP range requests) of each of Sarvam-1's
10 supported Indic languages, rather than the full per-language files
(which range from ~2GB to ~27GB each, ~233GB total for the whole corpus).

Output: data/CPT/<lang>_<Language>_200Mtok_slice.txt  (~1.4GB each, ~14GB total)

Sizing rationale: ~2B tokens total / 10 languages = 200M tokens/language.
Indic scripts run ~5-7 bytes/token in UTF-8 (multi-byte characters), so we
pull ~1.4GB per language (buffered above the ~1.2GB estimate at 6 bytes/token)
to comfortably cover 200M tokens after tokenization with whichever model
tokenizer is eventually used. Trim to an exact token count during the
tokenization step, once a specific tokenizer is chosen.
"""

import os
import urllib.request

BASE_URL = "https://huggingface.co/datasets/ai4bharat/IndicCorpV2/resolve/main/data"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CPT")

# (IndicCorpV2 file code, display name, output filename code)
# Hindi is split into hi-1/hi-2/hi-3 in the source repo; hi-1 alone is far
# larger (26.6GB) than our slice needs, so we only ever read from it.
LANGUAGES = [
    ("bn", "Bengali", "bn"),
    ("gu", "Gujarati", "gu"),
    ("hi-1", "Hindi", "hi"),
    ("kn", "Kannada", "kn"),
    ("ml", "Malayalam", "ml"),
    ("mr", "Marathi", "mr"),
    ("or", "Odia", "or"),
    ("pa", "Punjabi", "pa"),
    ("ta", "Tamil", "ta"),
    ("te", "Telugu", "te"),
]

BYTES_PER_LANGUAGE = 1_400_000_000  # ~1.4GB slice -> ~200M tokens at ~6-7 bytes/token


def download_slice(url: str, dest_path: str, num_bytes: int) -> int:
    """Download the first `num_bytes` of `url` to `dest_path` via an HTTP Range request."""
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{num_bytes - 1}"})
    with urllib.request.urlopen(req) as response:
        if response.status not in (200, 206):
            raise RuntimeError(f"Unexpected status {response.status} for {url}")
        with open(dest_path, "wb") as out_file:
            written = 0
            while True:
                chunk = response.read(1 << 20)  # 1MB chunks
                if not chunk:
                    break
                out_file.write(chunk)
                written += len(chunk)
    return written


def trim_to_last_newline(path: str) -> None:
    """A byte-range cutoff usually lands mid-line/mid-character. Truncate the
    file at the last complete newline so it ends on a clean line boundary."""
    with open(path, "rb") as f:
        data = f.read()
    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        print(f"  WARNING: no newline found in {path}, leaving file as-is")
        return
    if last_nl + 1 < len(data):
        with open(path, "r+b") as f:
            f.truncate(last_nl + 1)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file_code, display_name, out_code in LANGUAGES:
        filename = f"{out_code}_{display_name}_200Mtok_slice.txt"
        dest_path = os.path.join(OUTPUT_DIR, filename)
        url = f"{BASE_URL}/{file_code}.txt"

        print(f"Downloading {display_name} ({file_code}) -> {filename}")
        size = download_slice(url, dest_path, BYTES_PER_LANGUAGE)
        trim_to_last_newline(dest_path)
        final_size = os.path.getsize(dest_path)
        print(f"  {size:,} bytes downloaded, trimmed to {final_size:,} bytes")

    print("\nDone. Files written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
