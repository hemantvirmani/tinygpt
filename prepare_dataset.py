# prepare_dataset.py

import re
import argparse
from datasets import load_dataset
import random
import numpy as np
import tiktoken
import pandas as pd


# -----------------------------
# Config
# -----------------------------
OPENWEBTEXT_FRACTION = 0.02   # 2% of OpenWebText
MIN_LENGTH = 50
SEED = 42

TEXT_FILE = "dataset.txt"
BIN_FILE = "dataset.bin"


# -----------------------------
# Cleaning functions
# -----------------------------
def clean_text(text):
    text = text.strip()

    # remove section headers like == Title ==
    if text.startswith("=") and text.endswith("="):
        return ""

    # normalize weird tokens
    text = text.replace(" @-@", "-")
    text = text.replace(" @,@ ", ", ")
    text = text.replace(" @.@ ", ". ")

    # collapse whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def is_good(text):
    if len(text) < MIN_LENGTH:
        return False
    if text.count(" ") < 5:
        return False
    return True


# -----------------------------
# Load datasets
# -----------------------------
def load_data():
    print("Loading WikiText-103...")
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    print("Loading OpenWebText (sample)...")
    owt = load_dataset(
        "openwebtext",
        split=f"train[:{int(OPENWEBTEXT_FRACTION * 100)}%]"
    )

    return wiki["text"], owt["text"]


def load_parquet_files(paths):
    """Load text from local FineWeb-Edu parquet files."""
    texts = []
    for path in paths:
        print(f"Loading parquet file: {path}...")
        df = pd.read_parquet(path, columns=["text"])
        texts.extend(df["text"].tolist())
    print(f"Loaded {len(texts):,} samples from parquet files.")
    return texts


# -----------------------------
# Preprocessing pipeline
# -----------------------------
def preprocess(wiki_texts, owt_texts, extra_texts=None):
    print("Cleaning text...")

    all_texts = []

    for t in wiki_texts:
        ct = clean_text(t)
        if is_good(ct):
            all_texts.append(ct)

    for t in owt_texts:
        ct = clean_text(t)
        if is_good(ct):
            all_texts.append(ct)

    if extra_texts:
        for t in extra_texts:
            ct = clean_text(t)
            if is_good(ct):
                all_texts.append(ct)

    print(f"After cleaning: {len(all_texts)} samples")

    # Deduplicate (keep one copy)
    print("Deduplicating (keeping one copy)...")
    deduped = list(set(all_texts))

    print(f"After deduplication: {len(deduped)} samples")

    # Shuffle
    random.seed(SEED)
    random.shuffle(deduped)

    return deduped


# -----------------------------
# Save text
# -----------------------------
def save_text(texts, path=TEXT_FILE):
    print(f"Saving text to {path}...")
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n<|endoftext|>\n")
    print("Text file saved.")


# -----------------------------
# Tokenize + Save
# -----------------------------
def tokenize_and_save(texts, bin_file=BIN_FILE):
    print("Tokenizing...")
    enc = tiktoken.get_encoding("gpt2")

    full_text = "\n<|endoftext|>\n".join(texts)
    tokens = enc.encode(full_text, allowed_special={"<|endoftext|>"})

    print(f"Total tokens: {len(tokens):,}")

    tokens_np = np.array(tokens, dtype=np.uint16)

    print(f"Saving binary to {bin_file}...")
    tokens_np.tofile(bin_file)

    # sanity check
    print("\nSample decoded text:\n")
    print(enc.decode(tokens[:200]))

    print("\nToken + binary files saved.")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare dataset (wiki + owt). Optionally merge FineWeb-Edu parquet files into a second dataset.")
    parser.add_argument(
        "--hf-files",
        type=str,
        default="",
        help="Comma-separated paths to local FineWeb-Edu parquet files (e.g. 000_00000.parquet,001_00000.parquet). Produces dataset2.bin alongside the base dataset."
    )
    args = parser.parse_args()
    wiki_texts, owt_texts = load_data()

    if args.hf_files:
        parquet_paths = [p.strip() for p in args.hf_files.split(",") if p.strip()]
        fineweb_texts = load_parquet_files(parquet_paths)

        print("\n=== Building FineWeb-Edu dataset ===")
        cleaned = preprocess(wiki_texts, [], extra_texts=fineweb_texts)
        save_text(cleaned, path="dataset2.txt")
        tokenize_and_save(cleaned, bin_file="dataset2.bin")
    else:

        print("\n=== Building base dataset (wiki + owt) ===")
        cleaned = preprocess(wiki_texts, owt_texts)
        save_text(cleaned, path=TEXT_FILE)
        tokenize_and_save(cleaned, bin_file=BIN_FILE)

    print("\nDone.")


if __name__ == "__main__":
    main()
