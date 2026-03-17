# prepare_dataset.py

import re
from datasets import load_dataset
import random
import numpy as np
import tiktoken


# -----------------------------
# Config
# -----------------------------
OPENWEBTEXT_FRACTION = 0.02   # 2% of OpenWebText
MIN_LENGTH = 50
SEED = 42

TEXT_FILE = "dataset.txt"
BIN_FILE = "dataset.bin"
TOKENS_FILE = "tokens.npy"


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


# -----------------------------
# Preprocessing pipeline
# -----------------------------
def preprocess(wiki_texts, owt_texts):
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
def save_text(texts):
    print(f"Saving text to {TEXT_FILE}...")

    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n<|endoftext|>\n")

    print("Text file saved.")


# -----------------------------
# Tokenize + Save
# -----------------------------
def tokenize_and_save(texts):
    print("Tokenizing...")
    enc = tiktoken.get_encoding("gpt2")

    full_text = "\n<|endoftext|>\n".join(texts)
    tokens = enc.encode(full_text, allowed_special={"<|endoftext|>"})

    print(f"Total tokens: {len(tokens):,}")

    tokens_np = np.array(tokens, dtype=np.uint16)

    print(f"Saving tokens to {TOKENS_FILE}...")
    np.save(TOKENS_FILE, tokens_np)

    print(f"Saving binary to {BIN_FILE}...")
    tokens_np.tofile(BIN_FILE)

    # sanity check
    print("\nSample decoded text:\n")
    print(enc.decode(tokens[:200]))

    print("\nToken + binary files saved.")


# -----------------------------
# Main
# -----------------------------
def main():
    wiki_texts, owt_texts = load_data()
    cleaned_texts = preprocess(wiki_texts, owt_texts)

    save_text(cleaned_texts)
    tokenize_and_save(cleaned_texts)

    print("\nAll outputs generated successfully.")


if __name__ == "__main__":
    main()
    
