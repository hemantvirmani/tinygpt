# prepare_dataset.py
# !pip install datasets tiktoken numpy

import re
from datasets import load_dataset
import random
import numpy as np
import tiktoken


# -----------------------------
# Config
# -----------------------------
OPENWEBTEXT_FRACTION = 2   # 2% of OpenWebText
FINEWEB_FRACTION = 10      # more than 7% - atleast - ~1 parquet file out of ~13 in sample-10BT
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
        split=f"train[:{OPENWEBTEXT_FRACTION}%]"
    )

    print(f"Loading FineWeb-Edu sample-10BT ({FINEWEB_FRACTION}%)...")
    fineweb = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split=f"train[:{FINEWEB_FRACTION}%]"
    )

    return wiki["text"], owt["text"], fineweb["text"]


# -----------------------------
# Preprocessing pipeline
# -----------------------------
def preprocess(wiki_texts, owt_texts, fineweb_texts=None):
    all_texts = []

    print("Cleaning WikiText...")
    for t in wiki_texts:
        ct = clean_text(t)
        if is_good(ct):
            all_texts.append(ct)

    print("Cleaning OpenWebText...")
    for t in owt_texts:
        ct = clean_text(t)
        if is_good(ct):
            all_texts.append(ct)

    if fineweb_texts:
        print("Cleaning FineWeb-Edu text...")
        for t in fineweb_texts:
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
    wiki_texts, owt_texts, fineweb_texts = load_data()

    cleaned = preprocess(wiki_texts, owt_texts, fineweb_texts)
    save_text(cleaned, path=TEXT_FILE)
    tokenize_and_save(cleaned, bin_file=BIN_FILE)

    print("\nDone.")

if __name__ == "__main__":
    main()


def save_to_google_drive():
    from google.colab import drive
    import shutil
    drive.mount('/content/drive')
    shutil.copy(BIN_FILE, '/content/drive/MyDrive/dataset.bin')
    print("Saved to Google Drive.")

# Runpod Instructions:
# Get the file ID from the shareable link of dataset.bin in your Drive:
#https://drive.google.com/file/d/THIS_PART_IS_THE_ID/view
#pip install gdown
#gdown "https://drive.google.com/uc?id=YOUR_FILE_ID"