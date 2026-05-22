"""
LoRA Fine-Tuning: TinyGPT on Alpaca Cleaned (52K)

Identical pipeline to finetune_alpaca.py, but instead of updating all 163M
parameters we inject low-rank adapters into the attention projections and train
only those — typically < 1% of total parameters.

LoRA recap:
  For each target Linear W (frozen), add  ΔW = B·A  where A is (r × in),
  B is (out × r), and the effective update is scaled by alpha/r.
  B is zero-initialized → model starts exactly at pretrained weights.

Setup required before running on Kaggle:
  1. Turn on Internet in the Kaggle notebook settings
  2. Select a GPU accelerator
  3. Run Step 1 shell commands below to clone the repo and download weights
"""

# =============================================================================
# Step 1: Kaggle Setup
# =============================================================================
# Run these shell commands on Kaggle before importing:
#
#   pip install -q tiktoken datasets
#   mkdir -p /kaggle/working/models
#   test -f /kaggle/working/tinygpt/tinygpt.py || \
#       git clone https://github.com/hemantvirmani/tinygpt /kaggle/working/tinygpt
#   wget -q --show-progress \
#       -O /kaggle/working/models/tinygpt_pretrained_weights.pt \
#       "https://huggingface.co/hemantvirmani/tinyGPT/resolve/main/pretraining/PyTorch%20native/tinygpt_pretrained_weights.pt"
#   ls -lh /kaggle/working/models/tinygpt_pretrained_weights.pt

# =============================================================================
# Step 2: Config
# =============================================================================

import sys
import os
import random
import torch
import tiktoken
from datasets import load_dataset

torch.set_float32_matmul_precision("high")

# --- Paths ---
TINYGPT_DIR         = "/kaggle/working/tinygpt"
MODEL_DIR           = "/kaggle/working/models"
WEIGHTS_PATH        = f"{MODEL_DIR}/tinygpt_pretrained_weights.pt"
LORA_CHECKPOINT     = f"{MODEL_DIR}/tinygpt_lora_checkpoint.pt"
LORA_WEIGHTS_PATH   = f"{MODEL_DIR}/tinygpt_lora_weights.pt"
LORA_PLOT_PATH      = f"{MODEL_DIR}/lora_loss_curve.png"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Fine-tuning config ---
MAX_SEQ_LEN              = 512
VAL_SPLIT                = 0.1
LOAD_LORA_CHECKPOINT     = False   # set True to resume

# --- LoRA config ---
LORA_RANK    = 8     # rank of the low-rank decomposition; try 4, 8, 16, 32
LORA_ALPHA   = 16.0  # scaling factor; effective scale = alpha / rank
# Layers to adapt — c_attn (QKV) and c_proj (output) in every attention block
LORA_TARGETS = ("c_attn", "c_proj")

USE_BF16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
device   = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  |  BF16: {USE_BF16}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# Step 3: Load Model and Inject LoRA
# =============================================================================

sys.path.insert(0, TINYGPT_DIR)
import tinygpt

hparams = tinygpt.Hyperparameters(
    lr=2e-4,                  # LoRA adapters can tolerate a slightly higher LR
    weight_decay=0.0,         # no weight decay on low-rank matrices
    warmup_iters=100,
    max_iters=3000,
    batch_size=4,
    effective_batch_size=32,  # accumulation_steps = 32/4 = 8
)

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token

state = tinygpt.State(
    tokenizer=enc,
    train_data=None,
    val_data=None,
    vocab_size=enc.n_vocab,
)
tinygpt.G_DROPOUT_PROB = 0.05
model = tinygpt.TinyGPT(state).to(device)

# Load pretrained weights (base model — all parameters)
state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

# Inject LoRA — freezes everything except the new adapter matrices
tinygpt.inject_lora(model, target_modules=LORA_TARGETS, rank=LORA_RANK, alpha=LORA_ALPHA)
model.train()

# =============================================================================
# Step 4: Load and Format Alpaca Cleaned (same as finetune_alpaca.py)
# =============================================================================

print("Loading Alpaca Cleaned (52K)...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"Total examples: {len(dataset)}")


def format_and_tokenize(example):
    instruction = example["instruction"].strip()
    input_text  = example["input"].strip()
    response    = example["output"].strip()

    if input_text:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{response}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{response}"
        )

    tokens = enc.encode_ordinary(text)
    tokens.append(EOT)
    if len(tokens) > MAX_SEQ_LEN:
        return None
    return tokens


print("Tokenizing dataset...")
all_tokens = []
skipped = 0
for ex in dataset:
    tokens = format_and_tokenize(ex)
    if tokens is None:
        skipped += 1
        continue
    all_tokens.append(tokens)

print(f"Kept: {len(all_tokens)} | Skipped (too long): {skipped}")

split_idx  = int(len(all_tokens) * (1 - VAL_SPLIT))
train_data = all_tokens[:split_idx]
val_data   = all_tokens[split_idx:]
print(f"Train: {len(train_data)} | Val: {len(val_data)}")


def _get_batch(split):
    data  = train_data if split == "train" else val_data
    batch = random.sample(data, hparams.batch_size)
    max_len = max(len(s) for s in batch)

    x_list, y_list, mask_list = [], [], []
    for tokens in batch:
        pad_len = max_len - len(tokens)
        padded  = tokens + [EOT] * pad_len
        x_list.append(padded[:-1])
        y_list.append(padded[1:])
        mask_list.append([1] * (len(tokens) - 1) + [0] * pad_len)

    x    = torch.tensor(x_list,    dtype=torch.long).to(device)
    y    = torch.tensor(y_list,    dtype=torch.long).to(device)
    mask = torch.tensor(mask_list, dtype=torch.float).to(device)
    return x, y, mask

# =============================================================================
# Step 5: Baseline test
# =============================================================================

test_prompts = [
    "What is photosynthesis?",
    "Explain the water cycle in simple terms.",
    "Who was the first emperor of Rome?",
]


def test_model(model, prompts, label=""):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    model.eval()
    for prompt in prompts:
        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        print(f"\nQ: {prompt}")
        print(f"A: {model.generate_text(start_text=full_prompt, max_tokens=150, temperature=0.7)}")
        print("-" * 40)
    model.train()


test_model(model, test_prompts, label="BASELINE (Before LoRA Fine-Tuning)")

# =============================================================================
# Step 6: Training Loop
# =============================================================================

tinygpt.CHECKPOINT_FILE = LORA_CHECKPOINT
tinygpt.LOAD_CHECKPOINT = LOAD_LORA_CHECKPOINT
tinygpt.PLOT_TITLE      = "Fine-Tuning Loss — TinyGPT LoRA on Alpaca"
tinygpt.PLOT_PATH       = LORA_PLOT_PATH

model.hparams = hparams
model.train_loop(get_batch_fn=_get_batch)
print("\nTraining complete!")

# =============================================================================
# Step 7: Post-training test
# =============================================================================

test_model(model, test_prompts, label="AFTER LoRA Fine-Tuning")

# =============================================================================
# Step 8: Save — two options
#
# Option A: Save only the LoRA adapter weights (tiny — just A and B matrices).
#           Load by: inject_lora() on a fresh base model, then load this dict.
#
# Option B: Merge LoRA into the base weights and save a single standalone model.
#           Larger file, but no inject_lora() needed at inference.
# =============================================================================

# --- Option A: adapter-only (recommended for sharing / iterating) ---
lora_state = {
    name: param
    for name, param in model.named_parameters()
    if "lora_A" in name or "lora_B" in name
}
torch.save(lora_state, LORA_WEIGHTS_PATH)
size_mb = os.path.getsize(LORA_WEIGHTS_PATH) / 1024 / 1024
print(f"\nSaved LoRA adapter weights: {LORA_WEIGHTS_PATH} ({size_mb:.1f} MB)")
print(f"Adapter params: {sum(p.numel() for p in lora_state.values()):,}")

# --- Option B: merge and save full model ---
def merge_lora(model: torch.nn.Module) -> None:
    """Fold LoRA deltas into the base weight in-place, then remove the adapters."""
    for module in model.modules():
        if isinstance(module, tinygpt.LoRALinear):
            delta = (module.lora_B @ module.lora_A) * module.scale
            module.linear.weight.data += delta
            module.linear.weight.requires_grad_(True)
            module.lora_A.requires_grad_(False)
            module.lora_B.requires_grad_(False)


merged_path = f"{MODEL_DIR}/tinygpt_lora_merged_weights.pt"
best_ckpt   = torch.load(LORA_CHECKPOINT, map_location=device, weights_only=False)
model.load_state_dict(best_ckpt["model_state"])
merge_lora(model)
torch.save(model.state_dict(), merged_path)
size_mb = os.path.getsize(merged_path) / 1024 / 1024
print(f"Saved merged model weights: {merged_path} ({size_mb:.0f} MB)")
