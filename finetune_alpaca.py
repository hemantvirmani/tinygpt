"""
Instruction Fine-Tuning: TinyGPT on Alpaca Cleaned (52K)

Goal: Take the pretrained TinyGPT base model and instruction fine-tune it
using the Alpaca Cleaned dataset (52K examples), using a custom PyTorch training loop.

Setup required before running on Kaggle:
  1. Turn on Internet in the Kaggle notebook settings
  2. Select a GPU accelerator
  3. Run Step 1 to clone the TinyGPT repo and download the pretrained weights from Hugging Face
"""

# =============================================================================
# Step 1: Kaggle Setup and Download Base Model
# =============================================================================
# Run these shell commands on Kaggle before importing:
#
#   pip install -q tiktoken datasets
#   mkdir -p /kaggle/working/models
#   test -f /kaggle/working/tinygpt/tinygpt.py || \
#       git clone https://github.com/hemantvirmani/tinygpt /kaggle/working/tinygpt
#   wget -q --show-progress \
#       -O /kaggle/working/models/tinygpt_pretrained_weights.pt \
#       https://huggingface.co/hemantvirmani/tinyGPT/resolve/main/tinygpt_pretrained_weights.pt
#   ls -lh /kaggle/working/models/tinygpt_pretrained_weights.pt

# =============================================================================
# Step 2: Config
# =============================================================================

import sys
import os
import inspect
import random
import torch
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt
from datasets import load_dataset

# Match the pretraining script's matmul precision setting.
torch.set_float32_matmul_precision("high")

# --- Paths ---
TINYGPT_DIR  = "/kaggle/working/tinygpt"              # where tinygpt.py lives
MODEL_DIR    = "/kaggle/working/models"
WEIGHTS_PATH = f"{MODEL_DIR}/tinygpt_pretrained_weights.pt"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Fine-tuning Hyperparameters ---
BATCH_SIZE           = 4
EFFECTIVE_BATCH_SIZE = 64    # accumulation_steps = 64/4 = 16
MAX_STEPS            = 5000
LEARNING_RATE        = 1e-4  # lower than pretraining
WEIGHT_DECAY         = 0.01
GRAD_CLIP            = 1.0
WARMUP_STEPS         = 100
EVAL_STEPS           = 100   # evaluate every 100 steps
EVAL_ITERATIONS      = 50    # batches averaged during validation
MAX_SEQ_LEN          = 512   # shorter than pretraining 1024 — Alpaca examples are short
VAL_SPLIT            = 0.1
# BF16 requires Ampere (A100, RTX 3090+). Kaggle T4/P100 don't support it — auto-detect.
USE_BF16             = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"BF16: {USE_BF16}")

# =============================================================================
# Step 3: Load TinyGPT Model
# =============================================================================

sys.path.insert(0, TINYGPT_DIR)
import tinygpt

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token  # 50256 — used as padding and end-of-text

# Load model in TRAIN mode (not eval)
state = tinygpt.State(
    tokenizer=enc,
    train_data=None,
    val_data=None,
    vocab_size=enc.n_vocab
)
tinygpt.G_DROPOUT_PROB = 0.1   # add dropout regularization for fine-tuning
model = tinygpt.TinyGPT(state).to(device)

# Load pretrained weights (PyTorch native format).
# Strips _orig_mod. prefix if the checkpoint came from a torch.compile() run.
state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
if any(k.startswith("_orig_mod.") for k in state_dict):
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.train()  # switch to training mode

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded. Parameters: {total_params/1e6:.1f}M")

# =============================================================================
# Step 4: Load and Format Alpaca Cleaned
#
# The core of instruction fine-tuning — we define a prompt template that teaches
# the model what an instruction looks like, what a response looks like, and where
# it ends. The model learns this structure through repeated exposure.
# =============================================================================

print("Loading Alpaca Cleaned (52K)...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
print(f"Total examples: {len(dataset)}")
print(f"Sample: {dataset[0]}")


def format_and_tokenize(example):
    """Format an Alpaca example into instruction-response format and tokenize.

    Returns a list of token ids, or None if the example is too long.
    The EOT token marks the end of the response — critical for the model
    to learn where to stop generating.
    """
    instruction = example["instruction"].strip()
    input_text  = example["input"].strip()   # optional context (called "input" in Alpaca)
    response    = example["output"].strip()  # called "output" in Alpaca

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
    tokens.append(EOT)  # end of sequence marker

    if len(tokens) > MAX_SEQ_LEN:
        return None  # skip examples that are too long
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

print(f"Kept: {len(all_tokens)} examples | Skipped (too long): {skipped}")

# Train / val split
split_idx  = int(len(all_tokens) * (1 - VAL_SPLIT))
train_data = all_tokens[:split_idx]
val_data   = all_tokens[split_idx:]
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# Preview formatted example
print("\nFormatted example (decoded):")
print(enc.decode(train_data[0]))

# =============================================================================
# Step 5: Batch Sampler
#
# Since Alpaca examples have variable lengths, we pad shorter sequences to
# the longest in the batch. Padding tokens are masked out in the loss
# so they don't affect training.
# =============================================================================

def _get_batch(split):
    """Sample a random Alpaca batch, pad it, and return (x, y, mask)."""
    data = train_data if split == "train" else val_data
    batch = random.sample(data, BATCH_SIZE)

    max_len = max(len(s) for s in batch)

    x_list, y_list, mask_list = [], [], []
    for tokens in batch:
        pad_len = max_len - len(tokens)
        # x: input tokens (all but last)
        # y: target tokens (all but first, shifted by 1)
        padded = tokens + [EOT] * pad_len
        x = padded[:-1]
        y = padded[1:]
        # mask: 1 where tokens are real, 0 where padded
        mask = [1] * (len(tokens) - 1) + [0] * pad_len
        x_list.append(x)
        y_list.append(y)
        mask_list.append(mask)

    x    = torch.tensor(x_list, dtype=torch.long).to(device)
    y    = torch.tensor(y_list, dtype=torch.long).to(device)
    mask = torch.tensor(mask_list, dtype=torch.float).to(device)
    return x, y, mask


def _compute_loss(model, x, y, mask):
    """Forward pass + masked cross entropy loss.

    Unlike pretraining, we only train on real (non-padded) tokens.
    The mask zeroes out padding positions so they contribute nothing to
    the gradient.
    """
    logits = model(x)                          # (B, T, vocab)
    B, T, C = logits.shape
    loss = F.cross_entropy(
        logits.view(B * T, C),
        y.view(B * T),
        reduction="none"
    )                                          # (B*T,)
    loss = loss * mask.view(B * T)             # zero out padding positions
    return loss.sum() / mask.sum()             # mean over real tokens only


@torch.no_grad()
def _evaluate_loss(model, eval_iters=EVAL_ITERATIONS):
    """Averages loss over multiple batches for a more stable validation metric."""
    was_training = model.training
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        val_x, val_y, val_mask = _get_batch(split="val")
        if device == "cuda" and USE_BF16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses[k] = _compute_loss(model, val_x, val_y, val_mask)
        else:
            losses[k] = _compute_loss(model, val_x, val_y, val_mask)
    val_loss = losses.mean()
    if was_training:
        model.train()
    return val_loss

# =============================================================================
# Step 6: Baseline — Test BEFORE Fine-Tuning
# =============================================================================

test_prompts = [
    "What is photosynthesis?",
    "Explain the water cycle in simple terms.",
    "Who was the first emperor of Rome?",
]


def test_model(model, prompts, label=""):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    model.eval()
    for prompt in prompts:
        full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        print(f"\nQ: {prompt}")
        print(f"A: {model.generate_text(start_text=full_prompt, max_tokens=150, temperature=0.7)}")
        print("-" * 40)
    model.train()


test_model(model, test_prompts, label="BASELINE (Before Fine-Tuning)")

# =============================================================================
# Step 7: Optimizer and Scheduler
# =============================================================================

def _configure_optimizers(model, weight_decay, learning_rate, device_type):
    # All parameters that require gradients
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    # 2D parameters (weights in matmuls + embeddings) get weight decay;
    # 1D parameters (biases, layernorm scales) do not.
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params   = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Use fused AdamW when available on CUDA (faster kernel)
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
    )
    return optimizer


def _setup_training(model):
    optimizer = _configure_optimizers(
        model,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        device_type=device,
    )
    # Linear warmup from 10% of LR to full LR over WARMUP_STEPS
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_STEPS
    )
    # Cosine decay from full LR to 10% of LR over the remaining steps
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(MAX_STEPS - WARMUP_STEPS), eta_min=0.1 * LEARNING_RATE
    )
    # Chain them: warmup first, then cosine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[WARMUP_STEPS],
    )
    return optimizer, scheduler

# =============================================================================
# Step 8: Resume from Checkpoint (Optional)
#
# If resuming a previous fine-tuning run, set RESUME_CKPT to the checkpoint
# file path. Leave as None to start fresh.
# =============================================================================

RESUME_CKPT = None
# RESUME_CKPT = f"{MODEL_DIR}/tinygpt_finetuned_checkpoint_alpaca.pt"


def _maybe_load_checkpoint(model, optimizer=None, scheduler=None, resume_path=RESUME_CKPT):
    if not resume_path:
        print("Starting fresh from step 0.")
        return 0

    if not os.path.exists(resume_path):
        print(f"Checkpoint not found at {resume_path}. Starting fresh.")
        return 0

    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if scheduler is not None and ckpt.get("scheduler_state") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = int(ckpt.get("step", 0)) + 1
        print(f"Resumed from {resume_path} at step {start_step}")
    else:
        # Plain weights file (no optimizer/scheduler state)
        state_dict = ckpt
        start_step = 0
        print(f"Loaded weights from {resume_path}")

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return start_step


def _maybe_save_checkpoint(model, optimizer, scheduler=None, step=0, vocab_size=0,
                           val_loss=None, best_val_loss=float("inf"), best_ckpt_path=None):
    """Save only when val_loss improves. Deletes previous best to save disk space.

    This save-on-best strategy guarantees the saved file is always the best
    checkpoint regardless of when the best val loss occurs — solving the problem
    of periodic saves missing the true best.
    """
    if val_loss is None or val_loss >= best_val_loss:
        return best_val_loss, best_ckpt_path

    if best_ckpt_path and os.path.exists(best_ckpt_path):
        os.remove(best_ckpt_path)

    new_path = f"{MODEL_DIR}/tinygpt_finetuned_checkpoint_alpaca.pt"
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "vocab_size": vocab_size,
        "val_loss": val_loss,
    }
    if scheduler is not None:
        try:
            payload["scheduler_state"] = scheduler.state_dict()
        except Exception:
            payload["scheduler_state"] = None
    torch.save(payload, new_path)
    print(f"  Saved best checkpoint: {new_path} (val {val_loss:.4f})")
    return val_loss, new_path


def _plot_losses(steps, train_losses, val_losses):
    if not steps:
        return
    output_path = f"{MODEL_DIR}/finetune_loss_curve.png"
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="train")
    plt.plot(steps, val_losses, label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Fine-Tuning Loss — TinyGPT on Alpaca Cleaned")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Loss curve saved to {output_path}")

# =============================================================================
# Step 9: Training Loop
# =============================================================================

def train_loop(model):
    optimizer, scheduler = _setup_training(model)

    assert EFFECTIVE_BATCH_SIZE % BATCH_SIZE == 0, \
        "EFFECTIVE_BATCH_SIZE must be divisible by BATCH_SIZE"
    accumulation_steps = EFFECTIVE_BATCH_SIZE // BATCH_SIZE

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Effective batch size: {EFFECTIVE_BATCH_SIZE} (via {accumulation_steps} accumulation steps)")

    start_step = _maybe_load_checkpoint(model, optimizer, scheduler)

    steps, train_losses, val_losses = [], [], []
    best_val_loss, best_ckpt_path = float("inf"), None

    for step in range(start_step, MAX_STEPS):
        # 1. Accumulate gradients over multiple micro-batches
        optimizer.zero_grad(set_to_none=True)
        micro_step_loss = 0.0

        for _ in range(accumulation_steps):
            x, y, mask = _get_batch(split="train")
            if device == "cuda" and USE_BF16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = _compute_loss(model, x, y, mask)
            else:
                loss = _compute_loss(model, x, y, mask)
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            micro_step_loss += loss.item()

        avg_train_loss = micro_step_loss / accumulation_steps

        # 2. Clip and update weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # 3. Evaluate and checkpoint if best
        if (step + 1) % EVAL_STEPS == 0 or step == start_step:
            val_loss_val = _evaluate_loss(model, eval_iters=EVAL_ITERATIONS).item()
            steps.append(step + 1)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss_val)

            marker = " *** best ***" if val_loss_val < best_val_loss else ""
            print(f"step {step+1}: train {avg_train_loss:.4f} | val {val_loss_val:.4f}{marker}")

            best_val_loss, best_ckpt_path = _maybe_save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                step=step, vocab_size=enc.n_vocab,
                val_loss=val_loss_val, best_val_loss=best_val_loss, best_ckpt_path=best_ckpt_path,
            )

    print(f"\nBest val loss: {best_val_loss:.4f} — checkpoint: {best_ckpt_path}")
    _plot_losses(steps, train_losses, val_losses)
    return steps, train_losses, val_losses


steps, train_losses, val_losses = train_loop(model)
print("\nTraining complete!")

# =============================================================================
# Step 10: Plot Loss Curves
#
# The training loop already calls _plot_losses at the end.
# Re-run this line if you want to regenerate the plot without retraining.
# =============================================================================

_plot_losses(steps, train_losses, val_losses)

# =============================================================================
# Step 11: Test AFTER Fine-Tuning
#
# Same prompts as baseline. Compare the delta — that's your intuition moment.
# =============================================================================

test_model(model, test_prompts, label="AFTER Instruction Fine-Tuning")

# =============================================================================
# Step 12: Save Final Weights
#
# Load the best checkpoint and extract inference-only weights (strips optimizer
# and scheduler state). The resulting file is what you upload to Hugging Face
# or use with export_to_hf_alpaca.py.
# =============================================================================

best_ckpt = torch.load(
    f"{MODEL_DIR}/tinygpt_finetuned_checkpoint_alpaca.pt",
    map_location=device,
    weights_only=False,
)
best_state_dict = best_ckpt["model_state"]
best_step = best_ckpt.get("step", "?")
best_val  = best_ckpt.get("val_loss", "?")

final_weights_path = f"{MODEL_DIR}/tinygpt_finetuned_alpaca_weights.pt"
torch.save(best_state_dict, final_weights_path)
size_mb = os.path.getsize(final_weights_path) / 1024 / 1024
print(f"Saved: {final_weights_path} ({size_mb:.0f} MB)")
if isinstance(best_val, float):
    print(f"Best step: {best_step} | Best val loss: {best_val:.4f}")
else:
    print(f"Best step: {best_step}")

# =============================================================================
# What Just Happened?
#
# Stage                        | What the model learned
# -----------------------------|-----------------------------------------------
# Pretraining (base model)     | Predict next tokens from raw FineWeb-Edu text
# Instruction fine-tuning here | Recognize ### Instruction / ### Response
#                              | structure and generate helpful answers
#
# Dataset: yahma/alpaca-cleaned — 52K instruction/response pairs,
#          cleaned version of Stanford Alpaca.
#
# Next step: Take this instruction-tuned TinyGPT and do domain SFT on
#            Indian mythology Q&A pairs — or repeat this pipeline on
#            Qwen3-4B with LoRA for a production-grade result.
# =============================================================================
