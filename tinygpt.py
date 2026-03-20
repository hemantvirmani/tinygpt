# !pip install torch tiktoken requests huggingface_hub matplotlib
# BINARY_DATASET_FILENAME = "/kaggle/input/datasets/hemantvirmani/gpt-training-dataset/dataset.bin"
# CHECKPOINT_FILE = ""/kaggle/working/tinygpt_latest.pt"

from sched import scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, requests
import shutil
import argparse
from dataclasses import dataclass
import tiktoken
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

#Hyperparameters
G_BATCH_SIZE = 4
G_BLOCK_SIZE = 1024
G_N_EMBD = 768
G_MAX_ITERS = 40000
G_LR = 3e-4
G_N_LAYERS = 12
G_WEIGHT_DECAY = 0.1
G_GRAD_CLIP = 1.0
G_WARMPUP_ITERS = 4000 # 10% of G_MAX_ITERS
G_DROPOUT_PROB = 0.0
G_N_HEAD = 12
G_EFFECTIVE_BATCH_SIZE = 8
G_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#------Static Constants------#
G_SEED = 1947
G_SPLIT_RATIO = 0.8
BINARY_DATASET_FILENAME = "dataset.bin"
LOAD_CHECKPOINT = False
CHECKPOINT_FILE = "tinygpt_latest.pt"
#Random seed for reproducibility
torch.manual_seed(G_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(G_SEED)

@dataclass
class State:
    tokenizer: Any
    train_data: np.ndarray
    val_data: np.ndarray
    vocab_size: int

def build_state(split_ratio: float = G_SPLIT_RATIO, dataset_path: str = BINARY_DATASET_FILENAME) -> State:
    # Tokenizer and related objects
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    # Load training and validation dataset
    import numpy as np
    data = np.memmap(dataset_path, dtype=np.uint16, mode="r")
    data_len = len(data)
    split_idx = int(data_len * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return State(tokenizer=tokenizer, train_data=train_data, val_data=val_data, vocab_size=vocab_size)

#Lets do some Self Attention
class SelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.key = nn.Linear(G_N_EMBD, n_embd)
        self.query = nn.Linear(G_N_EMBD, n_embd)
        self.value = nn.Linear(G_N_EMBD, n_embd)

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(G_BLOCK_SIZE, G_BLOCK_SIZE))
        )
        self.dropout = nn.Dropout(G_DROPOUT_PROB)

    def forward(self, x): # Attention = softmax(similarity(q, k)) @ v
        B, T, C = x.shape

        # Compute key, query, value projections for self-attention.
        q = self.query(x) # `q` (query): what each token is "asking for".
        k = self.key(x)  # `k` (key): content to be compared/matched against the query.
        
        weights1 = q @ k.transpose(-2, -1) / (C**0.5)
        weights1 = weights1.masked_fill(self.mask[:T, :T] == 0, float('-inf')) #Mask ensures auto regressive behavior
        weights1 = F.softmax(weights1, dim=-1)
        weights1 = self.dropout(weights1)

        v = self.value(x) # `v` (value): information returned.
        out = weights1 @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(G_N_EMBD, G_N_EMBD)
        self.dropout = nn.Dropout(G_DROPOUT_PROB)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        assert n_embd % G_N_HEAD == 0
        # Key, Query, Value projections for all heads in one go
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(G_DROPOUT_PROB)
        self.resid_dropout = nn.Dropout(G_DROPOUT_PROB)
        
        self.register_buffer("mask", torch.tril(torch.ones(G_BLOCK_SIZE, G_BLOCK_SIZE))
                                        .view(1, 1, G_BLOCK_SIZE, G_BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.size()
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(G_N_EMBD, dim=2)
        k = k.view(B, T, G_N_HEAD, C // G_N_HEAD).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, G_N_HEAD, C // G_N_HEAD).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, G_N_HEAD, C // G_N_HEAD).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
# Transformer block: self-attention plus feed-forward network
class Block(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
 
        # Each head gets an equal portion of the embedding dimension
        self.ln1 = nn.LayerNorm(n_embd)
        #self.attention = MultiHeadAttention(G_N_HEAD, n_embd // G_N_HEAD)
        self.attention = CausalSelfAttention(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(G_DROPOUT_PROB),
        )

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x

#Lets Create the GPT model
class TinyGPT(nn.Module):
    def __init__(self, state: State):
        super().__init__()

        self.state = state
        self.token_embedding_table = nn.Embedding(state.vocab_size, G_N_EMBD)
        self.position_embedding_table = nn.Embedding(G_BLOCK_SIZE, G_N_EMBD)

        self.blocks = nn.Sequential(*[Block(G_N_EMBD) for _ in range(G_N_LAYERS)]) #Stacking multiple blocks for deeper architecture

        self.ln_f = nn.LayerNorm(G_N_EMBD)
        self.head = nn.Linear(G_N_EMBD, state.vocab_size)

        self.apply(self._init_weights)

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=G_DEVICE))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x) #normalizes the hidden states
        logits = self.head(x) #final projection to logits for each token in the vocabulary
        return logits

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    #Load the batch from the dataset
    def _get_batch(self, split: str = "train"):
        data = self.state.train_data if split == "train" else self.state.val_data
        ix = torch.randint(len(data) - G_BLOCK_SIZE - 1, (G_BATCH_SIZE,))

        # Convert the full batch slice in one go, then reshape to reduce overhead.
        ix_np = ix.cpu().numpy()
        offsets = np.arange(G_BLOCK_SIZE + 1)
        # Index memmap directly to avoid materializing the full dataset in RAM.
        batch = data[ix_np[:, None] + offsets]
        x = torch.from_numpy(batch[:, :-1]).long()
        y = torch.from_numpy(batch[:, 1:]).long()
        return x.to(G_DEVICE), y.to(G_DEVICE)

    def _compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self(x)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = y.view(B * T)
        return F.cross_entropy(logits, targets)

    def _setup_training(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=G_LR,
            weight_decay=G_WEIGHT_DECAY,
        ) #weighted decay for better generalization and AdamW optimizer for better convergence

        # 1. Define the Warmup Scheduler (Linear increase from a 10% of G_LR to G_LR)
        scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1,end_factor=1.0,
            total_iters=G_WARMPUP_ITERS)

        # 2. Define the Main Scheduler (Cosine decay from G_LR to eta_min)
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(G_MAX_ITERS - G_WARMPUP_ITERS),
            eta_min=1e-5)

        # 3. Combine them using SequentialLR - milestones=[500] means switch to the second scheduler at step 500
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[G_WARMPUP_ITERS]
        )

        return optimizer, scheduler

    @torch.no_grad()
    def _evaluate_loss(self, eval_iters=5) -> torch.Tensor:
        """Averages loss over multiple batches for a more stable validation metric."""
        was_training = self.training
        self.eval()
        losses = torch.zeros(eval_iters, device=G_DEVICE)
        for k in range(eval_iters):
            val_x, val_y = self._get_batch(split="val")
            losses[k] = self._compute_loss(val_x, val_y)
        val_loss = losses.mean()
        if was_training: self.train()
        return val_loss

    def train_loop(self) -> None:
        optimizer, scheduler = self._setup_training()

        # Target an effective batch size of 32 to reduce "waviness"
        # Since G_BATCH_SIZE is 4, we accumulate for 8 steps (4 * 8 = 32)
        assert G_EFFECTIVE_BATCH_SIZE % G_BATCH_SIZE == 0, "G_EFFECTIVE_BATCH_SIZE must be divisible by G_BATCH_SIZE"
        accumulation_steps = G_EFFECTIVE_BATCH_SIZE // G_BATCH_SIZE 

        print(f"Total parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")
        print(f"Effective batch size: {G_EFFECTIVE_BATCH_SIZE} (via {accumulation_steps} accumulation steps)")

        start_step = _maybe_load_checkpoint(self, optimizer)

        steps = []
        train_losses = []
        val_losses = []

        for step in range(start_step, G_MAX_ITERS):
            # 1. Accumulate gradients over multiple micro-batches
            optimizer.zero_grad(set_to_none=True)
            micro_step_loss = 0.0
            
            for _ in range(accumulation_steps):
                x, y = self._get_batch(split="train")
                loss = self._compute_loss(x, y)
                # Scale loss by accumulation steps so gradients are averaged correctly
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()
                micro_step_loss += loss.item()

            avg_train_loss = micro_step_loss / accumulation_steps

            # 2. Clip and update weights
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=G_GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            # 3. Logging and Checkpointing
            if (step + 1) % 100 == 0 or step == 0:
                val_loss = self._evaluate_loss(eval_iters=10)
                steps.append(step + 1)
                train_losses.append(avg_train_loss)
                val_losses.append(val_loss.item())
                print(f"step {step+1}: train {avg_train_loss:.4f} | val {val_loss.item():.4f}")

            _maybe_save_checkpoint(model=self, optimizer=optimizer,
                vocab_size=self.state.vocab_size, step=step)

        _plot_losses(steps, train_losses, val_losses)

    # Text Generation Function
    def generate_text(self, start_text, max_tokens=50):
        self.eval()

        tokens = self.state.tokenizer.encode(start_text)
        idx = torch.tensor(tokens).unsqueeze(0).to(G_DEVICE)

        for _ in range(max_tokens):
            idx_cond = idx[:, -G_BLOCK_SIZE:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        text = self.state.tokenizer.decode(idx[0].tolist())
        return text

def _maybe_load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: str | None = CHECKPOINT_FILE,
) -> int:
    if not resume_path or LOAD_CHECKPOINT == False:
        return 0
    
    ckpt = torch.load(resume_path, map_location=G_DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_step = int(ckpt.get("step", 0))
    print(f"Resumed from {resume_path} at step {start_step}")
    return start_step

def _maybe_save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    vocab_size: int,
    resume_path: str | None = CHECKPOINT_FILE,
) -> None:
    if resume_path is None or (step + 1) % 1000 != 0:
        return
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab_size": vocab_size,
        },
        resume_path,
    )
    print(f"Saved checkpoint: {resume_path}")

def _plot_losses(steps, train_losses, val_losses):
    if not steps:
        return

    output_path = "loss_curve.png"
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="train")
    plt.plot(steps, val_losses, label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    plt.close()

def main():
    # if dataset.bin does not exist, error out.
    if not os.path.exists(BINARY_DATASET_FILENAME):
        from huggingface_hub import hf_hub_download
        # Download the specific .bin file from your repository
        repo_id = "hemantvirmani/gpt-training-dataset"
        filename = "dataset.bin"

        print(f"Downloading {filename} from Hugging Face...")
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        shutil.copyfile(file_path, BINARY_DATASET_FILENAME)
    file_path = BINARY_DATASET_FILENAME

    #build the state and train the model
    state = build_state(dataset_path=file_path)
    model = TinyGPT(state).to(G_DEVICE)
    model.train_loop()

    #Lets generate some text from the trained model
    print(model.generate_text(start_text="USA is a country of ", max_tokens=1000))

if __name__ == "__main__":
    main()
