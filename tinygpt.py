# !pip install torch tiktoken requests huggingface_hub matplotlib

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
G_BATCH_SIZE = 64
G_BLOCK_SIZE = 128
G_N_EMBD = 256
G_MAX_ITERS = 10000
G_LR = 1e-4
G_N_LAYERS = 12
G_WEIGHT_DECAY = 0.1
G_GRAD_CLIP = 1.0

G_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G_SEED = 1947

#Static Constants
BINARY_DATASET_FILENAME = "dataset.bin"
LOAD_CHECKPOINT = False

#Random seed for reproducibility
torch.manual_seed(G_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(G_SEED)

@dataclass
class State:
    tokenizer: Any
    train_data: torch.Tensor
    val_data: torch.Tensor
    vocab_size: int

def build_state(split_ratio: float = 0.8, dataset_path: str = BINARY_DATASET_FILENAME) -> State:
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


#Load the batch from the dataset
def get_batch(state: State, split: str = "train"):
    data = state.train_data if split == "train" else state.val_data
    ix = torch.randint(len(data) - G_BLOCK_SIZE - 1, (G_BATCH_SIZE,))

    # Convert the full batch slice in one go, then reshape to reduce overhead.
    ix_np = ix.cpu().numpy()
    offsets = np.arange(G_BLOCK_SIZE + 1)
    # Index memmap directly to avoid materializing the full dataset in RAM.
    batch = data[ix_np[:, None] + offsets]
    x = torch.from_numpy(batch[:, :-1]).long()
    y = torch.from_numpy(batch[:, 1:]).long()
    return x.to(G_DEVICE), y.to(G_DEVICE)

#Lets do some Self Attention
class SelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(G_BLOCK_SIZE, G_BLOCK_SIZE))
        )

    def forward(self, x): # Attention = softmax(similarity(q, k)) @ v
        B, T, C = x.shape

        # Compute key, query, value projections for self-attention.
        q = self.query(x) # `q` (query): what each token is "asking for".
        k = self.key(x)  # `k` (key): content to be compared/matched against the query.
        
        weights1 = q @ k.transpose(-2, -1) / (C**0.5)
        weights1 = weights1.masked_fill(self.mask[:T, :T] == 0, float('-inf')) #Mask ensures auto regressive behavior
        weights1 = F.softmax(weights1, dim=-1)

        v = self.value(x) # `v` (value): information returned.
        out = weights1 @ v

        return out

# Transformer block: self-attention plus feed-forward network
class Block(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.sa = SelfAttention(n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd)
        )

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x

#Lets Create the GPT model
class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, G_N_EMBD)
        self.position_embedding_table = nn.Embedding(G_BLOCK_SIZE, G_N_EMBD)

        self.blocks = nn.Sequential(*[Block(G_N_EMBD) for _ in range(G_N_LAYERS)]) #Stacking multiple blocks for deeper architecture

        self.ln_f = nn.LayerNorm(G_N_EMBD)
        self.head = nn.Linear(G_N_EMBD, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=G_DEVICE))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x) #normalizes the hidden states
        logits = self.head(x) #final projection to logits for each token in the vocabulary
        return logits


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = y.view(B * T)
    return F.cross_entropy(logits, targets)


def maybe_load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: str | None,
) -> int:
    if not resume_path or LOAD_CHECKPOINT == False:
        return 0
    
    ckpt = torch.load(resume_path, map_location=G_DEVICE)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_step = int(ckpt.get("step", 0))
    print(f"Resumed from {resume_path} at step {start_step}")
    return start_step


def maybe_save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    resume_path: str | None,
    vocab_size: int,
) -> None:
    if resume_path is None or (step + 1) % 500 != 0:
        return
    checkpoint_dir = os.path.dirname(resume_path) or "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"tinygpt_latest.pt")
    torch.save(
        {
            "step": step + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab_size": vocab_size,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint: {ckpt_path}")


def evaluate_loss(model: nn.Module, state: State) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        val_x, val_y = get_batch(state, split="val")
        val_loss = compute_loss(model, val_x, val_y)
    if was_training:
        model.train()
    return val_loss

def plot_losses(steps, train_losses, val_losses):
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

def initialize_and_train(state: State,
                         resume_path: str | None = None) -> nn.Module:
    """Create, train, and return a `TinyGPT` model.

    Args:
        state: training state (tokenizer, data, vocab_size).
        resume_path: path to a checkpoint file to resume from.

    Returns:
        Trained `TinyGPT` instance (on `device`).
    """

    # Model Initialization
    model = TinyGPT(state.vocab_size).to(G_DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=G_LR, weight_decay=G_WEIGHT_DECAY) #weighted decay for better generalization and AdamW optimizer for better convergence
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( #  SCHEDULER: Define the Cosine Annealing curve for learning rate decay
        optimizer, T_max=G_MAX_ITERS, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    total_m = total_params / 1_000_000
    print(f"Total parameters: {total_m:.2f}M")

    start_step = maybe_load_checkpoint(model, optimizer, resume_path)

    steps = []
    train_losses = []
    val_losses = []

    for step in range(start_step, G_MAX_ITERS):
        optimizer.zero_grad(set_to_none=True) # More efficient than zero_grad() as it avoids unnecessary memory operations when gradients are not needed.

        x, y = get_batch(state, split="train")
        loss = compute_loss(model, x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=G_GRAD_CLIP) #GRADIENT CLIPPING: to prevent loss spikes
        optimizer.step()
        scheduler.step() # STEP THE SCHEDULER: Update the learning rate every iteration

        if (step + 1) % 100 == 0:
            val_loss = evaluate_loss(model, state)
            steps.append(step + 1)
            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            print(f"step {step+1}: train {loss.item():.4f} | val {val_loss.item():.4f}")

        maybe_save_checkpoint(
            model=model, optimizer=optimizer,
            vocab_size=state.vocab_size,
            step=step, resume_path=resume_path,
        )

    plot_losses(steps, train_losses, val_losses)

    return model


# Text Generation Function
def generateText(model, state: State, start_text, max_tokens=50):

    model.eval()

    tokens = state.tokenizer.encode(start_text)
    idx = torch.tensor(tokens).unsqueeze(0).to(G_DEVICE)

    for _ in range(max_tokens):
        idx_cond = idx[:, -G_BLOCK_SIZE:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    text = state.tokenizer.decode(idx[0].tolist())
    return text

def main():
    #basic argument parsing for checkpoint resume/save
    p = argparse.ArgumentParser(description="Train TinyGPT")
    p.add_argument("--checkpoint", metavar="PATH", help="Path to checkpoint to resume from (also enables saving)")
    checkpoint_path = p.parse_args().checkpoint
    file_path = BINARY_DATASET_FILENAME

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
    model = initialize_and_train(
        state,
        resume_path=checkpoint_path if checkpoint_path else None,
    )

    #Lets generate some text from the trained model
    sample = generateText(
        model, state,
        start_text="USA is a country of ",
        max_tokens=100,
    )

    #Lets see what we got!
    print("------ FINAL TEXT ------")
    print(sample)
    print("------------")

if __name__ == "__main__":
    main()
