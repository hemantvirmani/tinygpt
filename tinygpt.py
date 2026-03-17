# !pip install torch transformers requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, requests
import argparse
from dataclasses import dataclass
from transformers import AutoTokenizer

#Hyperparameters
G_BATCH_SIZE = 16
G_BLOCK_SIZE = 64
G_N_EMBD = 128
G_MAX_ITERS = 8000
G_LR = 5e-4
G_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G_SEED = 1947

torch.manual_seed(G_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(G_SEED)

@dataclass
class State:
    tokenizer: AutoTokenizer
    train_data: torch.Tensor
    val_data: torch.Tensor
    vocab_size: int

def build_state(split_ratio: float = 0.9) -> State:
    # Download dataset if needed
    if not os.path.exists("shakespeare.txt"):
        print("Downloading Tiny Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data = requests.get(url).text
        with open("shakespeare.txt", "w", encoding="utf-8") as f:
            f.write(data)

    # Load training dataset
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Tokenizer and related objects
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    data = torch.tensor(tokens)
    vocab_size = tokenizer.vocab_size
    split_idx = int(len(data) * split_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return State(tokenizer=tokenizer, train_data=train_data, val_data=val_data, vocab_size=vocab_size)


#Load the batch from the dataset
def get_batch(state: State, split: str = "train"):
    data = state.train_data if split == "train" else state.val_data
    ix = torch.randint(len(data) - G_BLOCK_SIZE, (G_BATCH_SIZE,))

    x = torch.stack([data[i:i+G_BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+G_BLOCK_SIZE+1] for i in ix])
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

        self.blocks = nn.Sequential(
            Block(G_N_EMBD),
            Block(G_N_EMBD),
            Block(G_N_EMBD)
        )
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
    if not resume_path:
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
    ckpt_path = os.path.join(checkpoint_dir, f"tinygpt_step_{step+1}.pt")
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

def initialize_and_train(state: State,
                         max_iters: int = G_MAX_ITERS,
                         resume_path: str | None = None) -> nn.Module:
    """Create, train, and return a `TinyGPT` model.

    Args:
        state: training state (tokenizer, data, vocab_size).
        max_iters: number of training iterations.
        resume_path: path to a checkpoint file to resume from.

    Returns:
        Trained `TinyGPT` instance (on `device`).
    """

    # Model Initialization
    model = TinyGPT(state.vocab_size).to(G_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=G_LR)

    total_params = sum(p.numel() for p in model.parameters())
    total_m = total_params / 1_000_000
    print(f"Total parameters: {total_m:.2f}M")

    start_step = maybe_load_checkpoint(model, optimizer, resume_path)

    for step in range(start_step, max_iters):
        x, y = get_batch(state, split="train")
        loss = compute_loss(model, x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            val_loss = evaluate_loss(model, state)
            print(f"step {step+1}: train {loss.item():.4f} | val {val_loss.item():.4f}")

        maybe_save_checkpoint(
            model=model, optimizer=optimizer,
            vocab_size=state.vocab_size,
            step=step, resume_path=resume_path,
        )

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

    #build the state and train the model
    state = build_state()
    model = initialize_and_train(
        state,
        resume_path=checkpoint_path if checkpoint_path else None,
    )

    #Lets generate some text from the trained model
    sample = generateText(
        model, state,
        start_text="To be, or not to be: that is the question:",
        max_tokens=100,
    )

    #Lets see what we got!
    print("------ FINAL TEXT ------")
    print(sample)
    print("------------")

if __name__ == "__main__":
    main()
