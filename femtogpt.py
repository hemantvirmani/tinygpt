# !pip install torch 
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Any


# hyperparameters
G_BATCH_SIZE = 64 # how many independent sequences will we process in parallel?
G_BLOCK_SIZE = 256 # what is the maximum context length for predictions?
G_N_EMBD = 384
G_MAX_ITERS = 2000
G_LR = 3e-4 #for simple models we can afford to go a bit higher. for complex models, it is usually best to start off around 1e-4
G_EVAL_ITERS = 200
G_N_HEAD = 6
G_N_LAYER = 6
G_DROPOUT = 0.2
# ------------
G_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
G_SEED = 1337
DATASET_FILENAME = "shakespeare.txt"
G_SPLIT_RATIO = 0.8

torch.manual_seed(G_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(G_SEED)

#Step 1: Create a local char based tokenizer and a state object.
class Tokenizer:
    def __init__(self, text: str):
        # here are all the unique characters that occur in this text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

    def encode(self, s: str):
        # encoder: take a string, output a list of integers
        return [self.stoi[c] for c in s]

    def decode(self, l):
        # decoder: take a list of integers, output a string
        return ''.join([self.itos[i] for i in l])


#State object
@dataclass
class State:
    tokenizer: Any
    train_data: torch.Tensor
    val_data: torch.Tensor
    vocab_size: int

#Step 2: Build the state (tokenizer + train/val splits)
def build_state(split_ratio: float = G_SPLIT_RATIO, dataset_path: str = DATASET_FILENAME) -> State:
    # Load the dataset
    # wget https://raw.githubusercontent.com/hemantvirmani/tinygpt/master/shakespeare.txt
    with open(dataset_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create the tokenizer
    tokenizer = Tokenizer(text)

    # Train and test splits
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return State(tokenizer=tokenizer, train_data=train_data, val_data=val_data, vocab_size=tokenizer.vocab_size)

# Function for data loading
def get_batch(state: State, split):
    # generate a small batch of data of inputs x and targets y
    data = state.train_data if split == 'train' else state.val_data
    ix = torch.randint(len(data) - G_BLOCK_SIZE, (G_BATCH_SIZE,))
    x = torch.stack([data[i:i+G_BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i+1:i+G_BLOCK_SIZE+1] for i in ix])
    x, y = x.to(G_DEVICE), y.to(G_DEVICE)
    return x, y

@torch.no_grad()
def estimate_loss(model: nn.Module, state: State):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(G_EVAL_ITERS)
        for k in range(G_EVAL_ITERS):
            X, Y = get_batch(state, split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(G_N_EMBD, head_size, bias=False)
        self.query = nn.Linear(G_N_EMBD, head_size, bias=False)
        self.value = nn.Linear(G_N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(G_BLOCK_SIZE, G_BLOCK_SIZE)))

        self.dropout = nn.Dropout(G_DROPOUT)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(G_N_EMBD, G_N_EMBD)
        self.dropout = nn.Dropout(G_DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(G_DROPOUT),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple model
class FemtoGPT(nn.Module):

    def __init__(self, state: State):
        super().__init__()
        self.state = state
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(state.vocab_size, G_N_EMBD)
        self.position_embedding_table = nn.Embedding(G_BLOCK_SIZE, G_N_EMBD)
        self.blocks = nn.Sequential(*[Block(G_N_EMBD, n_head=G_N_HEAD) for _ in range(G_N_LAYER)])
        self.ln_f = nn.LayerNorm(G_N_EMBD) # final layer norm
        self.lm_head = nn.Linear(G_N_EMBD, state.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=G_DEVICE)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -G_BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def initialize_and_train(state: State) -> nn.Module:

    model = FemtoGPT(state).to(G_DEVICE)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=G_LR)

    for iter in range(G_MAX_ITERS):

        # every once in a while evaluate the loss on train and val sets
        if (iter+1) % 100 == 0 or iter == 0:
            losses = estimate_loss(model, state)
            print(f"step {iter+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(state, 'train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model

def main():
    state = build_state()
    model = initialize_and_train(state)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=G_DEVICE)

    idx = model.generate(context, max_new_tokens=2000)[0].tolist()
    print(state.tokenizer.decode(idx))

if __name__ == "__main__":
    main()
