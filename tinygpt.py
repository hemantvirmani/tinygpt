# !pip install torch transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoTokenizer

#Hyperparameters
batch_size = 16
block_size = 64
n_embd = 128
max_iters = 2000 # cpu = 800 maybe?
lr = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

#load dataset
with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokens = tokenizer.encode(text)
data=torch.tensor(tokens)

vocab_size = tokenizer.vocab_size



#Load the batch
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

#Lets do some Self Attention
class SelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        
        weights1 = q @ k.transpose(-2, -1) / (C**0.5)
        weights1 = weights1.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        weights1 = F.softmax(weights1, dim=-1)

        v = self.value(x)
        out = weights1 @ v

        return out


#Lets do some Feed Forward Neural Network
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
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            Block(n_embd),
            Block(n_embd),
            Block(n_embd)
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape

        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=device))

        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

#        B, T, C = logits.shape
#        logits = logits.view(B*T, C)
#        targets = targets.view(B*T)
#        loss = F.cross_entropy(logits, targets)
#        return logits, loss

# Text Generation
def generate(model, tokenizer, start_text, max_tokens=50):

    model.eval()

    tokens = tokenizer.encode(start_text)
    idx = torch.tensor(tokens).unsqueeze(0).to(device)

    for _ in range(max_tokens):
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    text = tokenizer.decode(idx[0].tolist())

    return text

#Model Initialization
model = TinyGPT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for step in range(max_iters):

    x, y = get_batch()
    logits = model(x)

    B, T, C = logits.shape

    logits = logits.view(B*T, C)
    targets = y.view(B*T)

    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}: loss {loss.item()}")

        sample = generate(
            model, 
            tokenizer, 
            start_text="To be, or not to be: that is the question:", 
            max_tokens=60)
        
        print("------ SAMPLE TEXT ------")
        print(sample)
        print("------------")
