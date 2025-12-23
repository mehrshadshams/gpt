import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from utils import get_device
from dataset import ShakespeareDataset

#
batch_size = 64
block_size = 256
vocab_size = 65  # example vocab size
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = get_device()
n_embd = 384
n_layers = 6
n_head = 6
drop_out = 0.2
#

dataset = ShakespeareDataset("data/shakespeare.txt")
chars = sorted(list(set(dataset.text())))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(dataset.text()), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(100)
        for k in range(100):
            x, y = get_batch(split)
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)

        v = self.value(x)  # (B,T,head_size)
        out = wei @ v  # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa_head = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(drop_out),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos_emb = self.pos_embedding(torch.arange(T, device=device))  # T, C
        token_emb = self.token_embedding(idx)  # B, T, C

        x = token_emb + pos_emb  # B, T, C
        x = self.blocks(x)  # B, T, C
        logits = self.lm_head(self.ln_f(x))  # B, T, vocab_size
        B, T, C = logits.shape

        if targets is None:
            loss = None
        else:
            targets = targets.view(B * T)
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_crop = idx[:, -block_size:]
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :]  # B, C
            probs = F.softmax(logits, dim=-1)  # B, C
            idx_next = torch.multinomial(probs, num_samples=1)  # B, 1
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


start_time = time.time()

model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 32
for step in range(10000):
    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)
    if step % 3000 == 0:
        losses = estimate_loss(model)
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    logits, loss = model(xb, targets=yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step}: loss {loss.item()}")


input_data = torch.zeros((1, 1), dtype=torch.long).to(device)
print(decode(model.generate(input_data, max_new_tokens=400)[0].tolist()))

end_time = time.time()
print(f"Training and generation took {end_time - start_time:.2f} seconds")
