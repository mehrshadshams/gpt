import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
  block_size: int = 256
  vocab_size: int = 50257
  n_layer: int = 6
  n_head: int = 8
  n_embd: int = 512


class CasualSelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.n_head = config.n_head
    self.head_dim = config.n_embd // config.n_head
    self.scale = self.head_dim ** -0.5

    self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
    self.proj = nn.Linear(config.n_embd, config.n_embd)

    self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).unsqueeze(0).unsqueeze(0))

  def forward(self, x):
    B, T, C = x.size()
    qkv = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    attn_scores = (q @ k.transpose(-2, -1)) * self.scale
    attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
    attn_weights = torch.softmax(attn_scores, dim=-1)

    attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
    output = self.proj(attn_output)
    return output


class Block(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CasualSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = nn.Sequential(
      nn.Linear(config.n_embd, 4 * config.n_embd),
      nn.GELU(approximate='tanh'),
      nn.Linear(4 * config.n_embd, config.n_embd),
    )

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class GPT(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd),
      wpe = nn.Embedding(config.block_size, config.n_embd),
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embd),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

  def forward(self, x):
    # Forward pass implementation would go here
    pass
