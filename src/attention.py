import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_k = cfg.d_model

        self.W_q = nn.Linear(cfg.d_model, self.d_k, bias=False)
        self.W_k = nn.Linear(cfg.d_model, self.d_k, bias=False)
        self.W_v = nn.Linear(cfg.d_model, self.d_k, bias=False)

        # causal mask: upper triangle is True (will be filled with -inf)
        T = cfg.context_len
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        Q = self.W_q(x)  # (B, T, d_k)
        K = self.W_k(x)  # (B, T, d_k)
        V = self.W_v(x)  # (B, T, d_k)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, T, T)

        out = weights @ V  # (B, T, d_k)
        return out
