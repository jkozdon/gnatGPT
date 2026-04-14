import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        assert (
            cfg.d_model % cfg.n_heads == 0
        ), "size mismatch between d_model and n_heads"

        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        # Integer division!
        self.d_k = cfg.d_model // cfg.n_heads

        # nn.Linear goes in_features to out_features and in our
        # case they are the same
        self.W_q = nn.Linear(cfg.d_model, self.d_model, bias=False)
        self.W_k = nn.Linear(cfg.d_model, self.d_model, bias=False)
        self.W_v = nn.Linear(cfg.d_model, self.d_model, bias=False)
        self.W_o = nn.Linear(cfg.d_model, self.d_model, bias=False)

        # causal mask: upper triangle is True (will be filled with -inf)
        T = cfg.context_len
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        Q = self.W_q(x)  # (B, T, d_model)
        K = self.W_k(x)  # (B, T, d_model)
        V = self.W_v(x)  # (B, T, d_model)

        # Reshape: d_model → n_heads × d_k, then move head dim forward
        Q = Q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Now Q @ K^T is (B, n_heads, T, T) — all heads in one matmul
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        weights = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)

        out = weights @ V  # (B, n_heads, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, self.d_model)
        return self.W_o(out)
