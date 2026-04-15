import torch
import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # token embedding: lookup table, shape (vocab_size, d_model)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # positional encoding: computed once, stored as a buffer
        pe = self.make_pe(cfg.context_len, cfg.d_model)
        self.register_buffer("pe", pe)

    def make_pe(self, T, d_model):
        # build the (T, d_model) sinusoidal matrix
        pe = torch.zeros(T, d_model)
        positions = torch.arange(T).unsqueeze(1)  # size (T, 1)
        dims = torch.arange(0, d_model, 2)  # 0, 2, 4, ...
        freqs = torch.exp(-dims * math.log(10000) / d_model)
        pe[:, 0::2] = torch.sin(positions * freqs)
        pe[:, 1::2] = torch.cos(positions * freqs)
        return pe

    def forward(self, x):
        # x: (B, T) token ids
        # return: (B, T, d_model)
        _, T = x.shape
        tok = self.tok_emb(x)  # (B, T, d_model)
        return tok + self.pe[:T, :]  # broadcast add (T, d_model) + (B, T, d_model)
