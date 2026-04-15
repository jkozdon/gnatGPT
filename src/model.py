import torch.nn as nn
import torch.nn.functional as F

from config import Config
from embeddings import Embeddings
from block import TransformerBlock


class GnatGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embedding = Embeddings(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.model, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding <-> LM head
        self.lm_head.weight = self.embedding.tok_emb.weight

    def forward(self, idx, targets=None):
        # idx: (B, T) token ids
        x = self.embedding(idx)  # (B, T, d_model)

        for block in self.blocks:
            x = block(x)  # (B, T, d_model)

        x = self.norm(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss
