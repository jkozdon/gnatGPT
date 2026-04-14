import torch
from config import Config
from embeddings import Embeddings
from multi_head_attn import CausalSelfAttention
from ffn import FeedForward

cfg = Config()
emb = Embeddings(cfg)

ids = torch.randint(0, cfg.vocab_size, (2, cfg.context_len))  # fake batch
out = emb(ids)

attn = CausalSelfAttention(cfg)
out = attn(emb(ids))

ff = FeedForward(cfg)
ff(out)
