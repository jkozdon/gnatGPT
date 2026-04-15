import torch
from config import Config
from model import GnatGPT
from embeddings import Embeddings

# Expected loss if logits were uniform (all zero)
import math

cfg = Config()
model = GnatGPT(cfg)

# Random batch: B=2, T=16
idx = torch.randint(0, cfg.vocab_size, (2, 16))
targets = torch.randint(0, cfg.vocab_size, (2, 16))

logits, loss = model(idx, targets)

print("logits shape:", logits.shape)  # expect (2, 16, 65)
print("loss:        ", loss.item())  # expect ~ln(65) ≈ 4.17 for random weights

total = sum(p.numel() for p in model.parameters())
print(f"parameters: {total:,}")

print(model.lm_head.weight is model.embedding.tok_emb.weight)  # True

unique = sum(p.numel() for p in set(model.parameters()))
print(f"unique parameters: {unique:,}")  # slightly less than total


cfg = Config()
emb = Embeddings(cfg)

idx = torch.randint(0, cfg.vocab_size, (2, 16))
tok_out = emb(idx)

print("embedding mean:", tok_out.mean().item())
print("embedding std: ", tok_out.std().item())
print("pe max:        ", emb.pe.max().item())
print("pe min:        ", emb.pe.min().item())

cfg = Config()
model = GnatGPT(cfg)

idx = torch.randint(0, cfg.vocab_size, (2, 16))

# Check logit magnitude before any training
logits, _ = model(idx)
print("logits mean:", logits.mean().item())
print("logits std: ", logits.std().item())

print("expected loss at uniform logits:", math.log(cfg.vocab_size))


model.eval()  # disable dropout for this check
with torch.no_grad():
    x = model.embedding(idx)
    print(f"after embedding:  std={x.std().item():.3f}")
    for i, block in enumerate(model.blocks):
        x = block(x)
        print(f"after block {i}:    std={x.std().item():.3f}")
    x = model.norm(x)
    print(f"after final norm: std={x.std().item():.3f}")

print("lm_head weight std:", model.lm_head.weight.std().item())
print("tok_emb weight std:", model.embedding.tok_emb.weight.std().item())
