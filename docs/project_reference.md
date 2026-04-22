# gnatGPT — project reference

## Session plan

| Session | Title | Key deliverables | Est. time |
|---|---|---|---|
| 1 | Environment + PyTorch orientation | tensor ops, autograd intro, shape intuition | 45–60 min |
| 2 | Tokenizer + dataset pipeline | `tokenizer.py`, `dataset.py`, DataLoader | 45–60 min |
| 3 | Embeddings + causal self-attention | `embeddings.py`, `attention.py`, causal mask | 60–75 min |
| 4 | Multi-head attention + FFN + block | `multi_head_attn.py`, `ffn.py`, `block.py` | 60–75 min |
| 5 | Full model + forward pass | `model.py`, loss check, param count | 45–60 min |
| 6 | Training loop | `train.py`, AdamW, LR schedule, grad clipping | 60 min |
| 7 | First training run + generation | `generate.py`, checkpointing, sampling | 60 min |
| 8 | Scaling experiments | vary d_model, n_layers, d_ff; compare param counts and val loss | 60–90 min |
| 9 | Diagnostics + instrumentation | grad norm logging, loss curves, per-layer stats | 45–60 min |
| 10 | KV-cache | cached generation in `generate.py`, O(T) inference | 60–75 min |
| 11 | RoPE | replace sinusoidal PE in `embeddings.py`, rotary math | 60–75 min |
| 12+ | Extensions | BPE tokenizer, top-p sampling, Flash Attention | open-ended |

---

## Architecture overview

### Block structure (repeated N times)
```
x → LayerNorm → CausalMultiHeadAttention → + x  (residual)
  → LayerNorm → FeedForward               → + x  (residual)
```

### Full model
```
token_ids
  → TokenEmbedding + PositionalEncoding   # (B, T) → (B, T, d_model)
  → TransformerBlock × N                  # (B, T, d_model)
  → LayerNorm                             # (B, T, d_model)
  → Linear (LM head)                      # (B, T, vocab_size)
  → logits (used for cross-entropy loss, or softmax for generation)
```

---

## Key PyTorch concepts (quick reference)

### Tensor shapes
Batch dimension is always first. For this project:
- Token IDs: `(B, T)` — batch size × sequence length
- Embeddings / hidden states: `(B, T, d_model)`
- Attention weights: `(B, n_heads, T, T)`
- Logits: `(B, T, vocab_size)`

### Essential ops
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Matrix multiply (batched)
C = A @ B               # same as torch.matmul(A, B)

# Softmax along last dim
p = F.softmax(logits, dim=-1)

# Cross-entropy loss (expects logits, not probabilities)
loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

# Reshape without copying
x = x.view(B, T, n_heads, d_k).transpose(1, 2)   # (B, n_heads, T, d_k)

# Causal mask (upper-triangular, filled with -inf)
mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
```

### nn.Module pattern
```python
class MyLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear(x)
```

### Training loop skeleton
```python
model.train()
optimizer.zero_grad()
logits = model(x)                          # forward
loss = F.cross_entropy(...)                # loss
loss.backward()                            # backward (autograd)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()
```

---

## Attention: the math

Scaled dot-product attention for a single head:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √dₖ ) · V
```

- Q, K, V are linear projections of the input x: shape `(B, T, d_k)`
- `Q Kᵀ` gives an `(B, T, T)` matrix of raw attention scores
- Dividing by `√dₖ` prevents softmax saturation for large d_k
- Causal mask sets upper-triangle entries to `-inf` before softmax, so token i cannot attend to token j > i
- Result has same shape as V: `(B, T, d_k)`

Multi-head: run h independent heads with d_k = d_model / h, then concatenate and project back.

---

## Debugging shape errors

When you hit a shape mismatch, add these checks:
```python
print(x.shape)                  # quick check anywhere
assert x.shape == (B, T, d_model), f"got {x.shape}"
```

Common mistakes:
- Forgetting to transpose before matmul: `A @ B` vs `A @ B.transpose(-2, -1)`
- Wrong dim for softmax: should be `dim=-1` (over the last axis)
- Mixing up `view` and `reshape` — use `contiguous().view(...)` if you get a stride error
- Cross-entropy expects `(N, C)` logits and `(N,)` targets — flatten with `.view(-1, vocab_size)` and `.view(-1)`

---

## Hyperparameters (starting point)

```python
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 65        # character-level (~65 unique chars in Shakespeare)
    context_len: int = 128      # sequence length / block size
    d_model: int = 128          # embedding dimension
    n_heads: int = 4            # attention heads (d_model must be divisible by n_heads)
    n_layers: int = 4           # number of transformer blocks
    d_ff: int = 512             # feed-forward hidden dimension (typically 4 × d_model)
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 3e-4
    max_steps: int = 5000
    eval_interval: int = 500
    device: str = "cpu"         # switch to "cuda" or "mps" when available
```

Total parameters at these settings: approximately 2–4M.

---

## Resources

- PyTorch docs: https://pytorch.org/docs/stable/
- Attention is All You Need (original transformer paper): https://arxiv.org/abs/1706.03762
- Andrej Karpathy's nanoGPT (reference implementation): https://github.com/karpathy/nanoGPT
- Shakespeare dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
