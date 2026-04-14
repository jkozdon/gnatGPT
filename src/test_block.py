# scratch test — run from gnatGPT/src/
import torch
from config import Config
from block import TransformerBlock

cfg = Config()
block = TransformerBlock(cfg)

B, T = 2, cfg.context_len
x = torch.randn(B, T, cfg.d_model)
out = block(x)

print(out.shape)  # should be (2, 128, 128)
assert out.shape == (B, T, cfg.d_model)
print("block OK")
n_params = sum(p.numel() for p in block.parameters())
print(f"params per block: {n_params:,}")
