from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 65
    context_len: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 3e-4
    max_steps: int = 5000
    eval_interval: int = 500
    device: str = "cpu"
