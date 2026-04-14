import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

        self.expand = nn.Linear(cfg.d_model, cfg.d_ff)
        self.project = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        x = self.gelu(self.expand(x))
        return self.dropout(self.project(x))
