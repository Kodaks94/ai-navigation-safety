import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn

@dataclass
class PolicyConfig:
    patch: int = 9          # local window size (odd)
    d_model: int = 64
    nhead: int = 4
    layers: int = 2
    ff: int = 128
    dropout: float = 0.1
    actions: int = 8        # with diagonals

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerPolicy(nn.Module):
    def __init__(self, in_dim: int, cfg: PolicyConfig = PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        self.input = nn.Linear(in_dim, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)
        layer = nn.TransformerEncoderLayer(d_model=cfg.d_model, nhead=cfg.nhead, dim_feedforward=cfg.ff, dropout=cfg.dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg.layers)
        self.head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.ReLU(), nn.Linear(cfg.d_model, cfg.actions))

    def forward(self, tokens):
        # tokens: (B, T, D_in) where T = patch*patch
        h = self.input(tokens)
        h = self.pos(h)
        h = self.enc(h)
        # Use the CLS-style: mean pool tokens
        h = h.mean(dim=1)
        logits = self.head(h)
        return logits
