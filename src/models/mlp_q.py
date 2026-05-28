"""Vanilla 2-layer MLP Q-network."""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseQNetwork


class MLPQNetwork(BaseQNetwork):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
