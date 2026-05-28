"""Dueling DQN architecture: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))."""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseQNetwork


class DuelingQNetwork(BaseQNetwork):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        value = self.value(z)
        advantage = self.advantage(z)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
