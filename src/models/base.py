"""Abstract base class for Q-networks used by the bit-sequence DQN agent."""
from __future__ import annotations

import torch
import torch.nn as nn


class BaseQNetwork(nn.Module):
    """All Q-networks return ``[batch, n_actions=2]`` Q-values."""

    n_actions: int = 2

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError
