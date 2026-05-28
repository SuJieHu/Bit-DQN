"""Plain uniform-sampling replay buffer."""
from __future__ import annotations

import random
from collections import deque
from typing import Sequence, Tuple

import numpy as np

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class UniformReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        weights = np.ones(batch_size, dtype=np.float32)
        indices = np.arange(batch_size)
        return batch, weights, indices

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        return None

    def __len__(self) -> int:
        return len(self.buffer)
