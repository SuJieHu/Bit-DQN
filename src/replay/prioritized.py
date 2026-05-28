"""Prioritized Experience Replay (Schaul et al. 2016, proportional variant)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, transition: Transition) -> None:
        max_priority = float(self.priorities.max()) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        priorities = self.priorities[: len(self.buffer)]
        probs = priorities ** self.alpha
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        return batch, weights.astype(np.float32), indices

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(abs(priority) + 1e-5)

    def __len__(self) -> int:
        return len(self.buffer)
