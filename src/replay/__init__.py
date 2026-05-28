"""Replay buffers (uniform and prioritized)."""
from .uniform import UniformReplayBuffer
from .prioritized import PrioritizedReplayBuffer


def build_replay_buffer(capacity: int, prioritized: bool):
    if prioritized:
        return PrioritizedReplayBuffer(capacity)
    return UniformReplayBuffer(capacity)


__all__ = ["UniformReplayBuffer", "PrioritizedReplayBuffer", "build_replay_buffer"]
