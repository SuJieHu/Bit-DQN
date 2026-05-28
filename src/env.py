"""Gymnasium environment for the bit-sequence-generation task.

A target bit sequence of length ``n`` is sampled at every ``reset``. The agent
writes one bit per step (action 0 or 1) from position 0 to position n-1. The
episode terminates after exactly n steps. The default reward is sparse: +1 at
the terminal step iff the generated sequence equals the target, else 0.

Observation layout (length ``2n``):

    [ target_bits (n) , generated_so_far (n) ]

* generated bits not yet written are encoded as ``0.5`` so the network can
  tell "not yet written" apart from explicit 0s and 1s;
* the current write position ``t`` is therefore implicitly the index of the
  first ``0.5`` from the left; we do not store a separate position one-hot
  or progress scalar because they are functions of ``generated`` (and the
  network — be it MLP or Transformer — can recover them with a single
  layer).
"""
from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BitSequenceEnv(gym.Env):
    """Token-by-token bit-sequence generation environment."""

    metadata = {"render_modes": []}

    def __init__(self, sequence_length: int, reward_mode: str = "sparse", seed: int = 0):
        super().__init__()
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if reward_mode not in {"sparse", "shaped"}:
            raise ValueError(f"unknown reward_mode {reward_mode}")
        self.sequence_length = sequence_length
        self.reward_mode = reward_mode
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * sequence_length,),
            dtype=np.float32,
        )
        self.target = np.zeros(sequence_length, dtype=np.int64)
        self.generated = np.full(sequence_length, -1, dtype=np.int64)
        self.position = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        target = None if options is None else options.get("target")
        if target is None:
            self.target = self.rng.integers(0, 2, size=self.sequence_length, dtype=np.int64)
        else:
            self.target = np.array(target, dtype=np.int64)
        self.generated = np.full(self.sequence_length, -1, dtype=np.int64)
        self.position = 0
        return self._obs(), {}

    def step(self, action: int):
        action = int(action)
        correct = action == int(self.target[self.position])
        self.generated[self.position] = action
        self.position += 1
        terminated = self.position >= self.sequence_length
        reward = 0.0
        if self.reward_mode == "shaped":
            reward = (1.0 if correct else -0.25) / self.sequence_length
        if terminated:
            exact = bool(np.array_equal(self.generated, self.target))
            if self.reward_mode == "sparse":
                reward = 1.0 if exact else 0.0
            else:
                reward += 1.0 if exact else -0.25
        return self._obs(), float(reward), terminated, False, {}

    def _obs(self) -> np.ndarray:
        generated = self.generated.astype(np.float32)
        generated[generated < 0] = 0.5
        return np.concatenate(
            [self.target.astype(np.float32), generated]
        ).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return int(self.observation_space.shape[0])
