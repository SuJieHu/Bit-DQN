"""Encoder-only Transformer Q-network for the bit-sequence task.

Motivation
----------
The MLP variants flatten the observation into a single ``2n`` vector and
let a small feed-forward network do all the work. That is enough for small
``n`` but has two structural problems as ``n`` grows:

1. The MLP must implicitly learn that the i-th block of the observation
   corresponds to position i. With ``n = 50`` this is a 100-D vector and
   the relevant local interaction (target_bit_i vs generated_bit_i) is
   buried inside it.
2. The number of parameters in the first linear layer scales as
   ``hidden_dim * 2n``. The network capacity is spent on a permuted
   bag-of-positions encoding, not on the structure of the task.

The encoder-only Transformer instead represents the observation as a
**sequence of n tokens**, each carrying the per-position features
``(target_bit_i, generated_value_i, is_current_position_i)``, where
``is_current_position`` is derived inside the model as "this is the
leftmost 0.5 in generated". A learned positional embedding is added on
top. Self-attention can then look at any two positions in O(1) hops,
which is the right inductive bias for "compare target to generated,
conditioned on which slot I am about to write".

A note on naming
----------------
This file is the *encoder-only* Transformer (``arch: transformer_encoder``).
The full sequence-to-sequence Transformer (bidirectional encoder over the
target + causal decoder over generated, with cross-attention) lives in
:mod:`transformer_q` and is keyed as ``arch: transformer``.

This network is intentionally drop-in compatible with the existing
factory: same input dimension (the flat ``2n`` observation), same
output shape (``[batch, 2]`` Q-values). The flat vector is reshaped
internally.

Notes
-----
* We use a learned ``[CLS]``-style query token at the *current write
  position* by simply gathering the encoded token at that index, instead
  of pooling, so the head sees the contextualised representation of the
  exact slot it is acting on.
* For ``n = 1`` the position embedding has only one entry; the network
  still works but the Transformer is overkill at that scale - the MLP
  variants are preferable for small ``n`` and the user can pick the
  architecture per variant in the YAML config.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import BaseQNetwork


def _infer_n(input_dim: int) -> int:
    """Recover the sequence length from the flat observation dimension.

    The env layout is ``[target (n) | generated (n)]``, so ``input_dim == 2n``.
    """
    if input_dim % 2 != 0:
        raise ValueError(
            f"Cannot infer sequence length from input_dim={input_dim}; "
            "expected 2n (target | generated)."
        )
    return input_dim // 2


class TransformerEncoderQNetwork(BaseQNetwork):
    """Encoder-only self-attention Q-network for bit-sequence generation.

    Parameters
    ----------
    input_dim : int
        Flat observation length ``2n`` from :class:`BitSequenceEnv`.
    hidden_dim : int
        Width of the token embedding / transformer ``d_model``.
    n_heads : int
        Multi-head attention heads. Default 4. Must divide ``hidden_dim``.
    n_layers : int
        Number of stacked ``TransformerEncoderLayer`` blocks. Default 2.
    dropout : float
        Dropout rate inside attention / FFN. Default 0.0 — the task is
        deterministic per episode so we don't need regularisation noise.
    dueling : bool
        If True, use a value + advantage head instead of a single head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        dueling: bool = False,
    ):
        super().__init__(input_dim, hidden_dim)
        self.n = _infer_n(input_dim)
        self.d_model = hidden_dim
        self.dueling = dueling

        if hidden_dim % n_heads != 0:
            # Fallback to a divisor of hidden_dim to avoid a hard crash.
            for candidate in (4, 2, 1):
                if hidden_dim % candidate == 0:
                    n_heads = candidate
                    break

        # Per-token feature: (target_bit, generated_value, is_current_position)
        self.token_proj = nn.Linear(3, hidden_dim)
        self.position_embedding = nn.Embedding(max(self.n, 1), hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        if dueling:
            self.value_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_actions),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.n_actions),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.position_embedding.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _split_obs(self, x: torch.Tensor):
        n = self.n
        target = x[:, :n]
        generated = x[:, n : 2 * n]
        return target, generated

    @staticmethod
    def _unwritten_mask(generated: torch.Tensor) -> torch.Tensor:
        """Bool tensor: True wherever generated[i] is the sentinel 0.5."""
        return (generated > 0.25) & (generated < 0.75)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        target, generated = self._split_obs(x)

        # Derived "is_current_position" feature: the leftmost unwritten slot.
        unwritten = self._unwritten_mask(generated).float()
        unwritten_cumsum = torch.cumsum(unwritten, dim=1)
        is_current = (unwritten * (unwritten_cumsum == 1).float())

        token_feats = torch.stack([target, generated, is_current], dim=-1)
        tokens = self.token_proj(token_feats)

        positions = torch.arange(self.n, device=x.device).unsqueeze(0).expand(batch_size, -1)
        tokens = tokens + self.position_embedding(positions)

        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)

        # Gather the encoded token at the current write position. When the
        # episode has terminated (no 0.5 left) fall back to the last slot.
        has_current = is_current.sum(dim=1) > 0
        current_index = is_current.argmax(dim=1)
        current_index = torch.where(
            has_current,
            current_index,
            torch.full_like(current_index, self.n - 1),
        )
        index_gather = current_index.view(batch_size, 1, 1).expand(-1, 1, self.d_model)
        focused = encoded.gather(1, index_gather).squeeze(1)

        if self.dueling:
            value = self.value_head(focused)
            advantage = self.advantage_head(focused)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.head(focused)
