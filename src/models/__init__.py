"""Q-network architectures used by the bit-sequence DQN agent.

Add a new model by creating ``src/models/<name>.py`` with a subclass of
``BaseQNetwork`` and registering it in :func:`build_q_network`.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseQNetwork
from .mlp_q import MLPQNetwork
from .dueling_q import DuelingQNetwork
from .transformer_encoder_q import TransformerEncoderQNetwork
from .transformer_q import TransformerQNetwork


def build_q_network(
    input_dim: int,
    hidden_dim: int,
    dueling: bool,
    arch: str = "mlp",
    arch_kwargs: Optional[Dict[str, Any]] = None,
) -> BaseQNetwork:
    """Factory for Q-networks.

    ``arch`` selects the family; ``dueling`` is honoured for every family
    that supports it (currently mlp, transformer_encoder, transformer).
    For backward compatibility with the original MLP/Dueling pair,
    ``arch='mlp'`` with ``dueling=True`` still returns the dedicated
    DuelingQNetwork.

    Architectures
    -------------
    mlp                  : 2-layer feed-forward over the flat ``2n`` obs.
    transformer_encoder  : bidirectional self-attention over ``n`` tokens
                           carrying ``(target, generated, is_current)``.
                           Encoder-only.
    transformer          : full seq2seq Transformer -- bidirectional
                           encoder over the target + causal decoder over
                           generated with cross-attention. The
                           "principled" autoregressive variant.
    """
    arch_kwargs = dict(arch_kwargs or {})
    if arch == "mlp":
        if dueling:
            return DuelingQNetwork(input_dim, hidden_dim)
        return MLPQNetwork(input_dim, hidden_dim)
    if arch == "transformer_encoder":
        return TransformerEncoderQNetwork(
            input_dim,
            hidden_dim,
            dueling=dueling,
            **arch_kwargs,
        )
    if arch == "transformer":
        return TransformerQNetwork(
            input_dim,
            hidden_dim,
            dueling=dueling,
            **arch_kwargs,
        )
    raise ValueError(f"Unknown Q-network architecture: {arch}")


__all__ = [
    "BaseQNetwork",
    "MLPQNetwork",
    "DuelingQNetwork",
    "TransformerEncoderQNetwork",
    "TransformerQNetwork",
    "build_q_network",
]
