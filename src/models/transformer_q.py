"""Full encoder-decoder Transformer Q-network with masked attention.

Why an encoder-decoder architecture
-----------------------------------
The companion ``TransformerEncoderQNetwork`` is the encoder-only variant:
every position attends to every other position bidirectionally over a
single token stream that already co-locates ``(g_i, y_hat_i,
is_current_i)`` at index ``i``. It works because the unwritten slots
are filled with the ``0.5`` sentinel and self-attention can *learn* to
ignore them. But that inductive bias should be free, not learned. The
agent is autoregressive -- when deciding bit at position ``t`` it has
already committed to bits ``0..t-1`` and has not yet touched
``t+1..n-1`` -- so the principled architecture for sequence-to-sequence
goal-conditioned generation is:

* **bidirectional encoder over the target** (the goal is fully visible
  to the policy, otherwise the MDP is not Markov: two different targets
  whose generated prefixes coincide would yield the same state but
  different optimal actions and rewards);

* **causal decoder over the generated stream** (position ``t`` may
  attend to ``generated[0..t]`` only, never to the future), with
  **cross-attention to the target memory** so each decoder position
  can "look up" what the target says at any index.

This matches the standard seq2seq encoder-decoder used in machine
translation. The model is drop-in compatible with the existing factory:
same flat ``2n`` observation in, same ``[batch, 2]`` Q-values out. The
flat vector is split into target and generated internally.

A note on naming
----------------
This class is the *full* Transformer (encoder + decoder), and is hence
keyed as ``arch: transformer`` in the YAML configs. The encoder-only
variant lives in :mod:`transformer_encoder_q` and is keyed as
``arch: transformer_encoder``. A *decoder-only* configuration (the
GPT-style ``[target || generated]`` flat-causal arrangement) is not
implemented because it would be strictly weaker than both of the above
on this task: it would force a *causal* encoding of the target (losing
bidirectional goal visibility) without recovering the per-position
co-location that the encoder-only model already exploits.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseQNetwork
from .transformer_encoder_q import _infer_n


class TransformerQNetwork(BaseQNetwork):
    """Full Transformer: bidirectional target encoder + causal decoder
    over generated with cross-attention.

    Differences from :class:`TransformerEncoderQNetwork`:

    * Target and generated are two separate token streams (not packed
      into a single per-position feature vector).
    * The decoder uses a causal mask on the generated stream so position
      ``t`` cannot peek at ``generated[t+1..n-1]``.
    * Cross-attention to the encoder memory is bidirectional, i.e. any
      decoder position can attend to every target position -- the goal
      stays fully visible.
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
            for candidate in (4, 2, 1):
                if hidden_dim % candidate == 0:
                    n_heads = candidate
                    break

        # Target stream: a single 0/1 bit -> d_model, plus position embedding.
        self.target_proj = nn.Linear(1, hidden_dim)
        self.target_pos_embed = nn.Embedding(max(self.n, 1), hidden_dim)

        # Generated stream: value (0/0.5/1) and "is_current" flag.
        self.gen_proj = nn.Linear(2, hidden_dim)
        self.gen_pos_embed = nn.Embedding(max(self.n, 1), hidden_dim)

        # Bidirectional encoder over the target sequence.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.target_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Causal decoder over the generated sequence, attending to target memory.
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.generated_decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
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

        # Pre-build a static causal mask once. nn.Transformer expects -inf above
        # the diagonal for positions that must NOT be attended to.
        causal = torch.triu(
            torch.full((max(self.n, 1), max(self.n, 1)), float("-inf")),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.target_pos_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.gen_pos_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _split_obs(self, x: torch.Tensor):
        n = self.n
        return x[:, :n], x[:, n : 2 * n]

    @staticmethod
    def _unwritten_mask(generated: torch.Tensor) -> torch.Tensor:
        return (generated > 0.25) & (generated < 0.75)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        target, generated = self._split_obs(x)

        unwritten = self._unwritten_mask(generated).float()
        unwritten_cumsum = torch.cumsum(unwritten, dim=1)
        is_current = unwritten * (unwritten_cumsum == 1).float()

        positions = (
            torch.arange(self.n, device=x.device).unsqueeze(0).expand(batch_size, -1)
        )

        # Encode target (bidirectional).
        target_tokens = self.target_proj(target.unsqueeze(-1)) + self.target_pos_embed(positions)
        target_memory = self.target_encoder(target_tokens)

        # Decode generated (causal self-attention + cross-attention to target memory).
        gen_feat = torch.stack([generated, is_current], dim=-1)
        gen_tokens = self.gen_proj(gen_feat) + self.gen_pos_embed(positions)
        decoded = self.generated_decoder(
            tgt=gen_tokens,
            memory=target_memory,
            tgt_mask=self.causal_mask,
        )
        decoded = self.norm(decoded)

        # Gather the decoder output at the current write position. Mirrors
        # TransformerEncoderQNetwork: when the episode has terminated (no
        # 0.5 left) fall back to the last slot.
        has_current = is_current.sum(dim=1) > 0
        current_index = is_current.argmax(dim=1)
        current_index = torch.where(
            has_current,
            current_index,
            torch.full_like(current_index, self.n - 1),
        )
        idx = current_index.view(batch_size, 1, 1).expand(-1, 1, self.d_model)
        focused = decoded.gather(1, idx).squeeze(1)

        if self.dueling:
            value = self.value_head(focused)
            advantage = self.advantage_head(focused)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.head(focused)
