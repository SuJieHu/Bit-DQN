"""DQN agent: owns the online + target Q-networks, the replay buffer, the
optimiser, action selection, and one optimisation step.

The trainer drives the agent through episodes; the agent itself is stateless
w.r.t. logging so that swapping in another algorithm is local.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .config import ExperimentConfig
from .models import build_q_network
from .replay import build_replay_buffer

Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


def epsilon_by_step(cfg: ExperimentConfig, step: int) -> float:
    ratio = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + ratio * (cfg.epsilon_end - cfg.epsilon_start)


class DQNAgent:
    def __init__(self, cfg: ExperimentConfig, obs_dim: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        arch_kwargs = {
            "n_heads": cfg.transformer_heads,
            "n_layers": cfg.transformer_layers,
            "dropout": cfg.transformer_dropout,
        }
        self.q_net = build_q_network(
            obs_dim, cfg.hidden_dim, cfg.dueling, arch=cfg.arch, arch_kwargs=arch_kwargs
        ).to(device)
        self.target_net = build_q_network(
            obs_dim, cfg.hidden_dim, cfg.dueling, arch=cfg.arch, arch_kwargs=arch_kwargs
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay = build_replay_buffer(cfg.buffer_size, prioritized=cfg.per)
        self.global_step = 0

    def epsilon(self) -> float:
        return epsilon_by_step(self.cfg, self.global_step)

    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.q_net(obs_t), dim=1).item())

    def greedy(self, obs: np.ndarray) -> int:
        return self.act(obs, epsilon=0.0)

    def store(self, transition: Transition) -> None:
        self.replay.add(transition)

    def store_her(
        self,
        episode_transitions: List[Transition],
        generated: np.ndarray,
        sequence_length: int,
    ) -> None:
        """HER relabelling: pretend the bits we generated were the goal all
        along. Crucially this turns a failed sparse-reward trajectory into a
        successful one and so injects positive learning signal even when the
        agent never reaches the real target."""
        if not episode_transitions:
            return
        if random.random() > self.cfg.her_ratio:
            return
        hindsight_target = generated.astype(np.float32)
        n = sequence_length
        for obs, action, _, next_obs, done in episode_transitions:
            obs_h = obs.copy()
            next_obs_h = next_obs.copy()
            obs_h[:n] = hindsight_target
            next_obs_h[:n] = hindsight_target
            # Current write position = index of the first 0.5 in the
            # generated half of obs (obs is laid out as [target | generated]).
            generated_field = obs[n : 2 * n]
            unwritten = np.isclose(generated_field, 0.5)
            position = int(np.argmax(unwritten)) if unwritten.any() else n - 1
            correct = action == int(hindsight_target[position])
            if self.cfg.reward_mode == "shaped":
                reward = (1.0 if correct else -0.25) / n
                if done:
                    reward += 1.0
            else:
                reward = 1.0 if done else 0.0
            self.replay.add((obs_h, action, float(reward), next_obs_h, done))

    def optimize(self) -> Optional[float]:
        cfg = self.cfg
        if len(self.replay) < max(cfg.min_buffer_size, cfg.batch_size):
            return None
        batch, weights, indices = self.replay.sample(cfg.batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        device = self.device
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_obs_t = torch.tensor(np.array(next_obs), dtype=torch.float32, device=device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

        q_values = self.q_net(obs_t).gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            if cfg.double_dqn:
                next_actions = torch.argmax(self.q_net(next_obs_t), dim=1, keepdim=True)
                next_q = self.target_net(next_obs_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_obs_t).max(dim=1).values
            target = rewards_t + cfg.gamma * next_q * (1.0 - dones_t)
        td_error = target - q_values
        loss = (weights_t * F.smooth_l1_loss(q_values, target, reduction="none")).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()
        self.replay.update_priorities(indices, td_error.detach().abs().cpu().numpy())
        return float(loss.item())

    def maybe_update_target(self) -> None:
        if self.global_step % self.cfg.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def state_dict(self):
        return self.q_net.state_dict()
