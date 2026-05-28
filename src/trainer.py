"""Train a single variant across every n in [min_n, max_n] and aggregate
results into a few large files (instead of one folder per n).

Logging strategy
----------------
* One **wandb run per variant**.  Everything sent to wandb is a plain
  scalar so the run page only shows **curves** in the UI, with no tables,
  no Custom Charts and no entries in ``run.summary``:

    - ``train/n{n}/{episode_reward,epsilon,loss}`` --
      logged every episode, x-axis = ``train/n{n}/episode``.
    - ``eval/n{n}/{success_rate,bit_accuracy,prefix_accuracy}`` --
      logged once every ``eval_interval`` episodes, same x-axis.
    - ``length_curve/{success_rate,bit_accuracy,prefix_accuracy}`` --
      one point per n, x-axis = ``length_curve/n``.  This is the
      headline "metric vs n" curve.

  Every metric is registered with ``define_metric(..., summary="none")``
  so the wandb run summary stays empty; the per-n ``step_metric``
  guarantees that each n's panel starts at episode 1, independent of
  wandb's monotonically increasing internal step.

* No final tables, ``wandb.Table`` artefacts or ``wandb.plot.*`` Custom
  Charts are logged.  The local ``summary.csv`` / ``summary.json`` /
  ``success_vs_n.png`` cover the per-variant aggregate view for the
  written report.

* Local results: a single ``summary.json`` / ``summary.csv`` covering all n,
  a single ``training_curves.json`` keyed by n, a single ``examples.json``
  keyed by n, a single ``success_vs_n.png`` and one ``models/n{n}.pt`` per n.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from .agent import DQNAgent
from .config import ExperimentConfig, dump_config
from .env import BitSequenceEnv
from .utils import (
    average_metrics,
    compute_sequence_metrics,
    make_example,
    resolve_device,
    save_csv,
    save_json,
    set_seed,
)

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


def _eval_agent(agent: DQNAgent, length: int, eval_episodes: int, seed: int, save_examples: int = 0):
    env = BitSequenceEnv(length, reward_mode="sparse", seed=seed)
    metrics = []
    examples = []
    agent.q_net.eval()
    for episode in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.greedy(obs)
            obs, _, done, _, _ = env.step(action)
        metrics.append(compute_sequence_metrics(env.target, env.generated))
        if save_examples and len(examples) < save_examples:
            examples.append(make_example(length, env.target, env.generated, f"eval_{episode}"))
    agent.q_net.train()
    return average_metrics(metrics), examples


def _train_one_length(
    cfg: ExperimentConfig,
    length: int,
    device: torch.device,
    wandb_run,
) -> Tuple[Dict[str, float], List[dict], List[dict], DQNAgent]:
    """Train one Q-network on sequence length ``length`` and stream
    per-episode metrics to wandb as plain scalars.

    Each n gets its own step-metric (``train/n{length}/episode``) so its
    panel x-axis starts at 1 regardless of wandb's monotonically
    increasing internal step counter. ``summary="none"`` is used on every
    metric to keep run.summary completely empty (only curves remain in
    the wandb UI; no scalar leftovers under train/, eval/ or
    length_curve/)."""
    set_seed(cfg.seed + length)
    env = BitSequenceEnv(length, reward_mode=cfg.reward_mode, seed=cfg.seed + length)
    agent = DQNAgent(cfg, env.obs_dim, device)
    history: List[dict] = []

    if wandb_run is not None:
        episode_metric = f"train/n{length}/episode"
        wandb_run.define_metric(episode_metric, summary="none")
        wandb_run.define_metric(f"train/n{length}/*", step_metric=episode_metric, summary="none")
        wandb_run.define_metric(f"eval/n{length}/*", step_metric=episode_metric, summary="none")

    progress = trange(cfg.episodes, desc=f"{cfg.variant} n={length}", leave=False)
    for episode in progress:
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        losses = []
        episode_transitions = []
        while not done:
            epsilon = agent.epsilon()
            action = agent.act(obs, epsilon)
            next_obs, reward, done, _, _ = env.step(action)
            transition = (obs, action, reward, next_obs, done)
            agent.store(transition)
            episode_transitions.append(transition)
            loss = agent.optimize()
            if loss is not None:
                losses.append(loss)
            obs = next_obs
            episode_reward += reward
            agent.global_step += 1
            agent.maybe_update_target()
        if cfg.her:
            agent.store_her(episode_transitions, env.generated, length)

        record = {
            "episode": episode + 1,
            "episode_reward": float(episode_reward),
            "epsilon": float(agent.epsilon()),
            "loss": float(np.mean(losses)) if losses else float("nan"),
        }
        if (episode + 1) % cfg.eval_interval == 0 or episode + 1 == cfg.episodes:
            eval_metrics, _ = _eval_agent(
                agent, length, cfg.eval_episodes, cfg.seed + 10000 + episode, save_examples=0
            )
            record.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            progress.set_postfix(
                success=f"{eval_metrics['success_rate']:.2f}",
                bit_acc=f"{eval_metrics['bit_accuracy']:.2f}",
            )

        if wandb_run is not None:
            payload = {
                f"train/n{length}/episode": record["episode"],
                f"train/n{length}/episode_reward": record["episode_reward"],
                f"train/n{length}/epsilon": record["epsilon"],
            }
            if not np.isnan(record["loss"]):
                payload[f"train/n{length}/loss"] = record["loss"]
            if "eval_success_rate" in record:
                payload[f"eval/n{length}/success_rate"] = record["eval_success_rate"]
                payload[f"eval/n{length}/bit_accuracy"] = record["eval_bit_accuracy"]
                payload[f"eval/n{length}/prefix_accuracy"] = record["eval_prefix_accuracy"]
            wandb_run.log(payload)

        history.append(record)

    final_metrics, examples = _eval_agent(
        agent,
        length,
        cfg.eval_episodes,
        cfg.seed + 20000,
        save_examples=cfg.save_examples,
    )
    return final_metrics, history, examples, agent


def _plot_success_vs_n(path: Path, lengths: List[int], summary: List[dict]) -> None:
    success = [row["success_rate"] for row in summary]
    bit_acc = [row["bit_accuracy"] for row in summary]
    prefix_acc = [row["prefix_accuracy"] for row in summary]
    plt.figure(figsize=(9, 5))
    plt.plot(lengths, success, marker="o", label="success_rate")
    plt.plot(lengths, bit_acc, marker="s", label="bit_accuracy")
    plt.plot(lengths, prefix_acc, marker="^", label="prefix_accuracy")
    plt.xlabel("Sequence length n")
    plt.ylabel("Metric (averaged over eval episodes)")
    plt.title(f"{summary[0].get('variant', 'variant')}: metrics vs sequence length")
    plt.ylim(-0.02, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_training_curves(path: Path, training_curves: Dict[int, List[dict]], variant: str) -> None:
    if not training_curves:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    ns = sorted(training_curves.keys())
    for i, n in enumerate(ns):
        history = training_curves[n]
        episodes = [h["episode"] for h in history]
        rewards = [h.get("episode_reward", float("nan")) for h in history]
        ax.plot(episodes, rewards, color=cmap(i / max(1, len(ns) - 1)), label=f"n={n}", alpha=0.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("episode_reward")
    ax.set_title(f"{variant}: training reward curves per n")
    ax.grid(alpha=0.3)
    if len(ns) <= 12:
        ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(cfg: ExperimentConfig) -> List[dict]:
    device = resolve_device(cfg.device)
    print(f"[trainer] variant={cfg.variant} device={device} lengths={cfg.lengths}")

    out_dir = Path(cfg.results_dir) / cfg.run_name / cfg.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)
    dump_config(cfg, out_dir / "config.yaml")

    wandb_run = None
    if cfg.use_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=f"{cfg.run_name}-{cfg.variant}",
            group=cfg.wandb_group or cfg.variant,
            config=cfg.to_dict(),
            reinit=True,
        )
        # Cross-n curve (one point per length). ``summary="none"`` keeps
        # the run.summary panel empty -- the wandb UI only shows curves.
        wandb_run.define_metric("length_curve/n", summary="none")
        wandb_run.define_metric(
            "length_curve/*", step_metric="length_curve/n", summary="none"
        )

    summary_rows: List[dict] = []
    training_curves: Dict[int, List[dict]] = {}
    all_examples: Dict[int, List[dict]] = {}

    for length in cfg.lengths:
        final_metrics, history, examples, agent = _train_one_length(
            cfg, length, device, wandb_run
        )
        torch.save(agent.state_dict(), models_dir / f"n{length}.pt")
        training_curves[length] = history
        all_examples[length] = examples
        row = {
            "variant": cfg.variant,
            "sequence_length": length,
            **final_metrics,
        }
        summary_rows.append(row)
        print(
            f"[trainer] variant={cfg.variant} n={length}"
            f" success_rate={final_metrics['success_rate']:.3f}"
            f" bit_acc={final_metrics['bit_accuracy']:.3f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "length_curve/n": length,
                    "length_curve/success_rate": final_metrics["success_rate"],
                    "length_curve/bit_accuracy": final_metrics["bit_accuracy"],
                    "length_curve/prefix_accuracy": final_metrics["prefix_accuracy"],
                }
            )

    save_json(out_dir / "summary.json", summary_rows)
    save_csv(out_dir / "summary.csv", summary_rows)
    save_json(out_dir / "training_curves.json", {str(k): v for k, v in training_curves.items()})
    save_json(out_dir / "examples.json", {str(k): v for k, v in all_examples.items()})
    _plot_success_vs_n(out_dir / "success_vs_n.png", cfg.lengths, summary_rows)
    _plot_training_curves(out_dir / "training_curves.png", training_curves, cfg.variant)

    if wandb_run is not None:
        # No final tables / Custom Charts logged to wandb on purpose:
        # the per-n curves (train/n{k}/*, eval/n{k}/*) and the
        # ``length_curve/*`` scalars already cover everything the
        # professor needs to see. Tables are kept locally as
        # ``summary.csv`` / ``summary.json``.
        wandb_run.finish()

    return summary_rows
