"""Non-learning baselines: random and oracle copy.

These do not need training. We evaluate them with the same eval protocol as
DQN so the numbers are comparable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .config import ExperimentConfig, dump_config
from .utils import (
    average_metrics,
    compute_sequence_metrics,
    make_example,
    save_csv,
    save_json,
)

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None


def _eval_policy(policy: str, length: int, eval_episodes: int, seed: int, save_examples: int) -> Tuple[Dict[str, float], List[dict]]:
    rng = np.random.default_rng(seed)
    metrics = []
    examples: List[dict] = []
    for episode in range(eval_episodes):
        target = rng.integers(0, 2, size=length, dtype=np.int64)
        if policy == "random":
            generated = rng.integers(0, 2, size=length, dtype=np.int64)
        elif policy == "oracle":
            generated = target.copy()
        else:
            raise ValueError(f"Unknown baseline policy: {policy}")
        metrics.append(compute_sequence_metrics(target, generated))
        if len(examples) < save_examples:
            examples.append(make_example(length, target, generated, f"{policy}_{episode}"))
    return average_metrics(metrics), examples


def _plot(path: Path, lengths: List[int], summary: List[dict], variant: str) -> None:
    success = [row["success_rate"] for row in summary]
    bit_acc = [row["bit_accuracy"] for row in summary]
    prefix_acc = [row["prefix_accuracy"] for row in summary]
    plt.figure(figsize=(9, 5))
    plt.plot(lengths, success, marker="o", label="success_rate")
    plt.plot(lengths, bit_acc, marker="s", label="bit_accuracy")
    plt.plot(lengths, prefix_acc, marker="^", label="prefix_accuracy")
    plt.yscale("symlog", linthresh=1e-3)
    plt.xlabel("Sequence length n")
    plt.ylabel("Metric (averaged over eval episodes)")
    plt.title(f"{variant}: metrics vs sequence length")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def run_baseline(cfg: ExperimentConfig) -> List[dict]:
    if cfg.algorithm == "baseline_random":
        policy = "random"
    elif cfg.algorithm == "baseline_oracle":
        policy = "oracle"
    else:
        raise ValueError(f"run_baseline expects a baseline algorithm, got {cfg.algorithm}")

    out_dir = Path(cfg.results_dir) / cfg.run_name / cfg.variant
    out_dir.mkdir(parents=True, exist_ok=True)
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
        # Keep run.summary empty: only the length_curve scalars are logged
        # and they appear purely as a curve in the UI.
        wandb_run.define_metric("length_curve/n", summary="none")
        wandb_run.define_metric(
            "length_curve/*", step_metric="length_curve/n", summary="none"
        )

    summary: List[dict] = []
    all_examples: Dict[int, List[dict]] = {}
    for length in cfg.lengths:
        metrics, examples = _eval_policy(
            policy, length, cfg.eval_episodes, cfg.seed + length, cfg.save_examples
        )
        all_examples[length] = examples
        row = {"variant": cfg.variant, "sequence_length": length, **metrics}
        summary.append(row)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "length_curve/n": length,
                    "length_curve/success_rate": metrics["success_rate"],
                    "length_curve/bit_accuracy": metrics["bit_accuracy"],
                    "length_curve/prefix_accuracy": metrics["prefix_accuracy"],
                }
            )
        print(
            f"[baseline] policy={policy} n={length} "
            f"success_rate={metrics['success_rate']:.4f} "
            f"bit_accuracy={metrics['bit_accuracy']:.4f}"
        )

    save_json(out_dir / "summary.json", summary)
    save_csv(out_dir / "summary.csv", summary)
    save_json(out_dir / "examples.json", {str(k): v for k, v in all_examples.items()})
    _plot(out_dir / "success_vs_n.png", cfg.lengths, summary, cfg.variant)

    if wandb_run is not None:
        # length_curve/* scalars already give a clean curve of
        # success_rate / bit_accuracy / prefix_accuracy vs n; no
        # final tables are logged to wandb. summary.csv / summary.json
        # are kept locally for the report.
        wandb_run.finish()

    return summary
