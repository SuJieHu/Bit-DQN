"""Single entrypoint: ``python scripts/train.py --config configs/<name>.yaml``.

CLI overrides are minimal: only meta-options that you commonly toggle between
runs (wandb on/off, run name, results directory, max sequence length). All
algorithmic knobs live in the YAML config, so each run is fully described by
its config file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config  # noqa: E402
from src.baselines import run_baseline  # noqa: E402
from src.trainer import train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bit-sequence DQN trainer (YAML-config driven)")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="YAML config file name (under configs/) or absolute path.",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Force-enable wandb logging.")
    parser.add_argument("--no-wandb", action="store_true", help="Force-disable wandb logging.")
    parser.add_argument("--run-name", default=None, help="Override results subdirectory name.")
    parser.add_argument("--results-dir", default=None, help="Override results root directory.")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=None,
        help="Override max_sequence_length from the YAML (debug / smoke runs).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episodes per n (debug / smoke runs).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override eval_episodes (debug / smoke runs).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides: dict = {}
    if args.use_wandb:
        overrides["use_wandb"] = True
    if args.no_wandb:
        overrides["use_wandb"] = False
    for key in [
        "run_name",
        "results_dir",
        "max_sequence_length",
        "episodes",
        "eval_episodes",
        "seed",
        "device",
        "wandb_project",
        "wandb_entity",
    ]:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    cfg = load_config(args.config, overrides=overrides)
    print(f"[train] loaded config: {cfg.config_path}")
    print(f"[train] variant={cfg.variant} algorithm={cfg.algorithm}")
    print(f"[train] lengths={cfg.lengths} episodes={cfg.episodes} use_wandb={cfg.use_wandb}")

    if cfg.is_baseline:
        run_baseline(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
