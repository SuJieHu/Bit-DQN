"""YAML-driven experiment configuration.

A config file may contain a top-level ``extends: <other.yaml>`` key. The
referenced file is loaded first (recursively) and the current file overrides
its keys. This lets every variant override only what is different from
``configs/default.yaml`` while still being a self-contained, human-readable
record of one run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@dataclass
class ExperimentConfig:
    variant: str = "basic_dqn"
    algorithm: str = "dqn"

    max_sequence_length: int = 50
    min_sequence_length: int = 1

    episodes: int = 1200
    eval_episodes: int = 200
    eval_interval: int = 100

    seed: int = 42
    hidden_dim: int = 128
    arch: str = "mlp"                # mlp | transformer_encoder | transformer
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.0
    lr: float = 1.0e-3
    gamma: float = 0.98
    batch_size: int = 64
    buffer_size: int = 20000
    min_buffer_size: int = 500
    target_update_interval: int = 200

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000

    reward_mode: str = "sparse"

    double_dqn: bool = False
    dueling: bool = False
    per: bool = False
    her: bool = False
    her_ratio: float = 0.5

    device: str = "auto"
    results_dir: str = "results"
    run_name: str = "bit-dqn"
    save_examples: int = 12

    use_wandb: bool = False
    wandb_project: str = "bit-sequence-dqn"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None

    config_path: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def lengths(self) -> list[int]:
        return list(range(self.min_sequence_length, self.max_sequence_length + 1))

    @property
    def is_baseline(self) -> bool:
        return self.algorithm.startswith("baseline_")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _load_raw(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(path_or_name: str | Path) -> Path:
    """Locate a YAML config by absolute path, relative path under CONFIG_DIR,
    or bare name (with or without the .yaml extension). Subfolders are
    supported: ``mlp/basic_dqn`` resolves to ``configs/mlp/basic_dqn.yaml``."""
    p = Path(path_or_name)
    if p.is_file():
        return p.resolve()
    candidate = CONFIG_DIR / p
    if candidate.is_file():
        return candidate.resolve()
    # Append .yaml if it isn't there yet, *preserving* parent folders.
    if p.suffix != ".yaml":
        candidate = CONFIG_DIR / p.with_suffix(".yaml")
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find config: {path_or_name}")


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    out.update({k: v for k, v in override.items() if k != "extends"})
    return out


def _expand(path: Path, seen: set[Path]) -> Dict[str, Any]:
    if path in seen:
        raise ValueError(f"Circular extends detected at {path}")
    seen.add(path)
    raw = _load_raw(path)
    parent_key = raw.get("extends")
    if parent_key:
        parent_path = _resolve(parent_key) if not Path(parent_key).is_absolute() else Path(parent_key)
        if not parent_path.is_file():
            parent_path = (path.parent / parent_key).resolve()
        parent_cfg = _expand(parent_path, seen)
        return _merge(parent_cfg, raw)
    return raw


def load_config(path_or_name: str | Path, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Load a YAML config (with ``extends:`` chain) into an ExperimentConfig.

    ``overrides`` lets the CLI inject ad-hoc overrides like ``use_wandb=True``
    without touching the YAML file.
    """
    resolved = _resolve(path_or_name)
    merged = _expand(resolved, set())
    if overrides:
        merged = _merge(merged, overrides)
    known = {f.name for f in fields(ExperimentConfig)}
    known_kwargs = {k: v for k, v in merged.items() if k in known}
    extra = {k: v for k, v in merged.items() if k not in known and k != "extends"}
    cfg = ExperimentConfig(**known_kwargs, extra=extra, config_path=str(resolved))
    return cfg


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict()
    data.pop("extra", None)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
