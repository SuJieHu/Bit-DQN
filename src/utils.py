"""Shared utilities (seeding, device resolution, metrics, IO)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def compute_sequence_metrics(target: Sequence[int], generated: Sequence[int]) -> Dict[str, float]:
    target_arr = np.array(target, dtype=np.int64)
    gen_arr = np.array(generated, dtype=np.int64)
    matches = target_arr == gen_arr
    prefix = 0
    for ok in matches:
        if not ok:
            break
        prefix += 1
    return {
        "success_rate": float(np.all(matches)),
        "bit_accuracy": float(np.mean(matches)),
        "prefix_accuracy": float(prefix / len(target_arr)),
    }


def average_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    return {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}


def make_example(length: int, target: Sequence[int], generated: Sequence[int], sample_id: str) -> dict:
    target_list = [int(x) for x in target]
    generated_list = [int(x) for x in generated]
    match_mask = [int(a == b) for a, b in zip(target_list, generated_list)]
    metrics = compute_sequence_metrics(target_list, generated_list)
    return {
        "sample_id": sample_id,
        "sequence_length": length,
        "target": "".join(map(str, target_list)),
        "generated": "".join(map(str, generated_list)),
        "match_mask": "".join(map(str, match_mask)),
        **metrics,
    }


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
