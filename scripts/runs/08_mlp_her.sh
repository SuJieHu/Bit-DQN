#!/usr/bin/env bash
# MLP -- ablation: Hindsight Experience Replay only
# Pair against the MLP twins on GPU 5.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="5"
python scripts/train.py --config mlp/her "${COMMON_FLAGS[@]}" --device cuda "$@"
