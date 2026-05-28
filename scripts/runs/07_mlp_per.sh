#!/usr/bin/env bash
# MLP -- ablation: Prioritized Experience Replay only
# Pair against the MLP twins on GPU 4.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="4"
python scripts/train.py --config mlp/per "${COMMON_FLAGS[@]}" --device cuda "$@"
