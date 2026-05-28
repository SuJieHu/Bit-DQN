#!/usr/bin/env bash
# MLP -- ablation: dense per-bit reward shaping only
# Pair against the MLP twins on GPU 1.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="1"
python scripts/train.py --config mlp/reward_shaping "${COMMON_FLAGS[@]}" --device cuda "$@"
