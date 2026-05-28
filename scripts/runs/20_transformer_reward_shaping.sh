#!/usr/bin/env bash
# Full Transformer (encoder + decoder) -- ablation: dense per-bit reward shaping only
# Pair against the Full Transformer (encoder + decoder) twins on GPU 1.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="1"
python scripts/train.py --config transformer/reward_shaping "${COMMON_FLAGS[@]}" --device cuda "$@"
