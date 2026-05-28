#!/usr/bin/env bash
# Transformer encoder -- ablation: dense per-bit reward shaping only
# Pair against the Transformer encoder twins on GPU 1.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="1"
python scripts/train.py --config transformer_encoder/reward_shaping "${COMMON_FLAGS[@]}" --device cuda "$@"
