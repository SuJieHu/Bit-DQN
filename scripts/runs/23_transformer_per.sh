#!/usr/bin/env bash
# Full Transformer (encoder + decoder) -- ablation: Prioritized Experience Replay only
# Pair against the Full Transformer (encoder + decoder) twins on GPU 4.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="4"
python scripts/train.py --config transformer/per "${COMMON_FLAGS[@]}" --device cuda "$@"
