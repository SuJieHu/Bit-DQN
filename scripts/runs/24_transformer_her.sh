#!/usr/bin/env bash
# Full Transformer (encoder + decoder) -- ablation: Hindsight Experience Replay only
# Pair against the Full Transformer (encoder + decoder) twins on GPU 5.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="5"
python scripts/train.py --config transformer/her "${COMMON_FLAGS[@]}" --device cuda "$@"
