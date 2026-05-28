#!/usr/bin/env bash
# Transformer encoder -- ablation: Prioritized Experience Replay only
# Pair against the Transformer encoder twins on GPU 4.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="4"
python scripts/train.py --config transformer_encoder/per "${COMMON_FLAGS[@]}" --device cuda "$@"
