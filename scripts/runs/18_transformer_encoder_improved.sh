#!/usr/bin/env bash
# Transformer encoder -- all 5 tricks (shaped + Double + Dueling + PER + HER)
# Pair against the Transformer encoder twins on GPU 7.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="7"
python scripts/train.py --config transformer_encoder/improved "${COMMON_FLAGS[@]}" --device cuda "$@"
