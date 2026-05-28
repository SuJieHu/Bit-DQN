#!/usr/bin/env bash
# Full Transformer (encoder + decoder) -- all 5 tricks (shaped + Double + Dueling + PER + HER)
# Pair against the Full Transformer (encoder + decoder) twins on GPU 7.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="7"
python scripts/train.py --config transformer/improved "${COMMON_FLAGS[@]}" --device cuda "$@"
