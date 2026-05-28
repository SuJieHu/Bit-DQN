#!/usr/bin/env bash
# Full Transformer (encoder + decoder) -- ablation: Dueling head only
# Pair against the Full Transformer (encoder + decoder) twins on GPU 3.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="3"
python scripts/train.py --config transformer/dueling_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
