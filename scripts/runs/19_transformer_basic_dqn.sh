#!/usr/bin/env bash
# Full Transformer (encoder + decoder) -- control: no tricks, sparse reward
# Pair against the Full Transformer (encoder + decoder) twins on GPU 0.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="0"
python scripts/train.py --config transformer/basic_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
