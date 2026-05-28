#!/usr/bin/env bash
# Full Transformer (encoder + decoder) -- ablation: Double DQN only
# Pair against the Full Transformer (encoder + decoder) twins on GPU 2.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="2"
python scripts/train.py --config transformer/double_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
