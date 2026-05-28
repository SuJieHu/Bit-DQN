#!/usr/bin/env bash
# Transformer encoder -- ablation: Double DQN only
# Pair against the Transformer encoder twins on GPU 2.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="2"
python scripts/train.py --config transformer_encoder/double_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
