#!/usr/bin/env bash
# Transformer encoder -- control: no tricks, sparse reward
# Pair against the Transformer encoder twins on GPU 0.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="0"
python scripts/train.py --config transformer_encoder/basic_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
