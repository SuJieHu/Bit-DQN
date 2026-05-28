#!/usr/bin/env bash
# Transformer encoder -- ablation: Dueling head only
# Pair against the Transformer encoder twins on GPU 3.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="3"
python scripts/train.py --config transformer_encoder/dueling_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
