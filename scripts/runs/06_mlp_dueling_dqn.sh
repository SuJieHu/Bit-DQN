#!/usr/bin/env bash
# MLP -- ablation: Dueling head only
# Pair against the MLP twins on GPU 3.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="3"
python scripts/train.py --config mlp/dueling_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
