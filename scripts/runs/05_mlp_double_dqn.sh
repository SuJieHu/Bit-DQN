#!/usr/bin/env bash
# MLP -- ablation: Double DQN only
# Pair against the MLP twins on GPU 2.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="2"
python scripts/train.py --config mlp/double_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
