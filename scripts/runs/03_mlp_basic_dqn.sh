#!/usr/bin/env bash
# MLP -- control: no tricks, sparse reward
# Pair against the MLP twins on GPU 0.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="0"
python scripts/train.py --config mlp/basic_dqn "${COMMON_FLAGS[@]}" --device cuda "$@"
