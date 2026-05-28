#!/usr/bin/env bash
# MLP -- all 5 tricks (shaped + Double + Dueling + PER + HER)
# Pair against the MLP twins on GPU 7.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES="7"
python scripts/train.py --config mlp/improved "${COMMON_FLAGS[@]}" --device cuda "$@"
