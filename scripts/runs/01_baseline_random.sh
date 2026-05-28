#!/usr/bin/env bash
# Random baseline: success probability is 2^{-n}. CPU-only, no training.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES=""
python scripts/train.py --config baseline_random "${COMMON_FLAGS[@]}" "$@"
