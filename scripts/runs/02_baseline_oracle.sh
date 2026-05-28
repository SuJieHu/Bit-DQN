#!/usr/bin/env bash
# Oracle baseline: copies target bit at every position. ~100% success rate.
# CPU-only, no training.
source "$(dirname "$0")/_common.sh"

export CUDA_VISIBLE_DEVICES=""
python scripts/train.py --config baseline_oracle "${COMMON_FLAGS[@]}" "$@"
