#!/usr/bin/env bash
# Launch all 23 experiments in parallel as background jobs.
#
# All sibling jobs share ONE timestamp folder, so each invocation of
# launch_all.sh produces:
#   scripts/runs/logs/<YYYYmmdd_HHMMSS>/01_baseline_random.log
#   scripts/runs/logs/<YYYYmmdd_HHMMSS>/02_baseline_oracle.log
#   ...
#   scripts/runs/logs/<YYYYmmdd_HHMMSS>/26_transformer_improved.log
# Multiple invocations of launch_all.sh never overwrite each other.
#
# Layout (3 family triplets share each GPU; baselines are CPU-only):
#
#   trick slot           |  MLP    |  Transformer encoder | Full Transformer | GPU
#   ---------------------+---------+----------------------+------------------+-----
#   basic_dqn (control)  |  03     |  11                  |  19              |  0
#   reward_shaping       |  04     |  12                  |  20              |  1
#   double_dqn           |  05     |  13                  |  21              |  2
#   dueling_dqn          |  06     |  14                  |  22              |  3
#   per                  |  07     |  15                  |  23              |  4
#   her                  |  08     |  16                  |  24              |  5
#   improved (all 5)     |  10     |  18                  |  26              |  7
#
# Each GPU hosts exactly 3 jobs; peak VRAM per job is well under 4 GB so
# the 96 GB host is comfortable.
#
# Usage:
#   bash scripts/runs/launch_all.sh
#   tail -f scripts/runs/logs/<timestamp>/26_transformer_improved.log
#
# Stop everything:
#   pkill -f 'scripts/train.py'
set -euo pipefail

cd "$(dirname "$0")"

# One timestamp folder per launch_all invocation; exported so every child
# script (via _common.sh) writes into the same folder.
export RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/${RUN_TIMESTAMP}"
mkdir -p "$LOG_DIR"
echo "[launch_all] timestamp=${RUN_TIMESTAMP}"
echo "[launch_all] log_dir=${LOG_DIR}"

SCRIPTS=(
  01_baseline_random.sh
  02_baseline_oracle.sh

  03_mlp_basic_dqn.sh
  04_mlp_reward_shaping.sh
  05_mlp_double_dqn.sh
  06_mlp_dueling_dqn.sh
  07_mlp_per.sh
  08_mlp_her.sh
  10_mlp_improved.sh

  11_transformer_encoder_basic_dqn.sh
  12_transformer_encoder_reward_shaping.sh
  13_transformer_encoder_double_dqn.sh
  14_transformer_encoder_dueling_dqn.sh
  15_transformer_encoder_per.sh
  16_transformer_encoder_her.sh
  18_transformer_encoder_improved.sh

  19_transformer_basic_dqn.sh
  20_transformer_reward_shaping.sh
  21_transformer_double_dqn.sh
  22_transformer_dueling_dqn.sh
  23_transformer_per.sh
  24_transformer_her.sh
  26_transformer_improved.sh
)

run_bg() {
  local script="$1"
  local name
  name="$(basename "$script" .sh)"
  echo "[launch_all] starting $name"
  nohup bash "$script" >/dev/null 2>&1 &
  echo "[launch_all]   pid=$!"
}

for f in "${SCRIPTS[@]}"; do
  run_bg "$f"
done

echo
echo "All ${#SCRIPTS[@]} jobs spawned. Active processes:"
pgrep -af 'scripts/train.py' || true
echo
echo "Logs:             scripts/runs/${LOG_DIR}/<NN_name>.log"
echo "Follow one:       tail -f scripts/runs/${LOG_DIR}/26_transformer_improved.log"
echo "Watch all:        tail -f scripts/runs/${LOG_DIR}/*.log"
echo "Stop everything:  pkill -f 'scripts/train.py'"
