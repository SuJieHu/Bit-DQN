# Sourced by every NN_*.sh under scripts/runs/. Sets cwd, conda, proxy,
# the default flags shared between experiments, and -- importantly --
# redirects this script's stdout/stderr to a timestamped log file under
# scripts/runs/logs/ so that multiple invocations never overwrite each
# other. Direct invocations also see the same output on the terminal.
#
# DO NOT run this file directly.

set -euo pipefail

REPO_ROOT="/mnt/shenzhen2cephfs/mm-base-vision/suzzetehu/phd27fall"
CONDA_SH="/mnt/shenzhen2cephfs/mm-base-vision/suzzetehu/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="dqn"

export http_proxy="http://hk-mmhttpproxy.woa.com:11113"
export https_proxy="$http_proxy"
export all_proxy="$http_proxy"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_API_KEY="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"
fi

cd "$REPO_ROOT"
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# Reduce thread contention when many of these scripts run in parallel.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

# Timestamped log layout: scripts/runs/logs/<timestamp>/<NN_name>.log
# ${BASH_SOURCE[1]} is the caller script (e.g. 03_mlp_basic_dqn.sh).
# When launched via launch_all.sh, all sibling scripts share the same
# timestamp folder via the exported $RUN_TIMESTAMP. When launched
# directly, the script generates its own timestamp folder.
CALLER_SCRIPT="$(basename "${BASH_SOURCE[1]:-${0}}" .sh)"
TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/scripts/runs/logs/${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/${CALLER_SCRIPT}.log"
mkdir -p "$LOG_DIR"
# Tee so direct invocations still see the output on the terminal, while
# launch_all.sh (which redirects stdout to /dev/null) only writes to file.
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[run] script=${CALLER_SCRIPT} log=${LOG_FILE} started_at=${TIMESTAMP}"

# Default flags every experiment respects. wandb on/off is controlled by
# `use_wandb` in the YAML config (configs/default.yaml has it on by
# default); pass `--no-wandb` here to disable it locally.
COMMON_FLAGS=(
  --results-dir results
  --run-name bit-dqn
)
