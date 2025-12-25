#!/bin/bash
#
# Usage:
#   bash scripts/cluster_run_rl.sh /path/to/collab_env --config configs/examples/rl_qwen_baked_bell_pepper.yaml
#
# 环境变量：
#   RL_NUM_PROCS           (默认 1) accelerate --num_processes
#   RL_ACCELERATE_ARGS     (可选)   额外 accelerate 参数（空格分隔）
#   ACCELERATE_BIN         (默认 accelerate)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/cluster_run_rl.sh <collab_env_prefix> [main_rl args...]" >&2
    exit 1
fi

abs_path() {
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

COLLAB_ENV="$(abs_path "$1")"
shift || true

if [[ ! -x "$COLLAB_ENV/bin/python" ]]; then
    echo "[cluster-rl] 未在 $COLLAB_ENV 找到有效的 conda 环境，请先运行 cluster_env_setup.sh" >&2
    exit 1
fi

NUM_PROCS="${RL_NUM_PROCS:-1}"
ACCEL_BIN="${ACCELERATE_BIN:-accelerate}"
IFS=' ' read -r -a EXTRA_ACCEL <<< "${RL_ACCELERATE_ARGS:-}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate "$COLLAB_ENV"

set +e
"$ACCEL_BIN" launch --num_processes "$NUM_PROCS" "${EXTRA_ACCEL[@]}" -m collab_overcooked.main_rl "$@"
STATUS=$?
set -e
conda deactivate

exit $STATUS
