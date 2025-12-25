#!/bin/bash
#
# Usage:
#   bash scripts/cluster_run_sft.sh /path/to/collab_env <train_qwen_sft.py args>
#
# 环境变量：
#   SFT_NUM_PROCS            (默认 1) accelerate --num_processes
#   SFT_ACCELERATE_ARGS      (可选)  额外的 accelerate launch 参数，使用空格分隔
#   ACCELERATE_BIN           (默认 accelerate) 指定 accelerate CLI 路径

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/cluster_run_sft.sh <collab_env_prefix> [train args...]" >&2
    exit 1
fi

abs_path() {
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

COLLAB_ENV="$(abs_path "$1")"
shift || true

if [[ ! -x "$COLLAB_ENV/bin/python" ]]; then
    echo "[cluster-sft] 未在 $COLLAB_ENV 找到有效的 conda 环境，请先运行 cluster_env_setup.sh" >&2
    exit 1
fi

NUM_PROCS="${SFT_NUM_PROCS:-1}"
ACCEL_BIN="${ACCELERATE_BIN:-accelerate}"
IFS=' ' read -r -a EXTRA_ACCEL <<< "${SFT_ACCELERATE_ARGS:-}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate "$COLLAB_ENV"

set +e
"$ACCEL_BIN" launch --num_processes "$NUM_PROCS" "${EXTRA_ACCEL[@]}" scripts/train_qwen_sft.py "$@"
STATUS=$?
set -e
conda deactivate

exit $STATUS
