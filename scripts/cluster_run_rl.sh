#!/bin/bash
#
# Usage:
#   bash scripts/cluster_run_rl.sh  /mnt/volumes/ss-sai-bd-ga/zhangshuwen/collab-overcooked --config configs/examples/rl_qwen_baked_bell_pepper.yaml
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

ACCELERATE_BIN=/mnt/volumes/ss-sai-bd-ga/zhangshuwen/collab-overcooked/bin/accelerate
NUM_PROCS="${RL_NUM_PROCS:-1}"
ACCEL_BIN="${ACCELERATE_BIN:-accelerate}"
IFS=' ' read -r -a EXTRA_ACCEL <<< "${RL_ACCELERATE_ARGS:-}"
COLLECT_CFG="${RL_COLLECT_CONFIG:-}"
TRAIN_CFG="${RL_TRAIN_CONFIG:-}"
LOOP_ROUNDS="${RL_LOOP_ROUNDS:-1}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate "$COLLAB_ENV"

# 共享最新模型标记文件的默认路径，可在外部 export RL_LATEST_MODEL_FILE 自定义
: "${RL_LATEST_MODEL_FILE:=$(pwd)/runs/rl/latest_model.json}"

# If提供了双配置，则执行“采样->训练”循环；否则沿用单次启动。
if [[ -n "$COLLECT_CFG" || -n "$TRAIN_CFG" ]]; then
    if [[ -z "$COLLECT_CFG" || -z "$TRAIN_CFG" ]]; then
        echo "[cluster-rl] RL_COLLECT_CONFIG 和 RL_TRAIN_CONFIG 需同时设置" >&2
        conda deactivate
        exit 1
    fi
    STATUS=0
    for ((i=1; i<=LOOP_ROUNDS; i++)); do
        echo "[cluster-rl] Round $i collect -> $COLLECT_CFG"
        set +e
        "$ACCEL_BIN" launch --num_processes "$NUM_PROCS" "${EXTRA_ACCEL[@]}" -m collab_overcooked.main_rl --config "$COLLECT_CFG"
        STATUS=$?
        set -e
        if [[ $STATUS -ne 0 ]]; then
            echo "[cluster-rl] Collect failed (round $i), abort." >&2
            break
        fi

        echo "[cluster-rl] Round $i train -> $TRAIN_CFG"
        set +e
        "$COLLAB_ENV/bin/python" -m collab_overcooked.main_rl --config "$TRAIN_CFG"
        STATUS=$?
        set -e
        if [[ $STATUS -ne 0 ]]; then
            echo "[cluster-rl] Train failed (round $i), abort." >&2
            break
        fi
    done
else
    set +e
    "$ACCEL_BIN" launch --num_processes "$NUM_PROCS" "${EXTRA_ACCEL[@]}" -m collab_overcooked.main_rl "$@"
    STATUS=$?
    set -e
fi

conda deactivate

exit $STATUS
