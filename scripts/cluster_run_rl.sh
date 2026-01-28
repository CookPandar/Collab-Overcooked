#!/bin/bash
#
# Usage:
#   bash scripts/cluster_run_rl.sh  /mnt/volumes/ss-sai-bd-ga/zhangshuwen/collab-overcooked --config configs/examples/rl_qwen_baked_bell_pepper.yaml
#
# 环境变量：
#   RL_ACCELERATE_ARGS     (可选) 额外 accelerate 参数（空格分隔，三阶段共用）
#   RL_NUM_PROCS           (可选) 兼容旧用法：当未使用三阶段配置时作为 --num_processes
#   ACCELERATE_BIN         (默认 accelerate)

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/cluster_run_rl.sh <collab_env_prefix> [--collect-config cfg] [--train-config cfg] [--eval-config cfg] [--loop-rounds N] [-- main_rl args...]" >&2
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

ACCELERATE_BIN=../collab-overcooked/bin/accelerate
ACCEL_BIN="${ACCELERATE_BIN:-accelerate}"
IFS=' ' read -r -a EXTRA_ACCEL <<< "${RL_ACCELERATE_ARGS:-}"
COLLECT_CFG="${RL_COLLECT_CONFIG:-}"
TRAIN_CFG="${RL_TRAIN_CONFIG:-}"
EVAL_CFG="${RL_EVAL_CONFIG:-}"
LOOP_ROUNDS="${RL_LOOP_ROUNDS:-1}"
RUN_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --collect-config)
            if [[ $# -lt 2 ]]; then
                echo "[cluster-rl] --collect-config requires a path" >&2
                exit 1
            fi
            COLLECT_CFG="$2"
            shift 2
            ;;
        --train-config)
            if [[ $# -lt 2 ]]; then
                echo "[cluster-rl] --train-config requires a path" >&2
                exit 1
            fi
            TRAIN_CFG="$2"
            shift 2
            ;;
        --eval-config)
            if [[ $# -lt 2 ]]; then
                echo "[cluster-rl] --eval-config requires a path" >&2
                exit 1
            fi
            EVAL_CFG="$2"
            shift 2
            ;;
        --loop-rounds)
            if [[ $# -lt 2 ]]; then
                echo "[cluster-rl] --loop-rounds requires a value" >&2
                exit 1
            fi
            LOOP_ROUNDS="$2"
            shift 2
            ;;
        --)
            shift
            RUN_ARGS+=("$@")
            break
            ;;
        *)
            RUN_ARGS+=("$1")
            shift
            ;;
    esac
done

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate "$COLLAB_ENV"

# 共享最新模型标记文件的默认路径，可在外部 export RL_LATEST_MODEL_FILE 自定义
: "${RL_LATEST_MODEL_FILE:=$(pwd)/runs/rl/latest_model.json}"

#
# 三阶段资源分配默认策略：
#   - collect: 默认使用“当前可见 GPU 数量”的进程数（通常等于卡数）
#   - train:   1 卡（1 进程）
#   - eval:    1 卡（1 进程）
#
gpu_count() {
    # Prefer CUDA_VISIBLE_DEVICES if set (common in schedulers).
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        # Strip spaces; handle formats like "0,1,2,3" or "0"
        local cleaned="${CUDA_VISIBLE_DEVICES// /}"
        if [[ "$cleaned" == *","* ]]; then
            # Count commas + 1
            local commas="${cleaned//[^,]/}"
            echo $(( ${#commas} + 1 ))
            return 0
        fi
        echo 1
        return 0
    fi
    # Fallback: if nvidia-smi exists, count GPUs.
    if command -v nvidia-smi >/dev/null 2>&1; then
        local n
        n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n" =~ ^[0-9]+$ ]] && [[ "$n" -gt 0 ]]; then
            echo "$n"
            return 0
        fi
    fi
    # Conservative default.
    echo 1
}

COLLECT_NUM_PROCS="$(gpu_count)"
TRAIN_NUM_PROCS=1
EVAL_NUM_PROCS=1

# Allow explicit overrides from the job wrapper (useful when CUDA_VISIBLE_DEVICES is
# manipulated outside and we want to prevent launching more ranks than visible GPUs).
: "${RL_COLLECT_NUM_PROCS:=}"
: "${RL_TRAIN_NUM_PROCS:=}"
: "${RL_EVAL_NUM_PROCS:=}"
if [[ -n "$RL_COLLECT_NUM_PROCS" ]]; then
    COLLECT_NUM_PROCS="$RL_COLLECT_NUM_PROCS"
fi
if [[ -n "$RL_TRAIN_NUM_PROCS" ]]; then
    TRAIN_NUM_PROCS="$RL_TRAIN_NUM_PROCS"
fi
if [[ -n "$RL_EVAL_NUM_PROCS" ]]; then
    EVAL_NUM_PROCS="$RL_EVAL_NUM_PROCS"
fi

run_stage() {
    local stage_name="$1"
    local stage_cfg="$2"
    local stage_num_procs="$3"

    if [[ -z "$stage_cfg" ]]; then
        return 0
    fi

    # Resolve relative paths to absolute (prevents CWD differences in cluster jobs).
    if [[ -f "$stage_cfg" ]]; then
        stage_cfg="$(abs_path "$stage_cfg")"
    fi

    echo "[cluster-rl] ${stage_name} -> ${stage_cfg} (num_procs=${stage_num_procs})"
    set +e
    # 集群作业环境有时会预设 WORLD_SIZE/RANK/MASTER_* 等分布式变量（例如 8 卡作业），
    # 这会导致单进程（num_procs=1）启动时依然尝试 init_process_group，最终 rendezvous timeout。
    # 对 train/eval 的单进程阶段，显式清理这些环境变量，保证真正按单进程运行。
    if [[ "$stage_num_procs" == "1" ]]; then
        env \
            -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
            -u MASTER_ADDR -u MASTER_PORT \
            -u NODE_RANK -u GROUP_RANK -u ROLE_RANK \
            -u TORCHELASTIC_RUN_ID -u TORCHELASTIC_RESTART_COUNT -u TORCHELASTIC_MAX_RESTARTS \
            -u PET_RANK -u PET_NNODES -u PET_NODE_RANK -u PET_MASTER_ADDR -u PET_MASTER_PORT \
            "$ACCEL_BIN" launch --num_processes "$stage_num_procs" "${EXTRA_ACCEL[@]}" \
            -m collab_overcooked.main_rl --config "$stage_cfg" "${RUN_ARGS[@]}"
    else
        "$ACCEL_BIN" launch --num_processes "$stage_num_procs" "${EXTRA_ACCEL[@]}" \
            -m collab_overcooked.main_rl --config "$stage_cfg" "${RUN_ARGS[@]}"
    fi
    local stage_status=$?
    set -e
    return "$stage_status"
}

# 三阶段模式：任意一个阶段配置存在，就按 Round 循环执行 collect -> train -> eval（缺省阶段会跳过）。
if [[ -n "$COLLECT_CFG" || -n "$TRAIN_CFG" || -n "$EVAL_CFG" ]]; then
    STATUS=0
    for ((i=1; i<=LOOP_ROUNDS; i++)); do
        if [[ -n "$COLLECT_CFG" ]]; then
            run_stage "Round ${i} collect" "$COLLECT_CFG" "$COLLECT_NUM_PROCS"
            STATUS=$?
            if [[ $STATUS -ne 0 ]]; then
                echo "[cluster-rl] Collect failed (round $i), abort." >&2
                break
            fi
        fi

        if [[ -n "$TRAIN_CFG" ]]; then
            # 训练阶段也用 accelerate 启动，避免在集群环境中仅启动单进程却继承 WORLD_SIZE/RANK
            # 等分布式环境变量导致 init_process_group 等待其它 rank 最终 timeout。
            run_stage "Round ${i} train" "$TRAIN_CFG" "$TRAIN_NUM_PROCS"
            STATUS=$?
            if [[ $STATUS -ne 0 ]]; then
                echo "[cluster-rl] Train failed (round $i), abort." >&2
                break
            fi
        fi

        if [[ -n "$EVAL_CFG" ]]; then
            run_stage "Round ${i} eval" "$EVAL_CFG" "$EVAL_NUM_PROCS"
            STATUS=$?
            if [[ $STATUS -ne 0 ]]; then
                echo "[cluster-rl] Eval failed (round $i), abort." >&2
                break
            fi
        fi
    done
else
    # 兼容旧用法：把剩余参数原样透传给 main_rl（用户可能自己传 --config）
    LEGACY_NUM_PROCS="${RL_NUM_PROCS:-1}"
    set +e
    "$ACCEL_BIN" launch --num_processes "$LEGACY_NUM_PROCS" "${EXTRA_ACCEL[@]}" -m collab_overcooked.main_rl "${RUN_ARGS[@]}"
    STATUS=$?
    set -e
fi

conda deactivate

exit $STATUS
