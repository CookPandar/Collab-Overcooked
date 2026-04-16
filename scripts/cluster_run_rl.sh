#!/bin/bash
#
# Usage:
#   bash scripts/cluster_run_rl.sh /path/to/collab_env [--collect-config cfg] [--train-config cfg] [--eval-config cfg] [--loop-rounds N] [-- main_rl args...]
#
# Default behavior:
#   - start one vLLM server per GPU on ports 9000+
#   - bind each accelerate rank to one dedicated vLLM port
#   - synchronize collect/train/eval between rounds

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/cluster_run_rl.sh <collab_env_prefix> [--collect-config cfg] [--train-config cfg] [--eval-config cfg] [--loop-rounds N] [-- main_rl args...]" >&2
    exit 1
fi

abs_path() {
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

visible_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a ids <<< "${CUDA_VISIBLE_DEVICES}"
        echo "${#ids[@]}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local n
        n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n" =~ ^[0-9]+$ ]] && [[ "$n" -gt 0 ]]; then
            echo "$n"
            return
        fi
    fi
    echo 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENT_ROOT="${RL_EXPERIMENT_ROOT:-$REPO_ROOT}"
EXPERIMENT_ROOT="$(abs_path "$EXPERIMENT_ROOT")"
COLLAB_ENV="$(abs_path "$1")"
shift || true

ACCEL_BIN="${ACCELERATE_BIN:-$COLLAB_ENV/bin/accelerate}"
PY_BIN="$COLLAB_ENV/bin/python"
VLLM_ENV="${RL_VLLM_ENV:-/mnt/volumes/ss-sai-bd-ga/zhangshuwen/vllm018}"
VLLM_PY="$VLLM_ENV/bin/python"

if [[ ! -x "$COLLAB_ENV/bin/python" ]]; then
    echo "[cluster-rl] invalid env: $COLLAB_ENV/bin/python not found" >&2
    exit 1
fi
if [[ ! -x "$VLLM_PY" ]]; then
    echo "[cluster-rl] invalid vLLM env: $VLLM_PY not found" >&2
    exit 1
fi
NUM_PROCS="${RL_NUM_PROCS:-$(visible_gpu_count)}"
COLLECT_WORKERS="${RL_COLLECT_WORKERS:-$((NUM_PROCS * 4))}"
EVAL_WORKERS="${RL_EVAL_WORKERS:-$((NUM_PROCS * 4))}"
LOOP_ROUNDS="${RL_LOOP_ROUNDS:-1}"
EVAL_EVERY="${RL_EVAL_EVERY:-1}"
COLLECT_CFG="${RL_COLLECT_CONFIG:-$REPO_ROOT/configs/rl_qwen_collect_snapshot_kl.yaml}"
TRAIN_CFG="${RL_TRAIN_CONFIG:-$REPO_ROOT/configs/rl_qwen_train_kl.yaml}"
EVAL_CFG="${RL_EVAL_CONFIG:-$REPO_ROOT/configs/rl_qwen_eval_kl.yaml}"
VLLM_MODEL_PATH="${RL_VLLM_MODEL_PATH:-/mnt/volumes/ss-sai-bd-ga/zhangshuwen/models/qwen2.5-7b}"
VLLM_HOST="${RL_VLLM_HOST:-127.0.0.1}"
VLLM_START_PORT="${RL_VLLM_START_PORT:-9000}"
MASTER_PORT="${RL_MASTER_PORT:-29540}"
GPU_MEM="${RL_VLLM_GPU_MEM:-0.70}"
MAX_MODEL_LEN="${RL_VLLM_MAX_MODEL_LEN:-8192}"
MAX_LORAS="${RL_VLLM_MAX_LORAS:-2}"
MAX_LORA_RANK="${RL_VLLM_MAX_LORA_RANK:-0}"
ENFORCE_EAGER="${RL_VLLM_ENFORCE_EAGER:-1}"
SERVED_MODEL_NAME="${RL_VLLM_SERVED_MODEL_NAME:-qwen2.5-7B-instruct}"
API_KEY="${RL_VLLM_API_KEY:-YOUR_API_KEY}"
VLLM_MODE="${RL_VLLM_MODE:-balanced}"
IFS=' ' read -r -a EXTRA_ACCEL <<< "${RL_ACCELERATE_ARGS:-}"
RUN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --collect-config)
            COLLECT_CFG="$(abs_path "$2")"
            shift 2
            ;;
        --train-config)
            TRAIN_CFG="$(abs_path "$2")"
            shift 2
            ;;
        --eval-config)
            EVAL_CFG="$(abs_path "$2")"
            shift 2
            ;;
        --loop-rounds)
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

for cfg in "$COLLECT_CFG" "$TRAIN_CFG" "$EVAL_CFG"; do
    if [[ -n "$cfg" && ! -f "$cfg" ]]; then
        echo "[cluster-rl] config not found: $cfg" >&2
        exit 1
    fi
done
if [[ ! -d "$VLLM_MODEL_PATH" ]]; then
    echo "[cluster-rl] vLLM model path not found: $VLLM_MODEL_PATH" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export RL_REPO_ROOT="$REPO_ROOT"
export RL_EXPERIMENT_ROOT="$EXPERIMENT_ROOT"
export RL_VLLM_HOST="$VLLM_HOST"
export RL_VLLM_START_PORT="$VLLM_START_PORT"
export RL_VLLM_API_KEY="$API_KEY"
export RL_VLLM_MODEL_PATH="$VLLM_MODEL_PATH"
export RL_NUM_PROCS="$NUM_PROCS"
export RL_LATEST_MODEL_FILE="${RL_LATEST_MODEL_FILE:-$REPO_ROOT/runs/rl/latest_model_kl.json}"
export RL_PY_BIN="$PY_BIN"
export TOKENIZERS_PARALLELISM=false

case "$VLLM_MODE" in
    safe)
        export VLLM_COMPILE_BACKEND=none
        export VLLM_USE_TORCH_COMPILE=0
        export VLLM_TORCH_COMPILE=0
        export TORCH_COMPILE_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        export TORCHINDUCTOR_DISABLE=1
        export TORCHINDUCTOR_FREEZING=0
        ENFORCE_EAGER=1
        ;;
    fast)
        export VLLM_COMPILE_BACKEND=inductor
        export VLLM_USE_TORCH_COMPILE=1
        export VLLM_TORCH_COMPILE=1
        export TORCH_COMPILE_DISABLE=0
        export TORCHDYNAMO_DISABLE=0
        export TORCHINDUCTOR_DISABLE=0
        export TORCHINDUCTOR_FREEZING=1
        ;;
    balanced)
        export VLLM_COMPILE_BACKEND=none
        export VLLM_USE_TORCH_COMPILE=0
        export VLLM_TORCH_COMPILE=0
        export TORCH_COMPILE_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        export TORCHINDUCTOR_DISABLE=1
        export TORCHINDUCTOR_FREEZING=0
        ENFORCE_EAGER=1
        ;;
    *)
        echo "[cluster-rl] invalid RL_VLLM_MODE=$VLLM_MODE (expected safe|balanced|fast)" >&2
        exit 1
        ;;
esac

LOG_ROOT="$EXPERIMENT_ROOT/logs"
RUNS_ROOT="$EXPERIMENT_ROOT/runs/rl"
ROLLOUT_ROOT="$EXPERIMENT_ROOT/rollouts_kl"
ROLLOUT_EVAL_ROOT="$EXPERIMENT_ROOT/rollouts_eval_kl"

mkdir -p "$RUNS_ROOT" "$ROLLOUT_ROOT" "$ROLLOUT_EVAL_ROOT" "$LOG_ROOT/rl_vllm" "$LOG_ROOT/rl_workers"
find "$REPO_ROOT" -maxdepth 1 -name '.tmp_*.yaml' -delete 2>/dev/null || true

VLLM_PIDS=()
VLLM_PORTS=()
VLLM_ENGINE_PORTS=()
VLLM_INTERNAL_PORT_BASES=()
VLLM_INTERNAL_PORT_SPAN="${RL_VLLM_INTERNAL_PORT_SPAN:-20}"
TAIL_FAILED_LOGS="${RL_TAIL_FAILED_LOGS:-0}"
cleanup() {
    stop_vllm_servers
}
trap cleanup EXIT

kill_port_listener() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        local pids=()
        while IFS= read -r pid; do
            [[ -n "$pid" ]] && pids+=("$pid")
        done < <(lsof -tiTCP:"$port" -sTCP:LISTEN -Pn 2>/dev/null || true)
        if [[ "${#pids[@]}" -gt 0 ]]; then
            echo "[cluster-rl] clearing listener on port=$port pid=${pids[*]}"
            kill "${pids[@]}" 2>/dev/null || true
            sleep 1
            kill -9 "${pids[@]}" 2>/dev/null || true
        fi
    fi
}

pick_free_port() {
    python - <<'PY'
import socket
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

ensure_train_master_port() {
    local desired_port="$1"
    if command -v lsof >/dev/null 2>&1; then
        if lsof -tiTCP:"$desired_port" -sTCP:LISTEN -Pn >/dev/null 2>&1; then
            local new_port
            new_port="$(pick_free_port)"
            echo "[cluster-rl] MASTER_PORT $desired_port is busy, switching to $new_port"
            MASTER_PORT="$new_port"
            export MASTER_PORT
            return
        fi
    fi
    MASTER_PORT="$desired_port"
    export MASTER_PORT
}

port_is_listening() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$port" -sTCP:LISTEN -Pn >/dev/null 2>&1
        return $?
    fi
    python - <<PY
import socket, sys
sock = socket.socket()
sock.settimeout(0.2)
try:
    sock.connect(("127.0.0.1", int("$port")))
except OSError:
    sys.exit(1)
else:
    sock.close()
    sys.exit(0)
PY
}

reserve_vllm_port_block() {
    local base_port="${1:-$VLLM_START_PORT}"
    local count="${2:-$NUM_PROCS}"
    local max_tries=200
    local try_idx=0
    local per_gpu_internal_span="$VLLM_INTERNAL_PORT_SPAN"
    local block_span=$((200 + count * per_gpu_internal_span))
    while (( try_idx < max_tries )); do
        local candidate=$((base_port + try_idx * block_span))
        local ok=1
        for ((gpu=0; gpu<count; gpu++)); do
            local api_port=$((candidate + gpu))
            local engine_port=$((candidate + 100 + gpu))
            local internal_base=$((candidate + 200 + gpu * per_gpu_internal_span))
            if port_is_listening "$api_port" || port_is_listening "$engine_port"; then
                ok=0
                break
            fi
            for ((offset=0; offset<per_gpu_internal_span; offset++)); do
                local internal_port=$((internal_base + offset))
                if port_is_listening "$internal_port"; then
                    ok=0
                    break
                fi
            done
            if (( ok == 0 )); then
                break
            fi
        done
        if (( ok == 1 )); then
            VLLM_START_PORT="$candidate"
            export RL_VLLM_START_PORT="$VLLM_START_PORT"
            echo "[cluster-rl] reserved vLLM port block start=$VLLM_START_PORT count=$count internal_span=$per_gpu_internal_span"
            return 0
        fi
        try_idx=$((try_idx + 1))
    done
    echo "[cluster-rl] unable to reserve a free vLLM port block from base=$base_port count=$count" >&2
    return 1
}

stop_vllm_servers() {
    for pid in "${VLLM_PIDS[@]:-}"; do
        if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
            kill "$pid" || true
            wait "$pid" || true
        fi
    done
    for port in "${VLLM_PORTS[@]:-}"; do
        [[ -n "$port" ]] && kill_port_listener "$port"
    done
    for port in "${VLLM_ENGINE_PORTS[@]:-}"; do
        [[ -n "$port" ]] && kill_port_listener "$port"
    done
    for base in "${VLLM_INTERNAL_PORT_BASES[@]:-}"; do
        if [[ -n "$base" ]]; then
            for ((offset=0; offset<VLLM_INTERNAL_PORT_SPAN; offset++)); do
                kill_port_listener "$((base + offset))"
            done
        fi
    done
    VLLM_PIDS=()
    VLLM_PORTS=()
    VLLM_ENGINE_PORTS=()
    VLLM_INTERNAL_PORT_BASES=()
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local pid="$3"
    local log_file="$4"
    for _ in $(seq 1 180); do
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[cluster-rl] vLLM exited early on $host:$port; log=$log_file" >&2
            if [[ "$TAIL_FAILED_LOGS" == "1" ]]; then
                tail -n 100 "$log_file" >&2 || true
            else
                echo "[cluster-rl] inspect with: tail -n 100 $log_file" >&2
            fi
            return 1
        fi
        if python - <<PY
import socket, sys
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(("$host", int("$port")))
except OSError:
    sys.exit(1)
else:
    sock.close()
    sys.exit(0)
PY
        then
            return 0
        fi
        sleep 1
    done
    echo "[cluster-rl] timeout waiting for $host:$port; log=$log_file" >&2
    if [[ "$TAIL_FAILED_LOGS" == "1" ]]; then
        tail -n 100 "$log_file" >&2 || true
    else
        echo "[cluster-rl] inspect with: tail -n 100 $log_file" >&2
    fi
    return 1
}

start_vllm_servers() {
    local count="$1"
    local cfg_path="$2"
    local stage_name="$3"
    for ((gpu=0; gpu<count; gpu++)); do
        local port=$((VLLM_START_PORT + gpu))
        local engine_port=$((VLLM_START_PORT + 100 + gpu))
        local internal_port_base=$((VLLM_START_PORT + 200 + gpu * VLLM_INTERNAL_PORT_SPAN))
        VLLM_PORTS+=("$port")
        VLLM_ENGINE_PORTS+=("$engine_port")
        VLLM_INTERNAL_PORT_BASES+=("$internal_port_base")
        kill_port_listener "$port"
        kill_port_listener "$engine_port"
        for ((offset=0; offset<VLLM_INTERNAL_PORT_SPAN; offset++)); do
            kill_port_listener "$((internal_port_base + offset))"
        done
        local log_file="$LOG_ROOT/rl_vllm/vllm_${stage_name}_gpu${gpu}.log"
        echo "[cluster-rl] starting vLLM stage=$stage_name gpu=$gpu port=$port engine_port=$engine_port internal_port_base=$internal_port_base"
        "$VLLM_PY" "$REPO_ROOT/scripts/start_vllm_server.py" \
            --gpu "$gpu" \
            --port "$port" \
            --engine-port "$engine_port" \
            --internal-port-base "$internal_port_base" \
            --model "$VLLM_MODEL_PATH" \
            --config "$cfg_path" \
            --served-model-name "$SERVED_MODEL_NAME" \
            --gpu-memory-utilization "$GPU_MEM" \
            --max-model-len "$MAX_MODEL_LEN" \
            --api-key "$API_KEY" \
            --max-loras "$MAX_LORAS" \
            --max-lora-rank "$MAX_LORA_RANK" \
            $([[ "$ENFORCE_EAGER" == "1" ]] && echo "--enforce-eager") \
            >"$log_file" 2>&1 &
        VLLM_PIDS+=("$!")
    done

    for ((gpu=0; gpu<count; gpu++)); do
        local port=$((VLLM_START_PORT + gpu))
        wait_for_port "$VLLM_HOST" "$port" "${VLLM_PIDS[$gpu]}" "$LOG_ROOT/rl_vllm/vllm_${stage_name}_gpu${gpu}.log"
    done
}

ensure_vllm_servers() {
    local cfg_path="$1"
    local stage_name="$2"
    stop_vllm_servers
    start_vllm_servers "$NUM_PROCS" "$cfg_path" "$stage_name"
}

run_stage_workers() {
    local stage_name="$1"
    local cfg_path="$2"
    local total_workers="$3"
    local worker_pids=()
    local worker_logs=()
    local status=0
    mkdir -p "$LOG_ROOT/rl_workers"
    for ((worker_id=0; worker_id<total_workers; worker_id++)); do
        local rank=$((worker_id % NUM_PROCS))
        local log_file="$LOG_ROOT/rl_workers/${stage_name}_worker${worker_id}_gpu${rank}.log"
        worker_logs+=("$log_file")
        echo "[cluster-rl] starting worker stage=$stage_name worker=$worker_id gpu=$rank log=$log_file"
        env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK \
            CUDA_VISIBLE_DEVICES="$rank" \
            RL_WORKER_RANK="$rank" \
            RL_WORKER_ID="$worker_id" \
            RL_STAGE_PHASE="$stage_name" \
            RL_STAGE_ROUND_IDX="$i" \
            RL_LOOP_ROUND_IDX="$i" \
            "$PY_BIN" -u "$REPO_ROOT/scripts/rl_stage_runner.py" --config "$cfg_path" --stage "$stage_name" -- "${RUN_ARGS[@]}" \
            >"$log_file" 2>&1 &
        worker_pids+=("$!")
    done

    for idx in "${!worker_pids[@]}"; do
        local pid="${worker_pids[$idx]}"
        if ! wait "$pid"; then
            status=1
            echo "[cluster-rl] worker failed stage=$stage_name worker=$idx log=${worker_logs[$idx]}" >&2
            if [[ "$TAIL_FAILED_LOGS" == "1" ]]; then
                tail -n 80 "${worker_logs[$idx]}" >&2 || true
            else
                echo "[cluster-rl] inspect with: tail -n 80 ${worker_logs[$idx]}" >&2
            fi
        fi
    done
    return "$status"
}

aggregate_stage_metrics() {
    local stage_name="$1"
    local cfg_path="$2"
    if [[ "$stage_name" != "collect" && "$stage_name" != "eval" ]]; then
        return 0
    fi
    echo "[cluster-rl] aggregating stage metrics stage=$stage_name cfg=$cfg_path"
    "$PY_BIN" "$REPO_ROOT/scripts/aggregate_stage_metrics.py" \
        --config "$cfg_path" \
        --stage "$stage_name"
}

run_stage() {
    local stage_name="$1"
    local cfg_path="$2"
    if [[ -z "$cfg_path" ]]; then
        return 0
    fi
    if [[ "$stage_name" == "collect" || "$stage_name" == "eval" ]]; then
        reserve_vllm_port_block "$VLLM_START_PORT" "$NUM_PROCS" || return 1
        ensure_vllm_servers "$cfg_path" "$stage_name"
    else
        stop_vllm_servers
    fi
    echo "[cluster-rl] stage=$stage_name cfg=$cfg_path procs=$NUM_PROCS"
    if [[ "$stage_name" == "collect" || "$stage_name" == "eval" ]]; then
        local total_workers="$COLLECT_WORKERS"
        if [[ "$stage_name" == "eval" ]]; then
            total_workers="$EVAL_WORKERS"
        fi
        run_stage_workers "$stage_name" "$cfg_path" "$total_workers"
        local worker_status=$?
        if [[ $worker_status -eq 0 ]]; then
            aggregate_stage_metrics "$stage_name" "$cfg_path"
        fi
        return $worker_status
    else
        mkdir -p "$LOG_ROOT/rl_workers"
        local train_log="$LOG_ROOT/rl_workers/${stage_name}_accelerate.log"
        ensure_train_master_port "$MASTER_PORT"
        echo "[cluster-rl] accelerate log=$train_log"
        env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK \
        RL_STAGE_PHASE="$stage_name" \
        RL_STAGE_ROUND_IDX="$i" \
        RL_LOOP_ROUND_IDX="$i" \
        MASTER_PORT="$MASTER_PORT" \
        "$ACCEL_BIN" launch --num_processes "$NUM_PROCS" "${EXTRA_ACCEL[@]}" \
            "$REPO_ROOT/scripts/rl_stage_runner.py" --config "$cfg_path" --stage "$stage_name" -- "${RUN_ARGS[@]}" \
            2>&1 | tee "$train_log"
    fi
}

STATUS=0
for ((i=1; i<=LOOP_ROUNDS; i++)); do
    echo "[cluster-rl] round $i / $LOOP_ROUNDS"

    set +e
    run_stage collect "$COLLECT_CFG"
    STATUS=$?
    set -e
    if [[ $STATUS -ne 0 ]]; then
        echo "[cluster-rl] collect failed in round $i" >&2
        break
    fi

    set +e
    run_stage train "$TRAIN_CFG"
    STATUS=$?
    set -e
    if [[ $STATUS -ne 0 ]]; then
        echo "[cluster-rl] train failed in round $i" >&2
        break
    fi

    if [[ -n "$EVAL_CFG" ]] && (( EVAL_EVERY > 0 )) && (( i % EVAL_EVERY == 0 )); then
        set +e
        run_stage eval "$EVAL_CFG"
        STATUS=$?
        set -e
        if [[ $STATUS -ne 0 ]]; then
            echo "[cluster-rl] eval failed in round $i" >&2
            break
        fi
    fi
done

exit $STATUS
