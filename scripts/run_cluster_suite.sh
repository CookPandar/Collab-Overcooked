#!/bin/bash
#
# Usage:
#   bash scripts/run_cluster_suite.sh \
#       /path/to/vllm_env \
#       /path/to/collab_env \
#       team_model_name \
#       configs/model_configs.json \
#       assets/data/batch_results \
#       [gpu_mem_fraction] -- [additional run_model_suite.py args]
#
# 说明：
#   - `team_model_name` 必须在 `model_configs.json` 中有条目，映射到相应 YAML。
#   - YAML 的 `agents.agent_*` 需为本地 vLLM 服务提供字段：
#       - `base_url`（含端口）
#       - `local_model_path` / `model_dirname` / `model_path`（可被 vLLM 直接加载的 HF 模型目录，需包含 config.json/params.json）
#   - 可选：为每个 agent 配置 `cuda_visible_devices`（如 "0,1,2,3" 或 [0,1,2,3]），脚本会为该 vLLM 实例设置 CUDA_VISIBLE_DEVICES。
#   - 可选：`tensor_parallel_size`/`tp`；不配置时会自动取为 cuda_visible_devices 的 GPU 数量（即 vLLM 的张量并行大小）。

set -euo pipefail

abs_path() {
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

if [[ $# -lt 5 ]]; then
    cat <<'EOF' >&2
Usage: bash scripts/run_cluster_suite.sh <vllm_env> <collab_env> <team_model_name> <model_config.json> <output_dir> [gpu_mem] [-- run_model_suite args]
EOF
    exit 1
fi

VLLM_ENV="$(abs_path "$1")"; shift
COLLAB_ENV="$(abs_path "$1")"; shift
TEAM_MODEL="$1"; shift
MODEL_CONFIG="$1"; shift
OUTPUT_DIR="$1"; shift

GPU_MEM=0.9
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
    GPU_MEM="$1"
    shift
fi

if [[ "${1:-}" == "--" ]]; then
    shift
fi
SUITE_ARGS=("$@")

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
    echo "[cluster-suite] 未在 $VLLM_ENV 找到 vLLM 环境，请先运行 cluster_env_setup.sh" >&2
    exit 1
fi
if [[ ! -x "$COLLAB_ENV/bin/python" ]]; then
    echo "[cluster-suite] 未在 $COLLAB_ENV 找到 Collab 环境，请先运行 cluster_env_setup.sh" >&2
    exit 1
fi

VLLM_PY="$VLLM_ENV/bin/python"
COLLAB_PY="$COLLAB_ENV/bin/python"

export VLLM_COMPILE_BACKEND=none
export VLLM_USE_TORCH_COMPILE=0
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

CONFIG_PATH="$("$COLLAB_PY" -c '
import json, os, sys

mapping = json.load(open(sys.argv[1]))
team = sys.argv[2]
path = mapping.get(team)
if not path:
    raise SystemExit(f"Model {team!r} not found in {sys.argv[1]}")
print(os.path.abspath(path))
' "$MODEL_CONFIG" "$TEAM_MODEL")"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[cluster-suite] 模型配置不存在: $CONFIG_PATH" >&2
    exit 1
fi

SERVER_INFO="$("$COLLAB_PY" scripts/resolve_vllm_servers.py "$CONFIG_PATH")"

SERVER_LIST=()
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    SERVER_LIST+=("$line")
done <<<"$SERVER_INFO"

declare -a VLLM_PIDS=()
declare -a SERVER_LOGS=()

cleanup() {
    for pid in "${VLLM_PIDS[@]}"; do
        if ps -p "$pid" >/dev/null 2>&1; then
            echo "[cluster-suite] 停止 vLLM (PID $pid)"
            kill "$pid"
            wait "$pid" || true
        fi
    done
}
trap cleanup EXIT

start_vllm() {
    local model="$1" path="$2" host="$3" port="$4" cuda="$5" tp="$6"
    local log_file
    log_file="$(mktemp -t vllm_log.XXXXXX)"
    if [[ ! -d "$path" ]]; then
        echo "[cluster-suite] 模型目录不存在: $path (model=$model host=$host port=$port)" >&2
        exit 1
    fi
    if [[ ! -f "$path/config.json" && ! -f "$path/params.json" ]]; then
        echo "[cluster-suite] 模型目录缺少 config.json/params.json，vLLM 无法加载: $path" >&2
        echo "[cluster-suite] 这通常表示你传的是 LoRA adapter 或训练输出目录；请先 merge 成 HF 目录再 serve。" >&2
        exit 1
    fi
    local cuda_note=""
    if [[ -n "$cuda" ]]; then
        cuda_note=" CUDA_VISIBLE_DEVICES=$cuda tp=$tp"
    else
        cuda_note=" tp=$tp"
    fi
    echo "[cluster-suite] 启动 vLLM: $model (path=$path host=$host port=$port, log=$log_file)$cuda_note"
    if [[ -n "$cuda" ]]; then
        CUDA_VISIBLE_DEVICES="$cuda" \
        "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
            --model "$path" \
            --host "$host" \
            --port "$port" \
            --gpu-memory-utilization "$GPU_MEM" \
            --served-model-name "$model" \
            --data-parallel-size "${tp:-1}" \
            --max-model-len 8192 \
            --dtype auto \
            --api-key "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJHbW9oUjdNTTQ0cGpQTmIwZ2tKTjFIZ1J2bkJkcjdxQSJ9.0xbuBWNX5wKkvLrQTPo5xFMQ1t1-2MNIURnNQ4Q4KQM" \
            --enforce-eager \
            >"$log_file" 2>&1 &
    else
        "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
            --model "$path" \
            --host "$host" \
            --port "$port" \
            --gpu-memory-utilization "$GPU_MEM" \
            --served-model-name "$model" \
            --data-parallel-size "${tp:-1}" \
            --max-model-len 8192 \
            --dtype auto \
            --api-key "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJHbW9oUjdNTTQ0cGpQTmIwZ2tKTjFIZ1J2bkJkcjdxQSJ9.0xbuBWNX5wKkvLrQTPo5xFMQ1t1-2MNIURnNQ4Q4KQM" \
            --enforce-eager \
            >"$log_file" 2>&1 &
    fi
    VLLM_PIDS+=($!)
    SERVER_LOGS+=("$log_file")
}

wait_for() {
    local host="$1" port="$2" log="$3" pid="$4"
    echo "[cluster-suite] 等待 vLLM 在 $host:$port 启动..."
    local ready=0
    for _ in $(seq 1 120); do
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[cluster-suite] vLLM 进程已退出 (PID $pid)，启动失败；日志: $log" >&2
            tail -n 120 "$log" >&2 || true
            exit 1
        fi
        if python - <<PY
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("$host", $port))
    sys.exit(0)
except OSError:
    sys.exit(1)
PY
        then
            ready=1
            break
        fi
        sleep 1
    done
    if [[ $ready -ne 1 ]]; then
        echo "[cluster-suite] vLLM 未能在 120 秒内启动 (port $port)，详见 $log"
        tail -n 120 "$log" >&2 || true
        exit 1
    fi
}

SERVERS_STARTED=0
for entry in "${SERVER_LIST[@]}"; do
    [[ -z "$entry" ]] && continue
    IFS="|" read -r model path host port cuda tp <<<"$entry"
    start_vllm "$model" "$path" "$host" "$port" "${cuda:-}" "${tp:-1}"
    SERVERS_STARTED=$((SERVERS_STARTED + 1))
done

for idx in "${!SERVER_LOGS[@]}"; do
    log="${SERVER_LOGS[$idx]}"
    info="${SERVER_LIST[$idx]}"
    IFS="|" read -r model path host port cuda tp <<<"$info"
    wait_for "$host" "$port" "$log" "${VLLM_PIDS[$idx]}"
done

if [[ $SERVERS_STARTED -gt 0 ]]; then
    echo "[cluster-suite] vLLM 已全部就绪。"
fi

"$COLLAB_PY" scripts/run_model_suite.py \
    --models "$TEAM_MODEL" \
    --model-configs "$MODEL_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "${SUITE_ARGS[@]}"

echo "[cluster-suite] 批量测试完成。"
