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
#   - YAML 的 `agents.agent_*` 需为本地 vLLM 服务提供字段：`local_model_path`（合并后的模型目录）和 `base_url`（含端口）。
#   - 脚本会读取 YAML，为每个唯一的 (local_model_path, model, port) 启动 vLLM。

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

CONFIG_PATH="$("$COLLAB_PY" - <<PY "$MODEL_CONFIG" "$TEAM_MODEL"
import json, sys, os
mapping = json.load(open(sys.argv[1]))
team = sys.argv[2]
path = mapping.get(team)
if not path:
    raise SystemExit(f"Model '{team}' not found in {sys.argv[1]}")
print(os.path.abspath(path))
PY
)"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[cluster-suite] 模型配置不存在: $CONFIG_PATH" >&2
    exit 1
fi

SERVER_INFO="$("$COLLAB_PY" - <<'PY' "$CONFIG_PATH"
import sys, json, yaml, os
from urllib.parse import urlparse
cfg = yaml.safe_load(open(sys.argv[1]))
servers = {}
for key, agent in (cfg.get("agents") or {}).items():
    if not isinstance(agent, dict):
        continue
    local_path = agent.get("local_model_path") or agent.get("model_dirname")
    model = agent.get("model")
    base_url = agent.get("base_url")
    if not (local_path and model and base_url):
        continue
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    key = (os.path.abspath(local_path), model, host, port)
    servers[key] = {
        "path": os.path.abspath(local_path),
        "model": model,
        "host": host,
        "port": port,
    }
if not servers:
    raise SystemExit("No local_model_path + base_url entries found in config; nothing to serve.")
for srv in servers.values():
    print(f"{srv['model']}|{srv['path']}|{srv['host']}|{srv['port']}")
PY
)"

IFS=$'\n' read -r -d '' -a SERVER_LIST <<<"${SERVER_INFO}"$'\0'

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
    local model="$1" path="$2" host="$3" port="$4"
    local log_file
    log_file="$(mktemp -t vllm_log.XXXXXX)"
    echo "[cluster-suite] 启动 vLLM: $model (path=$path host=$host port=$port, log=$log_file)"
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
        --model "$path" \
        --host "$host" \
        --port "$port" \
        --gpu-memory-utilization "$GPU_MEM" \
        --served-model-name "$model" \
        --max-model-len 4096 \
        --dtype auto \
        --api-key "token-abc123" \
        --enforce-eager \
        >"$log_file" 2>&1 &
    VLLM_PIDS+=($!)
    SERVER_LOGS+=("$log_file")
}

wait_for() {
    local host="$1" port="$2" log="$3"
    echo "[cluster-suite] 等待 vLLM 在 $host:$port 启动..."
    local ready=0
    for _ in $(seq 1 120); do
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
        exit 1
    fi
}

SERVERS_STARTED=0
for entry in "${SERVER_LIST[@]}"; do
    [[ -z "$entry" ]] && continue
    IFS='|' read -r model path host port <<<"$entry"
    start_vllm "$model" "$path" "$host" "$port"
    SERVERS_STARTED=$((SERVERS_STARTED + 1))
done

for idx in "${!SERVER_LOGS[@]}"; do
    log="${SERVER_LOGS[$idx]}"
    info="${SERVER_LIST[$idx]}"
    IFS='|' read -r model path host port <<<"$info"
    wait_for "$host" "$port" "$log"
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
