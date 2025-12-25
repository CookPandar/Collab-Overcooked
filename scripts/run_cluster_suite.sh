#!/bin/bash
#
# Usage:
#   bash scripts/run_cluster_suite.sh \
#       /path/to/vllm_env \
#       /path/to/collab_env \
#       /path/to/model/or/hf/name \
#       served_model_name \
#       configs/model_configs.json \
#       assets/data/batch_results \
#       [port] [gpu_mem] -- [additional run_model_suite.py args]
#
# 说明：
#   - vllm_env / collab_env 需由 cluster_env_setup.sh 预先创建
#   - 其余参数与旧版 run_cluster_suite 基本一致

set -euo pipefail

abs_path() {
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

if [[ $# -lt 6 ]]; then
    cat <<'EOF' >&2
Usage: bash scripts/run_cluster_suite.sh <vllm_env> <collab_env> <model_path> <served_model_name> <model_config.json> <output_dir> [port] [gpu_mem] [-- run_model_suite args]
EOF
    exit 1
fi

VLLM_ENV="$(abs_path "$1")"; shift
COLLAB_ENV="$(abs_path "$1")"; shift
MODEL_PATH=$1; shift
MODEL_NAME=$1; shift
MODEL_CONFIG=$1; shift
OUTPUT_DIR=$1; shift

PORT=8000
GPU_MEM=0.9
if [[ $# -gt 0 && "${1:0:2}" != "--" && "${1:0:1}" != "-" ]]; then
    PORT=$1
    shift
fi
if [[ $# > 0 && "${1:0:2}" != "--" && "${1:0:1}" != "-" ]]; then
    GPU_MEM=$1
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

HOST="127.0.0.1"
API_KEY="${VLLM_API_KEY:-token-abc123}"
VLLM_PY="$VLLM_ENV/bin/python"
COLLAB_PY="$COLLAB_ENV/bin/python"

export VLLM_COMPILE_BACKEND=none
export VLLM_USE_TORCH_COMPILE=0
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

LOG_FILE="$(mktemp -t vllm_log.XXXXXX)"
echo "[cluster-suite] 日志: $LOG_FILE"

"$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM" \
    --served-model-name "$MODEL_NAME" \
    --max-model-len 4096 \
    --dtype auto \
    --api-key "$API_KEY" \
    --enforce-eager \
    >"$LOG_FILE" 2>&1 &
VLLM_PID=$!

cleanup() {
    if ps -p $VLLM_PID >/dev/null 2>&1; then
        echo "[cluster-suite] 停止 vLLM (PID $VLLM_PID)"
        kill $VLLM_PID
        wait $VLLM_PID || true
    fi
}
trap cleanup EXIT

echo "[cluster-suite] 等待 vLLM 在 $HOST:$PORT 启动..."
ready=0
for _ in $(seq 1 120); do
    if python - <<PY
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("$HOST", $PORT))
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
    echo "[cluster-suite] vLLM 未能在 120 秒内启动，详见 $LOG_FILE"
    exit 1
fi

echo "[cluster-suite] vLLM 已就绪，开始运行 run_model_suite.py"
"$COLLAB_PY" scripts/run_model_suite.py \
    --models "$MODEL_NAME" \
    --model-configs "$MODEL_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "${SUITE_ARGS[@]}"

echo "[cluster-suite] 批量测试完成。"
