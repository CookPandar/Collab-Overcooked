#!/bin/bash
set -euo pipefail
# Force vLLM to run in eager mode because torch.compile autotune kernels crash
# on some cluster driver/CUDA combos.
export VLLM_COMPILE_BACKEND=none
export VLLM_USE_TORCH_COMPILE=0
export VLLM_TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
# Usage:
#   bash scripts/run_cluster_suite.sh \
#       /path/to/qwen2.5-7B-instruct \
#       qwen2.5-7B-instruct \
#       configs/model_configs.json \
#       assets/data/batch_results \
#       8000 \
#       0.9 \
#       --max-workers 8 --repeats 1
#
# The remaining arguments after GPU memory are forwarded to run_model_suite.py.

MODEL_PATH=${1:?"Please provide model path (Hugging Face format)."}
MODEL_NAME=${2:?"Please provide served model name (e.g., qwen2.5-7B-instruct)."}
MODEL_CONFIG=${3:?"Please provide path to model_configs.json."}
OUTPUT_DIR=${4:?"Please provide output directory for batch results."}
PORT=${5:-8000}
GPU_MEM=${6:-0.9}
shift 6 || true
SUITE_ARGS=("$@")

HOST="127.0.0.1"
API_KEY="${VLLM_API_KEY:-token-abc123}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VLLM_ENV="$(cd "$SCRIPT_DIR/.." && pwd)/.vllm_env/bin/python"
VLLM_PYTHON_BIN="${VLLM_PYTHON:-}"
if [[ -z "$VLLM_PYTHON_BIN" ]]; then
    if [[ -x "$DEFAULT_VLLM_ENV" ]]; then
        VLLM_PYTHON_BIN="$DEFAULT_VLLM_ENV"
    else
        VLLM_PYTHON_BIN="python"
    fi
fi

echo "[cluster-suite] Starting vLLM server for $MODEL_NAME"
LOG_FILE="$(mktemp -t vllm_log.XXXXXX)"
echo "[cluster-suite] Using vLLM interpreter: $VLLM_PYTHON_BIN"
echo $"$LOG_FILE"
"$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
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
        echo "[cluster-suite] Stopping vLLM server (PID $VLLM_PID)"
        kill $VLLM_PID
        wait $VLLM_PID || true
    fi
}
trap cleanup EXIT

echo "[cluster-suite] Waiting for vLLM to become ready..."
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
if [ "$ready" -ne 1 ]; then
    echo "[cluster-suite] Failed to connect to $HOST:$PORT. See $LOG_FILE"
    exit 1
fi
echo "[cluster-suite] vLLM is ready on $HOST:$PORT"

echo "[cluster-suite] Launching batch evaluation..."
RUN_SUITE_PYTHON=${RUN_SUITE_PYTHON:-python}
echo "[cluster-suite] Using runner interpreter: $RUN_SUITE_PYTHON"
"$RUN_SUITE_PYTHON" scripts/run_model_suite.py \
    --models "$MODEL_NAME" \
    --model-configs "$MODEL_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    "${SUITE_ARGS[@]}"

echo "[cluster-suite] All tasks finished. Logs saved to $LOG_FILE"
