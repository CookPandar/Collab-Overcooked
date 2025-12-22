#!/bin/bash
# Bootstrap a clean machine (no conda / vLLM) and run the full evaluation suite.
#
# Usage:
#   bash scripts/install_and_run_suite.sh \
#       /path/to/qwen2.5-7B-instruct \
#       qwen2.5-7B-instruct \
#       configs/model_configs.json \
#       assets/data/batch_results \
#       8000 \
#       0.9 \
#       --max-workers 8 --repeats 1

set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
    echo "[install-run] python3 is required but not found."
    exit 1
fi

MODEL_PATH=${1:?"Please provide model path."}
MODEL_NAME=${2:?"Please provide served model name."}
MODEL_CONFIG=${3:?"Please provide model_configs.json path."}
OUTPUT_DIR=${4:?"Please provide output directory."}
PORT=${5:-8000}
GPU_MEM=${6:-0.9}
shift 6 || true
SUITE_ARGS=("$@")

MAIN_ENV_DIR=${MAIN_VENV_DIR:-".cluster_env"}
VLLM_ENV_DIR=${VLLM_VENV_DIR:-".vllm_env"}

abs_path() {
    python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

MAIN_ENV_PATH=$(abs_path "$MAIN_ENV_DIR")
VLLM_ENV_PATH=$(abs_path "$VLLM_ENV_DIR")

has_conda=0
if command -v conda >/dev/null 2>&1; then
    has_conda=1
fi

create_conda_env() {
    local prefix=$1
    local py=$2
    if [ -d "$prefix" ]; then
        if [ -d "$prefix/conda-meta" ]; then
            echo "[install-run] Using existing conda env at $prefix"
            return
        else
            echo "[install-run] Found legacy directory at $prefix (not a conda env). Recreating..."
            rm -rf "$prefix"
        fi
    else
        echo "[install-run] Creating conda env at $prefix"
    fi
    conda create -y -p "$prefix" "python=${py}"
}

activate_conda_env() {
    local prefix=$1
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$prefix"
}

setup_python_env() {
    local env_dir=$1
    local fallback_py=$2
    local setup_cmd=$3
    local py_bin

    if [ "$has_conda" -eq 1 ]; then
        create_conda_env "$env_dir" "$fallback_py"
        activate_conda_env "$env_dir"
        py_bin="$(python -c 'import sys; print(sys.executable)')"
    else
        if [ ! -d "$env_dir" ]; then
            echo "[install-run] Creating venv at $env_dir"
            python3 -m venv "$env_dir"
        fi
        # shellcheck disable=SC1090
        source "$env_dir/bin/activate"
        py_bin="$env_dir/bin/python"
    fi

    pip install --upgrade pip setuptools wheel
    eval "$setup_cmd"
    if [ "$has_conda" -eq 1 ]; then
        conda deactivate
else
    deactivate
fi
echo "$py_bin"
}

RUN_SUITE_PY_BIN="$(setup_python_env "$MAIN_ENV_PATH" "3.10" "pip install -e . && pip install openai==1.54.3 rich==13.5.2")"
VLLM_PY_BIN="$(setup_python_env "$VLLM_ENV_PATH" "3.10" "pip install --upgrade vllm")"

if [ "$has_conda" -eq 1 ]; then
    activate_conda_env "$MAIN_ENV_PATH"
else
    # shellcheck disable=SC1090
    source "$MAIN_ENV_PATH/bin/activate"
fi

export RUN_SUITE_PYTHON="$RUN_SUITE_PY_BIN"
export VLLM_PYTHON="$VLLM_PY_BIN"

bash scripts/run_cluster_suite.sh \
    "$MODEL_PATH" \
    "$MODEL_NAME" \
    "$MODEL_CONFIG" \
    "$OUTPUT_DIR" \
    "$PORT" \
    "$GPU_MEM" \
    "${SUITE_ARGS[@]}"
