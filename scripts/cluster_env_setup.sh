#!/bin/bash
#
# Usage:
#   bash scripts/cluster_env_setup.sh \
#       /path/to/collab_env \
#       /path/to/vllm_env \
#       [python_version]
#
# - collab_env: 供 SFT/RL/批量测试使用的主环境前缀（conda -p）
# - vllm_env:  只运行 vLLM 推理服务的轻量环境前缀
# - python_version: 可选，默认 3.10

set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
    echo "[cluster-env] conda 未安装，请先在节点上安装 Miniconda/Anaconda。"
    exit 1
fi

if [[ $# -lt 2 ]]; then
    echo "Usage: bash scripts/cluster_env_setup.sh <collab_env_prefix> <vllm_env_prefix> [python_version]" >&2
    exit 1
fi

abs_path() {
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COLLAB_ENV="$(abs_path "$1")"
VLLM_ENV="$(abs_path "$2")"
PY_VERSION="${3:-3.10}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

create_env() {
    local prefix=$1
    local kind=$2
    if [[ -d "$prefix/conda-meta" ]]; then
        echo "[cluster-env] 已检测到 $prefix ，跳过创建。"
        return
    fi
    echo "[cluster-env] 正在创建 $kind 环境 $prefix (python=${PY_VERSION})..."
    conda create -y -p "$prefix" "python=${PY_VERSION}"
    conda activate "$prefix"
    pip install --upgrade pip setuptools wheel
    if [[ "$kind" == "collab" ]]; then
        pip install -e "$PROJECT_ROOT"
        pip install transformers datasets accelerate peft
    else
        pip install vllm
    fi
    conda deactivate
    echo "[cluster-env] $kind 环境已准备完毕：$prefix"
}

create_env "$COLLAB_ENV" "collab"
create_env "$VLLM_ENV" "vllm"

echo "[cluster-env] 所有环境已就绪。"
