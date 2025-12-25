#!/bin/bash
#
# 下载 HuggingFace 上的 Qwen2.5 模型权重。
# 用法：
#   bash scripts/download_qwen25.sh [repo_id] [target_dir]
# 例如：
#   bash scripts/download_qwen25.sh Qwen/Qwen2.5-7B-Instruct models/qwen2.5-7b
#
# 依赖：
#   pip install huggingface_hub
#   若模型受限，请提前执行 `huggingface-cli login` 或设置 HF_TOKEN。

set -euo pipefail

REPO_ID="${1:-Qwen/Qwen2.5-7B-Instruct}"
TARGET_DIR="${2:-models/qwen2.5-7b-instruct}"

python - <<'PY' "$REPO_ID" "$TARGET_DIR"
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
target = sys.argv[2]
token = os.environ.get("HF_TOKEN")

print(f"[download] repo={repo_id} -> {target}")
snapshot_download(
    repo_id=repo_id,
    local_dir=target,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,
)
print("[download] 完成")
PY
