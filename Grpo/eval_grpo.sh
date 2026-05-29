#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash Grpo/eval_grpo.sh /path/to/python eval_config.yaml" >&2
  exit 1
fi

PYTHON_BIN="$1"
CONFIG_PATH="$2"

"$PYTHON_BIN" -m Grpo.train_grpo --config "$CONFIG_PATH"

