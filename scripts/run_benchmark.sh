#!/bin/bash
# Run all 30 Collab-Overcooked tasks for one model, then evaluate them.
# Usage: bash scripts/run_benchmark.sh <model> [episodes_per_task] [extra args for collab_overcooked.main]
# Example: bash scripts/run_benchmark.sh gpt-4o 10
#          bash scripts/run_benchmark.sh Qwen2.5-7B-Instruct 10 --local_server_api http://localhost:8000/v1
set -e
MODEL=${1:?"usage: bash scripts/run_benchmark.sh <model> [episodes_per_task] [extra args]"}
EPISODES=${2:-1}
shift $(( $# >= 2 ? 2 : 1 ))

TASKS=$(python -c "from collab_overcooked.config import ALL_TASKS; print(' '.join(ALL_TASKS))")
for order in ${TASKS}; do
    python -m collab_overcooked.main --gpt_model "${MODEL}" --order "${order}" --episode "${EPISODES}" "$@"
done

bash "$(dirname "$0")/run_evaluation.sh" "${MODEL}"
