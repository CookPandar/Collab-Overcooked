#!/bin/bash
# Evaluate the logs in ./data and summarize them in ./eval_result.
# Usage: bash scripts/run_evaluation.sh <model> [<model> ...]
set -e
MODELS=${@:-gpt-3.5-turbo-0125}

python -m collab_overcooked.evaluation.evaluation --test_mode build_in --model ${MODELS}
python -m collab_overcooked.evaluation.organize_result --model ${MODELS}
python -m collab_overcooked.evaluation.convert_result
