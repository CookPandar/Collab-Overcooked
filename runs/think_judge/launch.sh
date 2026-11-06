#!/bin/bash
# usage: launch.sh NAME GPU PORT MODE [WEIGHT] [extra key=value ...]
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
NAME=$1; GPU=$2; PORT=$3; MODE=$4; W=${5:-0.5}; shift 5
EXTRA=()
[ "$MODE" = none ] && EXTRA+=("judge=null")
mkdir -p runs/think_judge/rl/$NAME
CUDA_VISIBLE_DEVICES=$GPU nohup ~/collab_overcooked/bin/python -m baselines.rl.train --config runs/think_judge/configs/base.yaml \
  output_dir=runs/think_judge/rl/$NAME vllm_port=$PORT "think_reward={mode: $MODE, weight: $W}" "${EXTRA[@]}" "$@" \
  > runs/think_judge/rl/$NAME/train.log 2>&1 &
echo "$NAME pid $! gpu $GPU"
