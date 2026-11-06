#!/bin/bash
# zero-shot evaluation of checkpoints on levels 1-6: 30 episodes per (checkpoint, level) = 6 per order, later split
# into 3 groups of 10 for mean +- std. Horizon 50 for L1-4, 120 for L5-6 (the teacher needs ~75 steps there).
# usage: eval_all_levels.sh PORT PROTOCOL NAME=CHECKPOINT ...   (skips (checkpoint, level) pairs already done)
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
PORT=$1; PROTOCOL=$2; shift 2
TOKENS=384; [ "$PROTOCOL" = action_only ] && TOKENS=128
for L in 1 2 3 4 5 6; do
  H=50; [ $L -ge 5 ] && H=120
  for pair in "$@"; do
    NAME=${pair%%=*}; CK=${pair#*=}
    OUT=runs/think_judge/eval/all_levels/${NAME}_l$L.jsonl
    for try in 1 2 3 4 5; do
      [ -s $OUT ] && break
      rm -f $OUT.episodes.jsonl
      until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 26000); [ -n "$GPU" ]; do sleep 120; done
      ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --checkpoints $CK --level $L --episodes-per-task 6 \
        --horizon $H --seed 31000 --response-protocol $PROTOCOL --max-new-tokens $TOKENS --gpu $GPU --port $PORT \
        --kv-cache-gib 6 --workers 15 --vllm-python ~/vllm/bin/python --output $OUT || sleep 300
    done
  done
done
