#!/bin/bash
# usage: eval_queue50.sh GPU PORT RUN [RUN ...]: wait for 50 updates, then evaluate u30/u40/u50 on L3 and L2
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
GPU=$1; PORT=$2; shift 2
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
for RUN in "$@"; do
  D=runs/think_judge/rl/$RUN
  until [ -d $D/checkpoints/u00050 ] && [ "$(wc -l < $D/metrics.jsonl)" -ge 50 ]; do sleep 120; done
  for L in 3 2; do
    ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --checkpoints $D/checkpoints/u00030 $D/checkpoints/u00040 \
      $D/checkpoints/u00050 --level $L --episodes-per-task 4 --horizon 50 --gpu $GPU --port $PORT --kv-cache-gib 6 --workers 10 \
      --vllm-python ~/vllm/bin/python --output runs/think_judge/eval/${RUN}_l$L.jsonl >> runs/think_judge/eval/queue50_gpu$GPU.log 2>&1
  done
done
echo "queue on GPU $GPU done" >> runs/think_judge/eval/queue50_gpu$GPU.log
