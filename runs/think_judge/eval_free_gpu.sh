#!/bin/bash
# usage: eval_free_gpu.sh PORT RUN [RUN ...]: once a run has 50 updates, evaluate u30/u40/u50 on L2 then L3
# (20 episodes each) on any GPU with >= 26 GB free; retries a failed level
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
PORT=$1; shift
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
for RUN in "$@"; do
  D=runs/think_judge/rl/$RUN
  until [ -d $D/checkpoints/u00050 ] && [ "$(wc -l < $D/metrics.jsonl)" -ge 50 ]; do sleep 300; done
  for L in 2 3; do
    OUT=runs/think_judge/eval/${RUN}_l$L.jsonl
    for try in 1 2 3 4 5; do
      [ "$(wc -l < $OUT 2>/dev/null || echo 0)" -ge 3 ] && break
      rm -f $OUT $OUT.episodes.jsonl
      until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 26000); [ -n "$GPU" ]; do sleep 120; done
      ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --checkpoints $D/checkpoints/u00030 $D/checkpoints/u00040 \
        $D/checkpoints/u00050 --level $L --episodes-per-task 4 --horizon 50 --gpu $GPU --port $PORT --kv-cache-gib 6 \
        --workers 10 --vllm-python ~/vllm/bin/python --output $OUT || sleep 300
    done
  done
done
