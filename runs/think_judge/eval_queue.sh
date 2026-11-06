#!/bin/bash
# usage: eval_queue.sh GPU PORT RUN [RUN ...]: for each run, wait for 20 updates, then evaluate
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
GPU=$1; PORT=$2; shift 2
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
for RUN in "$@"; do
  D=runs/think_judge/rl/$RUN
  until [ -d $D/checkpoints/u00020 ] && [ "$(wc -l < $D/metrics.jsonl)" -ge 20 ]; do sleep 120; done
  for L in 3 2; do
    if [ $L = 3 ]; then CK="$D/checkpoints/u00010 $D/checkpoints/u00020"; else CK="$D/checkpoints/u00020"; fi
    ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --checkpoints $CK --level $L --episodes-per-task 4 \
      --horizon 50 --gpu $GPU --port $PORT --kv-cache-gib 6 --workers 10 --vllm-python ~/vllm/bin/python \
      --output runs/think_judge/eval/${RUN}_l$L.jsonl >> runs/think_judge/eval/queue_gpu$GPU.log 2>&1
  done
done
echo "queue on GPU $GPU done" >> runs/think_judge/eval/queue_gpu$GPU.log
