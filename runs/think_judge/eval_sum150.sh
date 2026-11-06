#!/bin/bash
# sum150: evaluate u50/75/100/125/150 on L2 (8 episodes per order = 40 each) on any GPU with >= 26 GB free
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
D=runs/think_judge/rl/l2_sum_w0.5/checkpoints
OUT=runs/think_judge/eval/l2_sum_w0.5_long_l2.jsonl
for try in 1 2 3 4 5; do
  [ "$(wc -l < $OUT 2>/dev/null || echo 0)" -ge 5 ] && break
  rm -f $OUT $OUT.episodes.jsonl
  until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 26000); [ -n "$GPU" ]; do sleep 120; done
  ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --checkpoints $D/u00050 $D/u00075 $D/u00100 $D/u00125 $D/u00150 \
    --level 2 --episodes-per-task 8 --horizon 50 --gpu $GPU --port 18994 --kv-cache-gib 6 \
    --workers 10 --vllm-python ~/vllm/bin/python --output $OUT || sleep 300
done
