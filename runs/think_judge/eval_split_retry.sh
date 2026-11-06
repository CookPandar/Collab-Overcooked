#!/bin/bash
# re-evaluate the split arm at u30/u40/u50 once some GPU has room for a vLLM server
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
D=runs/think_judge/rl/l2_split_w0.5/checkpoints
for L in 2 3; do
  OUT=runs/think_judge/eval/l2_split_w0.5_u30-50_l$L.jsonl
  while [ "$(wc -l < $OUT 2>/dev/null || echo 0)" -lt 3 ]; do
    rm -f $OUT
    until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 26000); [ -n "$GPU" ]; do sleep 120; done
    ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --checkpoints $D/u00030 $D/u00040 $D/u00050 --level $L \
      --episodes-per-task 4 --horizon 50 --gpu $GPU --port 18990 --kv-cache-gib 6 --workers 10 \
      --vllm-python ~/vllm/bin/python --output $OUT || sleep 300
  done
done
