#!/bin/bash
# u50 of the five think arms on L2 with fresh seeds, 16 episodes per order (80 per arm) for a low-noise comparison
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
R=runs/think_judge/rl
OUT=runs/think_judge/eval/u50_multi_l2.jsonl
while [ "$(wc -l < $OUT 2>/dev/null || echo 0)" -lt 5 ]; do
  rm -f $OUT $OUT.episodes.jsonl
  until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 26000); [ -n "$GPU" ]; do sleep 120; done
  ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --level 2 --episodes-per-task 16 --horizon 50 --seed 20261019 \
    --checkpoints $R/l2_none/checkpoints/u00050 $R/l2_split_w0.5/checkpoints/u00050 $R/l2_sum_w0.5/checkpoints/u00050 \
                  $R/l2_decoupled_w0.5/checkpoints/u00050 $R/l2_gated_w0.5/checkpoints/u00050 \
    --gpu $GPU --port 18991 --kv-cache-gib 6 --workers 16 --vllm-python ~/vllm/bin/python --output $OUT || sleep 300
done
