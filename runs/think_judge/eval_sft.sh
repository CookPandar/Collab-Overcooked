#!/bin/bash
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
for L in 2 3; do
  ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model /datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --checkpoints runs/think_judge/sft_think_l12 --level $L --episodes-per-task 4 --horizon 50 --gpu 4 --port 18900 --kv-cache-gib 6 \
    --workers 10 --vllm-python ~/vllm/bin/python --output runs/think_judge/eval/sft_l$L.jsonl
done
