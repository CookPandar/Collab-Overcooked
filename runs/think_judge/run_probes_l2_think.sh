#!/bin/bash
# think-before-answer probes on the L2 chain (reason, then the letter is read after the reasoning), circular
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
R=runs/think_judge/rl
until GPU=$(runs/think_judge/free_gpu.sh 25000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python runs/think_judge/probe_eval.py --gpu $GPU --port 18995 --workers 64 --think --per-qtype 100 \
  --probes eval/probes/probes_l2_v2.jsonl --out eval/probes/scores_l2_think.jsonl \
  base= sft_l12=runs/think_judge/sft_think_l12 gated_u20=$R/l2_gated_w0.5/checkpoints/u00020 \
  gated_u50=$R/l2_gated_w0.5/checkpoints/u00050 sum_u50=$R/l2_sum_w0.5/checkpoints/u00050
