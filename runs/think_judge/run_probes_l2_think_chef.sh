#!/bin/bash
# think-before-answer L2 probes for the Chef half (100 per qtype), completing the role-balanced think subset
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
R=runs/think_judge/rl
until GPU=$(runs/think_judge/free_gpu.sh 25000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python runs/think_judge/probe_eval.py --gpu $GPU --port 18997 --workers 64 --think --per-qtype 100 \
  --roles chef --probes eval/probes/probes_l2_v2.jsonl --out eval/probes/scores_l2_think_chef.jsonl \
  base= sft_l12=runs/think_judge/sft_think_l12 gated_u20=$R/l2_gated_w0.5/checkpoints/u00020 \
  gated_u50=$R/l2_gated_w0.5/checkpoints/u00050 sum_u50=$R/l2_sum_w0.5/checkpoints/u00050 \
  newsum_u50=$R/l2t_sum_matchfloor_g1/checkpoints/u00050 "$@"
