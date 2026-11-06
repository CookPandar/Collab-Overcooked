#!/bin/bash
# L2 probe chain (base -> SFT think L1+2 -> gated u20 -> gated u50, sum u50): direct letter scoring on all
# probes, then think-before-answer on 100 probes per qtype; both circular.
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
R=runs/think_judge/rl
CKPTS="base= sft_l12=runs/think_judge/sft_think_l12 gated_u20=$R/l2_gated_w0.5/checkpoints/u00020 gated_u50=$R/l2_gated_w0.5/checkpoints/u00050 sum_u50=$R/l2_sum_w0.5/checkpoints/u00050"
until GPU=$(runs/think_judge/free_gpu.sh 25000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python runs/think_judge/probe_eval.py --gpu $GPU --port 18995 --workers 48 \
  --probes eval/probes/probes_l2_v2.jsonl --out eval/probes/scores_l2_direct.jsonl $CKPTS
until GPU=$(runs/think_judge/free_gpu.sh 25000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python runs/think_judge/probe_eval.py --gpu $GPU --port 18995 --workers 64 --think --per-qtype 100 \
  --probes eval/probes/probes_l2_v2.jsonl --out eval/probes/scores_l2_think.jsonl $CKPTS
