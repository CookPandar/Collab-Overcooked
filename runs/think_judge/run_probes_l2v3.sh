#!/bin/bash
# role-balanced L2 probes (probes_l2_v3): direct letter scoring on all probes, then think-before-answer on 100 per
# (qtype, role); both circular.  usage: run_probes_l2v3.sh [extra NAME=CKPT ...]  (done names are skipped)
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
R=runs/think_judge/rl
CKPTS="base= sft_l12=runs/think_judge/sft_think_l12 newsum_u50=$R/l2t_sum_matchfloor_g1/checkpoints/u00050"
until GPU=$(runs/think_judge/free_gpu.sh 25000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python runs/think_judge/probe_eval.py --gpu $GPU --port 18997 --workers 48 \
  --probes eval/probes/probes_l2_v3.jsonl --out eval/probes/scores_l2v3_direct.jsonl $CKPTS "$@"
until GPU=$(runs/think_judge/free_gpu.sh 25000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python runs/think_judge/probe_eval.py --gpu $GPU --port 18997 --workers 64 --think --per-qtype 100 \
  --probes eval/probes/probes_l2_v3.jsonl --out eval/probes/scores_l2v3_think.jsonl $CKPTS "$@"
