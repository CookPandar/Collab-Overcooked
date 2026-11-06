#!/bin/bash
# add the new think-reward (scale match + grounded) L2 checkpoints to the L2 probe study: direct letter scoring on all
# probes and think-before-answer on 100 per qtype, appended to the existing score files (done names are skipped).
# usage: run_probes_l2_grounded.sh NAME=CKPT ...
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
until GPU=$(runs/think_judge/free_gpu.sh 25000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python runs/think_judge/probe_eval.py --gpu $GPU --port 18996 --workers 48 \
  --probes eval/probes/probes_l2_v2.jsonl --out eval/probes/scores_l2_direct.jsonl "$@"
until GPU=$(runs/think_judge/free_gpu.sh 25000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python runs/think_judge/probe_eval.py --gpu $GPU --port 18996 --workers 64 --think --per-qtype 100 \
  --probes eval/probes/probes_l2_v2.jsonl --out eval/probes/scores_l2_think.jsonl "$@"
