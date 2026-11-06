#!/bin/bash
# usage: run_branch_value.sh NAME CHECKPOINT PORT: on a GPU with >= 26 GB free runs branch_value.py; retries when
# the vLLM server cannot start (other jobs took the memory)
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
NAME=$1; CK=$2; PORT=$3
OUT=runs/think_judge/eval/branch_value/$NAME
for try in 1 2 3 4 5 6; do
  [ -s $OUT.jsonl ] && break
  until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 26000); [ -n "$GPU" ]; do sleep 120; done
  ~/collab_overcooked/bin/python runs/think_judge/branch_value.py --checkpoint $CK --name $NAME --gpu $GPU --port $PORT \
    --workers 32 >> $OUT.log 2>&1 || sleep 300
done
