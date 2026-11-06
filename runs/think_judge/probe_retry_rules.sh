#!/bin/bash
# usage: probe_retry_rules.sh NAME=CKPT ...  -> run_probes_l2v3.sh until every NAME is in both direct and think scores
cd ~/Collab-Overcooked-private
P=runs/think_judge/eval/probes
for try in 1 2 3 4 5 6 7 8 9 10; do
  echo "=== try $try $(date) : $*"
  bash runs/think_judge/run_probes_l2v3.sh "$@"
  missing=0
  for spec in "$@"; do n=${spec%%=*}
    for m in direct think; do grep -q "\"checkpoint\": \"$n\"" $P/scores_l2v3_$m.jsonl || { missing=1; echo "missing $n $m"; }; done
  done
  [ $missing = 0 ] && { echo "=== all done $(date)"; exit 0; }
  sleep 600
done
exit 1
