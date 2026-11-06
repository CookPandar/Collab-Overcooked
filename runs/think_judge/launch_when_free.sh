#!/bin/bash
# usage: launch_when_free.sh NAME PORT MODE WEIGHT TARGET_UPDATES [k=v ...]
# waits for a GPU with >= 38 GB free, runs (or resumes) the arm there, and retries after crashes (e.g. OOM caused
# by other jobs) until metrics.jsonl reaches TARGET_UPDATES
cd ~/Collab-Overcooked-private/runs/think_judge
NAME=$1; PORT=$2; MODE=$3; W=$4; TARGET=$5; shift 5
for try in $(seq 1 20); do
  # a restarted queue must not start a second copy of a run that is still training
  while pgrep -f "output_dir=runs/think_judge/rl/$NAME " > /dev/null; do sleep 120; done
  [ -f rl/$NAME/metrics.jsonl ] && [ "$(wc -l < rl/$NAME/metrics.jsonl)" -ge $TARGET ] && break
  # one queue at a time picks a GPU and waits until its new vLLM server holds its memory
  exec 9> launch_when_free.lock
  flock 9
  until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 38000); [ -n "$GPU" ]; do sleep 120; done
  [ -f rl/$NAME/train.log ] && cat rl/$NAME/train.log >> rl/$NAME/train_all.log
  echo "$(date) $NAME try $try on GPU $GPU" >> launch_when_free.log
  ./launch.sh $NAME $GPU $PORT $MODE $W updates=$TARGET "$@" > /dev/null
  sleep 180
  flock -u 9
  while pgrep -f "output_dir=runs/think_judge/rl/$NAME " > /dev/null; do sleep 120; done
done
echo "$(date) $NAME done" >> launch_when_free.log
