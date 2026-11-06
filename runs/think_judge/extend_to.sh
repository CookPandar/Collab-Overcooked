#!/bin/bash
# NOTE: never edit in place while running (write a copy and mv it over).
# extends a finished per-level run to more updates: waits for the running rl_then_eval.sh (PID) to finish its u50
# eval, keeps a copy of u00050 (pruned later), resumes with launch_when_free.sh (same args, new target), then evals.
# usage: extend_to.sh WAIT_PID NAME PORT EVAL_PORT EVAL_NAME TARGET [k=v ...]
cd ~/Collab-Overcooked-private/runs/think_judge
WPID=$1; NAME=$2; PORT=$3; EPORT=$4; ENAME=$5; TARGET=$6; shift 6
while kill -0 $WPID 2>/dev/null; do sleep 300; done
[ -d rl/$NAME/checkpoints/u00050 ] && [ ! -d rl/$NAME/keep_u00050 ] && cp -r rl/$NAME/checkpoints/u00050 rl/$NAME/keep_u00050
echo "$(date) $NAME extend to $TARGET" >> launch_when_free.log
./launch_when_free.sh $NAME $PORT none 0.5 $TARGET checkpoint_every=25 "$@"
CK=runs/think_judge/rl/$NAME/checkpoints/u$(printf %05d $TARGET)
[ -d ../../$CK ] || { echo "$(date) $NAME: no $CK, eval skipped" >> launch_when_free.log; exit 1; }
./eval_all_levels.sh $EPORT think_action $ENAME=$CK
echo "$(date) eval $ENAME done" >> launch_when_free.log
