#!/bin/bash
# per-level RL from a single-level SFT, then the L1-6 zero-shot eval of its last checkpoint (so finished runs are
# never left without the cross-level eval).
# usage: rl_then_eval.sh SFT_DIR NAME PORT EVAL_PORT PROTOCOL EVAL_NAME TARGET_UPDATES [k=v ...]
#   PROTOCOL: think_action | action_only;  EVAL_NAME: row name under eval/all_levels (e.g. think_l5rl_none_u50)
cd ~/Collab-Overcooked-private/runs/think_judge
SFT=$1; NAME=$2; PORT=$3; EPORT=$4; PROTOCOL=$5; ENAME=$6; TARGET=$7; shift 7
EXTRA=()
[ "$PROTOCOL" = action_only ] && EXTRA+=("response_protocol=action_only" "max_new_tokens=128")
./launch_after_sft.sh $SFT $NAME $PORT none 0.5 $TARGET "${EXTRA[@]}" "$@"
CK=runs/think_judge/rl/$NAME/checkpoints/u$(printf %05d $TARGET)
[ -d ../../$CK ] || { echo "$(date) $NAME: no $CK, eval skipped" >> launch_when_free.log; exit 1; }
./eval_all_levels.sh $EPORT $PROTOCOL $ENAME=$CK
echo "$(date) eval $ENAME done" >> launch_when_free.log
