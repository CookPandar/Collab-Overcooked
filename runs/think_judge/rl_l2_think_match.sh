#!/bin/bash
# L2 think RL with the think reward on the level of the action reward (scale: match with factor >= 1, weight 1) plus the judge's
# state-grounding score (grounding_weight 1), train dishes only, from the L1+2 think SFT (as l2_gated_w0.5 /
# l2_sum_w0.5); then eval_all_levels.sh on u50.   usage: rl_l2_think_match.sh MODE PORT EVAL_PORT
cd ~/Collab-Overcooked-private/runs/think_judge
MODE=$1; PORT=$2; EPORT=$3; NAME=l2t_${MODE}_matchfloor_g1
./launch_when_free.sh $NAME $PORT $MODE 1.0 50 \
  "tasks=[baked_pumpkin_slices, boiled_green_bean_slices, boiled_potato_slices]" \
  "think_reward={mode: $MODE, weight: 1.0, grounding_weight: 1.0, scale: match}"
[ -d rl/$NAME/checkpoints/u00050 ] && ./eval_all_levels.sh $EPORT think_action \
  think_${NAME}_u50=runs/think_judge/rl/$NAME/checkpoints/u00050
