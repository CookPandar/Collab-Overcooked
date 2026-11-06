#!/bin/bash
# NOTE: running wrappers read this file incrementally: never edit it in place, write a copy and mv it over.
# L2 think RL with the rule-verified think rewards (baselines/rl/rule_rewards.py) as an ablation: ARM ground | mate |
# plan (one reward) | full (the three summed) | full_v2 (ground + mate_v2 + plan) | none (no think reward, the baseline), mode sum, weight 1, scale match,
# train dishes only, from the L1+2 think SFT (as l2t_sum_matchfloor_g1); then eval_all_levels.sh on u50.
# usage: rl_l2_rules.sh ARM PORT EVAL_PORT [MODE]   MODE: sum (default, all tokens) | split (reasoning tokens only;
# the run is named l2r_ARM_split) -- with sum, a reward of the reasoning also moves the choice of the action
cd ~/Collab-Overcooked-private/runs/think_judge
ARM=$1; PORT=$2; EPORT=$3; TMODE=${4:-sum}; NAME=l2r_$ARM
[ $TMODE != sum ] && NAME=${NAME}_$TMODE
case $ARM in
  none) REWARD=("think_reward={mode: none}") ;;
  full) REWARD=("think_reward={mode: $TMODE, weight: 1.0, scale: match, rules: {ground: 1, mate: 1, plan: 1}}") ;;
  full_v2) REWARD=("think_reward={mode: $TMODE, weight: 1.0, scale: match, rules: {ground: 1, mate_v2: 1, plan: 1}}") ;;
  *) REWARD=("think_reward={mode: $TMODE, weight: 1.0, scale: match, rules: {$ARM: 1}}") ;;
esac
# the judge model only extracts the claims (gpt-5.6-sol, low effort: terra is too slow, the gateway times out)
[ $ARM != none ] && REWARD+=("judge={env_file: ~/ParaCook/.env.shuaiapi.local, model: gpt-5.6-sol, workers: 8, extra: {reasoning_effort: low}}")
MODE=$([ $ARM = none ] && echo none || echo $TMODE)
./launch_when_free.sh $NAME $PORT $MODE 1.0 50 \
  "tasks=[baked_pumpkin_slices, boiled_green_bean_slices, boiled_potato_slices]" "${REWARD[@]}"
[ -d rl/$NAME/checkpoints/u00050 ] && ./eval_all_levels.sh $EPORT think_action \
  think_${NAME}_u50=runs/think_judge/rl/$NAME/checkpoints/u00050
