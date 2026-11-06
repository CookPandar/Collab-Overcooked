#!/bin/bash
# NOTE: running wrappers read this file incrementally: never edit it in place, write a copy and mv it over.
# L2 think RL with the reward shaping of docs/reward_shaping.md (sections 5-6), train dishes only, from the L1+2
# think SFT (as l2r_none); then eval_all_levels.sh on u50. Run name l2s_ARM.
#   s0      none + intra_gamma 0.5 + repeat-message penalty 0.1 + advantage_norm role
#   s0a     none + advantage_norm role
#   s1      s0 + uptake collaboration reward (0.5 x the teammate's ITES gain when it executes the request in 5 steps)
#   s3      s1 + "chatter" penalty (a 3rd+ message since the last executed action: -0.1, no delivery share) + the
#           role normalization divides by max(std, 1) (advantage_std_floor 1: batches without success are only centered)
#           [10-28: s0 / s0a / s1 collapsed into talk; see docs/reward_shaping.md section 7]
#   s3n     s3 without advantage normalization (no baseline)
#   s2_rule s3 + rule think reward full_v2 (mean of ground, mate_v2, plan), split, ratio 0.25, centered per role x kind
#   s2_llm  s3 + LLM-judge think reward (as l2t_sum_matchfloor_g1: terra, grounding 1) split, ratio 0.25, role x kind
#   s4      rule version of the LLM judge (docs/reward_shaping.md section 8): the reward structure of none (no
#           shaping, no normalization) + a fixed think score per call category, centered per role, split, no judge
# usage: rl_l2_shaping.sh ARM PORT EVAL_PORT [AFTER]   AFTER: a run name whose training (50 updates) must finish first
cd ~/Collab-Overcooked-private/runs/think_judge
ARM=$1; PORT=$2; EPORT=$3; AFTER=$4; NAME=l2s_$ARM
if [ -n "$AFTER" ]; then
  until [ -f rl/$AFTER/metrics.jsonl ] && [ "$(wc -l < rl/$AFTER/metrics.jsonl)" -ge 50 ]; do sleep 300; done
fi
R0="format_penalty: 0.2, validator_penalty: 0.1, ites_weight: 1.0, tes_beta: 0.95"
S0="$R0, uncoop_penalty: 0.1, uncoop_kinds: [repeat]"
S1="$S0, collab_mode: uptake, collab_weight: 0.5, collab_window: 5"
S3="format_penalty: 0.2, validator_penalty: 0.1, ites_weight: 1.0, tes_beta: 0.95, uncoop_penalty: 0.1, uncoop_kinds: [repeat, chatter], chatter_limit: 2, collab_mode: uptake, collab_weight: 0.5, collab_window: 5"
THINK=("think_reward={mode: none}"); MODE=none; ARGS=()
case $ARM in
  s0) ARGS=("intra_gamma=0.5" "reward={$S0}" "advantage_norm=role") ;;
  s0a) ARGS=("advantage_norm=role") ;;
  s1) ARGS=("intra_gamma=0.5" "reward={$S1}" "advantage_norm=role") ;;
  s3|s2_rule|s2_llm) ARGS=("intra_gamma=0.5" "reward={$S3}" "advantage_norm=role" "advantage_std_floor=1.0") ;;
  s3n) ARGS=("intra_gamma=0.5" "reward={$S3}") ;;
  s4) ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
case $ARM in
  s2_rule) MODE=split
    THINK=("think_reward={mode: split, weight: 1.0, rules: {ground: 1, mate_v2: 1, plan: 1}, rules_combine: mean, scale: ratio, ratio: 0.25, center: role_type}"
           "judge={env_file: ~/ParaCook/.env.shuaiapi.local, model: gpt-5.6-sol, workers: 8, extra: {reasoning_effort: low}}") ;;
  s4) MODE=split
    THINK=("think_reward={mode: split, weight: 1.0, center: role, category: {act_progress: 0.8, act_noprogress: 0.1, act_regress: 0.1, act_rejected: -0.7, wait: 0.15, msg_pos1: 0.0, msg_pos2: -0.15, msg_pos3: -0.25, msg_repeat: -0.25}}" "judge=null") ;;
  s2_llm) MODE=split
    THINK=("think_reward={mode: split, weight: 1.0, grounding_weight: 1.0, scale: ratio, ratio: 0.25, center: role_type}"
           "judge={env_file: ~/ParaCook/.env.shuaiapi.local, model: gpt-5.6-terra, workers: 8}") ;;
esac
./launch_when_free.sh $NAME $PORT $MODE 1.0 50 \
  "tasks=[baked_pumpkin_slices, boiled_green_bean_slices, boiled_potato_slices]" "${ARGS[@]}" "${THINK[@]}"
[ -d rl/$NAME/checkpoints/u00050 ] && ./eval_all_levels.sh $EPORT think_action \
  think_${NAME}_u50=runs/think_judge/rl/$NAME/checkpoints/u00050
echo "$(date) eval think_${NAME}_u50 done" >> launch_when_free.log
