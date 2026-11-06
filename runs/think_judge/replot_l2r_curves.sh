#!/bin/bash
# regenerate eval/rule_rewards/train_curves_l2r.png; unfinished runs get "（至 uNN）" (old sum-mode ground is left out)
cd ~/Collab-Overcooked-private/runs/think_judge
lab() { n=$(cat rl/$1/metrics.jsonl 2>/dev/null | wc -l); [ "$n" -lt 50 ] && echo "$1=$2（至 u$n）" || echo "$1=$2"; }
RUNS=("$(lab l2r_none 'none（无思考奖励）')" "$(lab l2r_mate 'mate（旧版）')" "$(lab l2r_mate_v2 mate_v2)" "$(lab l2r_plan plan)"
      "$(lab l2r_full full)" "$(lab l2r_ground_split ground_split)")  # same order/colors as the probe charts
[ -s rl/l2r_full_v2_split/metrics.jsonl ] && RUNS+=("$(lab l2r_full_v2_split full_v2_split)")
python3 plot_rules_compare.py eval/rule_rewards/train_curves_l2r.png "L2 规则奖励消融：训练曲线（Qwen3-1.7B，只用 3 道训练菜；$(date '+%m-%d %H:%M')）" \
  "${RUNS[@]}" "--ref=l2t_sum_matchfloor_g1=LLM 评委 sum（参考）"
