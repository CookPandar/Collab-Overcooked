#!/bin/bash
# L2 PPO with a token-level critic (per-token GAE advantages, gamma = lam = 1, as verl/AgentGym-RL) that is warmed
# up with the actor frozen until its held-out explained variance averaged over the last 3 batches is >= 0.2 (at most 40
# critic-only updates), then 50 actor updates; rewards as l2r_none (no judge). usage: rl_l2_ppo_warm.sh NAME GPU PORT
cd ~/Collab-Overcooked-private/runs/think_judge
NAME=$1; GPU=$2; PORT=$3
./launch.sh $NAME $GPU $PORT none 0.0 updates=50 episodes_per_update=8 "horizon_schedule=[]" \
  "tasks=[baked_pumpkin_slices, boiled_green_bean_slices, boiled_potato_slices]" advantage_estimator=gae \
  "critic={level: token, lr: 1.0e-5, head_lr: 1.0e-3, epochs: 4, persistent: true, gamma: 1.0, lam: 1.0, cliprange_value: 0.5, warmup_ev: 0.2, warmup_window: 3, warmup_patience: 1, warmup_max: 40, replay: 4, gates: []}"
