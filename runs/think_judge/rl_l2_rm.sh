#!/bin/bash
# L2 think RL with the CollabRM reward model (~/CollabRM, LoRA on Qwen3-8B served by vLLM: ~/CollabRM/scripts/serve_rm.sh)
# as the think judge instead of the API LLM judge; otherwise as l2t_sum_matchfloor_g1 (think mode sum, weight 1,
# scale match). usage: rl_l2_rm.sh NAME GPU PORT RM_PORT
cd ~/Collab-Overcooked-private/runs/think_judge
NAME=$1; GPU=$2; PORT=$3; RM_PORT=$4
./launch.sh $NAME $GPU $PORT sum 1.0 updates=50 \
  "tasks=[baked_pumpkin_slices, boiled_green_bean_slices, boiled_potato_slices]" \
  "think_reward={mode: sum, weight: 1.0, grounding_weight: 0.0, scale: match}" \
  "judge={type: collab_rm, url: http://localhost:$RM_PORT, model: rm, workers: 48}"
