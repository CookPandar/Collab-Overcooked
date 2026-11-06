#!/bin/bash
# L1-6 eval of the per-level RL runs that finished without it (10-22); two streams, the second starts 10 min later
# so they do not grab the same GPU. l3v2_none is evaluated once its u50 checkpoint exists and training has exited.
cd ~/Collab-Overcooked-private/runs/think_judge
C=runs/think_judge/rl
( ./eval_all_levels.sh 18960 think_action think_l4rl_sum_u50=$C/l4_sum_w0.5/checkpoints/u00050
  until [ -d ../../$C/l3v2_none/checkpoints/u00050 ] && ! pgrep -f "output_dir=$C/l3v2_none " > /dev/null; do sleep 300; done
  ./eval_all_levels.sh 18960 think_action think_l3v2rl_none_u50=$C/l3v2_none/checkpoints/u00050 ) &
sleep 600
./eval_all_levels.sh 18962 action_only action_l4rl_u50=$C/l4_action_only/checkpoints/u00050 \
  action_l3v2rl_u50=$C/l3v2_action_only/checkpoints/u00050
wait
echo "$(date) eval_finished_20261022 done" >> launch_when_free.log
