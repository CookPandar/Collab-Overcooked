#!/bin/bash
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
until GPU=$(runs/think_judge/free_gpu.sh 26000); [ -n "$GPU" ]; do sleep 120; done
echo "$(date) imitation_acc on GPU $GPU" >> runs/think_judge/launch_when_free.log
~/collab_overcooked/bin/python runs/think_judge/imitation_acc.py --gpu $GPU --port 18990 --out runs/think_judge/eval/imitation_acc.jsonl  action_l2=runs/think_judge/sft_per_level/action_l2:2:action_only think_l2=runs/think_judge/sft_per_level/think_l2:2:think action_l3v2=runs/think_judge/sft_per_level/action_l3v2:3:action_only think_l3v2=runs/think_judge/sft_per_level/think_l3v2:3:think action_l4=runs/think_judge/sft_per_level/action_l4:4:action_only think_l4=runs/think_judge/sft_per_level/think_l4:4:think action_l5=runs/think_judge/sft_per_level/action_l5:5:action_only think_l5=runs/think_judge/sft_per_level/think_l5:5:think action_l6=runs/think_judge/sft_per_level/action_l6:6:action_only think_l6=runs/think_judge/sft_per_level/think_l6:6:think
