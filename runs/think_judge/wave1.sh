#!/bin/bash
cd ~/Collab-Overcooked-private/runs/think_judge
while [ ! -f sft_think_l12/Chef/adapter_model.safetensors ]; do sleep 30; done
sleep 20
pkill -f "sft_think_l12 --gradient-checkpointing"   # the first job would continue with the Assistant
cp -r sft_think_l12_asst/Assistant sft_think_l12/Assistant
./launch.sh l2_none 4 18110 none 0
./launch.sh l2_split_w0.5 2 18120 split 0.5
./launch.sh l2_sum_w0.5 5 18130 sum 0.5
