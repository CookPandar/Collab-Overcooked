#!/bin/bash
# usage: launch_after_sft.sh SFT_DIR NAME PORT MODE WEIGHT TARGET_UPDATES [k=v ...]
# waits until both adapters of SFT_DIR are written (sft_per_level.sh), then queues the RL arm with launch_when_free.sh
cd ~/Collab-Overcooked-private/runs/think_judge
SFT=$1; shift
until [ -f $SFT/Chef/adapter_model.safetensors ] && [ -f $SFT/Assistant/adapter_model.safetensors ]; do sleep 300; done
NAME=$1; PORT=$2; MODE=$3; W=$4; TARGET=$5; shift 5
exec ./launch_when_free.sh $NAME $PORT $MODE $W $TARGET \
  "adapters={chef: runs/think_judge/$SFT/Chef, assistant: runs/think_judge/$SFT/Assistant}" "$@"
