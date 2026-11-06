#!/bin/bash
# diagnostic SFT variant of a single-level action-only SFT: train (same trainer / hyperparameters as
# sft_per_level.sh unless overridden), then teacher-forced step accuracy (imitation_acc.py) and a 30-episode
# eval on its own level (same seed / horizon as eval_all_levels.sh).
# usage: sft_variant.sh NAME LEVEL PORT DATA_JSONL [trainer args ...]   (output: sft_variants/NAME)
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
TJ=runs/think_judge; FREE=$HOME/Collab-Overcooked-private/$TJ/free_gpu.sh
NAME=$1; L=$2; PORT=$3; DATA=$4; shift 4
OUT=$TJ/sft_variants/$NAME
mkdir -p $OUT
for try in 1 2 3 4 5; do
  [ -f $OUT/Chef/adapter_model.safetensors ] && [ -f $OUT/Assistant/adapter_model.safetensors ] && break
  exec 9> $TJ/launch_when_free.lock
  flock 9
  until GPU=$($FREE 30000); [ -n "$GPU" ]; do sleep 120; done
  echo "$(date) sft variant $NAME try $try on GPU $GPU" >> $TJ/launch_when_free.log
  CUDA_VISIBLE_DEVICES=$GPU ~/collab_overcooked/bin/python -m baselines.sft.train --model $M --data $DATA \
    --output $OUT --gradient-checkpointing "$@" >> $OUT/train.log 2>&1 &
  sleep 240; flock -u 9; wait
done
until GPU=$($FREE 26000); [ -n "$GPU" ]; do sleep 120; done
~/collab_overcooked/bin/python $TJ/imitation_acc.py --gpu $GPU --port $PORT --out $TJ/eval/imitation_acc_variants.jsonl \
  $NAME=$OUT:$L:action_only >> $OUT/eval.log 2>&1
H=50; [ $L -ge 5 ] && H=120
for try in 1 2 3; do
  [ -s $TJ/eval/variants/${NAME}_l$L.jsonl ] && break
  mkdir -p $TJ/eval/variants
  until GPU=$($FREE 26000); [ -n "$GPU" ]; do sleep 120; done
  ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --checkpoints $OUT --level $L --episodes-per-task 6 \
    --horizon $H --seed 31000 --response-protocol action_only --max-new-tokens 128 --gpu $GPU --port $PORT \
    --kv-cache-gib 6 --workers 15 --vllm-python ~/vllm/bin/python --output $TJ/eval/variants/${NAME}_l$L.jsonl \
    >> $OUT/eval.log 2>&1 || sleep 300
done
echo "$(date) sft variant $NAME done" >> $TJ/launch_when_free.log
