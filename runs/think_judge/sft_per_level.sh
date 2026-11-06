#!/bin/bash
# per-level SFT on the gpt-5.6-terra teacher data (runs/teacher_l456/data, train orders only), then zero-shot eval on
# levels 1-6 with eval_all_levels.sh. Same trainer and hyperparameters as sft_think_l123 (LoRA r16, lr 5e-5, 6 epochs),
# from the base model.   usage: sft_per_level.sh PROTOCOL LEVEL PORT [SUFFIX [trainer args ...]]   (PROTOCOL: think |
# action_only; SUFFIX names a retrain, e.g. v2 -> think_l3v2 on rebuilt data, e12 with --epochs 12)
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
P=$1; L=$2; PORT=$3; SUF=$4; shift 4 2>/dev/null || shift $#
TAG=$([ $P = think ] && echo think || echo action)
OUT=runs/think_judge/sft_per_level/${TAG}_l$L$SUF
mkdir -p $OUT
for try in 1 2 3 4 5; do
  [ -f $OUT/Chef/adapter_model.safetensors ] && [ -f $OUT/Assistant/adapter_model.safetensors ] && break
  exec 9> runs/think_judge/launch_when_free.lock
  flock 9
  until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 30000); [ -n "$GPU" ]; do sleep 120; done
  echo "$(date) sft ${TAG}_l$L$SUF try $try on GPU $GPU" >> runs/think_judge/launch_when_free.log
  CUDA_VISIBLE_DEVICES=$GPU ~/collab_overcooked/bin/python -m baselines.sft.train --model $M \
    --data runs/teacher_l456/data/$P/l${L}_train_both.jsonl --output $OUT --gradient-checkpointing "$@" >> $OUT/train.log 2>&1 &
  sleep 240; flock -u 9; wait
done
PROTOCOL=$P; [ "$P" = think ] && PROTOCOL=think_action
cd runs/think_judge && ./eval_all_levels.sh $PORT $PROTOCOL ${TAG}_sft_l$L$SUF=runs/think_judge/sft_per_level/${TAG}_l$L$SUF
echo "$(date) sft ${TAG}_l$L$SUF done" >> launch_when_free.log
