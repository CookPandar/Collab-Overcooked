#!/bin/bash
# episodes with observations for the role timeline analysis (role_timeline.py): 10 episodes (2 per order) per
# checkpoint on levels 2-4, same settings as eval_all_levels.sh (seed 31000, temperature 0.7, horizon 50)
cd ~/Collab-Overcooked-private
export PYTHONPATH=$HOME/Collab-Overcooked-private:$HOME/Collab-Overcooked-private/lib/overcooked_ai
M=/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
T=runs/think_judge
THINK="$T/sft_think_l123 $T/rl/l2_sum_w0.5/checkpoints/u00050 $T/rl/l2_gated_w0.5/checkpoints/u00050"
ACTION="runs/l4_transfer/ckpt/sft_l123_action $T/eval/all_levels/ckpt/action_l2rl_u50 runs/l4_transfer/ckpt/l3rl_u42"
for L in 3 4 2; do
  for P in think_action action_only; do
    OUT=$T/eval/trace/${P}_l$L.jsonl
    CKS=$([ $P = think_action ] && echo $THINK || echo $ACTION); TOKENS=$([ $P = think_action ] && echo 384 || echo 128)
    for try in 1 2 3 4 5; do
      [ "$(wc -l < $OUT 2>/dev/null || echo 0)" -ge 3 ] && break
      rm -f $OUT $OUT.episodes.jsonl
      until GPU=$($HOME/Collab-Overcooked-private/runs/think_judge/free_gpu.sh 26000); [ -n "$GPU" ]; do sleep 120; done
      ~/collab_overcooked/bin/python -m baselines.rl.evaluate --model $M --checkpoints $CKS --level $L --episodes-per-task 2 \
        --horizon 50 --seed 31000 --response-protocol $P --max-new-tokens $TOKENS --gpu $GPU --port 18970 --kv-cache-gib 6 \
        --workers 10 --vllm-python ~/vllm/bin/python --output $OUT --save-trace || sleep 300
    done
  done
done
