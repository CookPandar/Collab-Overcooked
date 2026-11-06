#!/bin/bash
# usage: start_judge.sh GPU MODEL_DIR LOG [extra vllm args]
GPU=$1; MODEL=$2; LOG=$3; shift 3
CUDA_VISIBLE_DEVICES=$GPU exec ~/vllm/bin/python -m vllm.entrypoints.openai.api_server --model "$MODEL" \
  --served-model-name judge --port 18500 --host 127.0.0.1 --max-model-len 8192 --gpu-memory-utilization 0.3 \
  --enable-prefix-caching --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}' "$@" > "$LOG" 2>&1
