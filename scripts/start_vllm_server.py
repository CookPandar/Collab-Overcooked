#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--engine-port', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--served-model-name', required=True)
    parser.add_argument('--gpu-memory-utilization', required=True)
    parser.add_argument('--max-model-len', required=True)
    parser.add_argument('--api-key', required=True)
    args = parser.parse_args()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['VLLM_DP_RANK'] = '0'
    env['VLLM_DP_SIZE'] = '1'

    cmd = [
        sys.executable,
        '-m', 'vllm.entrypoints.openai.api_server',
        '--model', args.model,
        '--served-model-name', args.served_model_name,
        '--host', '127.0.0.1',
        '--port', str(args.port),
        '--dtype', 'auto',
        '--max-model-len', str(args.max_model_len),
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--api-key', args.api_key,
        '--tensor-parallel-size', '1',
        '--enable-prefix-caching',
        '--enable-chunked-prefill',
        '--disable-log-stats',
        '--enforce-eager',
    ]
    return subprocess.call(cmd, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
