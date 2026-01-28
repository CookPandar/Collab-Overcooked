#!/bin/bash
#
# 在同一个集群作业脚本里同时跑两件事：
# - GPU 0-3：跑 RL（scripts/cluster_run_rl.sh）
# - GPU 4-7：跑 vLLM + 批量评测（scripts/run_cluster_suite.sh）
#
# 目标是避免两边抢 GPU，并且让 vLLM 在 RL 训练期间持续有真实负载（批量评测/回归测试）。
#
# Usage:
#   bash scripts/run_rl_and_vllm_suite_split_8g.sh \
#     <vllm_env> <collab_env> <suite_team_model> <model_configs.json> <suite_output_dir> \
#     --suite-args -- <run_model_suite.py args...> \
#     --rl-args -- <cluster_run_rl.sh args...>
#
# Example:
#   bash scripts/run_rl_and_vllm_suite_split_8g.sh \
#     /mnt/shared/envs/vllm \
#     /mnt/shared/envs/collab_overcooked \
#     qwen2.5-7B-instruct \
#     configs/model_configs.json \
#     assets/data/batch_results \
#     --suite-args -- --max-workers 8 --repeats 3 --temperatures 0 0.7 \
#     --rl-args -- --collect-config configs/rl_qwen_collect.yaml --train-config configs/rl_qwen_train.yaml --eval-config configs/rl_qwen_eval.yaml
#
# 可选环境变量：
#   SUITE_GPU_LIST   (默认自动从当前可见 GPU 中取后 4 张) vLLM 使用的 GPU 列表
#   SUITE_PORT       (默认 8100)      vLLM OpenAI API 起始端口（若需要 2 个 server，则使用 8100/8101）
#   SUITE_GPU_MEM    (默认 0.9)       vLLM gpu-memory-utilization
#   SUITE_LOOP       (默认 1)         1=在 RL 期间循环跑 suite；0=只跑一次 suite
#   RL_GPU_LIST      (默认自动从当前可见 GPU 中取前 4 张) RL 使用的 GPU 列表（通过 CUDA_VISIBLE_DEVICES 注入）
#
set -euo pipefail

: "${SPLIT_VERBOSE:=1}"

log() {
  if [[ "${SPLIT_VERBOSE}" != "0" ]]; then
    echo "$@"
  fi
}

usage() {
  cat <<'EOF' >&2
Usage:
  bash scripts/run_rl_and_vllm_suite_split_8g.sh <vllm_env> <collab_env> <suite_team_model> <model_configs.json> <suite_output_dir> \
    --suite-args -- <run_model_suite.py args...> \
    --rl-args -- <cluster_run_rl.sh args...>

Notes:
  - 默认从当前可见 GPU 列表中拆分：前 4 张给 RL，后 4 张给 vLLM（也可通过 RL_GPU_LIST / SUITE_GPU_LIST 覆盖）
  - vLLM/suite 会按模型目录自动选择：单模型=1 个 server；双模型=2 个 server（端口为 SUITE_PORT/SUITE_PORT+1）
  - 如需“控制台只保留 RL 输出”，可 export SPLIT_VERBOSE=0（脚本自身不打印）；
    suite 的输出默认会重定向到日志文件，不会刷屏。
EOF
}

if [[ $# -lt 5 ]]; then
  usage
  exit 1
fi

VLLM_ENV="$1"; shift
COLLAB_ENV="$1"; shift
SUITE_TEAM_MODEL="$1"; shift
MODEL_CONFIG="$1"; shift
SUITE_OUTPUT_DIR="$1"; shift

: "${SUITE_PORT:=8100}"
: "${SUITE_GPU_MEM:=0.9}"
: "${SUITE_LOOP:=1}"

SUITE_ARGS=()
RL_ARGS=()

mode=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite-args)
      mode="suite"
      shift
      if [[ "${1:-}" == "--" ]]; then
        shift
      fi
      ;;
    --rl-args)
      mode="rl"
      shift
      if [[ "${1:-}" == "--" ]]; then
        shift
      fi
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$mode" == "suite" ]]; then
        SUITE_ARGS+=("$1")
      elif [[ "$mode" == "rl" ]]; then
        RL_ARGS+=("$1")
      else
        echo "[split-8g] 解析失败：请先指定 --suite-args 或 --rl-args，再追加参数。" >&2
        usage
        exit 2
      fi
      shift
      ;;
  esac
done

if [[ ${#RL_ARGS[@]} -eq 0 ]]; then
  echo "[split-8g] 缺少 RL 参数：请在 --rl-args 后传入 cluster_run_rl.sh 的参数（例如 --collect-config ...）。" >&2
  exit 2
fi

if [[ "$SUITE_LOOP" != "0" && "$SUITE_LOOP" != "1" ]]; then
  echo "[split-8g] SUITE_LOOP 必须是 0 或 1，当前为: $SUITE_LOOP" >&2
  exit 2
fi

SUITE_PID=""
TMP_DIR=""
TMP_TEAM_NAME=""
TMP_CFG_PATH=""
TMP_MODEL_CONFIG=""

cleanup() {
  if [[ -n "${SUITE_PID:-}" ]]; then
    if ps -p "$SUITE_PID" >/dev/null 2>&1; then
      log "[split-8g] 终止 suite 后台进程 (PID $SUITE_PID)"
      kill "$SUITE_PID" >/dev/null 2>&1 || true
      wait "$SUITE_PID" >/dev/null 2>&1 || true
    fi
  fi
  if [[ -n "${TMP_DIR:-}" ]]; then
    rm -rf "$TMP_DIR" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

abs_path() {
  python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

gpu_list_from_env() {
  # Prefer scheduler-provided CUDA_VISIBLE_DEVICES, which may be non-0-based ids.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local cleaned="${CUDA_VISIBLE_DEVICES// /}"
    # allow either "0,1,2" or "0"
    echo "$cleaned"
    return 0
  fi
  # Fallback to 0..N-1 if nvidia-smi exists; keep as comma-separated ids.
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$n" =~ ^[0-9]+$ ]] && [[ "$n" -gt 0 ]]; then
      python - <<PY
n = int(${n})
print(",".join(str(i) for i in range(n)))
PY
      return 0
    fi
  fi
  echo ""
  return 0
}

gpu_count_from_list() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo 0
    return 0
  fi
  # count commas + 1
  local cleaned="${s// /}"
  if [[ "$cleaned" == *","* ]]; then
    local commas="${cleaned//[^,]/}"
    echo $(( ${#commas} + 1 ))
  else
    echo 1
  fi
}

gpu_slice_first_n() {
  local s="$1" n="$2"
  python - "$s" <<PY
import sys
s = sys.argv[1].strip()
n = int(${n})
ids = [x.strip() for x in s.split(",") if x.strip()]
print(",".join(ids[:n]))
PY
}

gpu_slice_last_n() {
  local s="$1" n="$2"
  python - "$s" <<PY
import sys
s = sys.argv[1].strip()
n = int(${n})
ids = [x.strip() for x in s.split(",") if x.strip()]
print(",".join(ids[-n:]))
PY
}

VISIBLE_GPU_LIST="$(gpu_list_from_env)"
VISIBLE_GPU_COUNT="$(gpu_count_from_list "$VISIBLE_GPU_LIST")"
if [[ "$VISIBLE_GPU_COUNT" -lt 8 ]]; then
  echo "[split-8g] 当前可见 GPU 数量不足 8（CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}，解析为: ${VISIBLE_GPU_LIST:-<empty>} count=${VISIBLE_GPU_COUNT}）。" >&2
  echo "[split-8g] 这会导致：RL/accelerate 可能启动超过可见卡数，引发 NCCL Duplicate GPU；vLLM 也可能 NVML index 越界。" >&2
  echo "[split-8g] 请确认你的作业确实申请了 8 张卡，并在作业开始时输出 CUDA_VISIBLE_DEVICES / nvidia-smi -L 核对。" >&2
  exit 2
fi

# Default split: first 4 -> RL, last 4 -> vLLM suite. Allow manual overrides via env.
: "${RL_GPU_LIST:=$(gpu_slice_first_n "$VISIBLE_GPU_LIST" 4)}"
: "${SUITE_GPU_LIST:=$(gpu_slice_last_n "$VISIBLE_GPU_LIST" 4)}"

log "[split-8g] visible GPUs: ${VISIBLE_GPU_LIST} (count=${VISIBLE_GPU_COUNT})"
log "[split-8g] RL GPUs:      ${RL_GPU_LIST}"
log "[split-8g] vLLM GPUs:    ${SUITE_GPU_LIST} (port=$SUITE_PORT, gpu_mem=$SUITE_GPU_MEM, loop=$SUITE_LOOP)"
log "[split-8g] suite model:  $SUITE_TEAM_MODEL"

make_suite_temp_config() {
  if [[ ! -x "$COLLAB_ENV/bin/python" ]]; then
    echo "[split-8g] 未在 $COLLAB_ENV 找到 Collab 环境 python" >&2
    exit 1
  fi
  local collab_py="$COLLAB_ENV/bin/python"

  TMP_DIR="$(mktemp -d -t split_8g_suite.XXXXXX)"
  TMP_CFG_PATH="$TMP_DIR/tmp_suite.yaml"
  TMP_MODEL_CONFIG="$TMP_DIR/tmp_model_configs.json"
  TMP_TEAM_NAME="__suite_${SUITE_TEAM_MODEL}_gpu${SUITE_GPU_LIST//,/}_p${SUITE_PORT}"

  # 解析 base team 的 YAML 路径并生成临时 YAML：
  # - 若 Chef/Assistant 指向同一个模型目录：两者共用同一个 base_url，只起 1 个 vLLM 实例（TP=4）
  # - 若 Chef/Assistant 指向不同模型目录：自动起 2 个 vLLM 实例，端口为 SUITE_PORT / SUITE_PORT+1，
  #   并把 SUITE_GPU_LIST 平分给两边（例如 4,5 和 6,7），避免端口/显卡冲突。
  "$collab_py" - \
    "$MODEL_CONFIG" \
    "$SUITE_TEAM_MODEL" \
    "$SUITE_GPU_LIST" \
    "$SUITE_PORT" \
    "$TMP_TEAM_NAME" \
    "$TMP_CFG_PATH" \
    "$TMP_MODEL_CONFIG" \
    <<'PY'
import json
import sys
from pathlib import Path

import yaml

model_config = Path(sys.argv[1]).expanduser().resolve()
team = sys.argv[2]
gpu_list = sys.argv[3]
base_port = int(sys.argv[4])
team_name = sys.argv[5]
out_cfg_path = Path(sys.argv[6]).expanduser().resolve()
out_map_path = Path(sys.argv[7]).expanduser().resolve()

mapping = json.loads(model_config.read_text())
base_cfg_path = mapping.get(team)
if not base_cfg_path:
    raise SystemExit(f"Model {team!r} not found in {str(model_config)!r}")
base_cfg_path = Path(base_cfg_path).expanduser().resolve()
if not base_cfg_path.exists():
    raise SystemExit(f"Base config not found: {str(base_cfg_path)}")


def split_gpus(gpu_str: str, parts: int):
    ids = [x.strip() for x in (gpu_str or "").split(",") if x.strip()]
    if parts <= 1:
        return [",".join(ids)]
    if len(ids) < parts:
        raise SystemExit(f"Need >= {parts} GPUs to split, got {len(ids)} from {gpu_str!r}")
    out = []
    # balanced split: e.g., 4 gpus into 2 -> 2+2
    base = len(ids) // parts
    rem = len(ids) % parts
    start = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        chunk = ids[start : start + size]
        out.append(",".join(chunk))
        start += size
    return out


def tp_from_gpu_list(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 1
    return len([x for x in s.split(",") if x.strip()])


cfg = yaml.safe_load(base_cfg_path.read_text())
if not isinstance(cfg, dict):
    raise SystemExit(f"Invalid YAML (not a mapping): {str(base_cfg_path)}")
agents = cfg.get("agents") or {}
if not isinstance(agents, dict):
    raise SystemExit(f"Invalid YAML: missing 'agents' mapping in {str(base_cfg_path)}")

agent_items = [
    (k, v) for k, v in agents.items() if isinstance(k, str) and k.startswith("agent_") and isinstance(v, dict)
]
if not agent_items:
    raise SystemExit(f"No agent_* entries found in {str(base_cfg_path)}")

def local_path(agent: dict) -> str:
    return str(agent.get("local_model_path") or agent.get("model_dirname") or agent.get("model_path") or "")

paths = [local_path(a) for _, a in agent_items]
unique_paths = {p for p in paths if p}

if len(unique_paths) <= 1:
    # 单模型：共用一个 vLLM server（TP=全部 GPU）
    base_url = f"http://127.0.0.1:{base_port}/v1"
    tp = tp_from_gpu_list(gpu_list)
    for _, agent in agent_items:
        agent["base_url"] = base_url
        agent["cuda_visible_devices"] = gpu_list
        agent["tensor_parallel_size"] = tp
    print(f"[split-8g] suite mode=single-server base_url={base_url} cuda={gpu_list} tp={tp}")
else:
    # 多模型：分成两个 vLLM server，端口 base_port/base_port+1，GPU 平分
    gpu_chunks = split_gpus(gpu_list, parts=len(agent_items))
    for idx, ((agent_key, agent), cuda_chunk) in enumerate(zip(agent_items, gpu_chunks)):
        url = f"http://127.0.0.1:{base_port + idx}/v1"
        tp = tp_from_gpu_list(cuda_chunk)
        agent["base_url"] = url
        agent["cuda_visible_devices"] = cuda_chunk
        agent["tensor_parallel_size"] = tp
        print(f"[split-8g] suite mode=multi-server {agent_key} base_url={url} cuda={cuda_chunk} tp={tp} path={local_path(agent)!r}")

out_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
out_map_path.write_text(json.dumps({team_name: str(out_cfg_path)}, ensure_ascii=False, indent=2))
print(f"[split-8g] suite tmp yaml: {out_cfg_path}")
print(f"[split-8g] suite tmp map:  {out_map_path}")
print(f"[split-8g] suite team:     {team_name}")
print(f"[split-8g] suite base_port: {base_port} gpu_list={gpu_list}")
PY
}

VLLM_ENV="$(abs_path "$VLLM_ENV")"
COLLAB_ENV="$(abs_path "$COLLAB_ENV")"
MODEL_CONFIG="$(abs_path "$MODEL_CONFIG")"
SUITE_OUTPUT_DIR="$(abs_path "$SUITE_OUTPUT_DIR")"

make_suite_temp_config

mkdir -p "$SUITE_OUTPUT_DIR" >/dev/null 2>&1 || true
: "${SUITE_CONSOLE_LOG:=$SUITE_OUTPUT_DIR/suite_console_${SUITE_TEAM_MODEL}_$(date +%Y%m%d-%H%M%S)_$$.log}"
touch "$SUITE_CONSOLE_LOG" 2>/dev/null || true
log "[split-8g] suite 控制台日志已重定向到: $SUITE_CONSOLE_LOG"

run_suite_once() {
  bash scripts/run_cluster_suite.sh \
    "$VLLM_ENV" \
    "$COLLAB_ENV" \
    "$TMP_TEAM_NAME" \
    "$TMP_MODEL_CONFIG" \
    "$SUITE_OUTPUT_DIR" \
    "$SUITE_GPU_MEM" \
    -- "${SUITE_ARGS[@]}"
}

run_suite_loop() {
  local iter=0
  while true; do
    iter=$((iter + 1))
    log "[split-8g] suite loop iteration=$iter start"
    set +e
    run_suite_once
    local st=$?
    set -e
    if [[ $st -ne 0 ]]; then
      log "[split-8g] suite iteration=$iter 失败（exit=$st），10 秒后重试..."
      sleep 10
    else
      log "[split-8g] suite loop iteration=$iter done"
    fi
  done
}

if [[ "$SUITE_LOOP" == "1" ]]; then
  run_suite_loop >>"$SUITE_CONSOLE_LOG" 2>&1 &
else
  run_suite_once >>"$SUITE_CONSOLE_LOG" 2>&1 &
fi
SUITE_PID=$!
log "[split-8g] suite PID=$SUITE_PID"

set +e
RL_COLLECT_NUM_PROCS="$(gpu_count_from_list "$RL_GPU_LIST")"
CUDA_VISIBLE_DEVICES="$RL_GPU_LIST" \
RL_COLLECT_NUM_PROCS="$RL_COLLECT_NUM_PROCS" \
bash scripts/cluster_run_rl.sh "$COLLAB_ENV" "${RL_ARGS[@]}"
RL_STATUS=$?
set -e
log "[split-8g] RL 结束，exit=$RL_STATUS；准备关闭 suite..."

exit "$RL_STATUS"
