#!/bin/bash
#
# Usage:
#   bash scripts/cluster_run_rl.sh /path/to/collab_env [--collect-config cfg] [--train-config cfg] [--eval-config cfg] [--skip-eval] [--resume] [--loop-rounds N] [-- main_rl args...]
#
# Default behavior:
#   - start one vLLM server per GPU on ports 9000+
#   - bind each accelerate rank to one dedicated vLLM port
#   - synchronize collect/train/eval between rounds
#   - set RL_GPU_IDS=2,4,7 to restrict the run to specific physical GPUs
#   - set RL_WORKERS_PER_GPU=2 for 3*2=6 collect/eval workers
#   - set RL_NOTIFY_EMAIL=you@example.com and SMTP envs to email on failure

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/cluster_run_rl.sh <collab_env_prefix> [--collect-config cfg] [--train-config cfg] [--eval-config cfg] [--skip-eval] [--resume] [--loop-rounds N] [-- main_rl args...]" >&2
    exit 1
fi

abs_path() {
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

visible_gpu_count() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a ids <<< "${CUDA_VISIBLE_DEVICES}"
        echo "${#ids[@]}"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        local n
        n="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
        if [[ "$n" =~ ^[0-9]+$ ]] && [[ "$n" -gt 0 ]]; then
            echo "$n"
            return
        fi
    fi
    echo 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENT_ROOT="${RL_EXPERIMENT_ROOT:-$REPO_ROOT}"
EXPERIMENT_ROOT="$(abs_path "$EXPERIMENT_ROOT")"
COLLAB_ENV="$(abs_path "$1")"
shift || true

ACCEL_BIN="${ACCELERATE_BIN:-$COLLAB_ENV/bin/accelerate}"
PY_BIN="$COLLAB_ENV/bin/python"
VLLM_ENV="${RL_VLLM_ENV:-/mnt/volumes/ss-sai-bd-ga/zhangshuwen/vllm018}"
VLLM_PY="$VLLM_ENV/bin/python"
REQUESTED_GPU_IDS="${RL_GPU_IDS:-${RL_VISIBLE_GPUS:-}}"
if [[ -n "$REQUESTED_GPU_IDS" ]]; then
    export CUDA_VISIBLE_DEVICES="$REQUESTED_GPU_IDS"
fi
GPU_SCOPE_RAW="${RL_CLEANUP_GPU_IDS:-${REQUESTED_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}}"
GPU_SCOPE_RAW="${GPU_SCOPE_RAW// /}"

if [[ ! -x "$COLLAB_ENV/bin/python" ]]; then
    echo "[cluster-rl] invalid env: $COLLAB_ENV/bin/python not found" >&2
    exit 1
fi
if [[ ! -x "$VLLM_PY" ]]; then
    echo "[cluster-rl] invalid vLLM env: $VLLM_PY not found" >&2
    exit 1
fi
NUM_PROCS="${RL_NUM_PROCS:-$(visible_gpu_count)}"
WORKERS_PER_GPU="${RL_WORKERS_PER_GPU:-4}"
COLLECT_WORKERS="${RL_COLLECT_WORKERS:-$((NUM_PROCS * WORKERS_PER_GPU))}"
EVAL_WORKERS="${RL_EVAL_WORKERS:-$((NUM_PROCS * WORKERS_PER_GPU))}"
COLLECT_MIN_SUCCESS_WORKERS="${RL_COLLECT_MIN_SUCCESS_WORKERS:-1}"
EVAL_MIN_SUCCESS_WORKERS="${RL_EVAL_MIN_SUCCESS_WORKERS:-1}"
LOOP_ROUNDS="${RL_LOOP_ROUNDS:-1}"
EVAL_EVERY="${RL_EVAL_EVERY:-5}"
COLLECT_CFG="${RL_COLLECT_CONFIG:-$REPO_ROOT/configs/rl_qwen_collect_snapshot_kl.yaml}"
TRAIN_CFG="${RL_TRAIN_CONFIG:-$REPO_ROOT/configs/rl_qwen_train_kl.yaml}"
EVAL_CFG="${RL_EVAL_CONFIG:-$REPO_ROOT/configs/rl_qwen_eval_kl.yaml}"
SKIP_EVAL="${RL_SKIP_EVAL:-0}"
RESUME_RUN="${RL_RESUME:-0}"
ONLY_STAGE="${RL_ONLY_STAGE:-}"
VLLM_MODEL_PATH="${RL_VLLM_MODEL_PATH:-/mnt/volumes/ss-sai-bd-ga/zhangshuwen/models/qwen2.5-7b}"
VLLM_HOST="${RL_VLLM_HOST:-127.0.0.1}"
VLLM_START_PORT="${RL_VLLM_START_PORT:-9000}"
VLLM_SPLIT_ACTOR_VALUE="${RL_VLLM_SPLIT_ACTOR_VALUE:-1}"
VLLM_VALUE_PORT_OFFSET="${RL_VLLM_VALUE_PORT_OFFSET:-1000}"
VLLM_START_RETRIES="${RL_VLLM_START_RETRIES:-3}"
VLLM_LIFECYCLE="${RL_VLLM_LIFECYCLE:-marshal}"
case "$VLLM_LIFECYCLE" in
    marshal|persistent|stage)
        ;;
    *)
        echo "[cluster-rl] invalid RL_VLLM_LIFECYCLE=$VLLM_LIFECYCLE (expected marshal|persistent|stage)" >&2
        exit 1
        ;;
esac
VLLM_PERSISTENT="${RL_VLLM_PERSISTENT:-0}"
if [[ "$VLLM_LIFECYCLE" != "stage" ]]; then
    VLLM_PERSISTENT=1
fi
VLLM_SLEEP_DURING_TRAIN="${RL_VLLM_SLEEP_DURING_TRAIN:-}"
if [[ -z "$VLLM_SLEEP_DURING_TRAIN" ]]; then
    if [[ "$VLLM_LIFECYCLE" == "marshal" ]]; then
        VLLM_SLEEP_DURING_TRAIN=1
    else
        VLLM_SLEEP_DURING_TRAIN=0
    fi
fi
VLLM_SLEEP_LEVEL="${RL_VLLM_SLEEP_LEVEL:-1}"
VLLM_SLEEP_MODE="${RL_VLLM_SLEEP_MODE:-abort}"
TRAIN_GPU_RESERVE_MB=0
if [[ -n "${RL_TRAIN_GPU_RESERVE_MB:-}" && "${RL_TRAIN_GPU_RESERVE_MB:-0}" != "0" ]]; then
    echo "[cluster-rl] RL_TRAIN_GPU_RESERVE_MB is deprecated and ignored; train no longer starts a GPU reservation process"
fi
TRAIN_MIN_FREE_MB="${RL_TRAIN_MIN_FREE_MB:-${RL_TRAIN_REQUIRED_FREE_MB:-0}}"
TRAIN_GPU_WAIT_SECONDS="${RL_TRAIN_GPU_WAIT_SECONDS:-0}"
TRAIN_STOP_VLLM_IF_SLEEP_INSUFFICIENT="${RL_TRAIN_STOP_VLLM_IF_SLEEP_INSUFFICIENT:-0}"
MASTER_PORT="${RL_MASTER_PORT:-29540}"
GPU_MEM="${RL_VLLM_GPU_MEM:-0.70}"
ACTOR_GPU_MEM="${RL_VLLM_ACTOR_GPU_MEM:-}"
VALUE_GPU_MEM="${RL_VLLM_VALUE_GPU_MEM:-}"
if [[ "$VLLM_SPLIT_ACTOR_VALUE" == "1" ]]; then
    ACTOR_GPU_MEM="${ACTOR_GPU_MEM:-0.46}"
    VALUE_GPU_MEM="${VALUE_GPU_MEM:-0.46}"
else
    ACTOR_GPU_MEM="${ACTOR_GPU_MEM:-$GPU_MEM}"
    VALUE_GPU_MEM="${VALUE_GPU_MEM:-$GPU_MEM}"
fi
MAX_MODEL_LEN="${RL_VLLM_MAX_MODEL_LEN:-8192}"
MAX_LORAS="${RL_VLLM_MAX_LORAS:-2}"
MAX_LORA_RANK="${RL_VLLM_MAX_LORA_RANK:-0}"
ENFORCE_EAGER="${RL_VLLM_ENFORCE_EAGER:-1}"
SERVED_MODEL_NAME="${RL_VLLM_SERVED_MODEL_NAME:-qwen2.5-7B-instruct}"
API_KEY="${RL_VLLM_API_KEY:-YOUR_API_KEY}"
VLLM_MODE="${RL_VLLM_MODE:-balanced}"
NOTIFY_EMAIL="${RL_NOTIFY_EMAIL:-}"
SMTP_HOST="${RL_SMTP_HOST:-smtp.qq.com}"
SMTP_PORT="${RL_SMTP_PORT:-465}"
SMTP_USER="${RL_SMTP_USER:-}"
SMTP_PASS="${RL_SMTP_PASS:-}"
NOTIFY_ON_SUCCESS="${RL_NOTIFY_ON_SUCCESS:-0}"
IFS=' ' read -r -a EXTRA_ACCEL <<< "${RL_ACCELERATE_ARGS:-}"
RUN_ARGS=()

physical_gpu_for_rank() {
    local rank="$1"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -r -a visible_ids <<< "${CUDA_VISIBLE_DEVICES}"
        if [[ "$rank" -ge 0 && "$rank" -lt "${#visible_ids[@]}" ]]; then
            echo "${visible_ids[$rank]}"
            return
        fi
    fi
    echo "$rank"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --collect-config)
            COLLECT_CFG="$(abs_path "$2")"
            shift 2
            ;;
        --train-config)
            TRAIN_CFG="$(abs_path "$2")"
            shift 2
            ;;
        --eval-config)
            EVAL_CFG="$(abs_path "$2")"
            shift 2
            ;;
        --loop-rounds)
            LOOP_ROUNDS="$2"
            shift 2
            ;;
        --skip-eval)
            SKIP_EVAL=1
            shift
            ;;
        --resume)
            RESUME_RUN=1
            shift
            ;;
        --)
            shift
            RUN_ARGS+=("$@")
            break
            ;;
        *)
            RUN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$SKIP_EVAL" == "1" ]]; then
    EVAL_CFG=""
fi
case "$ONLY_STAGE" in
    ""|collect|train|eval)
        ;;
    *)
        echo "[cluster-rl] invalid RL_ONLY_STAGE=$ONLY_STAGE (expected collect|train|eval)" >&2
        exit 1
        ;;
esac

LOG_ROOT="$EXPERIMENT_ROOT/logs"
RUNS_ROOT="$EXPERIMENT_ROOT/runs/rl"
ROLLOUT_ROOT="$EXPERIMENT_ROOT/rollouts_kl"
ROLLOUT_EVAL_ROOT="$EXPERIMENT_ROOT/rollouts_eval_kl"
STATE_ROOT="$LOG_ROOT/rl_runtime"
PRE_CLEANUP="${RL_PRE_CLEANUP:-1}"
if [[ "$VLLM_LIFECYCLE" == "marshal" ]]; then
    PRE_CLEANUP=0
fi

mkdir -p "$RUNS_ROOT" "$ROLLOUT_ROOT" "$ROLLOUT_EVAL_ROOT" "$LOG_ROOT/rl_vllm" "$LOG_ROOT/rl_workers" "$STATE_ROOT"
find "$REPO_ROOT" -maxdepth 1 -name '.tmp_*.yaml' -delete 2>/dev/null || true

detect_resume_round() {
    COLLECT_CFG_ENV="$COLLECT_CFG" \
    TRAIN_CFG_ENV="$TRAIN_CFG" \
    EVAL_CFG_ENV="$EVAL_CFG" \
    REPO_ROOT_ENV="$REPO_ROOT" \
    "$PY_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

import yaml


def resolve_path(raw: str, base: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def resolve_generated_path(raw: str, repo_root: Path, experiment_root: Path) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        return (experiment_root / path).resolve()
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return path
    return (experiment_root / rel).resolve()


def max_update_from_csv(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            values = []
            for row in reader:
                raw = (row or {}).get("update_idx")
                if raw is None or raw == "":
                    continue
                try:
                    values.append(int(raw))
                except ValueError:
                    continue
        return max(values) if values else 0
    except Exception:
        return 0


def max_update_from_latest(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8").strip())
    except Exception:
        return 0
    if isinstance(payload, dict):
        raw = payload.get("update_idx")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 0
    return 0


def inspect_cfg(raw_cfg: str, repo_root: Path, experiment_root: Path) -> int:
    if not raw_cfg:
        return 0
    cfg_path = Path(raw_cfg)
    if not cfg_path.exists():
        return 0
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return 0
    trainer = data.get("trainer") or {}
    best = 0
    latest_file = trainer.get("latest_model_path_file")
    if latest_file:
        best = max(
            best,
            max_update_from_latest(
                resolve_generated_path(str(latest_file), repo_root, experiment_root)
            ),
        )
    output_dir = trainer.get("output_dir")
    if output_dir:
        out = resolve_generated_path(str(output_dir), repo_root, experiment_root)
        for name in ("train_curve.csv", "reward_curve.csv", "performance_curve.csv"):
            best = max(best, max_update_from_csv(out / name))
    return best


repo_root = Path(os.environ["REPO_ROOT_ENV"]).resolve()
experiment_root = Path(os.environ.get("RL_EXPERIMENT_ROOT", str(repo_root))).resolve()
best = 0
for key in ("COLLECT_CFG_ENV", "TRAIN_CFG_ENV", "EVAL_CFG_ENV"):
    best = max(best, inspect_cfg(os.environ.get(key, ""), repo_root, experiment_root))
print(best)
PY
}

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export RL_REPO_ROOT="$REPO_ROOT"
export RL_EXPERIMENT_ROOT="$EXPERIMENT_ROOT"
export RL_VLLM_HOST="$VLLM_HOST"
export RL_VLLM_START_PORT="$VLLM_START_PORT"
export RL_VLLM_VALUE_START_PORT="$((VLLM_START_PORT + VLLM_VALUE_PORT_OFFSET))"
export RL_VLLM_API_KEY="$API_KEY"
export RL_VLLM_MODEL_PATH="$VLLM_MODEL_PATH"
export RL_NUM_PROCS="$NUM_PROCS"
export RL_LATEST_MODEL_FILE="${RL_LATEST_MODEL_FILE:-$REPO_ROOT/runs/rl/latest_model_kl.json}"
export RL_PY_BIN="$PY_BIN"
export TOKENIZERS_PARALLELISM=false

case "$VLLM_MODE" in
    safe)
        export VLLM_COMPILE_BACKEND=none
        export VLLM_USE_TORCH_COMPILE=0
        export VLLM_TORCH_COMPILE=0
        export TORCH_COMPILE_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        export TORCHINDUCTOR_DISABLE=1
        export TORCHINDUCTOR_FREEZING=0
        ENFORCE_EAGER=1
        ;;
    fast)
        export VLLM_COMPILE_BACKEND=inductor
        export VLLM_USE_TORCH_COMPILE=1
        export VLLM_TORCH_COMPILE=1
        export TORCH_COMPILE_DISABLE=0
        export TORCHDYNAMO_DISABLE=0
        export TORCHINDUCTOR_DISABLE=0
        export TORCHINDUCTOR_FREEZING=1
        ;;
    balanced)
        export VLLM_COMPILE_BACKEND=none
        export VLLM_USE_TORCH_COMPILE=0
        export VLLM_TORCH_COMPILE=0
        export TORCH_COMPILE_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        export TORCHINDUCTOR_DISABLE=1
        export TORCHINDUCTOR_FREEZING=0
        ENFORCE_EAGER=1
        ;;
    *)
        echo "[cluster-rl] invalid RL_VLLM_MODE=$VLLM_MODE (expected safe|balanced|fast)" >&2
        exit 1
        ;;
esac

VLLM_PIDS=()
VLLM_PGIDS=()
VLLM_PORTS=()
VLLM_ENGINE_PORTS=()
VLLM_INTERNAL_PORT_BASES=()
RESERVE_PIDS=()
VLLM_INTERNAL_PORT_SPAN="${RL_VLLM_INTERNAL_PORT_SPAN:-20}"
TAIL_FAILED_LOGS="${RL_TAIL_FAILED_LOGS:-0}"
cleanup() {
    if declare -F stop_gpu_reservations >/dev/null 2>&1; then
        stop_gpu_reservations
    fi
    if [[ "$VLLM_LIFECYCLE" == "stage" ]]; then
        if declare -F stop_vllm_servers >/dev/null 2>&1; then
            stop_vllm_servers
        fi
    else
        echo "[cluster-rl] keep vLLM servers alive lifecycle=$VLLM_LIFECYCLE"
        if declare -F save_runtime_state >/dev/null 2>&1; then
            save_runtime_state
        fi
    fi
}

send_failure_email() {
    local exit_code="$1"
    local reason="${2:-cluster_run_rl exited}"
    [[ -n "$NOTIFY_EMAIL" ]] || return 0
    if [[ -z "$SMTP_USER" || -z "$SMTP_PASS" ]]; then
        echo "[cluster-rl] notification skipped: RL_SMTP_USER/RL_SMTP_PASS not set" >&2
        return 0
    fi
    local latest_update="unknown"
    if [[ -f "$EXPERIMENT_ROOT/runs/grpo/latest_model.json" ]]; then
        latest_update="$("$PY_BIN" - <<PY 2>/dev/null || true
import json
from pathlib import Path
p = Path("$EXPERIMENT_ROOT/runs/grpo/latest_model.json")
try:
    print((json.loads(p.read_text()) or {}).get("update_idx", "unknown"))
except Exception:
    print("unknown")
PY
)"
    elif [[ -f "$EXPERIMENT_ROOT/runs/rl/latest_model.json" ]]; then
        latest_update="$("$PY_BIN" - <<PY 2>/dev/null || true
import json
from pathlib import Path
p = Path("$EXPERIMENT_ROOT/runs/rl/latest_model.json")
try:
    print((json.loads(p.read_text()) or {}).get("update_idx", "unknown"))
except Exception:
    print("unknown")
PY
)"
    fi
    local subject="[Collab-Overcooked] RL experiment failed"
    local body
    body="$(cat <<EOF
Experiment failed on $(hostname) at $(date).
Exit code: $exit_code
Reason: $reason
Experiment root: $EXPERIMENT_ROOT
Latest update: $latest_update
Logs:
- $LOG_ROOT/launch.log
- $LOG_ROOT/rl_workers
- $LOG_ROOT/rl_vllm
EOF
)"
    if SMTP_HOST="$SMTP_HOST" SMTP_PORT="$SMTP_PORT" SMTP_USER="$SMTP_USER" \
        SMTP_PASS="$SMTP_PASS" MAIL_TO="$NOTIFY_EMAIL" MAIL_SUBJECT="$subject" \
        MAIL_BODY="$body" "$PY_BIN" - <<'PY'
import os
import smtplib
import ssl
from email.message import EmailMessage

msg = EmailMessage()
msg["From"] = os.environ["SMTP_USER"]
msg["To"] = os.environ["MAIL_TO"]
msg["Subject"] = os.environ["MAIL_SUBJECT"]
msg.set_content(os.environ["MAIL_BODY"])

host = os.environ["SMTP_HOST"]
port = int(os.environ["SMTP_PORT"])
if port == 465:
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as server:
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        server.send_message(msg)
else:
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        server.send_message(msg)
PY
    then
        echo "[cluster-rl] sent failure notification to $NOTIFY_EMAIL"
    else
        echo "[cluster-rl] failed to send notification to $NOTIFY_EMAIL" >&2
    fi
}

send_success_email() {
    [[ "$NOTIFY_ON_SUCCESS" == "1" ]] || return 0
    [[ -n "$NOTIFY_EMAIL" ]] || return 0
    if [[ -z "$SMTP_USER" || -z "$SMTP_PASS" ]]; then
        return 0
    fi
    local subject="[Collab-Overcooked] RL experiment completed"
    local body="Experiment completed on $(hostname) at $(date).
Experiment root: $EXPERIMENT_ROOT
Logs: $LOG_ROOT"
    SMTP_HOST="$SMTP_HOST" SMTP_PORT="$SMTP_PORT" SMTP_USER="$SMTP_USER" \
        SMTP_PASS="$SMTP_PASS" MAIL_TO="$NOTIFY_EMAIL" MAIL_SUBJECT="$subject" \
        MAIL_BODY="$body" "$PY_BIN" - <<'PY' || true
import os
import smtplib
import ssl
from email.message import EmailMessage

msg = EmailMessage()
msg["From"] = os.environ["SMTP_USER"]
msg["To"] = os.environ["MAIL_TO"]
msg["Subject"] = os.environ["MAIL_SUBJECT"]
msg.set_content(os.environ["MAIL_BODY"])
host = os.environ["SMTP_HOST"]
port = int(os.environ["SMTP_PORT"])
if port == 465:
    with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=30) as server:
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        server.send_message(msg)
else:
    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        server.send_message(msg)
PY
}

EXIT_REASON="normal exit"
on_exit() {
    local code="$?"
    trap - EXIT INT TERM HUP QUIT
    cleanup
    if [[ "$code" -ne 0 ]]; then
        send_failure_email "$code" "$EXIT_REASON"
    else
        send_success_email
    fi
    exit "$code"
}

trap on_exit EXIT
trap 'EXIT_REASON="interrupted by SIGINT"; exit 130' INT
trap 'EXIT_REASON="terminated by SIGTERM"; exit 143' TERM
trap 'EXIT_REASON="terminated by SIGHUP"; exit 129' HUP
trap 'EXIT_REASON="terminated by SIGQUIT"; exit 131' QUIT

for cfg in "$COLLECT_CFG" "$TRAIN_CFG" "$EVAL_CFG"; do
    if [[ -n "$cfg" && ! -f "$cfg" ]]; then
        EXIT_REASON="config not found: $cfg"
        echo "[cluster-rl] config not found: $cfg" >&2
        exit 1
    fi
done
if [[ ! -d "$VLLM_MODEL_PATH" ]]; then
    EXIT_REASON="vLLM model path not found: $VLLM_MODEL_PATH"
    echo "[cluster-rl] vLLM model path not found: $VLLM_MODEL_PATH" >&2
    exit 1
fi

if [[ "$PRE_CLEANUP" == "1" && -x "$REPO_ROOT/scripts/cleanup_rl_processes.sh" ]]; then
    echo "[cluster-rl] pre-cleanup stale RL/vLLM processes under experiment_root=$EXPERIMENT_ROOT"
    RL_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
        RL_CLEANUP_GPU_IDS="$GPU_SCOPE_RAW" \
        RL_CLEANUP_PURGE_OUTPUTS=0 \
        RL_CLEANUP_FALLBACK_PORT_BLOCK=0 \
        "$REPO_ROOT/scripts/cleanup_rl_processes.sh" || true
fi

process_exists() {
    local pid="$1"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1
}

process_command() {
    local pid="$1"
    ps -p "$pid" -o command= 2>/dev/null || true
}

process_user() {
    local pid="$1"
    ps -p "$pid" -o user= 2>/dev/null | tr -d ' ' || true
}

process_pgid() {
    local pid="$1"
    ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true
}

value_in_csv() {
    local needle="$1"
    local csv="$2"
    [[ -n "$needle" && -n "$csv" ]] || return 1
    local item
    IFS=',' read -r -a items <<< "$csv"
    for item in "${items[@]}"; do
        [[ -n "$item" ]] || continue
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

csv_intersects() {
    local left="$1"
    local right="$2"
    [[ -n "$left" && -n "$right" ]] || return 1
    local item
    IFS=',' read -r -a items <<< "$left"
    for item in "${items[@]}"; do
        [[ -n "$item" ]] || continue
        value_in_csv "$item" "$right" && return 0
    done
    return 1
}

process_cuda_visible_devices() {
    local pid="$1"
    [[ -r "/proc/$pid/environ" ]] || return 0
    tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
        | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' \
        | head -n 1 \
        | tr -d ' '
}

pid_matches_gpu_scope() {
    local pid="$1"
    [[ -n "$GPU_SCOPE_RAW" ]] || return 0
    [[ -n "$pid" ]] || return 1
    local visible
    visible="$(process_cuda_visible_devices "$pid")"
    if [[ -n "$visible" ]]; then
        csv_intersects "$visible" "$GPU_SCOPE_RAW" && return 0
        return 1
    fi
    return 1
}

pid_in_list() {
    local needle="$1"
    shift || true
    local item
    for item in "$@"; do
        [[ -n "$item" && "$item" == "$needle" ]] && return 0
    done
    return 1
}

pid_is_ours_for_cleanup() {
    local pid="$1"
    [[ -n "$pid" ]] || return 1
    local current_user
    current_user="$(id -un)"
    [[ "$(process_user "$pid")" == "$current_user" ]] || return 1
    pid_matches_gpu_scope "$pid" || return 1
    pid_in_list "$pid" "${VLLM_PIDS[@]:-}" && return 0
    local pgid
    pgid="$(process_pgid "$pid")"
    pid_in_list "$pgid" "${VLLM_PGIDS[@]:-}" && return 0
    local cmd
    cmd="$(process_command "$pid")"
    [[ -n "$cmd" && ( "$cmd" == *"$EXPERIMENT_ROOT"* || "$cmd" == *"$STATE_ROOT"* ) ]]
}

kill_process_group() {
    local pgid="$1"
    [[ -n "$pgid" ]] || return 0
    kill -TERM -- "-$pgid" >/dev/null 2>&1 || true
    sleep 1
    kill -KILL -- "-$pgid" >/dev/null 2>&1 || true
}

write_list_file() {
    local path="$1"
    shift || true
    : > "$path"
    local item
    for item in "$@"; do
        [[ -n "$item" ]] && printf '%s\n' "$item" >> "$path"
    done
}

read_list_file() {
    local path="$1"
    [[ -f "$path" ]] || return 0
    while IFS= read -r item; do
        [[ -n "$item" ]] && printf '%s\n' "$item"
    done < "$path"
}

save_runtime_state() {
    printf 'experiment_root=%s\n' "$EXPERIMENT_ROOT" > "$STATE_ROOT/meta.env"
    printf 'log_root=%s\n' "$LOG_ROOT" >> "$STATE_ROOT/meta.env"
    printf 'num_procs=%s\n' "$NUM_PROCS" >> "$STATE_ROOT/meta.env"
    printf 'vllm_start_port=%s\n' "$VLLM_START_PORT" >> "$STATE_ROOT/meta.env"
    printf 'vllm_value_start_port=%s\n' "$((VLLM_START_PORT + VLLM_VALUE_PORT_OFFSET))" >> "$STATE_ROOT/meta.env"
    printf 'vllm_split_actor_value=%s\n' "$VLLM_SPLIT_ACTOR_VALUE" >> "$STATE_ROOT/meta.env"
    printf 'vllm_internal_port_span=%s\n' "$VLLM_INTERNAL_PORT_SPAN" >> "$STATE_ROOT/meta.env"
    write_list_file "$STATE_ROOT/vllm_pids.txt" "${VLLM_PIDS[@]:-}"
    write_list_file "$STATE_ROOT/vllm_pgids.txt" "${VLLM_PGIDS[@]:-}"
    write_list_file "$STATE_ROOT/vllm_ports.txt" "${VLLM_PORTS[@]:-}"
    write_list_file "$STATE_ROOT/vllm_engine_ports.txt" "${VLLM_ENGINE_PORTS[@]:-}"
    write_list_file "$STATE_ROOT/vllm_internal_bases.txt" "${VLLM_INTERNAL_PORT_BASES[@]:-}"
}

load_runtime_state() {
    [[ -f "$STATE_ROOT/meta.env" ]] || return 1
    local saved_num_procs=""
    local saved_start_port=""
    local saved_split=""
    local saved_span=""
    while IFS='=' read -r key value; do
        case "$key" in
            num_procs) saved_num_procs="$value" ;;
            vllm_start_port) saved_start_port="$value" ;;
            vllm_split_actor_value) saved_split="$value" ;;
            vllm_internal_port_span) saved_span="$value" ;;
        esac
    done < "$STATE_ROOT/meta.env"
    [[ "$saved_num_procs" == "$NUM_PROCS" ]] || return 1
    [[ "$saved_split" == "$VLLM_SPLIT_ACTOR_VALUE" ]] || return 1
    [[ -z "$saved_span" || "$saved_span" == "$VLLM_INTERNAL_PORT_SPAN" ]] || return 1
    [[ -n "$saved_start_port" ]] || return 1

    VLLM_START_PORT="$saved_start_port"
    export RL_VLLM_START_PORT="$VLLM_START_PORT"
    export RL_VLLM_VALUE_START_PORT="$((VLLM_START_PORT + VLLM_VALUE_PORT_OFFSET))"
    mapfile -t VLLM_PIDS < <(read_list_file "$STATE_ROOT/vllm_pids.txt")
    mapfile -t VLLM_PGIDS < <(read_list_file "$STATE_ROOT/vllm_pgids.txt")
    mapfile -t VLLM_PORTS < <(read_list_file "$STATE_ROOT/vllm_ports.txt")
    mapfile -t VLLM_ENGINE_PORTS < <(read_list_file "$STATE_ROOT/vllm_engine_ports.txt")
    mapfile -t VLLM_INTERNAL_PORT_BASES < <(read_list_file "$STATE_ROOT/vllm_internal_bases.txt")
    return 0
}

save_worker_state() {
    write_list_file "$STATE_ROOT/worker_pids.txt" "$@"
}

clear_worker_state() {
    : > "$STATE_ROOT/worker_pids.txt"
}

kill_port_listener() {
    local port="$1"
    if [[ "$VLLM_LIFECYCLE" == "marshal" ]]; then
        echo "[cluster-rl] lifecycle=marshal: not killing listener on port=$port"
        return 0
    fi
    if command -v lsof >/dev/null 2>&1; then
        local pids=()
        while IFS= read -r pid; do
            [[ -n "$pid" ]] && pids+=("$pid")
        done < <(lsof -tiTCP:"$port" -sTCP:LISTEN -Pn 2>/dev/null || true)
        if [[ "${#pids[@]}" -gt 0 ]]; then
            local owned_pids=()
            local pid
            for pid in "${pids[@]}"; do
                if pid_is_ours_for_cleanup "$pid"; then
                    owned_pids+=("$pid")
                else
                    echo "[cluster-rl] skip listener on port=$port pid=$pid because it is not owned by this run"
                fi
            done
            if [[ "${#owned_pids[@]}" -eq 0 ]]; then
                return 0
            fi
            echo "[cluster-rl] clearing owned listener on port=$port pid=${owned_pids[*]}"
            kill "${owned_pids[@]}" 2>/dev/null || true
            sleep 1
            kill -9 "${owned_pids[@]}" 2>/dev/null || true
        fi
    fi
}

pick_free_port() {
    python - <<'PY'
import socket
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
print(sock.getsockname()[1])
sock.close()
PY
}

port_is_bindable() {
    local port="$1"
    python - <<PY
import fcntl
import tempfile
from pathlib import Path
import socket, sys
lock_path = Path(tempfile.gettempdir()) / f"rl_vllm_port_{int("$port")}.lock"
lock_handle = lock_path.open("a+")
try:
    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(1)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", int("$port")))
except OSError:
    sys.exit(1)
else:
    sock.close()
    sys.exit(0)
PY
}

ensure_train_master_port() {
    local desired_port="$1"
    if ! port_is_bindable "$desired_port"; then
        local new_port
        new_port="$(pick_free_port)"
        echo "[cluster-rl] MASTER_PORT $desired_port is busy, switching to $new_port"
        MASTER_PORT="$new_port"
        export MASTER_PORT
        return
    fi
    MASTER_PORT="$desired_port"
    export MASTER_PORT
}

port_is_listening() {
    local port="$1"
    if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$port" -sTCP:LISTEN -Pn >/dev/null 2>&1
        return $?
    fi
    python - <<PY
import socket, sys
sock = socket.socket()
sock.settimeout(0.2)
try:
    sock.connect(("127.0.0.1", int("$port")))
except OSError:
    sys.exit(1)
else:
    sock.close()
    sys.exit(0)
PY
}

reserve_vllm_port_block() {
    local base_port="${1:-$VLLM_START_PORT}"
    local count="${2:-$NUM_PROCS}"
    local max_tries=200
    local try_idx=0
    local per_gpu_internal_span="$VLLM_INTERNAL_PORT_SPAN"
    local services_per_gpu=1
    if [[ "$VLLM_SPLIT_ACTOR_VALUE" == "1" ]]; then
        services_per_gpu=2
    fi
    local block_span=$((200 + count * services_per_gpu * per_gpu_internal_span))
    if [[ "$VLLM_SPLIT_ACTOR_VALUE" == "1" ]]; then
        block_span=$((VLLM_VALUE_PORT_OFFSET + 200 + count * services_per_gpu * per_gpu_internal_span))
    fi
    while (( try_idx < max_tries )); do
        local candidate=$((base_port + try_idx * block_span))
        local ok=1
        for ((gpu=0; gpu<count; gpu++)); do
            local api_port=$((candidate + gpu))
            local engine_port=$((candidate + 100 + gpu))
            local internal_base=$((candidate + 200 + gpu * per_gpu_internal_span))
            if ! port_is_bindable "$api_port" || ! port_is_bindable "$engine_port"; then
                ok=0
                break
            fi
            if [[ "$VLLM_SPLIT_ACTOR_VALUE" == "1" ]]; then
                local value_api_port=$((candidate + VLLM_VALUE_PORT_OFFSET + gpu))
                local value_engine_port=$((candidate + 100 + VLLM_VALUE_PORT_OFFSET + gpu))
                local value_internal_base=$((candidate + 200 + (count + gpu) * per_gpu_internal_span))
                if ! port_is_bindable "$value_api_port" || ! port_is_bindable "$value_engine_port"; then
                    ok=0
                    break
                fi
                for ((offset=0; offset<per_gpu_internal_span; offset++)); do
                    local value_internal_port=$((value_internal_base + offset))
                    if ! port_is_bindable "$value_internal_port"; then
                        ok=0
                        break
                    fi
                done
            fi
            for ((offset=0; offset<per_gpu_internal_span; offset++)); do
                local internal_port=$((internal_base + offset))
                if ! port_is_bindable "$internal_port"; then
                    ok=0
                    break
                fi
            done
            if (( ok == 0 )); then
                break
            fi
        done
        if (( ok == 1 )); then
            VLLM_START_PORT="$candidate"
            export RL_VLLM_START_PORT="$VLLM_START_PORT"
            export RL_VLLM_VALUE_START_PORT="$((VLLM_START_PORT + VLLM_VALUE_PORT_OFFSET))"
            echo "[cluster-rl] reserved vLLM port block start=$VLLM_START_PORT count=$count internal_span=$per_gpu_internal_span"
            save_runtime_state
            return 0
        fi
        try_idx=$((try_idx + 1))
    done
    echo "[cluster-rl] unable to reserve a free vLLM port block from base=$base_port count=$count" >&2
    return 1
}

stop_vllm_servers() {
    if [[ "$VLLM_LIFECYCLE" != "stage" ]]; then
        echo "[cluster-rl] lifecycle=$VLLM_LIFECYCLE: skip stopping vLLM servers"
        save_runtime_state
        return 0
    fi
    for idx in "${!VLLM_PIDS[@]}"; do
        local pid="${VLLM_PIDS[$idx]}"
        local pgid=""
        if [[ "$idx" -lt "${#VLLM_PGIDS[@]}" ]]; then
            pgid="${VLLM_PGIDS[$idx]}"
        fi
        if process_exists "$pid"; then
            if [[ -n "$pgid" ]]; then
                kill_process_group "$pgid"
            else
                kill "$pid" >/dev/null 2>&1 || true
                sleep 1
                kill -9 "$pid" >/dev/null 2>&1 || true
            fi
            wait "$pid" || true
        fi
    done
    for port in "${VLLM_PORTS[@]:-}"; do
        if [[ -n "$port" ]]; then
            kill_port_listener "$port"
        fi
    done
    for port in "${VLLM_ENGINE_PORTS[@]:-}"; do
        if [[ -n "$port" ]]; then
            kill_port_listener "$port"
        fi
    done
    for base in "${VLLM_INTERNAL_PORT_BASES[@]:-}"; do
        if [[ -n "$base" ]]; then
            for ((offset=0; offset<VLLM_INTERNAL_PORT_SPAN; offset++)); do
                local internal_port=$((base + offset))
                kill_port_listener "$internal_port"
            done
        fi
    done
    VLLM_PIDS=()
    VLLM_PGIDS=()
    VLLM_PORTS=()
    VLLM_ENGINE_PORTS=()
    VLLM_INTERNAL_PORT_BASES=()
    save_runtime_state
}

start_gpu_reservations() {
    return 0
}

stop_gpu_reservations() {
    local pid
    for pid in "${RESERVE_PIDS[@]:-}"; do
        if process_exists "$pid"; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
    sleep 1
    for pid in "${RESERVE_PIDS[@]:-}"; do
        if process_exists "$pid"; then
            kill -9 "$pid" >/dev/null 2>&1 || true
        fi
    done
    RESERVE_PIDS=()
}

gpu_free_mb() {
    local physical_gpu="$1"
    nvidia-smi --id="$physical_gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -n 1 \
        | tr -d ' '
}

wait_for_train_gpu_memory() {
    [[ "$TRAIN_MIN_FREE_MB" =~ ^[0-9]+$ ]] || return 0
    (( TRAIN_MIN_FREE_MB > 0 )) || return 0
    local deadline=0
    if [[ "$TRAIN_GPU_WAIT_SECONDS" =~ ^[0-9]+$ ]] && (( TRAIN_GPU_WAIT_SECONDS > 0 )); then
        deadline=$((SECONDS + TRAIN_GPU_WAIT_SECONDS))
    fi
    local fail_fast=0
    if [[ "$TRAIN_GPU_WAIT_SECONDS" =~ ^[0-9]+$ ]] && (( TRAIN_GPU_WAIT_SECONDS == 0 )); then
        fail_fast=1
    fi
    while true; do
        local ok=1
        for ((rank=0; rank<NUM_PROCS; rank++)); do
            local physical_gpu free_mb
            physical_gpu="$(physical_gpu_for_rank "$rank")"
            free_mb="$(gpu_free_mb "$physical_gpu")"
            if [[ ! "$free_mb" =~ ^[0-9]+$ ]]; then
                echo "[cluster-rl] unable to read free memory for gpu=$physical_gpu" >&2
                ok=0
                break
            fi
            if (( free_mb < TRAIN_MIN_FREE_MB )); then
                echo "[cluster-rl] waiting for train memory gpu=$physical_gpu free_mb=$free_mb required_mb=$TRAIN_MIN_FREE_MB"
                ok=0
            fi
        done
        (( ok == 1 )) && return 0
        if (( fail_fast == 1 )); then
            echo "[cluster-rl] insufficient train GPU memory; failing fast required_mb=$TRAIN_MIN_FREE_MB" >&2
            return 1
        fi
        if (( deadline > 0 && SECONDS >= deadline )); then
            echo "[cluster-rl] timeout waiting for train GPU memory required_mb=$TRAIN_MIN_FREE_MB" >&2
            return 1
        fi
        sleep 30
    done
}

wait_for_port() {
    local host="$1"
    local port="$2"
    local pid="$3"
    local log_file="$4"
    for _ in $(seq 1 180); do
        if ! ps -p "$pid" >/dev/null 2>&1; then
            echo "[cluster-rl] vLLM exited early on $host:$port; log=$log_file" >&2
            if [[ "$TAIL_FAILED_LOGS" == "1" ]]; then
                tail -n 100 "$log_file" >&2 || true
            else
                echo "[cluster-rl] inspect with: tail -n 100 $log_file" >&2
            fi
            return 1
        fi
        if python - <<PY
import socket, sys
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect(("$host", int("$port")))
except OSError:
    sys.exit(1)
else:
    sock.close()
    sys.exit(0)
PY
        then
            return 0
        fi
        sleep 1
    done
    echo "[cluster-rl] timeout waiting for $host:$port; log=$log_file" >&2
    if [[ "$TAIL_FAILED_LOGS" == "1" ]]; then
        tail -n 100 "$log_file" >&2 || true
    else
        echo "[cluster-rl] inspect with: tail -n 100 $log_file" >&2
    fi
    return 1
}

start_vllm_servers() {
    local count="$1"
    local cfg_path="$2"
    local stage_name="$3"
    for ((gpu=0; gpu<count; gpu++)); do
        local port=$((VLLM_START_PORT + gpu))
        local engine_port=$((VLLM_START_PORT + 100 + gpu))
        local internal_port_base=$((VLLM_START_PORT + 200 + gpu * VLLM_INTERNAL_PORT_SPAN))
        local roles=("both")
        if [[ "$VLLM_SPLIT_ACTOR_VALUE" == "1" ]]; then
            if [[ "$stage_name" == "eval" ]]; then
                roles=("actor")
            else
                roles=("actor" "value")
            fi
        fi
        local role
        for role in "${roles[@]}"; do
            local service_port="$port"
            local service_engine_port="$engine_port"
            local service_internal_port_base="$internal_port_base"
            local service_serial="${RL_VLLM_SERIALIZE_GENERATE:-1}"
            if [[ "$role" == "value" ]]; then
                service_port=$((VLLM_START_PORT + VLLM_VALUE_PORT_OFFSET + gpu))
                service_engine_port=$((VLLM_START_PORT + 100 + VLLM_VALUE_PORT_OFFSET + gpu))
                service_internal_port_base=$((VLLM_START_PORT + 200 + (count + gpu) * VLLM_INTERNAL_PORT_SPAN))
                service_serial=1
                service_gpu_mem="$VALUE_GPU_MEM"
            elif [[ "$role" == "actor" ]]; then
                service_serial="${RL_VLLM_ACTOR_SERIALIZE_GENERATE:-${RL_VLLM_SERIALIZE_GENERATE:-0}}"
                service_gpu_mem="$ACTOR_GPU_MEM"
            else
                service_gpu_mem="$GPU_MEM"
            fi
            VLLM_PORTS+=("$service_port")
            VLLM_ENGINE_PORTS+=("$service_engine_port")
            VLLM_INTERNAL_PORT_BASES+=("$service_internal_port_base")
            if [[ "$VLLM_LIFECYCLE" == "stage" ]]; then
                kill_port_listener "$service_port"
                kill_port_listener "$service_engine_port"
                for ((offset=0; offset<VLLM_INTERNAL_PORT_SPAN; offset++)); do
                    local internal_port=$((service_internal_port_base + offset))
                    kill_port_listener "$internal_port"
                done
            else
                if ! port_is_bindable "$service_port" || ! port_is_bindable "$service_engine_port"; then
                    echo "[cluster-rl] port busy while starting vLLM service_port=$service_port engine_port=$service_engine_port lifecycle=$VLLM_LIFECYCLE" >&2
                    return 1
                fi
            fi
            local log_file="$LOG_ROOT/rl_vllm/vllm_${stage_name}_${role}_gpu${gpu}.log"
            echo "[cluster-rl] starting vLLM stage=$stage_name role=$role gpu=$gpu port=$service_port engine_port=$service_engine_port internal_port_base=$service_internal_port_base serialize=$service_serial gpu_mem=$service_gpu_mem"
            if command -v setsid >/dev/null 2>&1; then
                RL_VLLM_SERIALIZE_GENERATE="$service_serial" \
                setsid "$VLLM_PY" "$REPO_ROOT/scripts/start_vllm_server.py" \
                    --gpu "$gpu" \
                    --port "$service_port" \
                    --engine-port "$service_engine_port" \
                    --internal-port-base "$service_internal_port_base" \
                    --model "$VLLM_MODEL_PATH" \
                    --config "$cfg_path" \
                    --service-role "$role" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --gpu-memory-utilization "$service_gpu_mem" \
                    --max-model-len "$MAX_MODEL_LEN" \
                    --api-key "$API_KEY" \
                    --max-loras "$MAX_LORAS" \
                    --max-lora-rank "$MAX_LORA_RANK" \
                    $([[ "$ENFORCE_EAGER" == "1" ]] && echo "--enforce-eager") \
                    >"$log_file" 2>&1 &
            else
                RL_VLLM_SERIALIZE_GENERATE="$service_serial" \
                "$VLLM_PY" "$REPO_ROOT/scripts/start_vllm_server.py" \
                    --gpu "$gpu" \
                    --port "$service_port" \
                    --engine-port "$service_engine_port" \
                    --internal-port-base "$service_internal_port_base" \
                    --model "$VLLM_MODEL_PATH" \
                    --config "$cfg_path" \
                    --service-role "$role" \
                    --served-model-name "$SERVED_MODEL_NAME" \
                    --gpu-memory-utilization "$service_gpu_mem" \
                    --max-model-len "$MAX_MODEL_LEN" \
                    --api-key "$API_KEY" \
                    --max-loras "$MAX_LORAS" \
                    --max-lora-rank "$MAX_LORA_RANK" \
                    $([[ "$ENFORCE_EAGER" == "1" ]] && echo "--enforce-eager") \
                    >"$log_file" 2>&1 &
            fi
            local vllm_pid="$!"
            local vllm_pgid=""
            if command -v ps >/dev/null 2>&1; then
                vllm_pgid="$(ps -o pgid= -p "$vllm_pid" 2>/dev/null | tr -d ' ' || true)"
            fi
            VLLM_PIDS+=("$vllm_pid")
            VLLM_PGIDS+=("$vllm_pgid")
            if ! wait_for_port "$VLLM_HOST" "$service_port" "$vllm_pid" "$log_file"; then
                return 1
            fi
        done
    done
    save_runtime_state

    local idx=0
    for ((gpu=0; gpu<count; gpu++)); do
        if [[ "$VLLM_SPLIT_ACTOR_VALUE" == "1" ]]; then
            local actor_port=$((VLLM_START_PORT + gpu))
            local value_port=$((VLLM_START_PORT + VLLM_VALUE_PORT_OFFSET + gpu))
            wait_for_port "$VLLM_HOST" "$actor_port" "${VLLM_PIDS[$idx]}" "$LOG_ROOT/rl_vllm/vllm_${stage_name}_actor_gpu${gpu}.log"
            idx=$((idx + 1))
            if [[ "$stage_name" != "eval" ]]; then
                wait_for_port "$VLLM_HOST" "$value_port" "${VLLM_PIDS[$idx]}" "$LOG_ROOT/rl_vllm/vllm_${stage_name}_value_gpu${gpu}.log"
                idx=$((idx + 1))
            fi
        else
            wait_for_port "$VLLM_HOST" "$((VLLM_START_PORT + gpu))" "${VLLM_PIDS[$idx]}" "$LOG_ROOT/rl_vllm/vllm_${stage_name}_both_gpu${gpu}.log"
            idx=$((idx + 1))
        fi
    done
}

vllm_servers_running() {
    if [[ "${#VLLM_PIDS[@]}" -eq 0 ]]; then
        return 1
    fi
    local pid
    for pid in "${VLLM_PIDS[@]}"; do
        if ! process_exists "$pid"; then
            return 1
        fi
    done
    local port
    for port in "${VLLM_PORTS[@]}"; do
        if ! port_is_listening "$port"; then
            return 1
        fi
    done
    return 0
}

restore_vllm_servers_if_running() {
    [[ "$VLLM_LIFECYCLE" != "stage" ]] || return 1
    load_runtime_state || return 1
    if vllm_servers_running; then
        echo "[cluster-rl] restored persistent vLLM servers start_port=$VLLM_START_PORT pids=${VLLM_PIDS[*]} ports=${VLLM_PORTS[*]}"
        return 0
    fi
    echo "[cluster-rl] persistent vLLM state exists but servers are not running; will start new servers"
    VLLM_PIDS=()
    VLLM_PGIDS=()
    VLLM_PORTS=()
    VLLM_ENGINE_PORTS=()
    VLLM_INTERNAL_PORT_BASES=()
    save_runtime_state
    return 1
}

reload_vllm_adapters() {
    local cfg_path="$1"
    local stage_name="${2:-}"
    echo "[cluster-rl] reloading vLLM adapters cfg=$cfg_path ports=${VLLM_PORTS[*]:-}"
    if [[ "$VLLM_SPLIT_ACTOR_VALUE" == "1" ]]; then
        "$PY_BIN" "$REPO_ROOT/scripts/reload_vllm_adapters.py" \
            --config "$cfg_path" \
            --host "$VLLM_HOST" \
            --start-port "$VLLM_START_PORT" \
            --count "$NUM_PROCS" \
            --service-role actor
        if [[ "$stage_name" != "eval" ]]; then
            "$PY_BIN" "$REPO_ROOT/scripts/reload_vllm_adapters.py" \
                --config "$cfg_path" \
                --host "$VLLM_HOST" \
                --start-port "$((VLLM_START_PORT + VLLM_VALUE_PORT_OFFSET))" \
                --count "$NUM_PROCS" \
                --service-role value
        fi
    else
        "$PY_BIN" "$REPO_ROOT/scripts/reload_vllm_adapters.py" \
            --config "$cfg_path" \
            --host "$VLLM_HOST" \
            --start-port "$VLLM_START_PORT" \
            --count "$NUM_PROCS" \
            --service-role both
    fi
}

ensure_train_memory_after_vllm_sleep() {
    [[ "$TRAIN_MIN_FREE_MB" =~ ^[0-9]+$ ]] || return 0
    (( TRAIN_MIN_FREE_MB > 0 )) || return 0
    local ok=1
    for ((rank=0; rank<NUM_PROCS; rank++)); do
        local physical_gpu free_mb
        physical_gpu="$(physical_gpu_for_rank "$rank")"
        free_mb="$(gpu_free_mb "$physical_gpu")"
        if [[ ! "$free_mb" =~ ^[0-9]+$ ]]; then
            echo "[cluster-rl] unable to read free memory after vLLM sleep gpu=$physical_gpu" >&2
            ok=0
            continue
        fi
        if (( free_mb < TRAIN_MIN_FREE_MB )); then
            echo "[cluster-rl] vLLM sleep left insufficient train memory gpu=$physical_gpu free_mb=$free_mb required_mb=$TRAIN_MIN_FREE_MB"
            ok=0
        fi
    done
    if (( ok == 1 )); then
        return 0
    fi
    if [[ "$TRAIN_STOP_VLLM_IF_SLEEP_INSUFFICIENT" == "1" && "$VLLM_LIFECYCLE" == "marshal" ]]; then
        echo "[cluster-rl] stopping this run's vLLM servers before train to free GPU memory"
        stop_vllm_servers
        return 0
    fi
    return 1
}

control_vllm_state() {
    local action="$1"
    [[ "$VLLM_LIFECYCLE" == "marshal" ]] || return 0
    [[ "$VLLM_SLEEP_DURING_TRAIN" == "1" ]] || return 0
    [[ "${#VLLM_PORTS[@]}" -gt 0 ]] || return 0
    local start_port="$VLLM_START_PORT"
    local count="${#VLLM_PORTS[@]}"
    local ports_csv
    ports_csv="$(IFS=','; echo "${VLLM_PORTS[*]}")"
    echo "[cluster-rl] vLLM $action ports=$ports_csv lifecycle=$VLLM_LIFECYCLE"
    if [[ "$action" == "sleep" ]]; then
        "$PY_BIN" "$REPO_ROOT/scripts/control_vllm_state.py" \
            --action sleep \
            --host "$VLLM_HOST" \
            --start-port "$start_port" \
            --count "$count" \
            --ports "$ports_csv" \
            --level "$VLLM_SLEEP_LEVEL" \
            --mode "$VLLM_SLEEP_MODE" \
            --ignore-unsupported
    else
        "$PY_BIN" "$REPO_ROOT/scripts/control_vllm_state.py" \
            --action wake \
            --host "$VLLM_HOST" \
            --start-port "$start_port" \
            --count "$count" \
            --ports "$ports_csv" \
            --ignore-unsupported
    fi
}

ensure_vllm_servers() {
    local cfg_path="$1"
    local stage_name="$2"
    if [[ "$VLLM_LIFECYCLE" != "stage" ]]; then
        if vllm_servers_running || restore_vllm_servers_if_running; then
            control_vllm_state wake
            reload_vllm_adapters "$cfg_path" "$stage_name"
            return $?
        fi
        start_vllm_servers "$NUM_PROCS" "$cfg_path" "$stage_name"
        control_vllm_state wake
        return $?
    fi
    if [[ "$VLLM_PERSISTENT" == "1" ]] && vllm_servers_running; then
        control_vllm_state wake
        reload_vllm_adapters "$cfg_path" "$stage_name"
        return $?
    fi
    stop_vllm_servers
    start_vllm_servers "$NUM_PROCS" "$cfg_path" "$stage_name"
}

run_stage_workers() {
    local stage_name="$1"
    local cfg_path="$2"
    local total_workers="$3"
    local worker_pids=()
    local worker_logs=()
    local status=0
    local success_count=0
    local failed_count=0
    local min_success=1
    if [[ "$stage_name" == "eval" ]]; then
        min_success="$EVAL_MIN_SUCCESS_WORKERS"
    else
        min_success="$COLLECT_MIN_SUCCESS_WORKERS"
    fi
    mkdir -p "$LOG_ROOT/rl_workers"
    for ((worker_id=0; worker_id<total_workers; worker_id++)); do
        local rank=$((worker_id % NUM_PROCS))
        local physical_gpu
        physical_gpu="$(physical_gpu_for_rank "$rank")"
        local log_file="$LOG_ROOT/rl_workers/${stage_name}_worker${worker_id}_gpu${rank}.log"
        worker_logs+=("$log_file")
        echo "[cluster-rl] starting worker stage=$stage_name worker=$worker_id rank=$rank gpu=$physical_gpu log=$log_file"
        env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK \
            CUDA_VISIBLE_DEVICES="$physical_gpu" \
            RL_WORKER_RANK="$rank" \
            RL_WORKER_ID="$worker_id" \
            RL_STAGE_PHASE="$stage_name" \
            RL_STAGE_ROUND_IDX="$i" \
            RL_LOOP_ROUND_IDX="$i" \
            "$PY_BIN" -u "$REPO_ROOT/scripts/rl_stage_runner.py" --config "$cfg_path" --stage "$stage_name" -- "${RUN_ARGS[@]}" \
            >"$log_file" 2>&1 &
        worker_pids+=("$!")
    done
    save_worker_state "${worker_pids[@]}"

    for idx in "${!worker_pids[@]}"; do
        local pid="${worker_pids[$idx]}"
        if wait "$pid"; then
            success_count=$((success_count + 1))
        else
            failed_count=$((failed_count + 1))
            status=1
            echo "[cluster-rl] worker failed stage=$stage_name worker=$idx log=${worker_logs[$idx]}" >&2
            if [[ "$TAIL_FAILED_LOGS" == "1" ]]; then
                tail -n 80 "${worker_logs[$idx]}" >&2 || true
            else
                echo "[cluster-rl] inspect with: tail -n 80 ${worker_logs[$idx]}" >&2
            fi
        fi
    done
    for pid in "${worker_pids[@]}"; do
        wait "$pid" >/dev/null 2>&1 || true
    done
    clear_worker_state
    if (( failed_count > 0 )); then
        echo "[cluster-rl] stage=$stage_name workers success=$success_count failed=$failed_count min_success=$min_success" >&2
    fi
    if (( success_count >= min_success )); then
        return 0
    fi
    return "$status"
}

aggregate_stage_metrics() {
    local stage_name="$1"
    local cfg_path="$2"
    if [[ "$stage_name" != "collect" && "$stage_name" != "eval" ]]; then
        return 0
    fi
    echo "[cluster-rl] aggregating stage metrics stage=$stage_name cfg=$cfg_path"
    "$PY_BIN" "$REPO_ROOT/scripts/aggregate_stage_metrics.py" \
        --config "$cfg_path" \
        --stage "$stage_name"
}

run_stage() {
    local stage_name="$1"
    local cfg_path="$2"
    if [[ -z "$cfg_path" ]]; then
        return 0
    fi
    if [[ "$stage_name" == "collect" || "$stage_name" == "eval" ]]; then
        local attempt=1
        local next_base_port="$VLLM_START_PORT"
        local reserved_start_port=""
        while (( attempt <= VLLM_START_RETRIES )); do
            if [[ "$VLLM_LIFECYCLE" != "stage" ]] && restore_vllm_servers_if_running; then
                reserved_start_port="$VLLM_START_PORT"
            elif [[ "$VLLM_PERSISTENT" == "1" ]] && vllm_servers_running; then
                reserved_start_port="$VLLM_START_PORT"
            elif [[ "$VLLM_LIFECYCLE" != "stage" ]]; then
                reserve_vllm_port_block "$next_base_port" "$NUM_PROCS" || return 1
                reserved_start_port="$VLLM_START_PORT"
            else
                stop_vllm_servers
                reserve_vllm_port_block "$next_base_port" "$NUM_PROCS" || return 1
                reserved_start_port="$VLLM_START_PORT"
            fi
            if ensure_vllm_servers "$cfg_path" "$stage_name"; then
                break
            fi
            echo "[cluster-rl] failed to start vLLM stage=$stage_name on port block start=$reserved_start_port attempt=$attempt/$VLLM_START_RETRIES" >&2
            if [[ "$VLLM_LIFECYCLE" == "stage" ]]; then
                stop_vllm_servers
            fi
            attempt=$((attempt + 1))
            next_base_port=$((reserved_start_port + 1))
        done
        if (( attempt > VLLM_START_RETRIES )); then
            echo "[cluster-rl] exhausted vLLM startup retries for stage=$stage_name" >&2
            return 1
        fi
    else
        if [[ "$VLLM_PERSISTENT" != "1" ]]; then
            stop_vllm_servers
        else
            echo "[cluster-rl] stage=$stage_name reuses persistent vLLM lifecycle=$VLLM_LIFECYCLE"
        fi
    fi
    echo "[cluster-rl] stage=$stage_name cfg=$cfg_path procs=$NUM_PROCS"
    if [[ "$stage_name" == "collect" || "$stage_name" == "eval" ]]; then
        start_gpu_reservations
        local total_workers="$COLLECT_WORKERS"
        if [[ "$stage_name" == "eval" ]]; then
            total_workers="$EVAL_WORKERS"
        fi
        run_stage_workers "$stage_name" "$cfg_path" "$total_workers"
        local worker_status=$?
        stop_gpu_reservations
        if [[ $worker_status -eq 0 ]]; then
            aggregate_stage_metrics "$stage_name" "$cfg_path" || worker_status=$?
        fi
        if [[ "$VLLM_PERSISTENT" != "1" ]]; then
            stop_vllm_servers
        else
            control_vllm_state sleep || worker_status=$?
        fi
        return $worker_status
    else
        stop_gpu_reservations
        control_vllm_state sleep
        ensure_train_memory_after_vllm_sleep || return 1
        wait_for_train_gpu_memory
        mkdir -p "$LOG_ROOT/rl_workers"
        local train_log="$LOG_ROOT/rl_workers/${stage_name}_accelerate.log"
        ensure_train_master_port "$MASTER_PORT"
        echo "[cluster-rl] using MASTER_PORT=$MASTER_PORT"
        echo "[cluster-rl] accelerate log=$train_log"
        env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u LOCAL_RANK \
        RL_STAGE_PHASE="$stage_name" \
        RL_STAGE_ROUND_IDX="$i" \
        RL_LOOP_ROUND_IDX="$i" \
        MASTER_PORT="$MASTER_PORT" \
        "$ACCEL_BIN" launch --num_processes "$NUM_PROCS" --main_process_port "$MASTER_PORT" "${EXTRA_ACCEL[@]}" \
            "$REPO_ROOT/scripts/rl_stage_runner.py" --config "$cfg_path" --stage "$stage_name" -- "${RUN_ARGS[@]}" \
            2>&1 | tee "$train_log"
    fi
}

STATUS=0
START_ROUND=1
END_ROUND="$LOOP_ROUNDS"
if [[ "$RESUME_RUN" == "1" ]]; then
    LAST_COMPLETED="$(detect_resume_round)"
    if [[ "$LAST_COMPLETED" =~ ^[0-9]+$ ]] && (( LAST_COMPLETED > 0 )); then
        START_ROUND=$((LAST_COMPLETED + 1))
    fi
    END_ROUND=$((START_ROUND + LOOP_ROUNDS - 1))
    echo "[cluster-rl] resume enabled last_completed=${LAST_COMPLETED:-0} start_round=$START_ROUND end_round=$END_ROUND"
fi
for ((i=START_ROUND; i<=END_ROUND; i++)); do
    echo "[cluster-rl] round $i / $END_ROUND"

    if [[ -n "$ONLY_STAGE" ]]; then
        case "$ONLY_STAGE" in
            collect)
                set +e
                run_stage collect "$COLLECT_CFG"
                STATUS=$?
                set -e
                ;;
            train)
                set +e
                run_stage train "$TRAIN_CFG"
                STATUS=$?
                set -e
                ;;
            eval)
                if [[ -z "$EVAL_CFG" ]]; then
                    echo "[cluster-rl] RL_ONLY_STAGE=eval requested but eval config is empty" >&2
                    STATUS=1
                else
                    set +e
                    run_stage eval "$EVAL_CFG"
                    STATUS=$?
                    set -e
                fi
                ;;
        esac
        if [[ $STATUS -ne 0 ]]; then
            echo "[cluster-rl] stage $ONLY_STAGE failed in round $i" >&2
        fi
        break
    fi

    set +e
    run_stage collect "$COLLECT_CFG"
    STATUS=$?
    set -e
    if [[ $STATUS -ne 0 ]]; then
        echo "[cluster-rl] collect failed in round $i" >&2
        break
    fi

    set +e
    run_stage train "$TRAIN_CFG"
    STATUS=$?
    set -e
    if [[ $STATUS -ne 0 ]]; then
        echo "[cluster-rl] train failed in round $i" >&2
        break
    fi

    if [[ -n "$EVAL_CFG" ]] && (( EVAL_EVERY > 0 )) && (( i % EVAL_EVERY == 0 )); then
        set +e
        run_stage eval "$EVAL_CFG"
        STATUS=$?
        set -e
        if [[ $STATUS -ne 0 ]]; then
            echo "[cluster-rl] eval failed in round $i" >&2
            break
        fi
    fi
done

exit $STATUS
