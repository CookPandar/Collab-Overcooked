#!/bin/bash

set -euo pipefail

abs_path() {
    python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPERIMENT_ROOT="${RL_EXPERIMENT_ROOT:-$REPO_ROOT}"
EXPERIMENT_ROOT="$(abs_path "$EXPERIMENT_ROOT")"
LOG_ROOT="$EXPERIMENT_ROOT/logs"
STATE_ROOT="$LOG_ROOT/rl_runtime"
TAIL_LINES="${RL_CLEANUP_TAIL_LINES:-40}"
PURGE_OUTPUTS="${RL_CLEANUP_PURGE_OUTPUTS:-1}"
KILL_BY_PORT="${RL_CLEANUP_KILL_BY_PORT:-1}"
FALLBACK_PORT_BLOCK="${RL_CLEANUP_FALLBACK_PORT_BLOCK:-0}"
PROTECTED_PORTS_RAW="${RL_CLEANUP_PROTECTED_PORTS:-8000,8001}"

port_is_protected() {
    local port="$1"
    [[ -n "$port" ]] || return 1
    local item
    IFS=',' read -r -a ports <<< "${PROTECTED_PORTS_RAW// /}"
    for item in "${ports[@]}"; do
        [[ -n "$item" ]] || continue
        if [[ "$item" == "$port" ]]; then
            return 0
        fi
    done
    return 1
}

process_exists() {
    local pid="$1"
    [[ -n "$pid" ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1
}

process_command() {
    local pid="$1"
    ps -p "$pid" -o command= 2>/dev/null || true
}

command_is_rl_related() {
    local cmd="$1"
    [[ -n "$cmd" ]] || return 1
    case "$cmd" in
        *"$REPO_ROOT"*|*"$EXPERIMENT_ROOT"*|*serve_rl_vllm.py*|*rl_stage_runner.py*|*collab_overcooked.main_rl*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

kill_process_group() {
    local pgid="$1"
    [[ -n "$pgid" ]] || return 0
    local matched=0
    while read -r _pid _pgid _cmd; do
        [[ "$_pgid" == "$pgid" ]] || continue
        if command_is_rl_related "$_cmd"; then
            matched=1
            break
        fi
    done < <(ps -eo pid=,pgid=,command= 2>/dev/null || true)
    if [[ "$matched" != "1" ]]; then
        echo "[cleanup-rl] skip pgid=$pgid because it no longer looks like this RL run"
        return 0
    fi
    echo "[cleanup-rl] killing process group pgid=$pgid"
    kill -TERM -- "-$pgid" >/dev/null 2>&1 || true
    sleep 1
    kill -KILL -- "-$pgid" >/dev/null 2>&1 || true
}

kill_pid_if_exists() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    if process_exists "$pid"; then
        local cmd
        cmd="$(process_command "$pid")"
        if ! command_is_rl_related "$cmd"; then
            echo "[cleanup-rl] skip pid=$pid because it no longer looks like this RL run"
            return 0
        fi
        echo "[cleanup-rl] killing pid=$pid"
        kill "$pid" >/dev/null 2>&1 || true
        sleep 1
        kill -9 "$pid" >/dev/null 2>&1 || true
    fi
}

kill_port_listener() {
    local port="$1"
    [[ -n "$port" ]] || return 0
    if port_is_protected "$port"; then
        echo "[cleanup-rl] skip protected port=$port"
        return 0
    fi
    if command -v lsof >/dev/null 2>&1; then
        local pids=()
        while IFS= read -r pid; do
            [[ -n "$pid" ]] && pids+=("$pid")
        done < <(lsof -tiTCP:"$port" -sTCP:LISTEN -Pn 2>/dev/null || true)
        if [[ "${#pids[@]}" -gt 0 ]]; then
            echo "[cleanup-rl] clearing listener on port=$port pid=${pids[*]}"
            kill "${pids[@]}" >/dev/null 2>&1 || true
            sleep 1
            kill -9 "${pids[@]}" >/dev/null 2>&1 || true
        fi
    fi
}

read_lines() {
    local path="$1"
    [[ -f "$path" ]] || return 0
    while IFS= read -r line; do
        [[ -n "$line" ]] && printf '%s\n' "$line"
    done < "$path"
}

remove_path_if_exists() {
    local path="$1"
    [[ -n "$path" ]] || return 0
    if [[ -e "$path" ]]; then
        echo "[cleanup-rl] removing $path"
        rm -rf "$path"
    fi
}

load_meta() {
    if [[ -f "$STATE_ROOT/meta.env" ]]; then
        # shellcheck disable=SC1090
        source "$STATE_ROOT/meta.env"
    fi
    NUM_PROCS_LOCAL="${num_procs:-${RL_NUM_PROCS:-0}}"
    START_PORT_LOCAL="${vllm_start_port:-${RL_VLLM_START_PORT:-0}}"
    INTERNAL_SPAN_LOCAL="${vllm_internal_port_span:-${RL_VLLM_INTERNAL_PORT_SPAN:-20}}"
}

cleanup_from_state() {
    local found=0
    if [[ -d "$STATE_ROOT" ]]; then
        if [[ -f "$STATE_ROOT/worker_pids.txt" ]]; then
            found=1
            while IFS= read -r pid; do
                kill_pid_if_exists "$pid"
            done < <(read_lines "$STATE_ROOT/worker_pids.txt")
        fi
        if [[ -f "$STATE_ROOT/vllm_pgids.txt" ]]; then
            found=1
            while IFS= read -r pgid; do
                kill_process_group "$pgid"
            done < <(read_lines "$STATE_ROOT/vllm_pgids.txt")
        fi
        if [[ -f "$STATE_ROOT/vllm_pids.txt" ]]; then
            found=1
            while IFS= read -r pid; do
                kill_pid_if_exists "$pid"
            done < <(read_lines "$STATE_ROOT/vllm_pids.txt")
        fi
        if [[ -f "$STATE_ROOT/vllm_ports.txt" ]]; then
            while IFS= read -r port; do
                kill_port_listener "$port"
            done < <(read_lines "$STATE_ROOT/vllm_ports.txt")
        fi
        if [[ -f "$STATE_ROOT/vllm_engine_ports.txt" ]]; then
            while IFS= read -r port; do
                kill_port_listener "$port"
            done < <(read_lines "$STATE_ROOT/vllm_engine_ports.txt")
        fi
        if [[ -f "$STATE_ROOT/vllm_internal_bases.txt" ]]; then
            while IFS= read -r base; do
                local offset
                for ((offset=0; offset<INTERNAL_SPAN_LOCAL; offset++)); do
                    kill_port_listener "$((base + offset))"
                done
            done < <(read_lines "$STATE_ROOT/vllm_internal_bases.txt")
        fi
    fi
    return "$found"
}

cleanup_from_port_block() {
    if [[ "$FALLBACK_PORT_BLOCK" != "1" ]]; then
        echo "[cleanup-rl] skip derived port-block cleanup (set RL_CLEANUP_FALLBACK_PORT_BLOCK=1 to enable)" >&2
        return 0
    fi
    if [[ "${NUM_PROCS_LOCAL:-0}" -le 0 || "${START_PORT_LOCAL:-0}" -le 0 ]]; then
        echo "[cleanup-rl] no runtime state and insufficient RL_NUM_PROCS/RL_VLLM_START_PORT to infer port block" >&2
        return 0
    fi
    if [[ "$KILL_BY_PORT" != "1" ]]; then
        echo "[cleanup-rl] derived port-block cleanup requires RL_CLEANUP_KILL_BY_PORT=1" >&2
        return 0
    fi
    echo "[cleanup-rl] no explicit state found; cleaning derived port block start=$START_PORT_LOCAL num_procs=$NUM_PROCS_LOCAL internal_span=$INTERNAL_SPAN_LOCAL"
    local gpu offset
    for ((gpu=0; gpu<NUM_PROCS_LOCAL; gpu++)); do
        kill_port_listener "$((START_PORT_LOCAL + gpu))"
        kill_port_listener "$((START_PORT_LOCAL + 100 + gpu))"
        local base=$((START_PORT_LOCAL + 200 + gpu * INTERNAL_SPAN_LOCAL))
        for ((offset=0; offset<INTERNAL_SPAN_LOCAL; offset++)); do
            kill_port_listener "$((base + offset))"
        done
    done
}

purge_runtime_outputs() {
    if [[ "$PURGE_OUTPUTS" != "1" ]]; then
        echo "[cleanup-rl] skip purging runtime outputs (RL_CLEANUP_PURGE_OUTPUTS=$PURGE_OUTPUTS)"
        return 0
    fi

    echo "[cleanup-rl] purging runtime outputs under $EXPERIMENT_ROOT"

    remove_path_if_exists "$LOG_ROOT/rl_workers"
    remove_path_if_exists "$LOG_ROOT/rl_vllm"
    remove_path_if_exists "$LOG_ROOT/rl_runtime"

    remove_path_if_exists "$EXPERIMENT_ROOT/rollouts_kl"
    remove_path_if_exists "$EXPERIMENT_ROOT/rollouts_eval_kl"

    find "$EXPERIMENT_ROOT/runs/rl" -type f \
        \( -name '*.csv' -o -name '*.jsonl' -o -name 'latest_model*.json' \) \
        -print -delete 2>/dev/null || true
    find "$EXPERIMENT_ROOT/results" -type f \
        \( -name '*.csv' -o -name '*.jsonl' -o -name '*.log' \) \
        -print -delete 2>/dev/null || true
}

show_recent_logs() {
    local kind="$1"
    local log_dir="$LOG_ROOT/$kind"
    [[ -d "$log_dir" ]] || return 0
    local latest
    latest="$(ls -1t "$log_dir" 2>/dev/null | head -n 3 || true)"
    [[ -n "$latest" ]] || return 0
    echo "[cleanup-rl] recent $kind logs:"
    local item
    while IFS= read -r item; do
        [[ -n "$item" ]] || continue
        echo "  $log_dir/$item"
        tail -n "$TAIL_LINES" "$log_dir/$item" 2>/dev/null || true
    done <<< "$latest"
}

load_meta

echo "[cleanup-rl] experiment_root=$EXPERIMENT_ROOT"
echo "[cleanup-rl] state_root=$STATE_ROOT"
echo "[cleanup-rl] kill_by_port=$KILL_BY_PORT"
echo "[cleanup-rl] fallback_port_block=$FALLBACK_PORT_BLOCK"
echo "[cleanup-rl] protected_ports=$PROTECTED_PORTS_RAW"

if ! cleanup_from_state; then
    cleanup_from_port_block
fi

show_recent_logs "rl_workers"
show_recent_logs "rl_vllm"
purge_runtime_outputs

echo "[cleanup-rl] done"
