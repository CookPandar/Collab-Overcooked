#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

def _extract_yaml_scalar_block(config_text: str, block_name: str, key: str) -> str:
    in_block = False
    block_indent = 0
    for raw_line in config_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if not in_block:
            if indent == 0 and stripped == f"{block_name}:":
                in_block = True
                block_indent = indent
            continue
        if indent <= block_indent:
            in_block = False
            if indent == 0 and stripped == f"{block_name}:":
                in_block = True
                block_indent = indent
            continue
        if stripped.startswith(f"{key}:"):
            value = stripped.split(":", 1)[1].strip()
            return value.strip("\"'")
    return ""


def _resolve_output_dir(config_path: Path, repo_root: Path) -> Path:
    explicit_output_dir = os.environ.get("RL_STAGE_OUTPUT_DIR", "").strip()
    if explicit_output_dir:
        return Path(explicit_output_dir).resolve()

    config_text = config_path.read_text(encoding="utf-8")
    output_dir = _extract_yaml_scalar_block(config_text, "trainer", "output_dir")
    if output_dir:
        path = Path(output_dir)
        return path if path.is_absolute() else (repo_root / path).resolve()

    order = _extract_yaml_scalar_block(config_text, "environment", "order") or "task"
    return (repo_root / "results" / f"mappo_{order}").resolve()


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _to_float(row: Dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw) if raw not in ("", None) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _aggregate_reward_curve(worker_dirs: List[Path], output_dir: Path) -> None:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for worker_dir in worker_dirs:
        for row in _load_csv_rows(worker_dir / "reward_curve.csv"):
            grouped[int(_to_float(row, "update_idx"))].append(row)
    if not grouped:
        return

    header = [
        "row_idx",
        "update_idx",
        "step",
        "num_transitions",
        "agent0_n",
        "agent1_n",
        "agent0_rl_sum",
        "agent0_rl_mean",
        "agent0_rl_nonzero_count",
        "agent0_rl_nonzero_ratio",
        "agent0_format_sum",
        "agent0_format_mean",
        "agent0_format_nonzero_count",
        "agent0_format_nonzero_ratio",
        "agent0_validator_sum",
        "agent0_validator_mean",
        "agent0_validator_nonzero_count",
        "agent0_validator_nonzero_ratio",
        "agent0_sequence_sum",
        "agent0_sequence_mean",
        "agent0_sequence_nonzero_count",
        "agent0_sequence_nonzero_ratio",
        "agent0_comm_sum",
        "agent0_comm_mean",
        "agent0_comm_nonzero_count",
        "agent0_comm_nonzero_ratio",
        "agent0_breakdown_total_sum",
        "agent0_breakdown_total_mean",
        "agent0_breakdown_total_nonzero_count",
        "agent0_breakdown_total_nonzero_ratio",
        "agent0_legacy_process_sum",
        "agent0_legacy_process_mean",
        "agent0_legacy_process_nonzero_count",
        "agent0_legacy_process_nonzero_ratio",
        "agent1_rl_sum",
        "agent1_rl_mean",
        "agent1_rl_nonzero_count",
        "agent1_rl_nonzero_ratio",
        "agent1_format_sum",
        "agent1_format_mean",
        "agent1_format_nonzero_count",
        "agent1_format_nonzero_ratio",
        "agent1_validator_sum",
        "agent1_validator_mean",
        "agent1_validator_nonzero_count",
        "agent1_validator_nonzero_ratio",
        "agent1_sequence_sum",
        "agent1_sequence_mean",
        "agent1_sequence_nonzero_count",
        "agent1_sequence_nonzero_ratio",
        "agent1_comm_sum",
        "agent1_comm_mean",
        "agent1_comm_nonzero_count",
        "agent1_comm_nonzero_ratio",
        "agent1_breakdown_total_sum",
        "agent1_breakdown_total_mean",
        "agent1_breakdown_total_nonzero_count",
        "agent1_breakdown_total_nonzero_ratio",
        "agent1_legacy_process_sum",
        "agent1_legacy_process_mean",
        "agent1_legacy_process_nonzero_count",
        "agent1_legacy_process_nonzero_ratio",
    ]
    output_path = output_dir / "reward_curve.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        cumulative_step = 0
        for row_idx, update_idx in enumerate(sorted(grouped.keys()), start=1):
            rows = grouped[update_idx]
            num_transitions = int(sum(_to_float(row, "num_transitions") for row in rows))
            agent0_n = int(sum(_to_float(row, "agent0_n") for row in rows))
            agent1_n = int(sum(_to_float(row, "agent1_n") for row in rows))
            cumulative_step += num_transitions
            agg = {
                "row_idx": row_idx,
                "update_idx": update_idx,
                "step": cumulative_step,
                "num_transitions": num_transitions,
                "agent0_n": agent0_n,
                "agent1_n": agent1_n,
            }
            sum_fields = [
                "agent0_rl_sum",
                "agent0_rl_nonzero_count",
                "agent0_format_sum",
                "agent0_format_nonzero_count",
                "agent0_validator_sum",
                "agent0_validator_nonzero_count",
                "agent0_sequence_sum",
                "agent0_sequence_nonzero_count",
                "agent0_comm_sum",
                "agent0_comm_nonzero_count",
                "agent0_breakdown_total_sum",
                "agent0_breakdown_total_nonzero_count",
                "agent0_legacy_process_sum",
                "agent0_legacy_process_nonzero_count",
                "agent1_rl_sum",
                "agent1_rl_nonzero_count",
                "agent1_format_sum",
                "agent1_format_nonzero_count",
                "agent1_validator_sum",
                "agent1_validator_nonzero_count",
                "agent1_sequence_sum",
                "agent1_sequence_nonzero_count",
                "agent1_comm_sum",
                "agent1_comm_nonzero_count",
                "agent1_breakdown_total_sum",
                "agent1_breakdown_total_nonzero_count",
                "agent1_legacy_process_sum",
                "agent1_legacy_process_nonzero_count",
            ]
            for key in sum_fields:
                agg[key] = sum(_to_float(row, key) for row in rows)
            agg["agent0_rl_mean"] = agg["agent0_rl_sum"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_rl_nonzero_ratio"] = agg["agent0_rl_nonzero_count"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_format_mean"] = agg["agent0_format_sum"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_format_nonzero_ratio"] = agg["agent0_format_nonzero_count"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_validator_mean"] = agg["agent0_validator_sum"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_validator_nonzero_ratio"] = agg["agent0_validator_nonzero_count"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_sequence_mean"] = agg["agent0_sequence_sum"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_sequence_nonzero_ratio"] = agg["agent0_sequence_nonzero_count"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_comm_mean"] = agg["agent0_comm_sum"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_comm_nonzero_ratio"] = agg["agent0_comm_nonzero_count"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_breakdown_total_mean"] = agg["agent0_breakdown_total_sum"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_breakdown_total_nonzero_ratio"] = (
                agg["agent0_breakdown_total_nonzero_count"] / agent0_n if agent0_n > 0 else 0.0
            )
            agg["agent0_legacy_process_mean"] = agg["agent0_legacy_process_sum"] / agent0_n if agent0_n > 0 else 0.0
            agg["agent0_legacy_process_nonzero_ratio"] = (
                agg["agent0_legacy_process_nonzero_count"] / agent0_n if agent0_n > 0 else 0.0
            )
            agg["agent1_rl_mean"] = agg["agent1_rl_sum"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_rl_nonzero_ratio"] = agg["agent1_rl_nonzero_count"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_format_mean"] = agg["agent1_format_sum"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_format_nonzero_ratio"] = agg["agent1_format_nonzero_count"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_validator_mean"] = agg["agent1_validator_sum"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_validator_nonzero_ratio"] = agg["agent1_validator_nonzero_count"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_sequence_mean"] = agg["agent1_sequence_sum"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_sequence_nonzero_ratio"] = agg["agent1_sequence_nonzero_count"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_comm_mean"] = agg["agent1_comm_sum"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_comm_nonzero_ratio"] = agg["agent1_comm_nonzero_count"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_breakdown_total_mean"] = agg["agent1_breakdown_total_sum"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_breakdown_total_nonzero_ratio"] = (
                agg["agent1_breakdown_total_nonzero_count"] / agent1_n if agent1_n > 0 else 0.0
            )
            agg["agent1_legacy_process_mean"] = agg["agent1_legacy_process_sum"] / agent1_n if agent1_n > 0 else 0.0
            agg["agent1_legacy_process_nonzero_ratio"] = (
                agg["agent1_legacy_process_nonzero_count"] / agent1_n if agent1_n > 0 else 0.0
            )
            writer.writerow(agg)


def _aggregate_performance_curve(worker_dirs: List[Path], output_dir: Path) -> None:
    grouped: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for worker_dir in worker_dirs:
        for row in _load_csv_rows(worker_dir / "performance_curve.csv"):
            grouped[int(_to_float(row, "update_idx"))].append(row)
    if not grouped:
        return

    header = [
        "row_idx",
        "update_idx",
        "model_update_idx",
        "env_steps",
        "episodes_completed",
        "success_episodes",
        "success_rate",
        "env_reward_sum",
        "avg_step_reward",
        "positive_reward_steps",
        "policy_calls",
        "avg_calls_per_step",
        "episode_return_sum",
        "avg_episode_return",
        "avg_episode_len",
        "agent0_custom_return_sum",
        "avg_agent0_custom_return",
        "agent1_custom_return_sum",
        "avg_agent1_custom_return",
        "team_custom_return_sum",
        "avg_team_custom_return",
    ]
    output_path = output_dir / "performance_curve.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row_idx, update_idx in enumerate(sorted(grouped.keys()), start=1):
            rows = grouped[update_idx]
            model_update_values = {
                row.get("model_update_idx", "")
                for row in rows
                if row.get("model_update_idx", "") not in ("", None)
            }
            model_update_idx = ""
            if len(model_update_values) == 1:
                model_update_idx = next(iter(model_update_values))
            env_steps = int(sum(_to_float(row, "env_steps") for row in rows))
            episodes = int(sum(_to_float(row, "episodes_completed") for row in rows))
            successes = int(sum(_to_float(row, "success_episodes") for row in rows))
            env_reward_sum = sum(_to_float(row, "env_reward_sum") for row in rows)
            pos_steps = int(sum(_to_float(row, "positive_reward_steps") for row in rows))
            policy_calls = int(sum(_to_float(row, "policy_calls") for row in rows))
            ep_return_sum = sum(_to_float(row, "episode_return_sum") for row in rows)
            ep_len_sum = sum(_to_float(row, "avg_episode_len") * _to_float(row, "episodes_completed") for row in rows)
            agent0_custom_sum = sum(_to_float(row, "agent0_custom_return_sum") for row in rows)
            agent1_custom_sum = sum(_to_float(row, "agent1_custom_return_sum") for row in rows)
            team_custom_sum = sum(_to_float(row, "team_custom_return_sum") for row in rows)
            agg = {
                "row_idx": row_idx,
                "update_idx": update_idx,
                "model_update_idx": model_update_idx,
                "env_steps": env_steps,
                "episodes_completed": episodes,
                "success_episodes": successes,
                "success_rate": (successes / episodes) if episodes > 0 else 0.0,
                "env_reward_sum": env_reward_sum,
                "avg_step_reward": (env_reward_sum / env_steps) if env_steps > 0 else 0.0,
                "positive_reward_steps": pos_steps,
                "policy_calls": policy_calls,
                "avg_calls_per_step": (policy_calls / env_steps) if env_steps > 0 else 0.0,
                "episode_return_sum": ep_return_sum,
                "avg_episode_return": (ep_return_sum / episodes) if episodes > 0 else 0.0,
                "avg_episode_len": (ep_len_sum / episodes) if episodes > 0 else 0.0,
                "agent0_custom_return_sum": agent0_custom_sum,
                "avg_agent0_custom_return": (agent0_custom_sum / episodes) if episodes > 0 else 0.0,
                "agent1_custom_return_sum": agent1_custom_sum,
                "avg_agent1_custom_return": (agent1_custom_sum / episodes) if episodes > 0 else 0.0,
                "team_custom_return_sum": team_custom_sum,
                "avg_team_custom_return": (team_custom_sum / episodes) if episodes > 0 else 0.0,
            }
            writer.writerow(agg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["collect", "eval"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    output_dir = _resolve_output_dir(config_path, repo_root)
    worker_root = output_dir / "__stage_workers__" / args.stage
    worker_dirs = sorted(
        path for path in worker_root.glob("worker_*") if path.is_dir()
    )
    if not worker_dirs:
        return 0
    _aggregate_reward_curve(worker_dirs, output_dir)
    _aggregate_performance_curve(worker_dirs, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
