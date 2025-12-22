#!/usr/bin/env python3
"""
Summarize success rates and process rewards per SFT split using evaluation logs.

Example
-------
python scripts/summarize_split_metrics.py \
    --log-root assets/data/batch_results/qwen2.5-7B-instruct/json \
    --train-data data/sft/train_level12.jsonl \
    --dev-data data/sft/dev_level12.jsonl \
    --test-data data/sft/test_level12.jsonl \
    --per-order-csv assets/data/batch_results/qwen2.5-7B-instruct/per_order_metrics.csv \
    --split-csv assets/data/batch_results/qwen2.5-7B-instruct/split_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate evaluation metrics by SFT split.")
    parser.add_argument(
        "--log-root",
        type=Path,
        required=True,
        help="Directory containing per-order JSON folders (e.g., assets/data/batch_results/<model>/json).",
    )
    parser.add_argument("--train-data", type=Path, help="JSONL file containing SFT training split.")
    parser.add_argument("--dev-data", type=Path, help="JSONL file containing SFT validation split.")
    parser.add_argument("--test-data", type=Path, help="JSONL file containing SFT test split.")
    parser.add_argument(
        "--per-order-csv",
        type=Path,
        help="Optional CSV path for per-order statistics.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        help="Optional CSV path for aggregated split statistics.",
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Allow orders missing from the provided split JSONLs (default: error).",
    )
    return parser.parse_args()


def default_agent_name(index: int) -> str:
    if index == 0:
        return "Chef"
    if index == 1:
        return "Assistant"
    return f"agent_{index}"


class OrderStats:
    def __init__(self) -> None:
        self.episodes = 0
        self.successes = 0
        self.scores: List[float] = []
        self.team_totals: List[float] = []
        self.agent_totals: Dict[int, List[float]] = defaultdict(list)
        self.agent_sequence_rewards: Dict[int, List[float]] = defaultdict(list)
        self.agent_format_penalties: Dict[int, List[float]] = defaultdict(list)
        self.agent_names: Dict[int, str] = {}

    def add_episode(
        self,
        success: bool,
        score: float,
        agent_totals: Dict[int, float],
        agent_sequences: Dict[int, float],
        agent_names: Dict[int, str],
        team_total: Optional[float],
        agent_format_penalties: Dict[int, float],
    ) -> None:
        self.episodes += 1
        if success:
            self.successes += 1
        self.scores.append(score)
        if team_total is not None:
            self.team_totals.append(team_total)
        for idx, total in agent_totals.items():
            self.agent_totals[idx].append(total)
        for idx, seq_val in agent_sequences.items():
            self.agent_sequence_rewards[idx].append(seq_val)
        for idx, penalty in agent_format_penalties.items():
            self.agent_format_penalties[idx].append(penalty)
        for idx, name in agent_names.items():
            if idx not in self.agent_names:
                self.agent_names[idx] = name or default_agent_name(idx)

    def merge(self, other: "OrderStats") -> None:
        self.episodes += other.episodes
        self.successes += other.successes
        self.scores.extend(other.scores)
        self.team_totals.extend(other.team_totals)
        for idx, values in other.agent_totals.items():
            self.agent_totals[idx].extend(values)
        for idx, values in other.agent_sequence_rewards.items():
            self.agent_sequence_rewards[idx].extend(values)
        for idx, values in other.agent_format_penalties.items():
            self.agent_format_penalties[idx].extend(values)
        for idx, name in other.agent_names.items():
            self.agent_names.setdefault(idx, name)

    def _summarize_list(self, values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"avg": None, "median": None, "min": None, "max": None}
        return {
            "avg": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    def summary(self) -> Dict[str, object]:
        agent_reward_summary: Dict[str, Dict[str, Optional[float]]] = {}
        for idx, values in self.agent_totals.items():
            name = self.agent_names.get(idx, default_agent_name(idx))
            agent_reward_summary[name] = self._summarize_list(values)

        agent_sequence_summary: Dict[str, Dict[str, Optional[float]]] = {}
        for idx, values in self.agent_sequence_rewards.items():
            name = self.agent_names.get(idx, default_agent_name(idx))
            agent_sequence_summary[name] = self._summarize_list(values)

        agent_penalty_summary: Dict[str, Dict[str, Optional[float]]] = {}
        for idx, values in self.agent_format_penalties.items():
            name = self.agent_names.get(idx, default_agent_name(idx))
            agent_penalty_summary[name] = self._summarize_list(values)

        summary = {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate(),
            "score": self._summarize_list(self.scores),
            "team_reward": self._summarize_list(self.team_totals),
            "agent_rewards": agent_reward_summary,
            "agent_sequence_rewards": agent_sequence_summary,
            "agent_format_penalties": agent_penalty_summary,
        }
        return summary

    def to_row(
        self,
        name: str,
        split: str,
        agent_columns: Iterable[Tuple[int, str]],
    ) -> Dict[str, Optional[float]]:
        summary = self.summary()
        row: Dict[str, Optional[float]] = {
            "name": name,
            "split": split,
            "episodes": summary["episodes"],
            "successes": summary["successes"],
            "success_rate": summary["success_rate"],
            "avg_score": summary["score"]["avg"],
            "median_score": summary["score"]["median"],
            "min_score": summary["score"]["min"],
            "max_score": summary["score"]["max"],
            "avg_team_reward": summary["team_reward"]["avg"],
            "total_team_reward": sum(self.team_totals) if self.team_totals else 0.0,
        }
        reward_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_rewards"]  # type: ignore[assignment]
        sequence_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_sequence_rewards"]  # type: ignore[assignment]
        penalty_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_format_penalties"]  # type: ignore[assignment]
        for idx, col_prefix in agent_columns:
            agent_name = self.agent_names.get(idx, default_agent_name(idx))
            reward_metrics = reward_summary.get(agent_name)
            seq_metrics = sequence_summary.get(agent_name)
            penalty_metrics = penalty_summary.get(agent_name)
            values = self.agent_totals.get(idx, [])
            row[f"{col_prefix}_avg_reward"] = reward_metrics["avg"] if reward_metrics else None
            row[f"{col_prefix}_total_reward"] = sum(values) if values else 0.0
            row[f"{col_prefix}_avg_sequence_reward"] = seq_metrics["avg"] if seq_metrics else None
            row[f"{col_prefix}_avg_format_reward"] = penalty_metrics["avg"] if penalty_metrics else None
        return row


def load_order_set(path: Optional[Path]) -> Tuple[set[str], Optional[str]]:
    if path is None:
        return set(), None
    orders: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = record.get("meta") or {}
            order = meta.get("order") or record.get("order")
            if order:
                orders.add(order)
    return orders, path.name


def build_order_split_map(args: argparse.Namespace) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for split_name, path in (
        ("train", args.train_data),
        ("dev", args.dev_data),
        ("test", args.test_data),
    ):
        orders, source = load_order_set(path)
        if not orders:
            continue
        for order in orders:
            if order in mapping and mapping[order] != split_name:
                print(
                    f"[split-map] Warning: order '{order}' already assigned to '{mapping[order]}', "
                    f"skipping duplicate in '{split_name}'."
                )
                continue
            mapping[order] = split_name
        print(f"[split-map] Loaded {len(orders)} unique orders for split '{split_name}' from {source}.")
    return mapping


def iter_log_files(root: Path) -> Iterable[Tuple[str, Path]]:
    for order_dir in sorted(root.iterdir()):
        if not order_dir.is_dir():
            continue
        for json_file in sorted(order_dir.glob("*.json")):
            yield order_dir.name, json_file


def extract_episode_totals(
    process_rewards: List[Dict],
) -> Tuple[Dict[int, float], Dict[int, str], float, Dict[int, float], Dict[int, float]]:
    agent_totals: Dict[int, float] = defaultdict(float)
    agent_sequences: Dict[int, float] = defaultdict(float)
    agent_format_penalties: Dict[int, float] = defaultdict(float)
    agent_names: Dict[int, str] = {}
    team_total = 0.0
    for step in process_rewards:
        per_agents = step.get("per_agent") or []
        for idx, agent in enumerate(per_agents):
            agent_totals.setdefault(idx, 0.0)
            agent_sequences.setdefault(idx, 0.0)
            agent_format_penalties.setdefault(idx, 0.0)
            seq_reward = float(agent.get("sequence_reward", 0.0))
            fmt_penalty = 0.0
            for penalty in agent.get("penalties") or []:
                if (penalty.get("type") or "").lower() == "format":
                    value = float(penalty.get("value", 0.0))
                    fmt_penalty += value
                    agent_format_penalties[idx] += value
            agent_sequences[idx] += seq_reward
            combined = seq_reward + fmt_penalty
            agent_totals[idx] += combined
            team_total += combined
            if idx not in agent_names:
                calls = agent.get("calls") or []
                name = calls[0].get("agent") if calls else None
                agent_names[idx] = name or default_agent_name(idx)
    return agent_totals, agent_names, team_total, agent_format_penalties, agent_sequences


def aggregate_order_metrics(log_root: Path) -> Dict[str, OrderStats]:
    if not log_root.exists():
        raise FileNotFoundError(f"Log root not found: {log_root}")
    order_metrics: Dict[str, OrderStats] = defaultdict(OrderStats)
    for order_name, json_path in iter_log_files(log_root):
        data = json.loads(json_path.read_text())
        success = bool(data.get("total_order_finished"))
        score = float(data.get("total_score", 0.0))
        process_rewards = data.get("process_rewards") or []
        agent_totals: Dict[int, float] = {}
        agent_names: Dict[int, str] = {}
        agent_format_penalties: Dict[int, float] = {}
        agent_sequences: Dict[int, float] = {}
        team_total: Optional[float] = None
        if process_rewards:
            (
                agent_totals,
                agent_names,
                team_total,
                agent_format_penalties,
                agent_sequences,
            ) = extract_episode_totals(process_rewards)
        order_metrics[order_name].add_episode(
            success,
            score,
            agent_totals,
            agent_sequences,
            agent_names,
            team_total,
            agent_format_penalties,
        )
    if not order_metrics:
        raise RuntimeError(f"No JSON logs were discovered under {log_root}")
    return order_metrics


def build_split_stats(order_metrics: Dict[str, OrderStats], mapping: Dict[str, str]) -> Dict[str, OrderStats]:
    split_stats: Dict[str, OrderStats] = defaultdict(OrderStats)
    for order_name, stats in order_metrics.items():
        split = mapping.get(order_name, "unknown")
        split_stats[split].merge(stats)
    overall = OrderStats()
    for stats in order_metrics.values():
        overall.merge(stats)
    split_stats["overall"] = overall
    return split_stats


def make_agent_columns(order_metrics: Dict[str, OrderStats]) -> List[Tuple[int, str]]:
    indices = sorted({idx for stats in order_metrics.values() for idx in stats.agent_totals.keys()})
    columns: List[Tuple[int, str]] = []
    for idx in indices:
        label = default_agent_name(idx).lower().replace(" ", "_")
        columns.append((idx, label))
    if not columns:
        columns.extend([(0, "chef"), (1, "assistant")])
    return columns


def write_csv(rows: List[Dict[str, Optional[float]]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[metrics] Saved CSV to {path}")


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_split_summary(split_stats: Dict[str, OrderStats]) -> None:
    for split_name in sorted(split_stats.keys()):
        stats = split_stats[split_name]
        summary = stats.summary()
        print(f"\n=== {split_name.upper()} ===")
        print(
            f"Episodes={summary['episodes']}  Successes={summary['successes']}  "
            f"SuccessRate={summary['success_rate']:.3f}"
        )
        score = summary["score"]
        team = summary["team_reward"]
        print(
            f"Score avg={_fmt(score['avg'])} / median={_fmt(score['median'])} "
            f"min={_fmt(score['min'])} max={_fmt(score['max'])}"
        )
        print(
            f"Team reward avg={_fmt(team['avg'])} median={_fmt(team['median'])} "
            f"min={_fmt(team['min'])} max={_fmt(team['max'])}"
        )
        agent_rewards: Dict[str, Dict[str, Optional[float]]] = summary["agent_rewards"]  # type: ignore[assignment]
        agent_sequences: Dict[str, Dict[str, Optional[float]]] = summary["agent_sequence_rewards"]  # type: ignore[assignment]
        agent_penalties: Dict[str, Dict[str, Optional[float]]] = summary["agent_format_penalties"]  # type: ignore[assignment]
        agent_names = sorted(set(agent_rewards.keys()) | set(agent_sequences.keys()) | set(agent_penalties.keys()))
        for agent_name in agent_names:
            reward_metrics = agent_rewards.get(agent_name)
            seq_metrics = agent_sequences.get(agent_name)
            penalty_metrics = agent_penalties.get(agent_name)
            agent_idx = next((i for i, n in stats.agent_names.items() if n == agent_name), None)
            total_val = sum(stats.agent_totals.get(agent_idx, [])) if agent_idx is not None else None
            print(
                f"- {agent_name:<10} avg_seq={_fmt(seq_metrics['avg'] if seq_metrics else None)} "
                f"avg_format={_fmt(penalty_metrics['avg'] if penalty_metrics else None)} "
                f"avg_reward={_fmt(reward_metrics['avg'] if reward_metrics else None)} "
                f"total_reward={_fmt(total_val)}"
            )


def main() -> None:
    args = parse_args()
    order_metrics = aggregate_order_metrics(args.log_root)
    mapping = build_order_split_map(args)
    missing_orders = sorted(o for o in order_metrics.keys() if o not in mapping)
    if missing_orders and not args.allow_unknown:
        preview = ", ".join(missing_orders[:10])
        raise ValueError(
            "未能在提供的 SFT 数据集中找到以下任务的 split 标签，请补充后重试："
            f"{preview}{' ...' if len(missing_orders) > 10 else ''}\n"
            "如需临时允许这些任务归入 unknown，请添加 --allow-unknown。"
        )
    if missing_orders and args.allow_unknown:
        print(
            f"[split-map] Warning: {len(missing_orders)} orders lack split mapping; "
            "they will be reported under 'unknown'."
        )
    split_stats = build_split_stats(order_metrics, mapping)
    print_split_summary(split_stats)

    agent_columns = make_agent_columns(order_metrics)
    per_order_rows: List[Dict[str, Optional[float]]] = []
    for order_name in sorted(order_metrics.keys()):
        split = mapping.get(order_name, "unknown")
        row = order_metrics[order_name].to_row(order_name, split, agent_columns)
        per_order_rows.append(row)
    if args.per_order_csv:
        write_csv(per_order_rows, args.per_order_csv)

    split_rows: List[Dict[str, Optional[float]]] = []
    for split_name in sorted(split_stats.keys()):
        row = split_stats[split_name].to_row(split_name, split_name, agent_columns)
        split_rows.append(row)
    if args.split_csv:
        write_csv(split_rows, args.split_csv)


if __name__ == "__main__":
    main()
