#!/usr/bin/env python3
"""
Aggregate Chef / Assistant process rewards from batch evaluation JSON logs.

Usage:
    python scripts/analyze_process_rewards.py \
        --input assets/data/batch_results/qwen2.5-7B-instruct/json \
        --output-csv assets/data/batch_results/qwen2.5-7B-instruct/process_reward_summary.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-agent process rewards from JSON logs.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing per-order JSON folders (e.g., assets/data/batch_results/<model>/json).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional path to save aggregated statistics as CSV.",
    )
    return parser.parse_args()


class RewardStats:
    def __init__(self) -> None:
        self.episode_count = 0
        self.agent_rewards: Dict[int, List[float]] = defaultdict(list)
        self.agent_names: Dict[int, str] = {}
        self.team_totals: List[float] = []

    def add_episode(self, agent_totals: Dict[int, float], agent_names: Dict[int, str], team_total: float) -> None:
        self.episode_count += 1
        for idx, total in agent_totals.items():
            self.agent_rewards[idx].append(total)
        for idx, name in agent_names.items():
            self.agent_names.setdefault(idx, name or f"agent_{idx}")
        self.team_totals.append(team_total)

    def summarize(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for idx, values in self.agent_rewards.items():
            name = self.agent_names.get(idx, f"agent_{idx}")
            summary[name] = {
                "episodes": len(values),
                "avg_reward": statistics.fmean(values) if values else 0.0,
                "median_reward": statistics.median(values) if values else 0.0,
                "min_reward": min(values) if values else 0.0,
                "max_reward": max(values) if values else 0.0,
            }
        if self.team_totals:
            summary["team_total"] = {
                "episodes": len(self.team_totals),
                "avg_reward": statistics.fmean(self.team_totals),
                "median_reward": statistics.median(self.team_totals),
                "min_reward": min(self.team_totals),
                "max_reward": max(self.team_totals),
            }
        return summary


def extract_episode_totals(process_rewards: List[Dict]) -> Tuple[Dict[int, float], Dict[int, str], float]:
    agent_totals: Dict[int, float] = defaultdict(float)
    agent_names: Dict[int, str] = {}
    team_total = 0.0
    for step in process_rewards:
        team_total += float(step.get("team_total", 0.0))
        per_agents = step.get("per_agent", [])
        for idx, agent in enumerate(per_agents):
            agent_totals[idx] += float(agent.get("total", 0.0))
            if idx not in agent_names:
                name = None
                calls = agent.get("calls") or []
                if calls:
                    name = calls[0].get("agent")
                agent_names[idx] = name or ("Chef" if idx == 0 else "Assistant")
    return agent_totals, agent_names, team_total


def iter_log_files(root: Path):
    for order_dir in sorted(root.iterdir()):
        if not order_dir.is_dir():
            continue
        order_name = order_dir.name
        for json_file in order_dir.glob("*.json"):
            yield order_name, json_file


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input}")

    stats_by_order: Dict[str, RewardStats] = defaultdict(RewardStats)
    overall_stats = RewardStats()

    for order_name, json_path in iter_log_files(args.input):
        data = json.loads(json_path.read_text())
        process_rewards = data.get("process_rewards") or []
        if not process_rewards:
            continue
        agent_totals, agent_names, team_total = extract_episode_totals(process_rewards)
        stats_by_order[order_name].add_episode(agent_totals, agent_names, team_total)
        overall_stats.add_episode(agent_totals, agent_names, team_total)

    if not stats_by_order:
        print("[reward-summary] No process reward entries were found.")
        return

    def print_summary(order: str, stats: RewardStats):
        summary = stats.summarize()
        print(f"\n=== {order} ===")
        print(f"Episodes: {stats.episode_count}")
        for agent_name, metrics in summary.items():
            print(
                f"{agent_name:<12} "
                f"episodes={metrics['episodes']:<3} "
                f"avg={metrics['avg_reward']:.3f} "
                f"median={metrics['median_reward']:.3f} "
                f"min={metrics['min_reward']:.3f} "
                f"max={metrics['max_reward']:.3f}"
            )
        return summary

    csv_rows: List[Dict[str, str]] = []
    for order_name in sorted(stats_by_order.keys()):
        summary = print_summary(order_name, stats_by_order[order_name])
        for agent_name, metrics in summary.items():
            row = {
                "order": order_name,
                "agent": agent_name,
                **{k: str(v) for k, v in metrics.items()},
            }
            csv_rows.append(row)

    print_summary("OVERALL", overall_stats)

    if args.output_csv and csv_rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["order", "agent", "episodes", "avg_reward", "median_reward", "min_reward", "max_reward"]
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_fh:
            import csv

            writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"[reward-summary] Saved CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
