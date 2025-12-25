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
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize per-agent process rewards from JSON logs.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing per-order JSON folders (e.g., assets/data/batch_results/<model>/json).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate per-order plots for process rewards.",
    )
    return parser.parse_args()


def default_agent_name(index: int) -> str:
    if index == 0:
        return "Chef"
    if index == 1:
        return "Assistant"
    return f"agent_{index}"


def infer_default_paths(input_dir: Path) -> Tuple[Path, Path]:
    base_dir = input_dir.resolve().parent
    csv_path = base_dir / "process_reward_summary.csv"
    plot_dir = base_dir / "process_reward_plots"
    return csv_path, plot_dir


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def resolve_agent_labels(stats_by_order: Dict[str, RewardStats]) -> List[str]:
    indexed_names: Dict[int, str] = {}
    for stats in stats_by_order.values():
        for idx, name in stats.agent_names.items():
            indexed_names[idx] = name
    if not indexed_names:
        indexed_names = {0: default_agent_name(0), 1: default_agent_name(1)}
    return [indexed_names[idx] for idx in sorted(indexed_names.keys())]


def plot_agent_metric_curves(
    order_summaries: Dict[str, Dict],
    agent_labels: List[str],
    metric_key: str,
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 未安装，跳过绘图。")
        return

    order_names = sorted(order_summaries.keys())
    if not order_names:
        print("[plot] 未找到可用的菜品条目，跳过绘图。")
        return

    x_idx = list(range(len(order_names)))
    plt.figure(figsize=(max(10, len(order_names) * 0.4), 5))
    has_series = False
    for agent_name in agent_labels:
        y_values: List[float] = []
        for order in order_names:
            metrics = order_summaries[order].get(metric_key, {}).get(agent_name)
            value = metrics.get(value_key) if metrics else None
            y_values.append(float(value) if value is not None else float("nan"))
        if all(math.isnan(v) for v in y_values):
            continue
        has_series = True
        plt.plot(x_idx, y_values, marker="o", label=agent_name)

    if not has_series:
        plt.close()
        print(f"[plot] {output_path.name} 缺少有效数据，跳过绘制。")
        return

    plt.xticks(x_idx, order_names, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title, pad=40)
    plt.grid(True, alpha=0.3)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=min(len(agent_labels), 4),
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[plot] Saved {output_path}")


def plot_process_rewards(
    order_summaries: Dict[str, Dict],
    stats_by_order: Dict[str, RewardStats],
    model_label: str,
    output_dir: Path,
) -> None:
    agent_labels = resolve_agent_labels(stats_by_order)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_agent_metric_curves(
        order_summaries,
        agent_labels,
        "agent_sequence_rewards",
        "avg_sequence_reward",
        f"{model_label}: Avg Process Reward",
        "Avg Process Reward",
        output_dir / "process_reward.png",
    )
    plot_agent_metric_curves(
        order_summaries,
        agent_labels,
        "agent_format_rewards",
        "avg_format_reward",
        f"{model_label}: Avg Format Reward",
        "Avg Format Reward",
        output_dir / "format_reward.png",
    )


class RewardStats:
    def __init__(self) -> None:
        self.episode_count = 0
        self.agent_rewards: Dict[int, List[float]] = defaultdict(list)
        self.agent_sequence_rewards: Dict[int, List[float]] = defaultdict(list)
        self.agent_format_rewards: Dict[int, List[float]] = defaultdict(list)
        self.agent_names: Dict[int, str] = {}
        self.team_totals: List[float] = []

    def add_episode(
        self,
        agent_totals: Dict[int, float],
        agent_sequences: Dict[int, float],
        agent_format_rewards: Dict[int, float],
        agent_names: Dict[int, str],
        team_total: float,
    ) -> None:
        self.episode_count += 1
        for idx, total in agent_totals.items():
            self.agent_rewards[idx].append(total)
        for idx, seq in agent_sequences.items():
            self.agent_sequence_rewards[idx].append(seq)
        for idx, fmt in agent_format_rewards.items():
            self.agent_format_rewards[idx].append(fmt)
        for idx, name in agent_names.items():
            self.agent_names.setdefault(idx, name or default_agent_name(idx))
        self.team_totals.append(team_total)

    def summarize(self) -> Dict[str, Dict[str, float]]:
        agent_summary: Dict[str, Dict[str, float]] = {}
        for idx, values in self.agent_rewards.items():
            name = self.agent_names.get(idx, default_agent_name(idx))
            agent_summary[name] = {
                "episodes": len(values),
                "avg_reward": statistics.fmean(values) if values else 0.0,
            }
        seq_summary: Dict[str, Dict[str, float]] = {}
        for idx, values in self.agent_sequence_rewards.items():
            name = self.agent_names.get(idx, default_agent_name(idx))
            seq_summary[name] = {
                "episodes": len(values),
                "avg_sequence_reward": statistics.fmean(values) if values else 0.0,
            }
        format_summary: Dict[str, Dict[str, float]] = {}
        for idx, values in self.agent_format_rewards.items():
            name = self.agent_names.get(idx, default_agent_name(idx))
            format_summary[name] = {
                "episodes": len(values),
                "avg_format_reward": statistics.fmean(values) if values else 0.0,
            }
        team_summary: Optional[Dict[str, float]] = None
        if self.team_totals:
            team_summary = {
                "episodes": len(self.team_totals),
                "avg_reward": statistics.fmean(self.team_totals),
            }
        return {
            "agent_rewards": agent_summary,
            "agent_sequence_rewards": seq_summary,
            "agent_format_rewards": format_summary,
            "team_reward": team_summary,
            "episodes": self.episode_count,
        }


def extract_episode_totals(
    process_rewards: List[Dict],
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, str], float]:
    agent_totals: Dict[int, float] = defaultdict(float)
    agent_sequences: Dict[int, float] = defaultdict(float)
    agent_format_rewards: Dict[int, float] = defaultdict(float)
    agent_names: Dict[int, str] = {}
    team_total = 0.0
    for step in process_rewards:
        team_total += float(step.get("team_total", 0.0))
        per_agents = step.get("per_agent") or []
        for idx, agent in enumerate(per_agents):
            agent_totals[idx] += float(agent.get("total", 0.0))
            agent_sequences[idx] += float(agent.get("sequence_reward", 0.0))
            for penalty in agent.get("penalties") or []:
                penalty_type = (penalty.get("type") or "").lower()
                if penalty_type == "format":
                    agent_format_rewards[idx] += float(penalty.get("value", 0.0))
            if idx not in agent_names:
                name = None
                calls = agent.get("calls") or []
                if calls:
                    name = calls[0].get("agent")
                agent_names[idx] = name or default_agent_name(idx)
    return agent_totals, agent_sequences, agent_format_rewards, agent_names, team_total


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
        (
            agent_totals,
            agent_sequences,
            agent_format_rewards,
            agent_names,
            team_total,
        ) = extract_episode_totals(process_rewards)
        stats_by_order[order_name].add_episode(
            agent_totals,
            agent_sequences,
            agent_format_rewards,
            agent_names,
            team_total,
        )
        overall_stats.add_episode(
            agent_totals,
            agent_sequences,
            agent_format_rewards,
            agent_names,
            team_total,
        )

    if not stats_by_order:
        print("[reward-summary] No process reward entries were found.")
        return

    def print_summary(order: str, stats: RewardStats):
        summary = stats.summarize()
        agent_rewards = summary["agent_rewards"]
        agent_sequences = summary["agent_sequence_rewards"]
        agent_formats = summary["agent_format_rewards"]
        print(f"\n=== {order} ===")
        print(f"Episodes: {stats.episode_count}")
        agent_names: List[str] = sorted(
            set(agent_rewards.keys()) | set(agent_sequences.keys()) | set(agent_formats.keys())
        )
        for agent_name in agent_names:
            reward_metrics = agent_rewards.get(agent_name)
            seq_metrics = agent_sequences.get(agent_name, {})
            fmt_metrics = agent_formats.get(agent_name, {})
            if not reward_metrics:
                continue
            print(
                f"{agent_name:<12} "
                f"episodes={reward_metrics['episodes']:<3} "
                f"avg={_fmt(reward_metrics.get('avg_reward'))} "
                f"seq_avg={_fmt(seq_metrics.get('avg_sequence_reward'))} "
                f"fmt_avg={_fmt(fmt_metrics.get('avg_format_reward'))}"
            )
        team_summary = summary.get("team_reward")
        if team_summary:
            print(
                f"{'team_total':<12} "
                f"episodes={team_summary['episodes']:<3} "
                f"avg={_fmt(team_summary.get('avg_reward'))}"
            )
        return summary

    csv_rows: List[Dict[str, Optional[float]]] = []
    order_summaries: Dict[str, Dict] = {}
    for order_name in sorted(stats_by_order.keys()):
        summary = print_summary(order_name, stats_by_order[order_name])
        order_summaries[order_name] = summary
        agent_rewards = summary["agent_rewards"]
        agent_sequences = summary["agent_sequence_rewards"]
        agent_formats = summary["agent_format_rewards"]
        for agent_name, metrics in agent_rewards.items():
            seq_metrics = agent_sequences.get(agent_name, {})
            fmt_metrics = agent_formats.get(agent_name, {})
            row: Dict[str, Optional[float]] = {
                "order": order_name,
                "agent": agent_name,
                "episodes": metrics.get("episodes"),
                "avg_reward": metrics.get("avg_reward"),
                "avg_sequence_reward": seq_metrics.get("avg_sequence_reward"),
                "avg_format_reward": fmt_metrics.get("avg_format_reward"),
            }
            csv_rows.append(row)

    print_summary("OVERALL", overall_stats)

    output_csv, default_plot_dir = infer_default_paths(args.input)
    if csv_rows:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "order",
            "agent",
            "episodes",
            "avg_reward",
            "avg_sequence_reward",
            "avg_format_reward",
        ]
        with output_csv.open("w", newline="", encoding="utf-8") as csv_fh:
            import csv

            writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
        print(f"[reward-summary] Saved CSV to {output_csv}")

    if args.plot:
        plot_dir = default_plot_dir
        model_label = args.input.resolve().parent.name
        plot_process_rewards(order_summaries, stats_by_order, model_label, plot_dir)


if __name__ == "__main__":
    main()
