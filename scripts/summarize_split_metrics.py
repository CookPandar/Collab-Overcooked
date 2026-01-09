#!/usr/bin/env python3
"""
Summarize success rates and process rewards per SFT split using evaluation logs.

Example
-------
python scripts/summarize_split_metrics.py \
    --log-root assets/data/batch_results/qwen2.5-7B-instruct/json \
    --train-data data/sft/train_level12.jsonl \
    --dev-data data/sft/dev_level12.jsonl \
    --test-data data/sft/test_level12.jsonl
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
        "--plot-per-order-csv",
        nargs="+",
        help="Plot per-order metrics from CSV files (format label=path or plain path).",
    )
    return parser.parse_args()


def infer_output_paths(log_root: Path) -> Tuple[Path, Path, Path]:
    base_dir = log_root.resolve().parent
    per_order_csv = base_dir / "per_order_metrics.csv"
    split_csv = base_dir / "split_metrics.csv"
    plot_dir = base_dir / "plots"
    return per_order_csv, split_csv, plot_dir


def default_agent_name(index: int) -> str:
    if index == 0:
        return "Chef"
    if index == 1:
        return "Assistant"
    return f"agent_{index}"


REFERENCE_DIR = (Path(__file__).resolve().parents[1] / "collab_overcooked" / "prompts" / "reference").resolve()


def load_reference_actions(order: str) -> Dict[int, List[List[str]]]:
    """
    Load reference demonstrations for a given order.
    Returns {agent_index: [reference_action_list, ...]}.
    """
    pattern = f"*_{order}_ref.txt"
    candidates = sorted(REFERENCE_DIR.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No reference file found for order '{order}' under {REFERENCE_DIR}")
    refs_by_agent: Dict[int, List[List[str]]] = {0: [], 1: []}
    for path in candidates:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        for _, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            agent0 = entry.get("agent_0")
            agent1 = entry.get("agent_1")
            if isinstance(agent0, list) and agent0:
                refs_by_agent[0].append([str(x).strip() for x in agent0 if str(x).strip()])
            if isinstance(agent1, list) and agent1:
                refs_by_agent[1].append([str(x).strip() for x in agent1 if str(x).strip()])
    return refs_by_agent


def normalize_action(action: str) -> str:
    return (action or "").strip().replace(" ", "")


def extract_executed_actions(data: Dict, agent_index: int) -> List[str]:
    total = data.get("total_action_list") or []
    if not isinstance(total, list) or agent_index >= len(total):
        return []
    actions = total[agent_index]
    if not isinstance(actions, list):
        return []
    items = []
    for item in actions:
        if isinstance(item, dict):
            items.append((int(item.get("timestamp", 0)), normalize_action(str(item.get("action") or ""))))
        elif isinstance(item, str):
            items.append((0, normalize_action(item)))
    items = [(t, a) for (t, a) in items if a]
    items.sort(key=lambda x: x[0])
    return [a for _, a in items]


def reference_progress(agent_actions: List[str], ref_actions: List[str]) -> int:
    """
    Return how many prefix steps of ref_actions are completed as an ordered subsequence
    within agent_actions (allows detours, but preserves order).
    """
    if not ref_actions or not agent_actions:
        return 0
    idx = 0
    for act in agent_actions:
        if idx >= len(ref_actions):
            break
        if act == ref_actions[idx]:
            idx += 1
    return idx


def best_reference_progress(order: str, agent_index: int, agent_actions: List[str]) -> Tuple[int, int, float]:
    """
    Return (matched_steps, ref_len, ratio) for the best reference demonstration.
    """
    refs = load_reference_actions(order).get(agent_index, [])
    best_steps = 0
    best_len = 0
    best_ratio = 0.0
    for ref in refs:
        ref_norm = [normalize_action(x) for x in ref]
        ref_norm = [x for x in ref_norm if x]
        if not ref_norm:
            continue
        steps = reference_progress(agent_actions, ref_norm)
        ratio = steps / len(ref_norm)
        # Prefer the reference that reaches the furthest step; use ratio as tie-breaker.
        if steps > best_steps or (steps == best_steps and ratio > best_ratio):
            best_ratio = ratio
            best_steps = steps
            best_len = len(ref_norm)
    return best_steps, best_len, best_ratio


class OrderStats:
    def __init__(self) -> None:
        self.episodes = 0
        self.successes = 0
        self.scores: List[float] = []
        self.team_totals: List[float] = []
        self.agent_totals: Dict[int, List[float]] = defaultdict(list)
        self.agent_sequence_rewards: Dict[int, List[float]] = defaultdict(list)
        self.agent_format_penalties: Dict[int, List[float]] = defaultdict(list)
        self.agent_ref_progress: Dict[int, List[float]] = defaultdict(list)
        self.agent_ref_steps: Dict[int, List[float]] = defaultdict(list)
        self.agent_ref_lens: Dict[int, List[float]] = defaultdict(list)
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
        agent_ref_progress: Dict[int, Tuple[int, int, float]],
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
        for idx, triple in agent_ref_progress.items():
            steps, ref_len, ratio = triple
            self.agent_ref_progress[idx].append(float(ratio))
            self.agent_ref_steps[idx].append(float(steps))
            self.agent_ref_lens[idx].append(float(ref_len))
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
        for idx, values in other.agent_ref_progress.items():
            self.agent_ref_progress[idx].extend(values)
        for idx, values in other.agent_ref_steps.items():
            self.agent_ref_steps[idx].extend(values)
        for idx, values in other.agent_ref_lens.items():
            self.agent_ref_lens[idx].extend(values)
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

        agent_progress_summary: Dict[str, Dict[str, Optional[float]]] = {}
        for idx, values in self.agent_ref_progress.items():
            name = self.agent_names.get(idx, default_agent_name(idx))
            agent_progress_summary[name] = self._summarize_list(values)

        agent_progress_steps_summary: Dict[str, Dict[str, Optional[float]]] = {}
        for idx, values in self.agent_ref_steps.items():
            name = self.agent_names.get(idx, default_agent_name(idx))
            agent_progress_steps_summary[name] = self._summarize_list(values)

        summary = {
            "episodes": self.episodes,
            "successes": self.successes,
            "success_rate": self.success_rate(),
            "score": self._summarize_list(self.scores),
            "team_reward": self._summarize_list(self.team_totals),
            "agent_rewards": agent_reward_summary,
            "agent_sequence_rewards": agent_sequence_summary,
            "agent_format_penalties": agent_penalty_summary,
            "agent_ref_progress": agent_progress_summary,
            "agent_ref_steps": agent_progress_steps_summary,
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
        progress_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_ref_progress"]  # type: ignore[assignment]
        progress_steps_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_ref_steps"]  # type: ignore[assignment]
        for idx, col_prefix in agent_columns:
            agent_name = self.agent_names.get(idx, default_agent_name(idx))
            reward_metrics = reward_summary.get(agent_name)
            seq_metrics = sequence_summary.get(agent_name)
            penalty_metrics = penalty_summary.get(agent_name)
            prog_metrics = progress_summary.get(agent_name)
            prog_steps_metrics = progress_steps_summary.get(agent_name)
            values = self.agent_totals.get(idx, [])
            row[f"{col_prefix}_avg_reward"] = reward_metrics["avg"] if reward_metrics else None
            row[f"{col_prefix}_total_reward"] = sum(values) if values else 0.0
            row[f"{col_prefix}_avg_sequence_reward"] = seq_metrics["avg"] if seq_metrics else None
            row[f"{col_prefix}_avg_format_reward"] = penalty_metrics["avg"] if penalty_metrics else None
            row[f"{col_prefix}_avg_ref_progress"] = prog_metrics["avg"] if prog_metrics else None
            row[f"{col_prefix}_avg_ref_steps"] = prog_steps_metrics["avg"] if prog_steps_metrics else None
            row[f"{col_prefix}_max_ref_progress"] = prog_metrics["max"] if prog_metrics else None
            row[f"{col_prefix}_max_ref_steps"] = prog_steps_metrics["max"] if prog_steps_metrics else None
        return row

    def agent_metric_row(
        self,
        name: str,
        split: str,
        agent_columns: Iterable[Tuple[int, str]],
    ) -> Dict[str, Optional[float]]:
        summary = self.summary()
        reward_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_rewards"]  # type: ignore[assignment]
        sequence_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_sequence_rewards"]  # type: ignore[assignment]
        penalty_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_format_penalties"]  # type: ignore[assignment]
        progress_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_ref_progress"]  # type: ignore[assignment]
        progress_steps_summary: Dict[str, Dict[str, Optional[float]]] = summary["agent_ref_steps"]  # type: ignore[assignment]
        row: Dict[str, Optional[float]] = {
            "name": name,
            "split": split,
            "success_rate": summary["success_rate"],
            "episodes": summary["episodes"],
        }
        for idx, col_prefix in agent_columns:
            agent_name = self.agent_names.get(idx, default_agent_name(idx))
            reward_metrics = reward_summary.get(agent_name)
            seq_metrics = sequence_summary.get(agent_name)
            penalty_metrics = penalty_summary.get(agent_name)
            prog_metrics = progress_summary.get(agent_name)
            prog_steps_metrics = progress_steps_summary.get(agent_name)
            values = self.agent_totals.get(idx, [])
            row[f"{col_prefix}_avg_reward"] = reward_metrics["avg"] if reward_metrics else None
            row[f"{col_prefix}_total_reward"] = sum(values) if values else 0.0
            row[f"{col_prefix}_avg_sequence_reward"] = seq_metrics["avg"] if seq_metrics else None
            row[f"{col_prefix}_avg_format_reward"] = penalty_metrics["avg"] if penalty_metrics else None
            row[f"{col_prefix}_avg_ref_progress"] = prog_metrics["avg"] if prog_metrics else None
            row[f"{col_prefix}_avg_ref_steps"] = prog_steps_metrics["avg"] if prog_steps_metrics else None
            row[f"{col_prefix}_max_ref_progress"] = prog_metrics["max"] if prog_metrics else None
            row[f"{col_prefix}_max_ref_steps"] = prog_steps_metrics["max"] if prog_steps_metrics else None
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


def extract_reference_progress(data: Dict, order_name: str) -> Dict[int, Tuple[int, int, float]]:
    progress: Dict[int, Tuple[int, int, float]] = {}
    for idx in (0, 1):
        actions = extract_executed_actions(data, idx)
        try:
            steps, ref_len, ratio = best_reference_progress(order_name, idx, actions)
        except FileNotFoundError:
            continue
        progress[idx] = (steps, ref_len, ratio)
    return progress


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
        agent_ref_progress = extract_reference_progress(data, order_name)
        order_metrics[order_name].add_episode(
            success,
            score,
            agent_totals,
            agent_sequences,
            agent_names,
            team_total,
            agent_format_penalties,
            agent_ref_progress,
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
        agent_progress: Dict[str, Dict[str, Optional[float]]] = summary["agent_ref_progress"]  # type: ignore[assignment]
        agent_progress_steps: Dict[str, Dict[str, Optional[float]]] = summary["agent_ref_steps"]  # type: ignore[assignment]
        agent_names = sorted(set(agent_rewards.keys()) | set(agent_sequences.keys()) | set(agent_penalties.keys()))
        for agent_name in agent_names:
            reward_metrics = agent_rewards.get(agent_name)
            seq_metrics = agent_sequences.get(agent_name)
            penalty_metrics = agent_penalties.get(agent_name)
            prog_metrics = agent_progress.get(agent_name)
            prog_steps_metrics = agent_progress_steps.get(agent_name)
            agent_idx = next((i for i, n in stats.agent_names.items() if n == agent_name), None)
            total_val = sum(stats.agent_totals.get(agent_idx, [])) if agent_idx is not None else None
            print(
                f"- {agent_name:<10} avg_seq={_fmt(seq_metrics['avg'] if seq_metrics else None)} "
                f"avg_format={_fmt(penalty_metrics['avg'] if penalty_metrics else None)} "
                f"avg_ref_progress={_fmt(prog_metrics['avg'] if prog_metrics else None)} "
                f"max_ref_progress={_fmt(prog_metrics['max'] if prog_metrics else None)} "
                f"avg_ref_steps={_fmt(prog_steps_metrics['avg'] if prog_steps_metrics else None)} "
                f"max_ref_steps={_fmt(prog_steps_metrics['max'] if prog_steps_metrics else None)} "
                f"avg_reward={_fmt(reward_metrics['avg'] if reward_metrics else None)} "
                f"total_reward={_fmt(total_val)}"
            )


def print_order_metrics(
    order_metrics: Dict[str, OrderStats],
    mapping: Dict[str, str],
    agent_columns: List[Tuple[int, str]],
) -> None:
    header_parts = ["order", "split", "success_rate"]
    for _, col_prefix in agent_columns:
        header_parts.extend(
            [
                f"{col_prefix}_avg_reward",
                f"{col_prefix}_total_reward",
                f"{col_prefix}_avg_sequence_reward",
                f"{col_prefix}_avg_format_reward",
                f"{col_prefix}_avg_ref_progress",
                f"{col_prefix}_avg_ref_steps",
                f"{col_prefix}_max_ref_progress",
                f"{col_prefix}_max_ref_steps",
            ]
        )
    print("\n=== PER-ORDER METRICS ===")
    print("\t".join(header_parts))
    for order_name in sorted(order_metrics.keys()):
        split = mapping.get(order_name)
        if split is None:
            continue
        row = order_metrics[order_name].agent_metric_row(order_name, split, agent_columns)
        values = [order_name, split, _fmt(row.get("success_rate"))]
        for _, col_prefix in agent_columns:
            values.append(_fmt(row.get(f"{col_prefix}_avg_reward")))
            values.append(_fmt(row.get(f"{col_prefix}_total_reward")))
            values.append(_fmt(row.get(f"{col_prefix}_avg_sequence_reward")))
            values.append(_fmt(row.get(f"{col_prefix}_avg_format_reward")))
            values.append(_fmt(row.get(f"{col_prefix}_avg_ref_progress")))
            values.append(_fmt(row.get(f"{col_prefix}_avg_ref_steps")))
            values.append(_fmt(row.get(f"{col_prefix}_max_ref_progress")))
            values.append(_fmt(row.get(f"{col_prefix}_max_ref_steps")))
        print("\t".join(values))


def parse_plot_sources(entries: List[str]) -> List[Tuple[str, Path]]:
    result: List[Tuple[str, Path]] = []
    for entry in entries:
        if "=" in entry:
            _, path_str = entry.split("=", 1)
        else:
            path_str = entry
        csv_path = Path(path_str.strip())
        label = csv_path.parent.name or csv_path.stem
        result.append((label, csv_path))
    return result


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_per_order_csv(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    rows: Dict[str, Dict[str, Optional[float]]] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            order = row.get("name") or row.get("order")
            split = (row.get("split") or "").strip().lower()
            if not order or split == "unknown":
                continue
            entry = {
                "split": split,
                "assistant_avg_reward": _to_float(row.get("assistant_avg_reward")),
                "assistant_avg_format_reward": _to_float(row.get("assistant_avg_format_reward")),
                "assistant_avg_sequence_reward": _to_float(row.get("assistant_avg_sequence_reward")),
                "assistant_avg_ref_progress": _to_float(row.get("assistant_avg_ref_progress")),
                "assistant_avg_ref_steps": _to_float(row.get("assistant_avg_ref_steps")),
                "assistant_max_ref_progress": _to_float(row.get("assistant_max_ref_progress")),
                "assistant_max_ref_steps": _to_float(row.get("assistant_max_ref_steps")),
                "chef_avg_reward": _to_float(row.get("chef_avg_reward")),
                "chef_avg_format_reward": _to_float(row.get("chef_avg_format_reward")),
                "chef_avg_sequence_reward": _to_float(row.get("chef_avg_sequence_reward")),
                "chef_avg_ref_progress": _to_float(row.get("chef_avg_ref_progress")),
                "chef_avg_ref_steps": _to_float(row.get("chef_avg_ref_steps")),
                "chef_max_ref_progress": _to_float(row.get("chef_max_ref_progress")),
                "chef_max_ref_steps": _to_float(row.get("chef_max_ref_steps")),
                "success_rate": _to_float(row.get("success_rate")),
            }
            rows[order] = entry
    return rows


def plot_per_order_metrics(datasets: List[Tuple[str, Dict[str, Dict[str, Optional[float]]]]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 未安装，跳过绘图。")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    order_names = sorted({order for _, data in datasets for order in data.keys()})
    if not order_names:
        print("[plot] 无可用的菜品数据，跳过绘图。")
        return

    metric_specs = [
        ("assistant_avg_reward", "Assistant Avg Reward"),
        ("assistant_avg_format_reward", "Assistant Avg Format Reward"),
        ("assistant_avg_sequence_reward", "Assistant Avg Process Reward"),
        ("assistant_max_ref_progress", "Assistant Max Reference Progress"),
        ("assistant_max_ref_steps", "Assistant Max Matched Reference Steps"),
        ("chef_avg_reward", "Chef Avg Reward"),
        ("chef_avg_format_reward", "Chef Avg Format Reward"),
        ("chef_avg_sequence_reward", "Chef Avg Process Reward"),
        ("chef_max_ref_progress", "Chef Max Reference Progress"),
        ("chef_max_ref_steps", "Chef Max Matched Reference Steps"),
        ("success_rate", "Success Rate"),
    ]

    x_idx = list(range(len(order_names)))
    n_series = max(1, len(datasets))
    jitter_span = 0.24
    step = jitter_span / max(1, n_series - 1) if n_series > 1 else 0.0
    offsets = [(-jitter_span / 2.0) + i * step for i in range(n_series)]
    markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">", "h", "H"]
    linestyles = ["-", "--", "-.", ":"]

    def _union_split(split_name: str) -> List[int]:
        indices: List[int] = []
        for idx, order in enumerate(order_names):
            for _, data in datasets:
                if data.get(order, {}).get("split") == split_name:
                    indices.append(idx)
                    break
        return indices

    highlight_indices = sorted(set(_union_split("test") + _union_split("dev")))

    for metric_key, title in metric_specs:
        plt.figure(figsize=(max(10, len(order_names) * 0.4), 5))
        for series_idx, (label, data) in enumerate(datasets):
            x_positions = [x + offsets[series_idx] for x in x_idx]
            y_values = []
            for order in order_names:
                val = data.get(order, {}).get(metric_key)
                y_values.append(float(val) if val is not None else float("nan"))
            line, = plt.plot(
                x_positions,
                y_values,
                label=label,
                alpha=0.85,
                linewidth=1.6,
                linestyle=linestyles[series_idx % len(linestyles)],
                marker=markers[series_idx % len(markers)],
                markersize=4.5,
            )
            color = line.get_color()
            plt.scatter(
                x_positions,
                y_values,
                color=color,
                s=18,
                alpha=0.85,
                zorder=4,
                edgecolors="white",
                linewidths=0.4,
            )
        for idx in highlight_indices:
            plt.axvline(
                x_idx[idx],
                color="gray",
                linestyle="--",
                linewidth=1.0,
                alpha=0.25,
                zorder=2,
            )
        plt.xticks(x_idx, order_names, rotation=45, ha="right")
        plt.ylabel(title)
        plt.title(title, pad=40)
        plt.grid(True, alpha=0.3)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.20),
            ncol=min(len(datasets), 3),
            frameon=False,
        )
        plt.tight_layout()
        plot_path = output_dir / f"{metric_key}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"[plot] Saved {plot_path}")


def main() -> None:
    args = parse_args()
    per_order_csv_path, split_csv_path, plot_output_dir = infer_output_paths(args.log_root)
    ran_plot = False
    if args.plot_per_order_csv:
        plot_entries = parse_plot_sources(args.plot_per_order_csv)
        datasets = []
        for label, csv_path in plot_entries:
            if not csv_path.exists():
                print(f"[plot] Warning: CSV not found at {csv_path}, skip.")
                continue
            data = load_per_order_csv(csv_path)
            if not data:
                print(f"[plot] Warning: No valid rows found in {csv_path}, skip.")
                continue
            datasets.append((label, data))
        if datasets:
            plot_per_order_metrics(datasets, plot_output_dir)
            ran_plot = True
        else:
            print("[plot] No datasets available for plotting.")

    order_metrics = aggregate_order_metrics(args.log_root)
    mapping = build_order_split_map(args)
    missing_orders = sorted(o for o in order_metrics.keys() if o not in mapping)
    if missing_orders:
        preview = ", ".join(missing_orders[:10])
        print(
            "[split-map] Warning: 以下任务未在 train/dev/test 数据集中找到标签，将被跳过: "
            f"{preview}{' ...' if len(missing_orders) > 10 else ''}"
        )
    split_stats = build_split_stats(order_metrics, mapping)
    agent_columns = make_agent_columns(order_metrics)
    print_order_metrics(order_metrics, mapping, agent_columns)
    print_split_summary(split_stats)
    per_order_rows: List[Dict[str, Optional[float]]] = []
    for order_name in sorted(order_metrics.keys()):
        split = mapping.get(order_name)
        if split is None:
            continue
        row = order_metrics[order_name].agent_metric_row(order_name, split, agent_columns)
        per_order_rows.append(row)
    if per_order_rows:
        write_csv(per_order_rows, per_order_csv_path)

    split_rows: List[Dict[str, Optional[float]]] = []
    for split_name in sorted(split_stats.keys()):
        row = split_stats[split_name].to_row(split_name, split_name, agent_columns)
        split_rows.append(row)
    if split_rows:
        write_csv(split_rows, split_csv_path)

    if args.plot_per_order_csv and not ran_plot:
        print("[plot] 未能绘制任何图表（数据为空）。")


if __name__ == "__main__":
    main()
