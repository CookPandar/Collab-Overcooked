#!/usr/bin/env python3
"""
Convert successful Collab-Overcooked runs into SFT-ready JSONL examples.

Example:
    python scripts/export_sft_dataset.py \
        --source-dir assets/data/batch_results \
        --models azure-gpt-4o \
        --levels 1 2 \
        --temperature 0.7 \
        --agents Chef Assistant \
        --output data/sft/gpt4o_level12.jsonl

Outputs are now split per agent role (e.g., `*_Chef.jsonl` / `*_Assistant.jsonl`
when requesting both agents).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Collab-Overcooked logs to SFT dataset.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("assets/data/batch_results"),
        help="Root directory that contains per-model batch results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model directories (under --source-dir) to process.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=[1, 2],
        help="Only include orders belonging to these levels (default: 1 2).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="If provided, only include runs whose inferred temperature matches this value.",
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        default=["Chef", "Assistant"],
        choices=["Chef", "Assistant"],
        help="Which agent roles to export (default: both).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write all samples to a single JSONL file.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        help="Write samples whose order belongs to the train split.",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        help="Write samples whose order belongs to the validation split.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        help="Write samples whose order belongs to the test split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Ratio of orders assigned to the train split (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Ratio of orders assigned to the validation split (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of orders assigned to the test split (default: 0.2).",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used when partitioning orders into splits.",
    )
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=Path("collab_overcooked/prompts/recipe"),
        help="Recipe directory used to map order names to level IDs.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap on number of exported examples.",
    )
    return parser.parse_args()


def load_order_levels(recipe_dir: Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for recipe_file in recipe_dir.glob("*.txt"):
        stem = recipe_file.stem
        parts = stem.split("_", 1)
        level = 0
        name = stem
        if len(parts) == 2 and parts[0].isdigit():
            level = int(parts[0])
            name = parts[1]
        elif len(parts) == 2:
            name = parts[1]
        mapping[name] = level
    return mapping


def parse_run_meta(run_id: str, order: str) -> Tuple[Optional[int], Optional[float]]:
    """Extract repeat index and numeric temperature from run_id."""
    if not run_id.startswith("worker"):
        return None, None
    core = run_id.split("-", 1)[0]
    body = core[len("worker") :]
    tokens = body.split("_")
    if len(tokens) < 2:
        return None, None
    try:
        repeat_idx = int(tokens[0])
    except ValueError:
        repeat_idx = None
    order_tokens = order.split("_")
    if not order_tokens:
        order_tokens = [order]
    if len(tokens) < 1 + len(order_tokens) + 1:
        temp_value = None
    else:
        job_index = tokens[-1]
        expected_order = tokens[-1 - len(order_tokens) : -1]
        if expected_order == order_tokens:
            temp_segments = tokens[1 : -1 - len(order_tokens)]
        else:
            temp_segments = tokens[1:-1]
        temp_value = None
        if temp_segments:
            temp_str = "_".join(temp_segments).replace("_", ".")
            try:
                temp_value = float(temp_str)
            except ValueError:
                temp_value = None
    return repeat_idx, temp_value


def iter_success_logs(model_dir: Path, order_levels: Dict[str, int]) -> Iterable[Tuple[str, int, Path, dict, Optional[float]]]:
    json_root = model_dir / "json"
    if not json_root.exists():
        return
    for order_dir in sorted(p for p in json_root.iterdir() if p.is_dir()):
        order = order_dir.name
        level = order_levels.get(order, -1)
        for log_file in sorted(order_dir.glob("*.json")):
            data = json.loads(log_file.read_text())
            if not data.get("total_order_finished"):
                continue
            run_id = log_file.stem
            _, temp_value = parse_run_meta(run_id, order)
            yield order, level, log_file, data, temp_value


def floats_close(a: float, b: float, tol: float = 1e-6) -> bool:

    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def build_sample(
    *,
    system_prompt: str,
    user_prompt: str,
    response: str,
    meta: Dict,
) -> Dict:
    return {
        "system": system_prompt.strip(),
        "prompt": user_prompt,
        "response": response,
        "meta": meta,
    }


def derive_agent_path(base: Path, agent: str) -> Path:
    """
    Derive an agent-specific output path. If `base` already looks like a file
    (has suffix), append `_<agent>` before the suffix. Otherwise, treat it as a
    directory and place `<agent>.jsonl` inside.
    """
    agent = agent.replace(" ", "_")
    if base.suffix:
        return base.with_name(f"{base.stem}_{agent}{base.suffix}")
    return base / f"{agent}.jsonl"


def gather_candidate_orders(
    args: argparse.Namespace,
    order_levels: Dict[str, int],
    target_levels: set,
) -> List[str]:
    orders: set[str] = set()
    for model_name in args.models:
        model_dir = args.source_dir / model_name
        if not model_dir.exists():
            continue
        json_root = model_dir / "json"
        if not json_root.exists():
            continue
        for order_dir in json_root.iterdir():
            if not order_dir.is_dir():
                continue
            order = order_dir.name
            level = order_levels.get(order, -1)
            if target_levels and level not in target_levels:
                continue
            include_order = False
            for log_file in order_dir.glob("*.json"):
                data = json.loads(log_file.read_text())
                if not data.get("total_order_finished"):
                    continue
                _, temp_value = parse_run_meta(log_file.stem, order)
                if args.temperature is not None:
                    if temp_value is None or not floats_close(temp_value, args.temperature):
                        continue
                include_order = True
                break
            if include_order:
                orders.add(order)
    return sorted(orders)


def assign_order_splits(
    orders: List[str],
    ratios: List[Tuple[str, float]],
    seed: int,
) -> Dict[str, str]:
    active = [(name, ratio) for name, ratio in ratios if ratio > 0]
    total_ratio = sum(r for _, r in active)
    if not active or total_ratio <= 0:
        return {}
    normalized = [(name, ratio / total_ratio) for name, ratio in active]
    rng = random.Random(seed)
    shuffled = orders[:]
    rng.shuffle(shuffled)
    total_orders = len(shuffled)
    order_split: Dict[str, str] = {}
    start = 0
    for idx, (name, ratio) in enumerate(normalized):
        if idx == len(normalized) - 1:
            size = total_orders - start
        else:
            size = min(total_orders - start, int(round(ratio * total_orders)))
        subset = shuffled[start : start + size]
        for order in subset:
            order_split[order] = name
        start += size
        if start >= total_orders:
            break
    return order_split


def export_samples(args: argparse.Namespace) -> int:
    if not (
        args.output
        or args.train_output
        or args.val_output
        or args.test_output
    ):
        raise ValueError("Please provide --output or at least one of --train-output/--val-output/--test-output.")

    order_levels = load_order_levels(args.recipe_dir)
    target_levels = set(args.levels or [])
    target_agents = set(args.agents or [])
    if not target_agents:
        target_agents = {"Chef", "Assistant"}
    agent_list = sorted(target_agents)

    split_configs = [
        ("train", args.train_output, args.train_ratio),
        ("val", args.val_output, args.val_ratio),
        ("test", args.test_output, args.test_ratio),
    ]
    active_split_configs = [(name, path, ratio) for name, path, ratio in split_configs if path and ratio > 0]
    order_split_map: Dict[str, str] = {}
    if active_split_configs:
        candidate_orders = gather_candidate_orders(args, order_levels, target_levels)
        ratios = [(name, ratio) for name, _, ratio in active_split_configs]
        order_split_map = assign_order_splits(candidate_orders, ratios, args.split_seed)
        for name, path, _ in active_split_configs:
            path.parent.mkdir(parents=True, exist_ok=True)
    writers: Dict[Tuple[str, str], Tuple[Path, TextIO]] = {}
    try:
        if args.output:
            for agent in agent_list:
                agent_path = derive_agent_path(args.output, agent)
                agent_path.parent.mkdir(parents=True, exist_ok=True)
                writers[("all", agent)] = (agent_path, agent_path.open("w", encoding="utf-8"))
        for name, path, _ in active_split_configs:
            for agent in agent_list:
                agent_path = derive_agent_path(path, agent)
                agent_path.parent.mkdir(parents=True, exist_ok=True)
                writers[(name, agent)] = (agent_path, agent_path.open("w", encoding="utf-8"))

        total_written = 0
        for model_name in args.models:
            model_dir = args.source_dir / model_name
            if not model_dir.exists():
                print(f"[export] Skip {model_name}: directory not found ({model_dir})")
                continue
            for order, level, log_path, payload, temp_value in iter_success_logs(model_dir, order_levels):
                if target_levels and level not in target_levels:
                    continue
                if args.temperature is not None:
                    if temp_value is None or not floats_close(temp_value, args.temperature):
                        continue
                split_name = None
                if order_split_map:
                    split_name = order_split_map.get(order)
                    if not split_name:
                        continue
                prompt_templates = payload.get("prompt_templates") or {}
                for turn in payload.get("content", []):
                    node = turn.get("content") or {}
                    observations = node.get("observation") or []
                    original_logs = node.get("original_log") or []
                    for agent_idx, calls in enumerate(original_logs):
                        if not calls:
                            continue
                        agent_name = calls[0].get("agent") or ("Chef" if agent_idx == 0 else "Assistant")
                        if target_agents and agent_name not in target_agents:
                            continue
                        system_prompt = prompt_templates.get(agent_name, {}).get("prompt", "").strip()
                        if not system_prompt:
                            continue
                        user_prompt = (
                            observations[agent_idx]
                            if agent_idx < len(observations)
                            else calls[0].get("input", "")
                        )
                        for call in calls:
                            sample = build_sample(
                                system_prompt=system_prompt,
                                user_prompt=call.get("input", user_prompt),
                                response=call.get("output", ""),
                                meta={
                                    "model": model_name,
                                    "order": order,
                                    "level": level,
                                    "agent": agent_name,
                                    "call_type": call.get("call_type"),
                                    "timestamp": call.get("timestamp"),
                                    "run_id": log_path.stem,
                                    "temperature": temp_value,
                                },
                            )
                            line = json.dumps(sample, ensure_ascii=False) + "\n"
                            writer_all = writers.get(("all", agent_name))
                            if writer_all:
                                writer_all[1].write(line)
                            if split_name:
                                writer_split = writers.get((split_name, agent_name))
                                if writer_split:
                                    writer_split[1].write(line)
                            total_written += 1
                            if args.max_samples and total_written >= args.max_samples:
                                return total_written
        return total_written
    finally:
        for _, handle in writers.values():
            handle.close()


def main() -> None:
    args = parse_args()
    total = export_samples(args)
    print(f"[export] Wrote {total} samples.")


if __name__ == "__main__":
    main()
