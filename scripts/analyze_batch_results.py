#!/usr/bin/env python3
"""
从 assets/data/batch_results/*/json 目录读取历史运行结果，统计：
1. 各模型在不同温度下的整体成功率
2. 各 level 的成功率，并输出折线图

示例：
    python scripts/analyze_batch_results.py \\
        --models azure-gpt-4o \\
        --output-dir assets/data/batch_results
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze batch evaluation logs.")
    parser.add_argument(
        "--models",
        nargs="*",
        help="Models to analyze (default: all subdirectories of --output-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/data/batch_results"),
        help="Root directory that contains per-model results.",
    )
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=Path("collab_overcooked/prompts/recipe"),
        help="Recipe directory used to map order name to level.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="If provided, only include runs whose inferred temperature matches.",
    )
    return parser.parse_args()


def load_order_levels(recipe_dir: Path) -> Dict[str, int]:
    """Derive mapping from order name to level using recipe filenames."""
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


def parse_run_meta(run_id: str) -> Tuple[int, str, Optional[float]]:
    """Extract repeat index, display label, and numeric temperature from run_id."""
    if not run_id.startswith("worker"):
        return -1, "unknown", None
    core = run_id[len("worker") :]
    parts = core.split("_")
    if len(parts) < 2:
        return -1, "unknown", None
    repeat = parts[0]
    job_index = parts[-1]
    temp_parts = parts[1:-1]
    if not temp_parts:
        safe_temp = "unknown"
        temp_value = None
    else:
        safe_temp = "_".join(temp_parts)
        as_float = safe_temp.replace("_", ".")
        try:
            temp_value = float(as_float)
        except ValueError:
            temp_value = None
    return int(repeat), safe_temp, temp_value


def safe_temp_label(safe_temp: str) -> str:
    if safe_temp == "unknown":
        return "unknown"
    float_str = safe_temp.replace("_", ".")
    try:
        value = float(float_str)
    except ValueError:
        return float_str
    if value.is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


def collect_entries(
    model_dir: Path,
    order_levels: Dict[str, int],
    target_temperature: Optional[float] = None,
) -> List[Dict]:
    entries: List[Dict] = []
    json_root = model_dir / "json"
    if not json_root.exists():
        return entries

    for order_dir in sorted(p for p in json_root.iterdir() if p.is_dir()):
        order = order_dir.name
        level = order_levels.get(order, -1)
        for log_file in sorted(order_dir.glob("*.json")):
            raw = json.loads(log_file.read_text())
            success = bool(raw.get("total_order_finished"))
            timestamps = raw.get("total_timestamp") or []
            steps = timestamps[-1] if timestamps else None
            run_id = log_file.stem.split("-", 1)[0]
            repeat_idx, safe_temp, temp_value = parse_run_meta(run_id)
            if target_temperature is not None:
                if temp_value is None or abs(temp_value - target_temperature) > 1e-6:
                    continue
            entries.append(
                {
                    "model": model_dir.name,
                    "temperature_label": safe_temp_label(safe_temp),
                    "order": order,
                    "level": level,
                    "success": success,
                    "steps": steps,
                    "log_path": str(log_file),
                    "repeat": repeat_idx,
                    "run_id": run_id,
                }
            )
    return entries


def aggregate(entries: List[Dict]):
    model_stats = defaultdict(lambda: {"success": 0, "total": 0})
    level_stats = defaultdict(lambda: {"success": 0, "total": 0})

    for item in entries:
        model = item["model"]
        model_stats[model]["total"] += 1
        if item["success"]:
            model_stats[model]["success"] += 1

        level = item["level"]
        level_key = (model, level)
        level_stats[level_key]["total"] += 1
        if item["success"]:
            level_stats[level_key]["success"] += 1

    return model_stats, level_stats


def save_overall_plot(model_stats, out_path: Path):
    labels = []
    rates = []
    for model, stats in sorted(model_stats.items()):
        label = model
        total = stats["total"]
        rate = stats["success"] / total if total else 0.0
        labels.append(label)
        rates.append(rate)

    if not labels:
        return

    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(labels, rates, color="#4E79A7")
    plt.ylim(0, 1.05)
    plt.ylabel("Success Rate")
    plt.xticks(rotation=20, ha="right")
    plt.title("Overall Success Rates")
    for idx, rate in enumerate(rates):
        plt.text(idx, min(1.0, rate + 0.02), f"{rate:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_level_plot(level_stats, out_path: Path):
    series_by_label: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for (model, level), stats in level_stats.items():
        if level < 0:
            continue
        total = stats["total"]
        rate = stats["success"] / total if total else 0.0
        series_by_label[model].append((level, rate))

    if not series_by_label:
        return

    plt.figure(figsize=(8, 4))
    for label, pairs in sorted(series_by_label.items()):
        pairs.sort(key=lambda x: x[0])
        xs = [level for level, _ in pairs]
        ys = [rate for _, rate in pairs]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xlabel("Level")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1.05)
    plt.title("Success Rate per Level")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    order_levels = load_order_levels(args.recipe_dir)

    model_dirs = []
    if args.models:
        for name in args.models:
            model_dirs.append(args.output_dir / name)
    else:
        model_dirs = [p for p in args.output_dir.iterdir() if p.is_dir()]

    if not model_dirs:
        raise SystemExit("No model directories found.")

    for model_dir in model_dirs:
        entries = collect_entries(model_dir, order_levels, target_temperature=args.temperature)
        if not entries:
            print(f"[analyze] Skip {model_dir.name}: no logs found.")
            continue

        model_stats, level_stats = aggregate(entries)
        summary_path = model_dir / "results.json"
        summary_path.write_text(json.dumps(entries, indent=2))

        overall = {}
        for model, stats in model_stats.items():
            total = stats["total"]
            overall[model] = {
                "success": stats["success"],
                "total": total,
                "rate": stats["success"] / total if total else 0.0,
            }

        by_level = defaultdict(dict)
        for (model, level), stats in level_stats.items():
            total = stats["total"]
            by_level[model][f"level_{level}"] = {
                "success": stats["success"],
                "total": total,
                "rate": stats["success"] / total if total else 0.0,
            }

        aggregate_path = model_dir / "aggregate.json"
        aggregate_path.write_text(json.dumps({"overall": overall, "by_level": by_level}, indent=2))

        save_overall_plot(model_stats, model_dir / "success_rates.png")
        save_level_plot(level_stats, model_dir / "level_success.png")
        print(f"[analyze] Saved stats for {model_dir.name} -> {aggregate_path}")


if __name__ == "__main__":
    main()
