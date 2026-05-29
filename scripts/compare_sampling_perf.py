#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _to_float(row: Dict[str, str], key: str) -> float:
    raw = row.get(key, "")
    try:
        return float(raw) if raw not in ("", None) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _latest_row(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[-1]


def _summarize(root: Path) -> Dict[str, float]:
    perf_path = root / "runs" / "rl" / "collect" / "performance_curve.csv"
    if not perf_path.exists():
        # Also support passing the collect output directory directly.
        perf_path = root / "performance_curve.csv"
    if not perf_path.exists():
        raise FileNotFoundError(f"performance_curve.csv not found under {root}")
    row = _latest_row(perf_path)
    return {
        "update_idx": _to_float(row, "update_idx"),
        "rollout_wall_time_sec": _to_float(row, "rollout_wall_time_sec"),
        "env_steps": _to_float(row, "env_steps"),
        "policy_calls": _to_float(row, "policy_calls"),
        "actor_llm_calls": _to_float(row, "actor_llm_calls"),
        "avg_actor_llm_seconds": _to_float(row, "avg_actor_llm_seconds"),
        "value_llm_calls": _to_float(row, "value_llm_calls"),
        "avg_value_llm_seconds": _to_float(row, "avg_value_llm_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare collect sampling performance across experiment roots."
    )
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()

    rows: List[Dict[str, str]] = []
    for root in args.roots:
        stats = _summarize(root.resolve())
        wall = stats["rollout_wall_time_sec"]
        policy_calls = stats["policy_calls"]
        rows.append(
            {
                "root": str(root),
                "update_idx": f"{stats['update_idx']:.0f}",
                "wall_sec": f"{wall:.2f}",
                "env_steps": f"{stats['env_steps']:.0f}",
                "policy_calls": f"{policy_calls:.0f}",
                "policy_calls_per_sec": f"{policy_calls / wall if wall > 0 else 0.0:.4f}",
                "actor_calls": f"{stats['actor_llm_calls']:.0f}",
                "avg_actor_sec": f"{stats['avg_actor_llm_seconds']:.2f}",
                "value_calls": f"{stats['value_llm_calls']:.0f}",
                "avg_value_sec": f"{stats['avg_value_llm_seconds']:.2f}",
            }
        )

    headers = [
        "root",
        "update_idx",
        "wall_sec",
        "env_steps",
        "policy_calls",
        "policy_calls_per_sec",
        "actor_calls",
        "avg_actor_sec",
        "value_calls",
        "avg_value_sec",
    ]
    print(",".join(headers))
    for row in rows:
        print(",".join(row[h] for h in headers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
