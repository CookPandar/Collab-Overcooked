#!/usr/bin/env python3
"""Filter an RL snapshot dataset by timestep and emit a smaller JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter snapshot JSONL rows by timestep."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Source snapshot JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Filtered snapshot JSONL output path.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        required=True,
        help="Keep rows whose snapshot timestep is in this set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def extract_timestep(row: Dict[str, Any]) -> Optional[int]:
    snapshot = row.get("snapshot") or row.get("state") or {}
    metadata = row.get("metadata") or {}
    candidates = [
        snapshot.get("timestep"),
        (snapshot.get("env_state") or {}).get("timestep"),
        metadata.get("local_step"),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input snapshot file not found: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output}. Use --overwrite to replace it."
        )

    keep: Set[int] = {int(step) for step in args.timesteps}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            timestep = extract_timestep(row)
            if timestep not in keep:
                continue
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[filter_snapshot_dataset] input={args.input} output={args.output} "
        f"kept={written}/{total} timesteps={sorted(keep)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
