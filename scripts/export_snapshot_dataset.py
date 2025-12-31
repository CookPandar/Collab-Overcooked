#!/usr/bin/env python3
"""Run the RL session to export environment snapshots for off-policy training."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from collab_overcooked.main import load_config_from_yaml
from collab_overcooked.training.mappo_qwen import MAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the RL YAML config used for session construction.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .jsonl file where snapshots will be written.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of snapshots (steps) to export.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Optional cap on episodes to roll out (0 means unlimited).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config_from_yaml(args.config)
    env_cfg = deepcopy(config.get("environment", {}))
    trainer_cfg = deepcopy(config.get("trainer", {}))
    trainer_cfg["collect_only"] = False
    trainer_cfg["train_only"] = False
    trainer_cfg.setdefault("total_updates", 1)

    trainer = MAPPOTrainer(env_cfg, trainer_cfg, full_config=config)
    if trainer.session is None:
        raise RuntimeError("Trainer session is not initialized; check the config flags.")
    session = trainer.session

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Snapshot file {output_path} already exists. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    steps_written = 0
    episodes = 0
    with output_path.open("w", encoding="utf-8") as writer:
        while steps_written < args.num_steps:
            snapshot = session.capture_snapshot()
            payload = {
                "snapshot": snapshot,
                "metadata": {
                    "step_index": steps_written,
                    "episode_index": episodes,
                },
            }
            writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
            writer.flush()

            step_result = session.step()
            steps_written += 1
            if step_result.done:
                session.reset()
                episodes += 1
                if args.max_episodes and episodes >= args.max_episodes:
                    break

    trainer.accelerator.print(
        f"[SnapshotExport] Wrote {steps_written} snapshots to {output_path} (episodes={episodes})."
    )


if __name__ == "__main__":
    main()
