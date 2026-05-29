#!/usr/bin/env python3
"""Standalone GRPO entrypoint for Collab-Overcooked."""

from __future__ import annotations

from argparse import ArgumentParser

from collab_overcooked.main_rl import run


def main() -> None:
    parser = ArgumentParser(description="Run Collab-Overcooked GRPO training/eval.")
    parser.add_argument("--config", required=True, help="Path to a YAML config with trainer.type=grpo.")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()

