#!/usr/bin/env python3
"""Decode one critic input sample from a rollout cache into a text file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one decoded critic input from a rollout cache."
    )
    parser.add_argument(
        "rollout_cache",
        help="Path to a rollout cache .pt file, for example runs/rl/train/initial_rollout_cache/rollout_rank0_u00001.pt",
    )
    parser.add_argument(
        "--model-path",
        default="/datacache/LLMs/Qwen2.5-7B-Instruct",
        help="Tokenizer/model path used to decode critic_input_ids.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Transition index to export.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output text file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_path = Path(args.rollout_cache).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    transitions = torch.load(rollout_path, map_location="cpu")
    if not isinstance(transitions, list):
        raise TypeError(f"Expected a list in rollout cache, got {type(transitions)!r}")
    if not transitions:
        raise ValueError(f"Rollout cache is empty: {rollout_path}")
    if args.index < 0 or args.index >= len(transitions):
        raise IndexError(
            f"Index {args.index} out of range for {rollout_path} with {len(transitions)} transitions"
        )

    sample = transitions[args.index]
    if not isinstance(sample, dict):
        raise TypeError(f"Expected transition dict, got {type(sample)!r}")
    critic_input_ids = sample.get("critic_input_ids")
    if critic_input_ids is None:
        raise ValueError(f"Transition {args.index} has no critic_input_ids")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    critic_text = tokenizer.decode(critic_input_ids, skip_special_tokens=True)

    header = [
        f"rollout_cache: {rollout_path}",
        f"index: {args.index}",
        f"agent_index: {sample.get('agent_index')}",
        f"timestep: {sample.get('timestep')}",
        f"reward: {sample.get('reward')}",
        f"value: {sample.get('value')}",
        f"critic_tokens: {int(critic_input_ids.numel())}",
        "",
        "===== Decoded Critic Input =====",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(header) + critic_text + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
