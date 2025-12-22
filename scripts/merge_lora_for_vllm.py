#!/usr/bin/env python3
"""
Utility to merge a LoRA adapter checkpoint with its base model so that vLLM (or
any Hugging Face-compatible runtime) can load the resulting directory directly.

Example:
    python scripts/merge_lora_for_vllm.py \
        --base /path/to/Qwen/Qwen2.5-7B-Instruct \
        --adapter runs/qwen2.5-sft-level12/20251218-215320/epoch_02 \
        --output runs/qwen2.5-sft-level12/20251218-215320/epoch_02-merged
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("peft is required to merge LoRA adapters. Install with `pip install peft`.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model for vLLM serving.")
    parser.add_argument("--base", required=True, type=Path, help="Path to the base HF checkpoint.")
    parser.add_argument("--adapter", required=True, type=Path, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output", required=True, type=Path, help="Directory to save the merged model.")
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["float32", "float16", "bfloat16", "auto", None],
        help="Torch dtype for loading/merging. Defaults to base model dtype.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to host the merge process. 'cuda' requires an available GPU.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_str: Optional[str]):
    if not dtype_str or dtype_str == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[dtype_str]


def copy_extra_tokenizer_files(src: Path, dst: Path) -> None:
    """Copy auxiliary tokenizer files (e.g., chat templates) from adapter -> output."""
    extra_files = [
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "spiece.model",
    ]
    for name in extra_files:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch_dtype = resolve_dtype(args.dtype)
    device_map = "auto" if args.device == "cuda" and torch.cuda.is_available() else None

    print(f"[merge] Loading base model from {args.base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if device_map is None:
        base_model = base_model.to(args.device)

    print(f"[merge] Loading LoRA adapter from {args.adapter}")
    peft_model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        torch_dtype=torch_dtype,
    )

    print("[merge] Merging and unloading adapter weights...")
    merged_model = peft_model.merge_and_unload()

    print(f"[merge] Saving merged checkpoint to {args.output}")
    merged_model.save_pretrained(args.output)

    print("[merge] Saving tokenizer artifacts...")
    if (args.adapter / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    copy_extra_tokenizer_files(args.adapter, args.output)

    print("[merge] Done. The output directory can now be served directly by vLLM.")


if __name__ == "__main__":
    main()
