#!/usr/bin/env python3
"""
Evaluate an SFT model on the exported JSONL test set.

Metrics
-------
- Teacher-forcing loss / perplexity on the reference assistant tokens.
- Generation metrics: format pass rate, Action exact match rate.
- Language drift: fraction of outputs containing Chinese characters.

Examples
--------
python scripts/sft/eval_qwen_sft.py \
  --model /path/to/assistant_merged \
  --data data/sft/test_level12_Assistant.jsonl \
  --max-samples 200 \
  --temperature 0.0

If you only have a LoRA adapter dir (no config.json), provide the base:
python scripts/sft/eval_qwen_sft.py \
  --model /path/to/Assistant_adapter \
  --base-model /path/to/Qwen2.5-7B-Instruct \
  --data data/sft/test_level12_Assistant.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ACTION_RE = re.compile(r"(?m)^\s*Action:\s*(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen SFT checkpoints on JSONL test data.")
    parser.add_argument("--model", required=True, type=Path, help="HF model dir (merged) or LoRA adapter dir.")
    parser.add_argument("--base-model", type=Path, help="Base HF model dir/name when --model is a LoRA adapter.")
    parser.add_argument("--data", required=True, type=Path, help="JSONL file (e.g., data/sft/test_level12_Assistant.jsonl).")
    parser.add_argument("--max-length", type=int, default=2048, help="Max total tokens for loss computation.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for generation.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for loss computation.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = all).")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    parser.add_argument(
        "--dump-jsonl",
        type=Path,
        help="Optional path to write debug JSONL with mismatches / format failures / Chinese outputs.",
    )
    parser.add_argument(
        "--cjk-threshold",
        type=float,
        default=0.01,
        help="Mark output as Chinese when CJK char ratio >= threshold.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    return parser.parse_args()


def load_jsonl(path: Path, max_samples: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples and len(records) >= max_samples:
                break
    return records


def canonicalize_action(action: str) -> str:
    action = action.strip().strip("`").strip()
    action = re.sub(r"\s+", " ", action)
    # Remove spaces around structural punctuation to reduce false mismatches.
    action = re.sub(r"\s*([(),;])\s*", r"\1", action)
    return action


def extract_action(text: str) -> Optional[str]:
    match = _ACTION_RE.search(text)
    if not match:
        return None
    action = match.group(1).strip()
    action = canonicalize_action(action)
    return action if action else None


def is_format_ok(text: str) -> bool:
    return all(tag in text for tag in ("Think:", "Recent Goal:", "Action:"))


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(_CJK_RE.findall(text))
    return cjk / max(1, len(text))


@dataclass
class LossBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    target_tokens: int


def _apply_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def build_loss_example(
    tokenizer,
    record: Dict[str, Any],
    max_length: int,
) -> Tuple[List[int], List[int], int]:
    system_prompt = record.get("system") or ""
    user_prompt = record.get("prompt") or ""
    assistant_ref = record.get("response") or ""

    messages_prompt: List[Dict[str, str]] = []
    if system_prompt:
        messages_prompt.append({"role": "system", "content": system_prompt})
    messages_prompt.append({"role": "user", "content": user_prompt})

    prompt_text = _apply_chat(tokenizer, messages_prompt, add_generation_prompt=True)
    full_text = _apply_chat(
        tokenizer,
        messages_prompt + [{"role": "assistant", "content": assistant_ref}],
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
    else:
        prompt_len = len(prompt_ids)

    labels = full_ids.copy()
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    target_tokens = sum(1 for v in labels if v != -100)
    return full_ids, labels, target_tokens


def collate_loss_batch(examples: List[Tuple[List[int], List[int], int]], pad_id: int) -> LossBatch:
    max_len = max(len(ids) for ids, _, _ in examples)
    batch_input, batch_mask, batch_labels = [], [], []
    total_targets = 0
    for ids, labels, target_tokens in examples:
        total_targets += target_tokens
        pad = max_len - len(ids)
        batch_input.append(ids + [pad_id] * pad)
        batch_mask.append([1] * len(ids) + [0] * pad)
        batch_labels.append(labels + [-100] * pad)
    return LossBatch(
        input_ids=torch.tensor(batch_input, dtype=torch.long),
        attention_mask=torch.tensor(batch_mask, dtype=torch.long),
        labels=torch.tensor(batch_labels, dtype=torch.long),
        target_tokens=total_targets,
    )


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_path = args.model
    is_adapter = (model_path / "adapter_config.json").exists() or (model_path / "adapter_model.safetensors").exists()
    if is_adapter:
        if not args.base_model:
            raise SystemExit("Detected LoRA adapter directory; please pass --base-model.")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto" if args.device == "cuda" else None,
        )
        try:
            from peft import PeftModel
        except Exception as exc:
            raise SystemExit("peft is required to load LoRA adapters. Please install peft in the env.") from exc
        model = PeftModel.from_pretrained(base, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto" if args.device == "cuda" else None,
        )

    model.eval()
    return model, tokenizer


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.data, args.max_samples)
    if not records:
        raise SystemExit(f"No records loaded from {args.data}")

    model, tokenizer = load_model_and_tokenizer(args)
    device = torch.device("cpu")
    if args.device == "cuda" and torch.cuda.is_available():
        device = next(model.parameters()).device

    # Loss / perplexity
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        examples = [build_loss_example(tokenizer, rec, args.max_length) for rec in batch]
        loss_batch = collate_loss_batch(examples, pad_id=tokenizer.pad_token_id)
        loss_batch.input_ids = loss_batch.input_ids.to(device)
        loss_batch.attention_mask = loss_batch.attention_mask.to(device)
        loss_batch.labels = loss_batch.labels.to(device)

        with torch.no_grad():
            out = model(
                input_ids=loss_batch.input_ids,
                attention_mask=loss_batch.attention_mask,
                labels=loss_batch.labels,
            )
            loss = float(out.loss)
        if loss_batch.target_tokens:
            total_nll += loss * loss_batch.target_tokens
            total_tokens += loss_batch.target_tokens

    mean_loss = total_nll / max(1, total_tokens)
    ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")

    # Generation metrics
    fmt_ok = 0
    action_match = 0
    action_total = 0
    zh_count = 0
    gen_count = 0
    debug_rows: List[Dict[str, Any]] = []

    do_sample = args.temperature > 1e-6
    for rec in records:
        system_prompt = rec.get("system") or ""
        user_prompt = rec.get("prompt") or ""
        ref = rec.get("response") or ""

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        prompt_text = _apply_chat(tokenizer, messages, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
            )
        out_text = tokenizer.decode(gen_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        gen_count += 1

        fmt_pass = is_format_ok(out_text)
        if fmt_pass:
            fmt_ok += 1
        zh_ratio = chinese_ratio(out_text)
        zh_flag = zh_ratio >= args.cjk_threshold
        if zh_flag:
            zh_count += 1

        ref_action = extract_action(ref)
        pred_action = extract_action(out_text)
        if ref_action is not None:
            action_total += 1
            if pred_action == ref_action:
                action_match += 1

        if args.dump_jsonl:
            reasons: List[str] = []
            if not fmt_pass:
                reasons.append("format_fail")
            if ref_action is not None and pred_action != ref_action:
                reasons.append("action_mismatch")
            if zh_flag:
                reasons.append("chinese_output")
            if reasons:
                meta = rec.get("meta") or {}
                debug_rows.append(
                    {
                        "reasons": reasons,
                        "meta": meta,
                        "ref_action": ref_action,
                        "pred_action": pred_action,
                        "pred_text": out_text,
                    }
                )

    print(f"[eval] samples={len(records)}")
    print(f"[eval] teacher_forcing_loss={mean_loss:.6f} perplexity={ppl:.3f} (tokens={total_tokens})")
    print(f"[eval] format_pass_rate={fmt_ok/max(1,gen_count):.3f} ({fmt_ok}/{gen_count})")
    print(f"[eval] action_exact_match={action_match/max(1,action_total):.3f} ({action_match}/{action_total})")
    print(f"[eval] chinese_output_rate={zh_count/max(1,gen_count):.3f} ({zh_count}/{gen_count})")
    if args.dump_jsonl:
        args.dump_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_jsonl.open("w", encoding="utf-8") as fh:
            for row in debug_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[eval] dump_jsonl={args.dump_jsonl} (rows={len(debug_rows)})")


if __name__ == "__main__":
    main()
