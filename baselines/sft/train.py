"""LoRA SFT of the Chef and Assistant adapters on data from baselines.sft.collect.

    python -m baselines.sft.train --model /path/to/Qwen3-1.7B --data data/sft/gpt4o_think.jsonl \
        --output runs/sft_think

Each role gets its own LoRA adapter (r=16, alpha=32 on q/k/v/o), saved as <output>/Chef and
<output>/Assistant: the `adapters` of an RL config. The prompt is tokenized exactly as during RL rollouts
(chat template, enable_thinking=False) and the loss covers the response tokens and the end-of-turn token.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

ROLE_ADAPTERS = {"chef": "Chef", "assistant": "Assistant"}


def encode(tokenizer, example, max_length):
    prompt = tokenizer.apply_chat_template(example["messages"], add_generation_prompt=True, tokenize=True,
                                           enable_thinking=False)
    if hasattr(prompt, "keys"):
        prompt = prompt["input_ids"]
    response = tokenizer(example["response"], add_special_tokens=False)["input_ids"]
    response.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    if len(prompt) + len(response) > max_length:
        return None
    return torch.tensor(list(prompt) + response), len(prompt)


def train_role(model_path, examples, output, args):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = [x for x in (encode(tokenizer, e, args.max_length) for e in examples) if x is not None]
    print(f"{output.name}: {len(data)} examples ({len(examples) - len(data)} longer than {args.max_length})")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(args.device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(data) / args.batch_size) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(args.warmup_ratio * steps), steps)
    rng = random.Random(args.seed)
    log = (output.parent / f"{output.name}_train_log.jsonl").open("w")
    step = 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = [data[i] for i in order[start:start + args.batch_size]]
            n_tokens = sum(len(ids) - prompt_len for ids, prompt_len in batch)
            total = 0.0
            for ids, prompt_len in batch:  # one sequence per forward pass: no padding
                ids = ids.to(args.device)
                logits = model(input_ids=ids[None]).logits[0, prompt_len - 1:-1].float()
                loss = torch.nn.functional.cross_entropy(logits, ids[prompt_len:], reduction="sum") / n_tokens
                loss.backward()
                total += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            log.write(json.dumps({"step": step, "epoch": epoch, "loss": total, "grad_norm": float(grad_norm),
                                  "lr": scheduler.get_last_lr()[0]}) + "\n")
            if step % 20 == 0:
                print(f"{output.name} epoch {epoch} step {step}/{steps} loss {total:.4f}", flush=True)
        model.save_pretrained(str(output.parent / "epochs" / f"epoch{epoch}" / output.name))
    model.save_pretrained(str(output))
    log.close()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, nargs="+", help="JSONL files from baselines.sft.collect")
    parser.add_argument("--output", required=True)
    parser.add_argument("--roles", nargs="+", default=["chef", "assistant"])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16, help="sequences per optimizer step")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    examples = [json.loads(line) for path in args.data for line in open(path)]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for role in args.roles:
        train_role(args.model, [e for e in examples if e["role"] == role], output / ROLE_ADAPTERS[role], args)


if __name__ == "__main__":
    main()
