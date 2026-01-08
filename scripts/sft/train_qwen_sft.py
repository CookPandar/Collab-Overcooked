#!/usr/bin/env python3
"""
Lightweight SFT runner for Qwen2.5 models using exported JSONL data.

Example:
    python scripts/train_qwen_sft.py \
        --data-path data/sft/gpt4o_level12.jsonl \
        --model-name Qwen/Qwen2.5-7B-Instruct \
        --output-dir runs/qwen2.5-sft-level12 \
        --epochs 1 \
        --per-device-train-batch-size 1 \
        --gradient-accumulation-steps 16 \
        --learning-rate 5e-5 \
        --max-length 4096 \
        --use-lora \
        --agents Chef Assistant

The script will train one LoRA head per agent and store results under
`<output-dir>/<timestamp>/<AgentName>/`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import datasets
import inspect

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

try:
    from peft import LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = None
    get_peft_model = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 on exported SFT data.")
    parser.add_argument("--data-path", type=Path, help="Single JSONL dataset (backward-compatible).")
    parser.add_argument(
        "--train-data",
        type=Path,
        help="JSONL file for training split.",
    )
    parser.add_argument(
        "--eval-data",
        type=Path,
        help="JSONL file for validation split.",
    )
    parser.add_argument(
        "--test-data",
        type=Path,
        help="JSONL file for held-out test split (not used for training).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model name or local path.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=float, default=1.0, help="Number of training epochs.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum token length after formatting.")
    parser.add_argument("--eval-ratio", type=float, default=0.0, help="Optional validation split (0-1). Ignored when --eval-data provided.")
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Enable LoRA fine-tuning (requires peft).",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 training if available.")
    parser.add_argument("--fp16", action="store_true", help="Use float16 training if available.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Save a full model snapshot under <output-dir>/epoch_<N> at the end of each epoch.",
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Save per-step loss curves + CSV to <output-dir>/training_metrics.{png,csv}.",
    )
    parser.add_argument(
        "--agents",
        nargs="*",
        default=["Chef", "Assistant"],
        choices=["Chef", "Assistant"],
        help="Agent roles to train (each receives its own LoRA head).",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> datasets.Dataset:
    return datasets.load_dataset("json", data_files={"data": str(path)})["data"]


def load_datasets(args: argparse.Namespace) -> Dict[str, datasets.Dataset]:
    result: Dict[str, datasets.Dataset] = {}
    if args.train_data and args.eval_data:
        result["train"] = load_jsonl(args.train_data)
        result["eval"] = load_jsonl(args.eval_data)
    elif args.train_data:
        result["train"] = load_jsonl(args.train_data)
    elif args.data_path:
        result["train"] = load_jsonl(args.data_path)
    else:
        raise ValueError("Please provide --data-path or --train-data.")

    if args.eval_data and "eval" not in result:
        result["eval"] = load_jsonl(args.eval_data)

    if args.test_data:
        result["test"] = load_jsonl(args.test_data)

    return result


def format_example(example: Dict, tokenizer: AutoTokenizer, max_length: int) -> Dict:
    messages: List[Dict[str, str]] = []
    system_prompt = example.get("system")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": example.get("prompt", "")})
    messages.append({"role": "assistant", "content": example.get("response", "")})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def maybe_split_dataset(ds: datasets.Dataset, ratio: float, seed: int) -> Dict[str, datasets.Dataset]:
    if not ratio:
        return {"train": ds}
    split = ds.train_test_split(test_size=ratio, seed=seed)
    return {"train": split["train"], "eval": split["test"]}


def extract_agent_from_meta(meta) -> Optional[str]:
    if isinstance(meta, dict):
        return meta.get("agent")
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed.get("agent")
    return None


def filter_dataset_by_agent(ds: Optional[datasets.Dataset], agent: str) -> Optional[datasets.Dataset]:
    if ds is None:
        return None
    return ds.filter(lambda example: extract_agent_from_meta(example.get("meta")) == agent)


def _should_enable_device_map_auto() -> bool:
    """Only use device_map='auto' when running single-process training."""
    for key in ("WORLD_SIZE", "ACCELERATE_WORLD_SIZE", "SLURM_NTASKS", "MPI_WORLD_SIZE"):
        env_val = os.environ.get(key)
        if env_val and int(env_val) > 1:
            return False
    # LOCAL_RANK will be defined under torchrun/accelerate multi-proc launches.
    local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
    if local_rank and int(local_rank) > 0:
        return False
    return True


def build_model(args: argparse.Namespace) -> AutoModelForCausalLM:
    device_map = "auto" if _should_enable_device_map_auto() else None
    if device_map is None:
        print("[train] Detected distributed launch; loading model without device_map='auto'.")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map=device_map,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        # 可选：有些模型/Trainer需要这个来省显存
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("peft is required for LoRA fine-tuning. Install with `pip install peft`.")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    return model


class EpochSaveCallback(TrainerCallback):
    """Save tokenizer + model weights to output_dir/epoch_<n> at every epoch end."""

    def __init__(self, base_dir: Path, tokenizer: AutoTokenizer):
        self.base_dir = Path(base_dir)
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_float = state.epoch
        if epoch_float is None:
            return
        epoch_idx = max(1, int(round(epoch_float)))
        save_dir = self.base_dir / f"epoch_{epoch_idx:02d}"
        save_dir.mkdir(parents=True, exist_ok=True)
        model = kwargs.get("model")
        if model is not None:
            model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)


def _extract_metric_series(
    log_history: List[Dict],
) -> Tuple[List[int], List[float], List[int], List[float], List[Dict[str, float]]]:
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    records: List[Dict[str, float]] = []
    for record in log_history:
        step = record.get("step") or record.get("global_step")
        if step is None:
            continue
        log_entry = {
            "step": step,
            "epoch": record.get("epoch"),
            "learning_rate": record.get("learning_rate"),
            "loss": record.get("loss"),
            "eval_loss": record.get("eval_loss"),
        }
        if "loss" in record:
            train_steps.append(step)
            train_losses.append(record["loss"])
        if "eval_loss" in record:
            eval_steps.append(step)
            eval_losses.append(record["eval_loss"])
        records.append(log_entry)
    return train_steps, train_losses, eval_steps, eval_losses, records


def save_metric_plots(log_history: List[Dict], run_dir: Path) -> None:
    train_steps, train_losses, eval_steps, eval_losses, records = _extract_metric_series(log_history)
    if not train_losses and not eval_losses:
        print("[train] No loss entries found in log history; skip plotting.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[train] matplotlib not installed; skipping metric plots.")
        return
    plt.figure(figsize=(8, 5))
    if train_losses:
        plt.plot(train_steps, train_losses, label="train_loss")
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label="eval_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = run_dir / "training_metrics.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"[train] Saved metric plot to {plot_path}")

    if records:
        csv_path = run_dir / "training_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as csv_fh:
            fieldnames = ["step", "epoch", "learning_rate", "loss", "eval_loss"]
            writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
            writer.writeheader()
            for entry in records:
                writer.writerow(entry)
        print(f"[train] Saved per-step metrics to {csv_path}")


def main() -> None:
    args = parse_args()
    dataset_dict = load_datasets(args)
    if "train" not in dataset_dict:
        raise ValueError("Training dataset is required.")
    if "eval" not in dataset_dict:
        split = maybe_split_dataset(dataset_dict["train"], args.eval_ratio, args.seed)
        dataset_dict["train"] = split["train"]
        if "eval" in split:
            dataset_dict["eval"] = split["eval"]

    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_dir) / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(batch: Dict) -> Dict:
        return format_example(batch, tokenizer, args.max_length)

    base_train_dataset = dataset_dict["train"]
    base_eval_dataset = dataset_dict.get("eval")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    target_agents = args.agents or []
    if not target_agents:
        target_agents = ["Chef", "Assistant"]

    base_training_kwargs = dict(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
    )

    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "save_strategy" in sig_params:
        base_training_kwargs["save_strategy"] = "epoch"

    trained_agents: List[str] = []
    for agent in target_agents:
        agent_train = filter_dataset_by_agent(base_train_dataset, agent)
        if agent_train is None or len(agent_train) == 0:
            print(f"[train] Skip agent {agent}: no training samples found.")
            continue
        agent_eval = filter_dataset_by_agent(base_eval_dataset, agent)
        if agent_eval is not None and len(agent_eval) == 0:
            agent_eval = None

        tokenized_train = agent_train.map(
            preprocess,
            remove_columns=agent_train.column_names,
            desc=f"Tokenizing train set ({agent})",
        )
        tokenized_eval = None
        if agent_eval is not None:
            tokenized_eval = agent_eval.map(
                preprocess,
                remove_columns=agent_eval.column_names,
                desc=f"Tokenizing eval set ({agent})",
            )

        agent_run_dir = run_dir / agent.replace(" ", "_")
        agent_run_dir.mkdir(parents=True, exist_ok=True)

        training_kwargs = dict(base_training_kwargs)
        training_kwargs["output_dir"] = str(agent_run_dir)
        if tokenized_eval is not None:
            if "evaluation_strategy" in sig_params:
                training_kwargs["evaluation_strategy"] = "epoch"
            elif "do_eval" in sig_params:
                training_kwargs["do_eval"] = True
        else:
            if "evaluation_strategy" in sig_params:
                training_kwargs["evaluation_strategy"] = "no"
            elif "do_eval" in sig_params:
                training_kwargs["do_eval"] = False

        model = build_model(args)
        training_args = TrainingArguments(**training_kwargs)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )
        if args.save_every_epoch:
            trainer.add_callback(EpochSaveCallback(agent_run_dir, tokenizer))
        trainer.train()
        trainer.save_model(str(agent_run_dir))
        tokenizer.save_pretrained(str(agent_run_dir))
        if args.plot_metrics:
            save_metric_plots(trainer.state.log_history, agent_run_dir)
        trained_agents.append(agent)
        del trainer
        del model
        try:
            import torch  # type: ignore

            torch.cuda.empty_cache()
        except Exception:
            pass

    if not trained_agents:
        raise RuntimeError("No agents were trained; please verify datasets and --agents selection.")
    print(f"[train] Finished SFT for agents: {', '.join(trained_agents)}. Results saved under {run_dir}.")


if __name__ == "__main__":
    main()
