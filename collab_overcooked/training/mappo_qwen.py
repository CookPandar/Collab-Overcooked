"""MAPPO trainer for Collab-Overcooked using the full LLM prompt pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model

from .main_session import CollabMainSession
from ..main import convert_yaml_to_variant


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Text generation actor-critic
# ---------------------------------------------------------------------------


@dataclass
class TextTransition:
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    log_prob: float
    value: float
    reward: float
    done: float
    agent_index: int
    entropy: float
    format_reward: float = 0.0
    validator_reward: float = 0.0
    process_reward: float = 0.0


class TextRolloutBuffer:
    def __init__(self):
        self.storage: List[TextTransition] = []

    def add(self, **kwargs):
        self.storage.append(TextTransition(**kwargs))

    def clear(self):
        self.storage.clear()


class LMGenerationResult:
    def __init__(
        self,
        text: str,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        log_prob: float,
        entropy: float,
        value: float,
    ) -> None:
        self.text = text
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.log_prob = log_prob
        self.entropy = entropy
        self.value = value


class QwenLMActorCritic(nn.Module):
    """Actor-critic built on top of HF causal LM (Think/Action generation)."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        eval_batch_size: int = 4,
        lora_cfg: Optional[Dict[str, Any]] = None,
        lora_path: Optional[str] = None,
        fix_mistral_regex: Optional[bool] = None,
    ) -> None:
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        tokenizer_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        # Mistral 系列需要修复分词 regex，否则会错码
        if fix_mistral_regex is True or (
            fix_mistral_regex is None and "mistral" in model_path.lower()
        ):
            tokenizer_kwargs["fix_mistral_regex"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": self.dtype,
        }
        # 有 flash-attn2 则启用，否则退回 sdpa，避免缺依赖时报错
        if torch.cuda.is_available():
            try:
                import flash_attn  # type: ignore

                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                model_kwargs["attn_implementation"] = "sdpa"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        # LoRA support: either load existing adapters or create new ones.
        self.is_lora = bool(lora_cfg) or bool(lora_path)
        if self.is_lora:
            if lora_path:
                self.model = PeftModel.from_pretrained(
                    self.model, lora_path, is_trainable=True
                )
            else:
                target_modules = lora_cfg.get(
                    "target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
                )
                lora_config = LoraConfig(
                    r=int(lora_cfg.get("r", 16)),
                    lora_alpha=int(lora_cfg.get("alpha", 32)),
                    lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=target_modules,
                )
                self.model = get_peft_model(self.model, lora_config)
            # 冻结基座权重，仅训练 LoRA + value head
            for name, param in self.model.named_parameters():
                if "lora_" not in name:
                    param.requires_grad = False

        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, device=device, dtype=self.dtype)
        # value head总是训练
        for p in self.value_head.parameters():
            p.requires_grad = True

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.to(device)

    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def act(self, prompt: str) -> LMGenerationResult:
        inputs = self._prepare_inputs(prompt)
        gen_out = self.model.generate(
            **inputs,
            do_sample=True,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
        )
        full_seq = gen_out.sequences[0]
        prompt_len = inputs["input_ids"].shape[1]
        generated = full_seq[prompt_len:]
        response_text = self.tokenizer.decode(
            generated, skip_special_tokens=True
        )

        log_probs = []
        entropies = []
        for step, logits in enumerate(gen_out.scores):
            probs = torch.log_softmax(logits[0], dim=-1)
            token_id = generated[step]
            log_probs.append(probs[token_id])
            dist = torch.distributions.Categorical(logits=logits[0])
            entropies.append(dist.entropy())
        total_log_prob = torch.stack(log_probs).sum().item() if log_probs else 0.0
        avg_entropy = (
            torch.stack(entropies).mean().item() if entropies else 0.0
        )

        seq = full_seq.unsqueeze(0)
        outputs = self.model(seq, use_cache=False, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :]
        value = self.value_head(hidden).squeeze(-1).item()

        return LMGenerationResult(
            response_text,
            prompt_ids=inputs["input_ids"].squeeze(0).cpu(),
            response_ids=generated.cpu(),
            log_prob=total_log_prob,
            entropy=avg_entropy,
            value=value,
        )

    def evaluate_batch(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
    ):
        device = self.device
        chunk_log_probs = []
        chunk_entropies = []
        chunk_values = []
        for start in range(0, len(prompt_tensors), self.eval_batch_size):
            end = start + self.eval_batch_size
            lp, ent, val = self._evaluate_chunk(
                prompt_tensors[start:end], response_tensors[start:end], device
            )
            chunk_log_probs.append(lp)
            chunk_entropies.append(ent)
            chunk_values.append(val)
        return (
            torch.cat(chunk_log_probs, dim=0),
            torch.cat(chunk_entropies, dim=0),
            torch.cat(chunk_values, dim=0),
        )

    def _evaluate_chunk(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        device: torch.device,
    ):
        pad_id = self.tokenizer.pad_token_id
        prompt_lengths = torch.tensor([len(t) for t in prompt_tensors], device=device)
        response_lengths = torch.tensor([len(t) for t in response_tensors], device=device)
        combined = [
            torch.cat([p.to(device), r.to(device)], dim=0)
            for p, r in zip(prompt_tensors, response_tensors)
        ]
        input_ids = pad_sequence(combined, batch_first=True, padding_value=pad_id)
        attention_mask = input_ids.ne(pad_id).long()
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        log_probs = []
        entropies = []
        values = []
        for i in range(input_ids.size(0)):
            p_len = int(prompt_lengths[i].item())
            r_len = int(response_lengths[i].item())
            if r_len == 0:
                log_probs.append(torch.tensor(0.0, device=device))
                entropies.append(torch.tensor(0.0, device=device))
                values.append(self.value_head(hidden_states[i : i + 1, -1, :]).squeeze(0))
                continue
            start = max(p_len - 1, 0)
            end = start + r_len
            token_logits = logits[i, start:end, :]
            token_ids = input_ids[i, p_len : p_len + r_len]
            logprob = torch.log_softmax(token_logits, dim=-1)
            gathered = logprob.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
            log_probs.append(gathered.sum())
            dists = torch.distributions.Categorical(logits=token_logits)
            entropies.append(dists.entropy().mean())
            last_index = p_len + r_len - 1
            value_vec = hidden_states[i, last_index, :].unsqueeze(0)
            values.append(self.value_head(value_vec).squeeze(0))
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)

    def forward(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
    ):
        """DDP forward pass delegates to evaluate_batch."""
        return self.evaluate_batch(prompt_tensors, response_tensors)


# ---------------------------------------------------------------------------
# MAPPO Trainer
# ---------------------------------------------------------------------------


class MAPPOTrainer:
    """MAPPO loop that fine-tunes the LLM used during evaluation."""

    def __init__(
        self,
        env_config: Dict[str, Any],
        trainer_cfg: Dict[str, Any],
        full_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        if full_config is None:
            raise ValueError("RL configs must provide the full YAML (with agents).")
        self.trainer_cfg = dict(trainer_cfg)
        env_latest_file = os.getenv("RL_LATEST_MODEL_FILE", "").strip()
        cfg_latest_file = self.trainer_cfg.get("latest_model_path_file", "")
        latest_file = cfg_latest_file or env_latest_file
        self.latest_model_path_file: Optional[Path] = Path(latest_file) if latest_file else None
        self.export_latest_dir: Optional[Path] = None
        self.export_interval = max(1, int(self.trainer_cfg.get("export_interval", 1)))
        self.merge_lora_after_train = bool(
            self.trainer_cfg.get(
                "merge_lora_after_train",
                bool(self.trainer_cfg.get("lora_path")) or bool(self.trainer_cfg.get("lora")),
            )
        )
        self.collect_only = bool(trainer_cfg.get("collect_only", False))
        self.train_only = bool(trainer_cfg.get("train_only", False))
        self.rollout_dir = Path(trainer_cfg.get("rollout_dir", "rollouts"))
        self.cleanup_rollouts = bool(trainer_cfg.get("cleanup_rollouts", False))

        self.gamma = trainer_cfg.get("gamma", 0.99)
        self.gae_lambda = trainer_cfg.get("gae_lambda", 0.95)
        self.steps_per_update = trainer_cfg.get("steps_per_update", 256)
        self.local_steps_per_update = max(
            1, math.ceil(self.steps_per_update / self.accelerator.num_processes)
        )
        self.train_batch_size = int(trainer_cfg.get("train_batch_size", 32))
        self.total_updates = trainer_cfg.get("total_updates", 1000)
        self.clip_coef = trainer_cfg.get("clip_coef", 0.2)
        self.entropy_coef = trainer_cfg.get("entropy_coef", 0.01)
        self.value_coef = trainer_cfg.get("value_coef", 0.5)
        self.max_grad_norm = trainer_cfg.get("max_grad_norm", 0.5)
        self.model_path = trainer_cfg["model_path"]
        self.lora_cfg = trainer_cfg.get("lora", {})
        self.lora_path = trainer_cfg.get("lora_path", None)

        if self.latest_model_path_file and self.latest_model_path_file.exists():
            self._apply_model_override_from_file(self.trainer_cfg)

        self.output_dir = Path(
            trainer_cfg.get(
                "output_dir",
                Path("results") / f"mappo_{env_config.get('order', 'task')}",
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.trainer_cfg.get("export_latest_dir"):
            self.export_latest_dir = Path(self.trainer_cfg["export_latest_dir"])
        elif self.export_latest_dir is None and (latest_file or self.merge_lora_after_train):
            # 默认导出到输出目录下的 export_latest，方便采样端重载
            self.export_latest_dir = self.output_dir / "export_latest"
        interval = trainer_cfg.get(
            "save_interval", trainer_cfg.get("checkpoint_interval", 0)
        )
        self.checkpoint_interval = int(interval) if interval else 0

        self.text_policy = QwenLMActorCritic(
            model_path=self.model_path,
            device=self.device,
            max_new_tokens=trainer_cfg.get("max_new_tokens", 512),
            temperature=trainer_cfg.get("generation_temperature", 0.7),
            eval_batch_size=trainer_cfg.get("evaluation_batch_size", 4),
            lora_cfg=self.lora_cfg,
            lora_path=self.lora_path,
            fix_mistral_regex=trainer_cfg.get("fix_mistral_regex", None),
        )
        self.policy_tokenizer = self.text_policy.tokenizer
        # 仅优化可训练参数（LoRA + value head）
        trainable_params = [p for p in self.text_policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=trainer_cfg.get("lr", 1e-5)
        )
        self.text_policy, self.optimizer = self.accelerator.prepare(
            self.text_policy, self.optimizer
        )
        variant = convert_yaml_to_variant(full_config)
        variant["yaml_config"] = full_config
        if not self.train_only:
            self.session = CollabMainSession(
                variant=variant,
                policy_fn=self._policy_call,
            )
        else:
            self.session = None
        self.buffer = TextRolloutBuffer()

    # ------------------------------------------------------------------
    def _apply_model_override_from_file(self, cfg: Dict[str, Any]) -> None:
        """Allow dynamic model_path / lora_path override via marker file."""
        if not self.latest_model_path_file:
            return
        try:
            content = self.latest_model_path_file.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if not content:
            return
        override = None
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                override = parsed
        except Exception:
            override = None
        if override:
            merged_path = override.get("merged_model_path")
            if self.collect_only and merged_path:
                cfg["model_path"] = merged_path
                cfg["lora_path"] = None
            else:
                if override.get("model_path"):
                    cfg["model_path"] = override["model_path"]
                if "lora_path" in override:
                    cfg["lora_path"] = override["lora_path"]
            return
        # fallback：纯字符串表示 model_path
        cfg["model_path"] = content
        cfg["lora_path"] = cfg.get("lora_path", None)

    # ------------------------------------------------------------------
    def _policy_call(self, agent_index: int, messages, context):
        assert self.text_policy is not None
        from ..agents.utils import convert_messages_to_prompt

        prompt_text = convert_messages_to_prompt(messages)
        tokenizer = self.policy_tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            chat_prompt = prompt_text
        model = self.accelerator.unwrap_model(self.text_policy)
        print(
            "[MAPPOTrainer] policy_call agent="
            f"{agent_index} prompt_chars={len(chat_prompt)}"
        )
        result = model.act(chat_prompt)
        print(
            "[MAPPOTrainer] policy_call agent="
            f"{agent_index} generated_tokens={len(result.response_ids)}"
        )
        metadata = {
            "prompt_ids": result.prompt_ids,
            "response_ids": result.response_ids,
            "log_prob": result.log_prob,
            "value": result.value,
            "entropy": result.entropy,
            "token_count": len(result.response_ids),
        }
        return result.text, metadata

    # ------------------------------------------------------------------
    def train(self):
        if self.collect_only or self.train_only:
            self.rollout_dir.mkdir(parents=True, exist_ok=True)

        # Collect-only mode: just roll out and save to disk.
        if self.collect_only:
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            for update_idx in range(1, self.total_updates + 1):
                self.collect_rollout()
                self.log_rewards(update_idx, self.buffer.storage)
                self.save_rollout(self.buffer.storage, update_idx)
                self.buffer.clear()
                self.accelerator.wait_for_everyone()
                self.accelerator.print(f"[Collect] saved rollout u{update_idx:05d}")
            return

        # Train-only mode: load rollouts from disk and update policy.
        if self.train_only:
            for update_idx in range(1, self.total_updates + 1):
                transitions = self.load_rollouts()
                if not transitions:
                    self.accelerator.print("[TrainOnly] No rollouts found; stopping.")
                    break
                loss_dict = self.update_policy(transitions)
                self.log_rewards(update_idx, transitions)
                if self.cleanup_rollouts and self.accelerator.is_main_process:
                    self._cleanup_rollout_files()
                self.accelerator.print(
                    f"[TrainOnly] Update {update_idx} "
                    f"loss={loss_dict['loss']:.4f} policy={loss_dict['policy']:.4f} "
                    f"value={loss_dict['value']:.4f} entropy={loss_dict['entropy']:.4f}"
                )
                self._maybe_export_latest(update_idx)
            return

        for update_idx in range(1, self.total_updates + 1):
            self.collect_rollout()
            self.log_rewards(update_idx, self.buffer.storage)
            loss_dict = self.update_policy(self.buffer.storage)

            self.accelerator.print(
                f"[MAPPO] Update {update_idx}/{self.total_updates} "
                f"loss={loss_dict['loss']:.4f} policy={loss_dict['policy']:.4f} "
                f"value={loss_dict['value']:.4f} entropy={loss_dict['entropy']:.4f}"
            )
            self._maybe_export_latest(update_idx)
            if self.checkpoint_interval and update_idx % self.checkpoint_interval == 0:
                self.save_checkpoint(tag=f"checkpoint_{update_idx:05d}")
        self.accelerator.wait_for_everyone()
        self.save_checkpoint(tag="final")

    # ------------------------------------------------------------------
    def collect_rollout(self):
        assert self.session is not None
        self.buffer.clear()
        step_count = 0
        local_target = self.local_steps_per_update
        while step_count < local_target:
            step_result = self.session.step()
            for record in step_result.policy_records:
                reward = getattr(record, "reward", step_result.reward)
                done = float(getattr(record, "done", step_result.done))
                meta = record.metadata
                breakdown = (meta or {}).get("reward_breakdown") or {}
                raw_entry = breakdown.get("raw") or {}
                fmt_reward = float(
                    breakdown.get("format_reward", raw_entry.get("format_reward", 0.0) or 0.0)
                )
                validator_reward = float(
                    breakdown.get("validator_reward", raw_entry.get("validator_reward", 0.0) or 0.0)
                )
                seq_reward = float(
                    breakdown.get("sequence_reward", raw_entry.get("sequence_reward", 0.0) or 0.0)
                )
                if "communication_reward" in breakdown:
                    process_reward = float(breakdown.get("communication_reward", 0.0) or 0.0)
                else:
                    process_reward = float(
                        raw_entry.get("total", seq_reward + fmt_reward + validator_reward)
                        or (seq_reward + fmt_reward + validator_reward)
                    )
                self.buffer.add(
                    prompt_ids=meta["prompt_ids"],
                    response_ids=meta["response_ids"],
                    log_prob=meta["log_prob"],
                    value=meta["value"],
                    reward=reward,
                    done=done,
                    agent_index=record.agent_index,
                    entropy=meta.get("entropy", 0.0),
                    format_reward=fmt_reward,
                    validator_reward=validator_reward,
                    process_reward=process_reward,
                )
                step_count += 1
            if step_result.done:
                self.session.reset()
            if not step_result.policy_records:
                step_count += 1

    # ------------------------------------------------------------------
    def compute_advantages(self, transitions: List[TextTransition]):
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        values = torch.tensor([t.value for t in transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            next_value = values[idx + 1] if idx + 1 < len(values) else 0.0
            delta = rewards[idx] + self.gamma * next_value * (1 - dones[idx]) - values[idx]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[idx]) * gae
            advantages[idx] = gae
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def update_policy(self, transitions: List[TextTransition]):
        assert self.text_policy is not None
        prompt_tensors = [t.prompt_ids for t in transitions]
        response_tensors = [t.response_ids for t in transitions]
        old_log_probs = torch.tensor(
            [t.log_prob for t in transitions], dtype=torch.float32, device=self.device
        )
        old_values = torch.tensor(
            [t.value for t in transitions], dtype=torch.float32, device=self.device
        )
        advantages, returns = self.compute_advantages(transitions)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        batch_size = max(1, self.train_batch_size)
        num_transitions = len(transitions)
        num_minibatches = math.ceil(num_transitions / batch_size)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0

        self.optimizer.zero_grad()
        for start in range(0, num_transitions, batch_size):
            end = min(start + batch_size, num_transitions)
            batch_prompts = prompt_tensors[start:end]
            batch_responses = response_tensors[start:end]
            batch_old_log_probs = old_log_probs[start:end]
            batch_adv = advantages[start:end]
            batch_returns = returns[start:end]

            log_probs, entropies, values = self.text_policy(batch_prompts, batch_responses)
            log_probs = log_probs.to(self.device, dtype=torch.float32)
            entropies = entropies.to(self.device, dtype=torch.float32)
            values = values.to(self.device, dtype=torch.float32)

            ratios = torch.exp(log_probs - batch_old_log_probs)
            surr1 = ratios * batch_adv
            surr2 = torch.clamp(ratios, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, batch_returns)
            entropy_loss = -entropies.mean()
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.accelerator.backward(loss / num_minibatches)

            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value += value_loss.item()
            total_entropy += entropies.mean().item()

        self.accelerator.clip_grad_norm_(self.text_policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        avg_loss = total_loss / num_minibatches if num_minibatches > 0 else 0.0
        avg_policy = total_policy / num_minibatches if num_minibatches > 0 else 0.0
        avg_value = total_value / num_minibatches if num_minibatches > 0 else 0.0
        avg_entropy = total_entropy / num_minibatches if num_minibatches > 0 else 0.0

        return {
            "loss": avg_loss,
            "policy": avg_policy,
            "value": avg_value,
            "entropy": avg_entropy,
        }

    def log_rewards(self, update_idx: int, transitions: List[TextTransition]):
        if not self.accelerator.is_main_process:
            return
        num = len(transitions)
        agent_stats: Dict[int, Dict[str, float]] = {}
        for t in transitions:
            idx = int(t.agent_index)
            stats = agent_stats.setdefault(
                idx,
                {"format": 0.0, "validator": 0.0, "process": 0.0},
            )
            stats["format"] += float(getattr(t, "format_reward", 0.0))
            stats["validator"] += float(getattr(t, "validator_reward", 0.0))
            stats["process"] += float(getattr(t, "process_reward", getattr(t, "reward", 0.0)))
        # 近似步数：以 steps_per_update 为步长
        step_est = (update_idx - 1) * self.steps_per_update + num
        log_path = self.output_dir / "reward_curve.csv"
        header = (
            "update_idx,step,num_transitions,"
            "agent0_format,agent0_validator,agent0_process,"
            "agent1_format,agent1_validator,agent1_process"
        )
        if not log_path.exists():
            log_path.write_text(header + "\n", encoding="utf-8")
        else:
            try:
                with log_path.open("r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
            except OSError:
                first_line = ""
            if first_line != header:
                backup_path = log_path.with_suffix(log_path.suffix + ".bak")
                try:
                    log_path.replace(backup_path)
                except OSError:
                    pass
                log_path.write_text(header + "\n", encoding="utf-8")
        a0_stats = agent_stats.get(0, {"format": 0.0, "validator": 0.0, "process": 0.0})
        a1_stats = agent_stats.get(1, {"format": 0.0, "validator": 0.0, "process": 0.0})
        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{update_idx},{step_est},{num},"
                f"{a0_stats['format']},{a0_stats['validator']},{a0_stats['process']},"
                f"{a1_stats['format']},{a1_stats['validator']},{a1_stats['process']}\n"
            )

    # ------------------------------------------------------------------
    def _maybe_export_latest(self, update_idx: int):
        if self.export_latest_dir is None:
            return
        if update_idx % self.export_interval != 0:
            return
        if not self.accelerator.is_main_process:
            return
        self._export_latest_model(update_idx)

    def _export_latest_model(self, update_idx: int):
        """Export当前权重到指定目录，并更新latest标记供采样端重载。"""
        assert self.text_policy is not None
        self.export_latest_dir.mkdir(parents=True, exist_ok=True)
        marker_path = (
            self.latest_model_path_file
            if self.latest_model_path_file
            else self.export_latest_dir / "latest_model.json"
        )
        model_dir = self.export_latest_dir / (
            "adapter_u" + f"{update_idx:05d}"
            if self.lora_path or self.lora_cfg
            else "model_u" + f"{update_idx:05d}"
        )
        merged_dir = self.export_latest_dir / f"merged_u{update_idx:05d}"

        unwrapped: QwenLMActorCritic = self.accelerator.unwrap_model(self.text_policy)
        hf_model = unwrapped.model
        tokenizer = unwrapped.tokenizer

        self.accelerator.print(f"[Export] Saving adapter/model to {model_dir}")
        hf_model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        payload: Dict[str, Any] = {
            "model_path": str(self.model_path),
            "lora_path": str(model_dir) if unwrapped.is_lora else None,
            "merged_model_path": None,
            "update_idx": update_idx,
        }

        if self.merge_lora_after_train and unwrapped.is_lora:
            self.accelerator.print(f"[Export] Merging LoRA into base -> {merged_dir}")
            merge_dtype = unwrapped.dtype if torch.cuda.is_available() else None
            device_map = "auto" if torch.cuda.is_available() else None
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=merge_dtype,
                device_map=device_map,
            )
            merged = PeftModel.from_pretrained(base_model, model_dir)
            merged = merged.merge_and_unload()
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            payload["merged_model_path"] = str(merged_dir)
            # 采样端优先用 merge 后的完整模型
            payload["model_path"] = str(merged_dir)
            payload["lora_path"] = None
            self._cleanup_old_merged_dirs(merged_dir)
        elif not unwrapped.is_lora:
            payload["model_path"] = str(model_dir)
            payload["lora_path"] = None

        marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.accelerator.print(f"[Export] latest model info -> {marker_path}")

    # ------------------------------------------------------------------
    def _cleanup_old_merged_dirs(self, keep_dir: Path):
        if not self.export_latest_dir:
            return
        keep_dir = keep_dir.resolve()
        for path in self.export_latest_dir.glob("merged_u*"):
            try:
                if path.resolve() != keep_dir and path.exists():
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    def _transition_to_dict(self, t: TextTransition):
        def _cpu(x):
            return x.cpu() if isinstance(x, torch.Tensor) else x

        return {
            "prompt_ids": _cpu(t.prompt_ids),
            "response_ids": _cpu(t.response_ids),
            "log_prob": t.log_prob,
            "value": t.value,
            "reward": t.reward,
            "done": t.done,
            "agent_index": t.agent_index,
            "entropy": t.entropy,
            "format_reward": t.format_reward,
            "validator_reward": t.validator_reward,
            "process_reward": t.process_reward,
        }

    def _dict_to_transition(self, d: Dict[str, Any]) -> TextTransition:
        return TextTransition(
            prompt_ids=d["prompt_ids"],
            response_ids=d["response_ids"],
            log_prob=float(d["log_prob"]),
            value=float(d["value"]),
            reward=float(d["reward"]),
            done=float(d["done"]),
            agent_index=int(d["agent_index"]),
            entropy=float(d["entropy"]),
            format_reward=float(d.get("format_reward", 0.0)),
            validator_reward=float(d.get("validator_reward", 0.0)),
            process_reward=float(d.get("process_reward", d.get("reward", 0.0))),
        )

    def save_rollout(self, transitions: List[TextTransition], update_idx: int):
        if not transitions:
            return
        path = self.rollout_dir / f"rollout_rank{self.accelerator.process_index}_u{update_idx:05d}.pt"
        payload = [self._transition_to_dict(t) for t in transitions]
        torch.save(payload, path)

    def load_rollouts(self) -> List[TextTransition]:
        self.accelerator.wait_for_everyone()
        files = sorted(self.rollout_dir.glob("rollout_rank*_u*.pt"))
        if not files:
            return []
        # 仅加载最新一轮（最大 u 值）的采样文件，避免历史数据无限累积。
        u_pattern = re.compile(r"_u(\d+)\.pt$")
        max_u = -1
        selected = []
        for p in files:
            m = u_pattern.search(p.name)
            if not m:
                continue
            u_idx = int(m.group(1))
            if u_idx > max_u:
                max_u = u_idx
                selected = [p]
            elif u_idx == max_u:
                selected.append(p)
        if max_u < 0:
            return []
        transitions: List[TextTransition] = []
        for p in sorted(selected):
            data = torch.load(p, map_location="cpu")
            transitions.extend([self._dict_to_transition(d) for d in data])
        return transitions

    def _cleanup_rollout_files(self):
        if not self.accelerator.is_main_process:
            return
        for p in self.rollout_dir.glob("rollout_rank*_u*.pt"):
            p.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str):
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(ckpt_dir)
        if self.accelerator.is_main_process:
            metadata = {
                "model_path": self.model_path,
                "tag": tag,
                "trainer": {
                    k: v
                    for k, v in self.trainer_cfg.items()
                    if k
                    not in {
                        "model_path",
                        "output_dir",
                        "save_interval",
                        "checkpoint_interval",
                    }
                },
            }
            meta_path = ckpt_dir / "metadata.json"
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


__all__ = ["MAPPOTrainer"]
