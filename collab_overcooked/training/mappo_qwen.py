"""MAPPO trainer for Collab-Overcooked using the full LLM prompt pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer

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
    ) -> None:
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": self.dtype,
        }
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except Exception:
            model_kwargs.pop("attn_implementation", None)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        )
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, device=device, dtype=self.dtype)
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
        outputs = self.model(
            seq,
            output_hidden_states=True,
        )
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

        self.gamma = trainer_cfg.get("gamma", 0.99)
        self.gae_lambda = trainer_cfg.get("gae_lambda", 0.95)
        self.steps_per_update = trainer_cfg.get("steps_per_update", 256)
        self.local_steps_per_update = max(
            1, math.ceil(self.steps_per_update / self.accelerator.num_processes)
        )
        self.total_updates = trainer_cfg.get("total_updates", 1000)
        self.clip_coef = trainer_cfg.get("clip_coef", 0.2)
        self.entropy_coef = trainer_cfg.get("entropy_coef", 0.01)
        self.value_coef = trainer_cfg.get("value_coef", 0.5)
        self.max_grad_norm = trainer_cfg.get("max_grad_norm", 0.5)
        self.model_path = trainer_cfg["model_path"]

        self.output_dir = Path(
            trainer_cfg.get(
                "output_dir",
                Path("results") / f"mappo_{env_config.get('order', 'task')}",
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        )
        self.optimizer = torch.optim.AdamW(
            self.text_policy.parameters(), lr=trainer_cfg.get("lr", 1e-5)
        )
        self.text_policy, self.optimizer = self.accelerator.prepare(
            self.text_policy, self.optimizer
        )
        variant = convert_yaml_to_variant(full_config)
        variant["yaml_config"] = full_config
        self.session = CollabMainSession(
            variant=variant,
            policy_fn=self._policy_call,
        )
        self.buffer = TextRolloutBuffer()

    # ------------------------------------------------------------------
    def _policy_call(self, agent_index: int, messages, context):
        assert self.text_policy is not None
        from ..agents.utils import convert_messages_to_prompt

        prompt = convert_messages_to_prompt(messages)
        model = self.accelerator.unwrap_model(self.text_policy)
        print(
            "[MAPPOTrainer] policy_call agent="
            f"{agent_index} prompt_chars={len(prompt)}"
        )
        result = model.act(prompt)
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
        for update_idx in range(1, self.total_updates + 1):
            self.collect_rollout()
            loss_dict = self.update_policy()

            self.accelerator.print(
                f"[MAPPO] Update {update_idx}/{self.total_updates} "
                f"loss={loss_dict['loss']:.4f} policy={loss_dict['policy']:.4f} "
                f"value={loss_dict['value']:.4f} entropy={loss_dict['entropy']:.4f}"
            )
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
                self.buffer.add(
                    prompt_ids=meta["prompt_ids"],
                    response_ids=meta["response_ids"],
                    log_prob=meta["log_prob"],
                    value=meta["value"],
                    reward=reward,
                    done=done,
                    agent_index=record.agent_index,
                    entropy=meta.get("entropy", 0.0),
                )
                step_count += 1
            if step_result.done:
                self.session.reset()
            if not step_result.policy_records:
                step_count += 1

    # ------------------------------------------------------------------
    def compute_advantages(self):
        rewards = torch.tensor([t.reward for t in self.buffer.storage], dtype=torch.float32)
        values = torch.tensor([t.value for t in self.buffer.storage], dtype=torch.float32)
        dones = torch.tensor([t.done for t in self.buffer.storage], dtype=torch.float32)
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
    def update_policy(self):
        assert self.text_policy is not None
        prompt_tensors = [t.prompt_ids for t in self.buffer.storage]
        response_tensors = [t.response_ids for t in self.buffer.storage]
        old_log_probs = torch.tensor(
            [t.log_prob for t in self.buffer.storage], dtype=torch.float32, device=self.device
        )
        old_values = torch.tensor(
            [t.value for t in self.buffer.storage], dtype=torch.float32, device=self.device
        )
        advantages, returns = self.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        log_probs, entropies, values = self.text_policy(prompt_tensors, response_tensors)
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        self.accelerator.clip_grad_norm_(self.text_policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "entropy": entropies.mean().item(),
        }

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
