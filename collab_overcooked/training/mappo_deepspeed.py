"""DeepSpeed-enabled MAPPO trainer for Collab-Overcooked.

This module mirrors :mod:`collab_overcooked.training.mappo_qwen` but replaces the
Accelerate dependency with an explicit DeepSpeed engine so that the underlying
LLM can be sharded across multiple GPUs (ZeRO + tensor/pipeline parallel).
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

from .main_session import CollabMainSession
from ..main import convert_yaml_to_variant


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def init_distributed():
    """Initialize the default DeepSpeed/torch distributed process group."""
    if dist.is_available() and dist.is_initialized():
        return
    deepspeed.init_distributed()


def load_deepspeed_config(config: Any) -> Dict[str, Any]:
    """Load a DeepSpeed config from JSON / dict / path."""
    if isinstance(config, (str, Path)):
        path = Path(config)
        data = json.loads(path.read_text())
        data["_config_path"] = str(path)
        return data
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"Unsupported DeepSpeed config type: {type(config)}")


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def broadcast_object(obj: Any, src: int = 0) -> Any:
    """Broadcast an arbitrary picklable python object from ``src``."""
    if not dist.is_available() or not dist.is_initialized():
        return obj
    objs = [obj]
    dist.broadcast_object_list(objs, src=src)
    return objs[0]


def extract_parallel_sizes(config: Dict[str, Any], world_size: int) -> Tuple[int, int, int]:
    """Derive tensor / pipeline parallel sizes from DeepSpeed config."""
    tp_cfg = config.get("tensor_parallel", {}) or {}
    pp_cfg = config.get("pipeline", {}) or {}
    tp_size = int(tp_cfg.get("tp_size", 1) or 1)
    pp_size = int(pp_cfg.get("parallel_size", 1) or 1)
    total_mp = tp_size * pp_size
    if world_size and total_mp > world_size:
        raise ValueError(
            f"MP (tp={tp_size}, pp={pp_size}) exceeds world_size={world_size}"
        )
    if world_size and total_mp > 0 and world_size % total_mp != 0:
        raise ValueError(
            f"world_size {world_size} must be divisible by tp*pp={total_mp}"
        )
    dp_groups = world_size // total_mp if world_size else 1
    return tp_size, pp_size, dp_groups


# ---------------------------------------------------------------------------
# Dataclasses shared with MAPPO
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


# ---------------------------------------------------------------------------
# DeepSpeed actor-critic
# ---------------------------------------------------------------------------


class QwenActorCriticModule(nn.Module):
    """Simple container that keeps the HF causal LM and value head together."""

    def __init__(self, model_path: str, dtype: torch.dtype):
        super().__init__()
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=dtype,
        )
        hidden_size = self.policy_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, dtype=dtype)

    def forward(self, *args, **kwargs):  # type: ignore[override]
        return self.policy_model(*args, **kwargs)


class DeepSpeedQwenActorCritic:
    """LLM policy/value wrapper managed by DeepSpeed."""

    def __init__(
        self,
        model_path: str,
        ds_config: Any,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        eval_batch_size: int = 4,
    ) -> None:
        init_distributed()
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.ds_config = load_deepspeed_config(ds_config)
        self.tp_size, self.pp_size, self.dp_groups = extract_parallel_sizes(
            self.ds_config, self.world_size
        )
        self.dp_group_size = max(1, self.world_size // max(1, self.tp_size * self.pp_size))
        self.is_dp_leader = (self.rank % max(1, self.tp_size * self.pp_size)) == 0

        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        module = QwenActorCriticModule(model_path=model_path, dtype=self.dtype)
        self.engine, _, _, _ = deepspeed.initialize(
            model=module,
            model_parameters=module.parameters(),
            config=self.ds_config,
        )
        self.module = self.engine.module
        self.device = self.engine.device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.uses_zero3 = (
            self.ds_config.get("zero_optimization", {}).get("stage", 0) >= 3
        )

    # ------------------------------------------------------------------
    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    @contextmanager
    def _zero3_context(self):
        if not self.uses_zero3:
            yield
            return
        params = list(self.module.parameters())
        with deepspeed.zero.GatheredParameters(params, modifier_rank=self.rank):
            yield

    def act(self, prompt: str) -> LMGenerationResult:
        """Sample Think/Action text for a given prompt."""
        inputs = self._prepare_inputs(prompt)
        with torch.no_grad():
            with self._zero3_context():
                gen_out = self.module.policy_model.generate(
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
            dist_obj = torch.distributions.Categorical(logits=logits[0])
            entropies.append(dist_obj.entropy())
        total_log_prob = torch.stack(log_probs).sum().item() if log_probs else 0.0
        avg_entropy = (
            torch.stack(entropies).mean().item() if entropies else 0.0
        )

        seq = full_seq.unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.module.policy_model(
                seq,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-1][:, -1, :]
        value_scalar = self.module.value_head(hidden).squeeze(-1).item()

        # Only data-parallel leader decodes to text; broadcast to keep ranks in sync for TP/PP setups.
        payload = {
            "text": response_text if self.is_dp_leader else "",
            "prompt_ids": inputs["input_ids"].squeeze(0).cpu(),
            "response_ids": generated.cpu(),
            "log_prob": total_log_prob,
            "entropy": avg_entropy,
            "value": value_scalar,
        }
        payload = broadcast_object(payload, src=0)

        return LMGenerationResult(
            payload["text"],
            prompt_ids=payload["prompt_ids"],
            response_ids=payload["response_ids"],
            log_prob=payload["log_prob"],
            entropy=payload["entropy"],
            value=payload["value"],
        )

    # ------------------------------------------------------------------
    def evaluate_batch(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
    ):
        chunk_log_probs = []
        chunk_entropies = []
        chunk_values = []
        for start in range(0, len(prompt_tensors), self.eval_batch_size):
            end = start + self.eval_batch_size
            lp, ent, val = self._evaluate_chunk(
                prompt_tensors[start:end],
                response_tensors[start:end],
                self.device,
            )
            chunk_log_probs.append(lp)
            chunk_entropies.append(ent)
            chunk_values.append(val)
        return (
            torch.cat(chunk_log_probs, dim=0) if chunk_log_probs else torch.tensor([], device=self.device),
            torch.cat(chunk_entropies, dim=0) if chunk_entropies else torch.tensor([], device=self.device),
            torch.cat(chunk_values, dim=0) if chunk_values else torch.tensor([], device=self.device),
        )

    def _evaluate_chunk(
        self,
        prompt_tensors: Sequence[torch.Tensor],
        response_tensors: Sequence[torch.Tensor],
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
        with self._zero3_context():
            outputs = self.module.policy_model(
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
                values.append(self.module.value_head(hidden_states[i : i + 1, -1, :]).squeeze(0))
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
            values.append(self.module.value_head(value_vec).squeeze(0))
        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)

    def forward(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
    ):
        return self.evaluate_batch(prompt_tensors, response_tensors)


# ---------------------------------------------------------------------------
# MAPPO trainer (DeepSpeed)
# ---------------------------------------------------------------------------


class DeepSpeedMAPPOTrainer:
    """MAPPO loop that uses DeepSpeed to shard the policy model."""

    def __init__(
        self,
        env_config: Dict[str, Any],
        trainer_cfg: Dict[str, Any],
        full_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        init_distributed()
        if full_config is None:
            raise ValueError("RL configs must provide the full YAML (with agents).")

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.is_main = self.rank == 0

        self.trainer_cfg = dict(trainer_cfg)
        self.ds_config = trainer_cfg.get(
            "deepspeed_config", "configs/deepspeed/mappo_zero3.json"
        )
        self.collect_only = bool(trainer_cfg.get("collect_only", False))
        self.train_only = bool(trainer_cfg.get("train_only", False))
        self.rollout_dir = Path(trainer_cfg.get("rollout_dir", "rollouts"))
        self.cleanup_rollouts = bool(trainer_cfg.get("cleanup_rollouts", True))

        self.gamma = trainer_cfg.get("gamma", 0.99)
        self.gae_lambda = trainer_cfg.get("gae_lambda", 0.95)
        self.steps_per_update = trainer_cfg.get("steps_per_update", 256)
        horizon_cfg = env_config.get("horizon", 10)
        self.rollout_horizon = (
            int(horizon_cfg) if horizon_cfg is not None else None
        )
        self.local_steps_per_update = max(
            1, math.ceil(self.steps_per_update / max(1, self.world_size))
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
                Path("results") / f"dsmappo_{env_config.get('order', 'task')}",
            )
        )
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = int(
            trainer_cfg.get(
                "save_interval",
                trainer_cfg.get("checkpoint_interval", 0),
            )
            or 0
        )

        self.text_policy = DeepSpeedQwenActorCritic(
            model_path=self.model_path,
            ds_config=self.ds_config,
            max_new_tokens=trainer_cfg.get("max_new_tokens", 512),
            temperature=trainer_cfg.get("generation_temperature", 0.7),
            eval_batch_size=trainer_cfg.get("evaluation_batch_size", 4),
        )
        self.engine = self.text_policy.engine

        variant = convert_yaml_to_variant(full_config)
        variant["yaml_config"] = full_config
        self.session: Optional[CollabMainSession]
        if not self.train_only:
            self.session = CollabMainSession(
                variant=variant,
                policy_fn=self._policy_call,
            )
        else:
            self.session = None

        self.buffer = TextRolloutBuffer()

    # ------------------------------------------------------------------
    def _rollout_reached_horizon(self, step_result: SessionStep) -> bool:
        if self.rollout_horizon is None:
            return False
        timestep = None
        if isinstance(step_result.observation, dict):
            timestep = step_result.observation.get("timestep")
        if timestep is None:
            timestep = getattr(step_result, "timestep", None)
        if timestep is None:
            return False
        return int(timestep) >= self.rollout_horizon

    # ------------------------------------------------------------------
    def _policy_call(self, agent_index: int, messages, context):
        from ..agents.utils import convert_messages_to_prompt

        prompt_text = convert_messages_to_prompt(messages)
        tokenizer = self.text_policy.tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            chat_prompt = prompt_text
        result = self.text_policy.act(chat_prompt)
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
        if self.collect_only:
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            for update_idx in range(1, self.total_updates + 1):
                if self.is_main:
                    print(f"[Collect] update {update_idx}/{self.total_updates}")
                self.collect_rollout()
                self.save_rollout(self.buffer.storage, update_idx)
                self.buffer.clear()
            return

        if self.train_only:
            losses = []
            for update_idx in range(1, self.total_updates + 1):
                transitions = self.load_rollouts()
                if not transitions:
                    if self.is_main:
                        print("[TrainOnly] No rollouts found; stopping.")
                    break
                loss_dict = self.update_policy(transitions)
                losses.append(loss_dict)
                if self.cleanup_rollouts and self.is_main:
                    self._cleanup_rollout_files()
                if self.is_main:
                    print(
                        f"[TrainOnly] Update {update_idx} "
                        f"loss={loss_dict['loss']:.4f} policy={loss_dict['policy']:.4f} "
                        f"value={loss_dict['value']:.4f} entropy={loss_dict['entropy']:.4f}"
                    )
            return

        for update_idx in range(1, self.total_updates + 1):
            if self.is_main:
                self.collect_rollout()
            else:
                # Non-main ranks wait for rollout synchronization.
                self._sync_barrier()

            transitions = self._broadcast_rollout()
            loss_dict = self.update_policy(transitions)

            if self.is_main:
                print(
                    f"[DSMAPPo] Update {update_idx}/{self.total_updates} "
                    f"loss={loss_dict['loss']:.4f} policy={loss_dict['policy']:.4f} "
                    f"value={loss_dict['value']:.4f} entropy={loss_dict['entropy']:.4f}"
                )
                if self.checkpoint_interval and update_idx % self.checkpoint_interval == 0:
                    self.save_checkpoint(tag=f"checkpoint_{update_idx:05d}")
        self.save_checkpoint(tag="final")

    # ------------------------------------------------------------------
    def _sync_barrier(self):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _broadcast_rollout(self) -> List[TextTransition]:
        data = self.buffer.storage if self.is_main else None
        shared = broadcast_object(data, src=0)
        if shared is None:
            shared = []
        if not self.is_main:
            # Ensure followers reset their local buffer view.
            self.buffer.storage = list(shared)
        return list(shared)

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
            if self._rollout_reached_horizon(step_result):
                if not step_result.done:
                    self.session.reset()
                break
        self._sync_barrier()

    # ------------------------------------------------------------------
    def compute_advantages(self, transitions: Iterable[TextTransition]):
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.engine.device)
        values = torch.tensor([t.value for t in transitions], dtype=torch.float32, device=self.engine.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.engine.device)
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
        if not transitions:
            return {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0}
        prompt_tensors = [t.prompt_ids for t in transitions]
        response_tensors = [t.response_ids for t in transitions]
        old_log_probs = torch.tensor(
            [t.log_prob for t in transitions], dtype=torch.float32, device=self.engine.device
        )
        old_values = torch.tensor(
            [t.value for t in transitions], dtype=torch.float32, device=self.engine.device
        )
        advantages, returns = self.compute_advantages(transitions)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        log_probs, entropies, values = self.text_policy(prompt_tensors, response_tensors)
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.engine.backward(loss)
        if self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.engine.module.parameters(), self.max_grad_norm)
        self.engine.step()

        return {
            "loss": loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "entropy": entropies.mean().item(),
        }

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str):
        if not self.is_main:
            return
        ckpt_dir = self.output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.engine.save_checkpoint(str(ckpt_dir))
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
            "deepspeed_config": self.ds_config,
        }
        meta_path = ckpt_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
        )

    def save_rollout(self, transitions: List[TextTransition], update_idx: int):
        if not transitions:
            return
        path = self.rollout_dir / f"rollout_rank{self.rank}_u{update_idx:05d}.pt"
        payload = [self._transition_to_dict(t) for t in transitions]
        torch.save(payload, path)
        if self.is_main:
            print(f"[Collect] Saved {len(payload)} transitions to {path}")

    def load_rollouts(self) -> List[TextTransition]:
        files = sorted(self.rollout_dir.glob("rollout_rank*_u*.pt"))
        transitions: List[TextTransition] = []
        for p in files:
            data = torch.load(p, map_location="cpu")
            transitions.extend([self._dict_to_transition(d) for d in data])
        return transitions

    def _cleanup_rollout_files(self):
        for p in self.rollout_dir.glob("rollout_rank*_u*.pt"):
            p.unlink(missing_ok=True)


__all__ = ["DeepSpeedMAPPOTrainer"]
