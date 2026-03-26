"""MAPPO trainer for Collab-Overcooked using the full LLM prompt pipeline."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
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
    timestep: Optional[int] = None
    critic_input_ids: Optional[torch.Tensor] = None
    format_reward: float = 0.0
    validator_reward: float = 0.0
    process_reward: float = 0.0
    # Reward breakdown (optional; depends on env/validator providing reward_breakdown)
    # NOTE: `reward` is still the scalar used for RL updates; these fields are only for logging/analysis.
    sequence_reward: float = 0.0
    communication_reward: float = 0.0
    breakdown_total_reward: float = 0.0


@dataclass
class AdapterSpec:
    name: str
    lora_path: Optional[str] = None
    lora_config: Optional[Dict[str, Any]] = None


@dataclass
class SnapshotRecord:
    snapshot: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        critic_input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self.text = text
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.log_prob = log_prob
        self.entropy = entropy
        self.value = value
        self.critic_input_ids = critic_input_ids


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
        actor_adapters: Optional[Dict[int, AdapterSpec]] = None,
        critic_adapter: Optional[AdapterSpec] = None,
        fix_mistral_regex: Optional[bool] = None,
    ) -> None:
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.actor_adapters = actor_adapters or {}
        self.actor_adapter_names = {
            idx: spec.name for idx, spec in self.actor_adapters.items()
        }
        self.critic_adapter = critic_adapter
        self.critic_adapter_name = critic_adapter.name if critic_adapter else None
        self.default_lora_cfg = lora_cfg or {}
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
        self.is_lora = False
        if self.actor_adapters or self.critic_adapter:
            self._initialize_multi_adapters(lora_path=lora_path)
        else:
            self.is_lora = bool(self.default_lora_cfg) or bool(lora_path)
            if self.is_lora:
                if lora_path:
                    self.model = PeftModel.from_pretrained(
                        self.model, lora_path, is_trainable=True
                    )
                else:
                    lora_config = self._build_lora_config(self.default_lora_cfg)
                    self.model = get_peft_model(self.model, lora_config)
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

    def _build_lora_config(self, override: Optional[Dict[str, Any]] = None) -> LoraConfig:
        cfg = dict(self.default_lora_cfg or {})
        if override:
            cfg.update(override)
        target_modules = cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],
        )
        return LoraConfig(
            r=int(cfg.get("r", 16)),
            lora_alpha=int(cfg.get("alpha", 32)),
            lora_dropout=float(cfg.get("dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    def _initialize_multi_adapters(self, lora_path: Optional[str] = None):
        specs: List[AdapterSpec] = [
            self.actor_adapters[idx]
            for idx in sorted(self.actor_adapters.keys())
        ]
        extra: List[AdapterSpec] = []
        if self.critic_adapter:
            extra.append(self.critic_adapter)
        queue = [spec for spec in specs + extra if spec is not None]
        if not queue:
            return
        primary = next((spec for spec in queue if spec.lora_path), None)
        if primary is None and lora_path:
            queue[0].lora_path = lora_path
            primary = queue[0]
        if primary is None:
            primary = queue[0]
        self.model = self._attach_adapter(primary, self.model)
        loaded = {primary.name}
        for spec in queue:
            if spec.name in loaded:
                continue
            self._attach_adapter(spec, self.model)
            loaded.add(spec.name)
        self.is_lora = True
        for name, param in self.model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False

    def _attach_adapter(self, spec: AdapterSpec, base_model):
        if spec.lora_path:
            lora_path = Path(spec.lora_path)
            if lora_path.is_dir() and not (lora_path / "adapter_config.json").exists():
                candidate = lora_path / spec.name
                if (candidate / "adapter_config.json").exists():
                    lora_path = candidate
            if isinstance(base_model, PeftModel):
                base_model.load_adapter(
                    str(lora_path),
                    adapter_name=spec.name,
                    is_trainable=True,
                )
            else:
                base_model = PeftModel.from_pretrained(
                    base_model,
                    str(lora_path),
                    adapter_name=spec.name,
                    is_trainable=True,
                )
        else:
            lora_config = self._build_lora_config(spec.lora_config)
            if isinstance(base_model, PeftModel):
                base_model.add_adapter(spec.name, lora_config)
            else:
                base_model = get_peft_model(
                    base_model,
                    lora_config,
                    adapter_name=spec.name,
                )
        base_model.set_adapter(spec.name)
        return base_model

    def _use_adapter(self, adapter_name: Optional[str]):
        if not self.is_lora or not isinstance(self.model, PeftModel):
            return nullcontext()
        if adapter_name:
            self.model.set_adapter(adapter_name)
            return nullcontext()
        return self.model.disable_adapter()

    def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _use_critic_adapter(self):
        if self.critic_adapter_name:
            return self._use_adapter(self.critic_adapter_name)
        if self.is_lora and isinstance(self.model, PeftModel):
            return self.model.disable_adapter()
        return nullcontext()

    def _evaluate_value_inputs(
        self, input_tensors: List[torch.Tensor], use_critic_adapter: bool = True
    ) -> torch.Tensor:
        if not input_tensors:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        pad_id = self.tokenizer.pad_token_id
        inputs = pad_sequence(input_tensors, batch_first=True, padding_value=pad_id)
        attention_mask = inputs.ne(pad_id).long()
        context = self._use_critic_adapter() if use_critic_adapter else nullcontext()
        with context:
            outputs = self.model(
                inputs,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
        hidden_states = outputs.hidden_states[-1]
        lengths = attention_mask.sum(dim=-1).clamp(min=1) - 1
        gather_index = lengths.view(-1, 1, 1).expand(-1, 1, hidden_states.size(-1))
        last_hidden = hidden_states.gather(dim=1, index=gather_index).squeeze(1)
        return self.value_head(last_hidden).squeeze(-1)

    @torch.no_grad()
    def evaluate_text_value(self, prompt: str) -> Tuple[torch.Tensor, float]:
        inputs = self._prepare_inputs(prompt)
        input_ids = inputs["input_ids"].squeeze(0).cpu()
        value = self._evaluate_value_inputs(
            [input_ids.to(self.device)], use_critic_adapter=True
        )[0].item()
        return input_ids, value

    @torch.no_grad()
    def act(
        self,
        prompt: str,
        agent_index: int,
        critic_prompt: Optional[str] = None,
    ) -> LMGenerationResult:
        adapter_name = self.actor_adapter_names.get(agent_index)
        critic_input_ids = None
        with self._use_adapter(adapter_name):
            inputs = self._prepare_inputs(prompt)
            # Transformers requires `temperature` to be strictly positive whenever it is set.
            # For deterministic/greedy decoding, set `do_sample=False` and do not pass
            # temperature at all.
            do_sample = self.temperature is not None and float(self.temperature) > 0.0
            generate_kwargs: Dict[str, Any] = {
                **inputs,
                "do_sample": bool(do_sample),
                "max_new_tokens": self.max_new_tokens,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            if do_sample:
                generate_kwargs["temperature"] = float(self.temperature)
            gen_out = self.model.generate(**generate_kwargs)
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

        prompt_ids = inputs["input_ids"].squeeze(0).cpu()
        response_ids = generated.cpu()
        value = 0.0
        if critic_prompt:
            critic_inputs = self._prepare_inputs(critic_prompt)
            critic_input_ids = critic_inputs["input_ids"].squeeze(0).cpu()
            value = self._evaluate_value_inputs(
                [critic_input_ids.to(self.device)], use_critic_adapter=True
            )[0].item()
        else:
            seq = full_seq.unsqueeze(0)
            outputs = self.model(seq, use_cache=False, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, -1, :]
            value = self.value_head(hidden).squeeze(-1).item()
            if self.critic_adapter_name and self.critic_adapter_name != adapter_name:
                value = self._evaluate_value_single(prompt_ids, response_ids)

        return LMGenerationResult(
            response_text,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            log_prob=total_log_prob,
            entropy=avg_entropy,
            value=value,
            critic_input_ids=critic_input_ids,
        )

    def _evaluate_value_single(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> float:
        prompt = prompt_ids.to(self.device)
        response = response_ids.to(self.device)
        with self._use_adapter(self.critic_adapter_name):
            _, _, values = self._evaluate_chunk(
                [prompt],
                [response],
                need_policy=False,
                need_value=True,
            )
        return values[0].item()

    def evaluate_batch(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        agent_indices: List[int],
        critic_tensors: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        num_samples = len(prompt_tensors)
        prompt_device = [t.to(self.device) for t in prompt_tensors]
        response_device = [t.to(self.device) for t in response_tensors]
        critic_device = (
            [t.to(self.device) if t is not None else None for t in critic_tensors]
            if critic_tensors is not None
            else [None] * num_samples
        )
        log_probs_buf: List[Optional[torch.Tensor]] = [None] * num_samples
        entropy_buf: List[Optional[torch.Tensor]] = [None] * num_samples
        values_buf: List[Optional[torch.Tensor]] = [None] * num_samples

        groups: Dict[Optional[str], List[int]] = {}
        if self.actor_adapters:
            for idx, agent_idx in enumerate(agent_indices):
                adapter_name = self.actor_adapter_names.get(agent_idx)
                groups.setdefault(adapter_name, []).append(idx)
        else:
            groups = {None: list(range(num_samples))}

        critic_indices = [idx for idx, tensor in enumerate(critic_device) if tensor is not None]
        if critic_indices:
            critic_values = self._evaluate_value_inputs(
                [critic_device[idx] for idx in critic_indices if critic_device[idx] is not None],
                use_critic_adapter=True,
            )
            for local_idx, global_idx in enumerate(critic_indices):
                values_buf[global_idx] = critic_values[local_idx]
        missing_value_indices = [
            idx for idx, value in enumerate(values_buf) if value is None
        ]

        critic_matches_actor = False
        if self.critic_adapter_name and self.actor_adapter_names:
            critic_matches_actor = all(
                self.actor_adapter_names.get(agent_idx) == self.critic_adapter_name
                for agent_idx in agent_indices
            )
        need_actor_values = bool(missing_value_indices) and (
            self.critic_adapter_name is None or critic_matches_actor
        )
        for adapter_name, sample_indices in groups.items():
            subset_prompts = [prompt_device[i] for i in sample_indices]
            subset_responses = [response_device[i] for i in sample_indices]
            with self._use_adapter(adapter_name):
                logp, ent, vals = self._evaluate_chunk(
                    subset_prompts,
                    subset_responses,
                    need_policy=True,
                    need_value=need_actor_values,
                )
            for local_idx, global_idx in enumerate(sample_indices):
                log_probs_buf[global_idx] = logp[local_idx]
                entropy_buf[global_idx] = ent[local_idx]
                if need_actor_values and global_idx in missing_value_indices:
                    values_buf[global_idx] = vals[local_idx]

        if self.critic_adapter_name and not critic_matches_actor and missing_value_indices:
            with self._use_adapter(self.critic_adapter_name):
                _, _, critic_vals = self._evaluate_chunk(
                    [prompt_device[idx] for idx in missing_value_indices],
                    [response_device[idx] for idx in missing_value_indices],
                    need_policy=False,
                    need_value=True,
                )
            for local_idx, global_idx in enumerate(missing_value_indices):
                values_buf[global_idx] = critic_vals[local_idx]

        return (
            torch.stack([t for t in log_probs_buf if t is not None]),
            torch.stack([t for t in entropy_buf if t is not None]),
            torch.stack([t for t in values_buf if t is not None]),
        )

    def _evaluate_chunk(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        need_policy: bool = True,
        need_value: bool = True,
    ):
        device = self.device
        pad_id = self.tokenizer.pad_token_id
        prompt_lengths = torch.tensor([len(t) for t in prompt_tensors], device=device)
        response_lengths = torch.tensor([len(t) for t in response_tensors], device=device)
        combined = [
            torch.cat([p, r], dim=0)
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
        logits = outputs.logits if need_policy else None
        hidden_states = outputs.hidden_states[-1]
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        for i in range(input_ids.size(0)):
            p_len = int(prompt_lengths[i].item())
            r_len = int(response_lengths[i].item())
            if need_policy:
                if r_len == 0:
                    log_probs.append(torch.tensor(0.0, device=device))
                    entropies.append(torch.tensor(0.0, device=device))
                else:
                    start = max(p_len - 1, 0)
                    end = start + r_len
                    token_logits = logits[i, start:end, :]
                    token_ids = input_ids[i, p_len : p_len + r_len]
                    logprob = torch.log_softmax(token_logits, dim=-1)
                    gathered = logprob.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
                    log_probs.append(gathered.sum())
                    dists = torch.distributions.Categorical(logits=token_logits)
                    entropies.append(dists.entropy().mean())
            if need_value:
                last_index = max(p_len + r_len - 1, 0)
                value_vec = hidden_states[i, last_index, :].unsqueeze(0)
                values.append(self.value_head(value_vec).squeeze(0))
        lp_tensor = torch.stack(log_probs) if need_policy else None
        ent_tensor = torch.stack(entropies) if need_policy else None
        val_tensor = torch.stack(values) if need_value else None
        return lp_tensor, ent_tensor, val_tensor

    def forward(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        agent_indices: Optional[List[int]] = None,
        critic_tensors: Optional[List[Optional[torch.Tensor]]] = None,
    ):
        """DDP forward pass delegates to evaluate_batch."""
        if agent_indices is None:
            agent_indices = [0] * len(prompt_tensors)
        return self.evaluate_batch(
            prompt_tensors,
            response_tensors,
            agent_indices,
            critic_tensors=critic_tensors,
        )


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
        if full_config is None:
            raise ValueError("RL configs must provide the full YAML (with agents).")
        self.trainer_cfg = dict(trainer_cfg)

        # Multi-adapter (multi-agent) LoRA setups naturally lead to some trainable adapter
        # parameters being unused on a given rank/iteration (e.g., if a rank only sees
        # agent0 samples in that step). Enable unused parameter detection in DDP to
        # prevent "Expected to have finished reduction..." errors.
        ddp_find_unused_default = bool(self.trainer_cfg.get("actor_adapters")) or bool(
            self.trainer_cfg.get("critic_adapter")
        )
        ddp_find_unused = bool(
            self.trainer_cfg.get(
                "ddp_find_unused_parameters", ddp_find_unused_default
            )
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.device = self.accelerator.device

        env_latest_file = os.getenv("RL_LATEST_MODEL_FILE", "").strip()
        cfg_latest_file = self.trainer_cfg.get("latest_model_path_file", "")
        latest_file = cfg_latest_file or env_latest_file
        self.latest_model_path_file: Optional[Path] = Path(latest_file) if latest_file else None
        self.export_latest_dir: Optional[Path] = None
        self.export_interval = max(1, int(self.trainer_cfg.get("export_interval", 1)))
        self.collect_only = bool(trainer_cfg.get("collect_only", False))
        self.train_only = bool(trainer_cfg.get("train_only", False))
        self.rollout_dir = Path(trainer_cfg.get("rollout_dir", "rollouts"))
        self.cleanup_rollouts = bool(trainer_cfg.get("cleanup_rollouts", True))
        self.agents_cfg = full_config.get("agents", {})
        self.agent_roles = self._extract_agent_roles(self.agents_cfg)
        self.critic_role_prompt = self._load_critic_role_prompt()

        if self.latest_model_path_file and self.latest_model_path_file.exists():
            self._apply_model_override_from_file(self.trainer_cfg)

        self.actor_adapters: Dict[int, AdapterSpec] = self._build_actor_specs(
            self.trainer_cfg, self.agents_cfg
        )
        self.critic_adapter: Optional[AdapterSpec] = self._build_critic_spec(self.trainer_cfg)
        cap_value = int(trainer_cfg.get("max_records_per_step", 0))
        self.max_records_per_step: Optional[int] = cap_value if cap_value > 0 else None
        self._record_cap_warned = False
        snapshot_paths = trainer_cfg.get("off_policy_snapshots") or trainer_cfg.get("snapshot_files") or []
        if isinstance(snapshot_paths, (str, Path)):
            snapshot_paths = [snapshot_paths]
        elif not isinstance(snapshot_paths, list):
            snapshot_paths = []
        self.snapshot_records: List[SnapshotRecord] = (
            self._load_snapshot_dataset(snapshot_paths) if snapshot_paths else []
        )
        self.snapshot_cycle = bool(trainer_cfg.get("snapshot_cycle", True))
        self._snapshot_cursor = 0

        self.gamma = float(trainer_cfg.get("gamma", 0.99))
        # Optional: discount within a single env timestep across multiple LLM calls,
        # while preventing the number of calls in a timestep from amplifying discount
        # across timesteps (see `discount_reset_per_timestep`).
        self.gamma_call = float(trainer_cfg.get("gamma_call", self.gamma))
        self.discount_reset_per_timestep = bool(
            trainer_cfg.get("discount_reset_per_timestep", False)
        )
        self.gae_lambda = float(trainer_cfg.get("gae_lambda", 0.95))
        self.steps_per_update = trainer_cfg.get("steps_per_update", 256)
        horizon_cfg = env_config.get("horizon", 10)
        self.rollout_horizon = (
            int(horizon_cfg) if horizon_cfg is not None else None
        )
        self.local_steps_per_update = max(
            1, math.ceil(self.steps_per_update / self.accelerator.num_processes)
        )
        self.train_batch_size = int(trainer_cfg.get("train_batch_size", 32))
        self.update_epochs = int(trainer_cfg.get("update_epochs", 4))
        self.clip_coef = trainer_cfg.get("clip_coef", 0.2)
        self.value_clip_coef = trainer_cfg.get(
            "value_clip_coef", trainer_cfg.get("cliprange_value", self.clip_coef)
        )
        self.shuffle_minibatches = bool(trainer_cfg.get("shuffle_minibatches", True))
        self.gradient_accumulation_steps = max(
            1, int(trainer_cfg.get("gradient_accumulation_steps", 8))
        )
        target_kl_cfg = trainer_cfg.get("target_kl", None)
        self.target_kl = float(target_kl_cfg) if target_kl_cfg is not None else None
        self.entropy_coef = trainer_cfg.get("entropy_coef", 0.01)
        self.value_coef = trainer_cfg.get("value_coef", 0.5)
        self.max_grad_norm = trainer_cfg.get("max_grad_norm", 0.5)
        self.model_path = trainer_cfg["model_path"]
        self.lora_cfg = trainer_cfg.get("lora", {})
        self.lora_path = trainer_cfg.get("lora_path", None)

        self.output_dir = Path(
            trainer_cfg.get(
                "output_dir",
                Path("results") / f"mappo_{env_config.get('order', 'task')}",
            )
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_rollout_cache_updates = int(
            trainer_cfg.get("initial_rollout_cache_updates", 3)
        )
        self.reuse_initial_rollout_cache = bool(
            trainer_cfg.get("reuse_initial_rollout_cache", True)
        )
        cache_dir_cfg = trainer_cfg.get("initial_rollout_cache_dir")
        self.initial_rollout_cache_dir = (
            Path(cache_dir_cfg) if cache_dir_cfg else self.output_dir / "initial_rollout_cache"
        )
        if self.initial_rollout_cache_updates > 0:
            self.initial_rollout_cache_dir.mkdir(parents=True, exist_ok=True)
        if self.trainer_cfg.get("export_latest_dir"):
            self.export_latest_dir = Path(self.trainer_cfg["export_latest_dir"])
        elif self.export_latest_dir is None and latest_file:
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
            actor_adapters=self.actor_adapters,
            critic_adapter=self.critic_adapter,
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
        self._last_rollout_stats: Dict[str, Any] = {}
        self._current_update_idx: Optional[int] = None
        self._episode_counter = 0
        self._episode_log_path: Optional[Path] = None

        if self.output_dir:
            self._episode_log_path = (
                self.output_dir
                / f"episode_return_curve_rank{self.accelerator.process_index}.csv"
            )

    def _reset_rollout_stats(self):
        self._last_rollout_stats = {
            "env_steps": 0,
            "episodes_completed": 0,
            "success_episodes": 0,
            "env_reward_sum": 0.0,
            "positive_reward_steps": 0,
            "policy_calls": 0,
            "episode_return_sum": 0.0,
            "episode_lengths_sum": 0,
            "agent0_custom_return_sum": 0.0,
            "agent1_custom_return_sum": 0.0,
            "team_custom_return_sum": 0.0,
        }
        self._current_episode_return = 0.0
        self._current_episode_length = 0
        self._current_episode_has_positive = False
        self._current_episode_custom_stats = {
            "agent0_total": 0.0,
            "agent0_sequence": 0.0,
            "agent0_format": 0.0,
            "agent0_validator": 0.0,
            "agent0_comm": 0.0,
            "agent1_total": 0.0,
            "agent1_sequence": 0.0,
            "agent1_format": 0.0,
            "agent1_validator": 0.0,
            "agent1_comm": 0.0,
            "team_total": 0.0,
        }

    def _prepare_csv_log(self, path: Path, header: str) -> Tuple[int, Optional[Dict[str, str]]]:
        """Ensure the CSV header exists and return the next monotonic row index."""
        last_row: Optional[Dict[str, str]] = None
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(header + "\n", encoding="utf-8")
            return 1, None

        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            lines = []

        if not lines or lines[0] != header:
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                path.replace(backup_path)
            except OSError:
                pass
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(header + "\n", encoding="utf-8")
            return 1, None

        if len(lines) > 1:
            columns = header.split(",")
            values = lines[-1].split(",")
            if len(values) == len(columns):
                last_row = dict(zip(columns, values))
        return len(lines), last_row

    def _log_episode_return(
        self,
        episode_return: float,
        episode_len: int,
        had_positive: bool,
        custom_stats: Dict[str, float],
    ):
        if not self._episode_log_path:
            return
        header = (
            "row_idx,update_idx,episode_idx,episode_return,episode_len,had_positive,rank,"
            "agent0_custom_return,agent0_sequence_sum,agent0_format_sum,agent0_validator_sum,agent0_comm_sum,"
            "agent1_custom_return,agent1_sequence_sum,agent1_format_sum,agent1_validator_sum,agent1_comm_sum,"
            "team_custom_return"
        )
        row_idx, _ = self._prepare_csv_log(self._episode_log_path, header)
        update_idx = self._current_update_idx if self._current_update_idx is not None else -1
        rank = self.accelerator.process_index
        self._episode_counter += 1
        with self._episode_log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{row_idx},{update_idx},{self._episode_counter},{episode_return},"
                f"{episode_len},{1 if had_positive else 0},{rank},"
                f"{custom_stats.get('agent0_total', 0.0)},{custom_stats.get('agent0_sequence', 0.0)},"
                f"{custom_stats.get('agent0_format', 0.0)},{custom_stats.get('agent0_validator', 0.0)},"
                f"{custom_stats.get('agent0_comm', 0.0)},"
                f"{custom_stats.get('agent1_total', 0.0)},{custom_stats.get('agent1_sequence', 0.0)},"
                f"{custom_stats.get('agent1_format', 0.0)},{custom_stats.get('agent1_validator', 0.0)},"
                f"{custom_stats.get('agent1_comm', 0.0)},{custom_stats.get('team_total', 0.0)}\n"
            )

    def _extract_step_custom_reward_stats(
        self, process_reward: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        stats = {
            "agent0_total": 0.0,
            "agent0_sequence": 0.0,
            "agent0_format": 0.0,
            "agent0_validator": 0.0,
            "agent0_comm": 0.0,
            "agent1_total": 0.0,
            "agent1_sequence": 0.0,
            "agent1_format": 0.0,
            "agent1_validator": 0.0,
            "agent1_comm": 0.0,
            "team_total": 0.0,
        }
        if not process_reward or not isinstance(process_reward, dict):
            return stats
        per_agent = process_reward.get("per_agent") or []
        for agent_idx in range(min(len(per_agent), 2)):
            reward_entry = per_agent[agent_idx]
            if not isinstance(reward_entry, dict):
                continue
            calls = reward_entry.get("calls") or []
            seq = sum(float(call.get("sequence_reward", 0.0) or 0.0) for call in calls)
            fmt = sum(float(call.get("format_reward", 0.0) or 0.0) for call in calls)
            validator = sum(float(call.get("validator_reward", 0.0) or 0.0) for call in calls)
            comm = sum(float(call.get("communication_reward", 0.0) or 0.0) for call in calls)
            total = seq + fmt + validator + comm
            prefix = f"agent{agent_idx}"
            stats[f"{prefix}_total"] = total
            stats[f"{prefix}_sequence"] = seq
            stats[f"{prefix}_format"] = fmt
            stats[f"{prefix}_validator"] = validator
            stats[f"{prefix}_comm"] = comm
            stats["team_total"] += total
        return stats

    def _update_rollout_stats(
        self,
        step_reward: float,
        done: bool,
        policy_calls: int,
        process_reward: Optional[Dict[str, Any]] = None,
    ):
        stats = self._last_rollout_stats
        stats["env_steps"] += 1
        stats["env_reward_sum"] += float(step_reward)
        stats["policy_calls"] += int(policy_calls)
        if float(step_reward) > 0:
            stats["positive_reward_steps"] += 1
            self._current_episode_has_positive = True
        self._current_episode_return += float(step_reward)
        self._current_episode_length += 1
        step_custom_stats = self._extract_step_custom_reward_stats(process_reward)
        for key, value in step_custom_stats.items():
            self._current_episode_custom_stats[key] += float(value)
        if done:
            stats["episodes_completed"] += 1
            if self._current_episode_has_positive:
                stats["success_episodes"] += 1
            stats["episode_return_sum"] += float(self._current_episode_return)
            stats["episode_lengths_sum"] += int(self._current_episode_length)
            stats["agent0_custom_return_sum"] += float(
                self._current_episode_custom_stats["agent0_total"]
            )
            stats["agent1_custom_return_sum"] += float(
                self._current_episode_custom_stats["agent1_total"]
            )
            stats["team_custom_return_sum"] += float(
                self._current_episode_custom_stats["team_total"]
            )
            self._log_episode_return(
                episode_return=self._current_episode_return,
                episode_len=self._current_episode_length,
                had_positive=self._current_episode_has_positive,
                custom_stats=dict(self._current_episode_custom_stats),
            )
            self._current_episode_return = 0.0
            self._current_episode_length = 0
            self._current_episode_has_positive = False
            for key in list(self._current_episode_custom_stats.keys()):
                self._current_episode_custom_stats[key] = 0.0

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

    def _extract_agent_roles(self, agents_cfg: Dict[str, Any]) -> Dict[int, str]:
        roles: Dict[int, str] = {}
        if not isinstance(agents_cfg, dict):
            return roles
        for key, cfg in agents_cfg.items():
            if not isinstance(cfg, dict) or not key.startswith("agent_"):
                continue
            try:
                idx = int(key.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            roles[idx] = cfg.get("role") or key
        return roles

    def _resolve_agent_index(self, key: Any) -> Optional[int]:
        if isinstance(key, int):
            return key
        if not isinstance(key, str):
            return None
        if key.startswith("agent_"):
            try:
                return int(key.split("_", 1)[1])
            except (ValueError, IndexError):
                return None
        for idx, role in self.agent_roles.items():
            if role and key.lower() == role.lower():
                return idx
        return None

    def _build_actor_specs(
        self, trainer_cfg: Dict[str, Any], agents_cfg: Dict[str, Any]
    ) -> Dict[int, AdapterSpec]:
        specs: Dict[int, AdapterSpec] = {}
        cfg_value = trainer_cfg.get("actor_adapters") or {}
        if isinstance(cfg_value, dict):
            for key, value in cfg_value.items():
                if not isinstance(value, dict):
                    continue
                idx = self._resolve_agent_index(key)
                if idx is None:
                    continue
                adapter_name = (
                    value.get("adapter_name")
                    or value.get("name")
                    or self.agent_roles.get(idx)
                    or f"agent_{idx}"
                )
                specs[idx] = AdapterSpec(
                    name=adapter_name,
                    lora_path=value.get("lora_path"),
                    lora_config=value.get("lora_config"),
                )
        if specs:
            return specs
        if not isinstance(agents_cfg, dict):
            return specs
        for key, cfg in agents_cfg.items():
            if not isinstance(cfg, dict) or not key.startswith("agent_"):
                continue
            try:
                idx = int(key.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            path = cfg.get("rl_lora_path") or cfg.get("lora_path")
            if not path:
                continue
            specs[idx] = AdapterSpec(
                name=cfg.get("adapter_name") or cfg.get("role") or key,
                lora_path=path,
                lora_config=None,
            )
        return specs

    def _build_critic_spec(self, trainer_cfg: Dict[str, Any]) -> Optional[AdapterSpec]:
        critic_cfg = trainer_cfg.get("critic_adapter")
        if isinstance(critic_cfg, dict):
            return AdapterSpec(
                name=critic_cfg.get("adapter_name", "critic"),
                lora_path=critic_cfg.get("lora_path"),
                lora_config=critic_cfg.get("lora_config"),
            )
        critic_path = trainer_cfg.get("critic_lora_path")
        if critic_path:
            return AdapterSpec(name="critic", lora_path=critic_path, lora_config=None)
        return None

    def _load_critic_role_prompt(self) -> str:
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "prompts"
            / "critic"
            / "centralized_critic_role.txt"
        )
        try:
            content = prompt_path.read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        if content:
            return content
        return (
            "You are a centralized critic. You do not generate a new action. "
            "You estimate how good the current transition is for future cumulative return "
            "using the global state, both agents' context, and the current output."
        )

    def _prompt_root(self) -> Path:
        return Path(__file__).resolve().parents[1] / "prompts" / "gpt"

    def _read_prompt_file(self, filename: str) -> str:
        path = self._prompt_root() / filename
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _recipe_access_text(self, role_name: str) -> str:
        lowered = (role_name or "").lower()
        if "chef" in lowered:
            return "You have access to the recipe."
        return "You do not have direct recipe access and may need to ask the chef."

    def _format_rule_template(self, template: str, role_name: str, teammate_name: str) -> str:
        content = template or ""
        replacements = {
            "{role}": role_name,
            "{teammate}": teammate_name,
            "{has_recipe}": self._recipe_access_text(role_name),
            "{recipe}": "[Recipe omitted in critic input]",
            "{skill}": "[Skill section omitted in this template]",
            "{communication_rule}": "[Communication rule omitted in this template]",
        }
        for key, value in replacements.items():
            content = content.replace(key, value)
        return content.strip()

    def _build_game_rule_block(self) -> str:
        env_rule = self._format_rule_template(
            self._read_prompt_file("environment_rule.txt"),
            "Chef",
            "Assistant",
        )
        return self._compact_text(env_rule, limit=2600) if env_rule else ""

    def _extract_action_space(self, role_name: str) -> str:
        skill_file = "chef_skill.txt" if "chef" in role_name.lower() else "assistant_skill.txt"
        raw = self._read_prompt_file(skill_file)
        if not raw:
            return ""
        content = raw
        start = content.find("**'Operation actions'**:")
        if start != -1:
            content = content[start:]
        return self._compact_text(content.strip(), limit=2400)

    @staticmethod
    def _compact_text(text: Optional[str], limit: int = 800) -> str:
        content = (text or "").strip()
        if len(content) <= limit:
            return content
        return content[: limit - 3].rstrip() + "..."

    def _agent_label(self, agent_index: int) -> str:
        return self.agent_roles.get(agent_index) or f"agent_{agent_index}"

    def _format_global_observation(self, observation: Optional[Dict[str, Any]]) -> str:
        if not isinstance(observation, dict):
            return "[MISSING]"
        players = observation.get("players") or {}
        lines = [
            f"timestep: {observation.get('timestep')}",
            f"orders: {observation.get('orders')}",
            f"grid: {self._compact_text(str(observation.get('grid', '')), limit=600)}",
            f"pot_states: {self._compact_text(str(observation.get('pot_states', {})), limit=400)}",
            f"counters: {self._compact_text(str(observation.get('counters', {})), limit=400)}",
        ]
        for idx in (0, 1):
            player = players.get(idx) or players.get(str(idx)) or {}
            lines.append(
                f"player_{idx}: pos={player.get('position')} orient={player.get('orientation')} "
                f"obj={player.get('object')}"
            )
        return "\n".join(lines)

    def _build_critic_prompt(
        self,
        agent_index: int,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
        output_text: str,
    ) -> str:
        shared_traces = context.get("shared_agent_traces") or {}
        current_user = next(
            (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
            "",
        )
        role_name = self._agent_label(agent_index)
        teammate_index = 1 - int(agent_index)
        teammate_name = self._agent_label(teammate_index)
        teammate_trace = shared_traces.get(teammate_index) or {}
        game_rules = self._build_game_rule_block()
        acting_actions = self._extract_action_space(role_name)
        teammate_actions = self._extract_action_space(teammate_name)
        sections = [
            self.critic_role_prompt,
            "",
            "[Global Observation]",
            self._format_global_observation(context.get("global_observation")),
            "",
            "[Game Rules]",
            game_rules or "[MISSING]",
            "",
            f"[{role_name} Action Space]",
            acting_actions or "[MISSING]",
            "",
            f"[{teammate_name} Action Space]",
            teammate_actions or "[MISSING]",
            "",
            "[Current Transition]",
            f"acting_agent: {role_name} (index={agent_index})",
            f"call_type: {context.get('call_type')}",
            "acting_agent_observation_excerpt:\n"
            + self._compact_text(context.get("observation_excerpt") or current_user, limit=1400),
            "acting_agent_output:\n" + self._compact_text(output_text, limit=1200),
            "",
            "[Teammate Recent Context]",
            f"teammate_agent: {teammate_name} (index={teammate_index})",
            f"teammate_recent_call_type: {teammate_trace.get('call_type', '[MISSING]')}",
            "teammate_recent_observation_excerpt:\n"
            + self._compact_text(
                teammate_trace.get("observation_excerpt", "[MISSING]"), limit=1000
            ),
            "teammate_recent_output:\n"
            + self._compact_text(
                teammate_trace.get("response_excerpt", "[MISSING]"), limit=1000
            ),
            "",
            "[Evaluation Focus]",
            "Estimate the value of this transition for future cumulative return. "
            "Focus on task progress, coordination quality, rule compliance, repeated communication, "
            "format errors, validator errors, and whether the output is an effective embodied action "
            "or an effective communication move under the stated constraints.",
        ]
        return "\n".join(sections)

    def _load_snapshot_dataset(self, paths: List[Union[str, Path]]) -> List[SnapshotRecord]:
        dataset: List[SnapshotRecord] = []
        for raw in paths:
            path = Path(raw).expanduser()
            if path.is_dir():
                candidates = sorted(path.glob("*.jsonl"))
                if not candidates:
                    candidates = sorted(path.glob("*.json"))
                if not candidates:
                    candidates = sorted(path.glob("*.pt"))
                for candidate in candidates:
                    dataset.extend(self._read_snapshot_file(candidate))
            else:
                dataset.extend(self._read_snapshot_file(path))
        if dataset and self.accelerator.is_main_process:
            self.accelerator.print(
                f"[MAPPO] Loaded {len(dataset)} snapshot states from {len(paths)} path(s)."
            )
        return dataset

    def _read_snapshot_file(self, path: Path) -> List[SnapshotRecord]:
        records: List[SnapshotRecord] = []
        if not path.exists():
            return records
        suffix = path.suffix.lower()
        try:
            if suffix in {".jsonl"}:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        data = json.loads(line)
                        snapshot = data.get("snapshot") or data.get("state")
                        if snapshot:
                            records.append(
                                SnapshotRecord(
                                    snapshot=snapshot,
                                    metadata=data.get("metadata", {}),
                                )
                            )
            elif suffix in {".json"}:
                with path.open("r", encoding="utf-8") as handle:
                    content = json.load(handle)
                if isinstance(content, dict):
                    snapshot = content.get("snapshot") or content.get("state")
                    if snapshot:
                        records.append(
                            SnapshotRecord(
                                snapshot=snapshot,
                                metadata=content.get("metadata", {}),
                            )
                        )
                elif isinstance(content, list):
                    for entry in content:
                        if not isinstance(entry, dict):
                            continue
                        snapshot = entry.get("snapshot") or entry.get("state")
                        if snapshot:
                            records.append(
                                SnapshotRecord(
                                    snapshot=snapshot,
                                    metadata=entry.get("metadata", {}),
                                )
                            )
            elif suffix == ".pt":
                data = torch.load(path, map_location="cpu")
                if isinstance(data, list):
                    for entry in data:
                        if not isinstance(entry, dict):
                            continue
                        snapshot = entry.get("snapshot") or entry.get("state")
                        if snapshot:
                            records.append(
                                SnapshotRecord(
                                    snapshot=snapshot,
                                    metadata=entry.get("metadata", {}),
                                )
                            )
                elif isinstance(data, dict):
                    snapshot = data.get("snapshot") or data.get("state")
                    if snapshot:
                        records.append(
                            SnapshotRecord(
                                snapshot=snapshot,
                                metadata=data.get("metadata", {}),
                            )
                        )
            else:
                return records
        except Exception as exc:
            if self.accelerator.is_main_process:
                self.accelerator.print(f"[MAPPO] Failed to read snapshot file {path}: {exc}")
        return records

    def _next_snapshot_record(self) -> Optional[SnapshotRecord]:
        if not self.snapshot_records:
            return None
        if self._snapshot_cursor >= len(self.snapshot_records):
            if not self.snapshot_cycle:
                return None
            self._snapshot_cursor = 0
        record = self.snapshot_records[self._snapshot_cursor]
        self._snapshot_cursor += 1
        return record

    # ------------------------------------------------------------------
    def _apply_model_override_from_file(self, cfg: Dict[str, Any]) -> None:
        """Allow dynamic model_path / lora_path override via marker file.

        The marker file may contain either a legacy single-model payload:
          {"model_path": "...", "lora_path": "..."}

        Or a multi-adapter payload (for 2-agent setups) such as:
          {
            "model_path": "...",
            "actor_adapters": {"0": {"adapter_name": "Chef", "lora_path": "..."}, ...},
            "critic_adapter": {"adapter_name": "critic", "lora_path": "..."}
          }
        """
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
            if override.get("model_path"):
                cfg["model_path"] = override["model_path"]
            lora_path = override.get("lora_path")
            if lora_path:
                cfg["lora_path"] = lora_path

            # Multi-adapter override: patch per-agent adapter paths when present.
            actor_override = override.get("actor_adapters")
            if isinstance(actor_override, dict) and isinstance(cfg.get("actor_adapters"), dict):
                for key, value in actor_override.items():
                    if not isinstance(value, dict):
                        continue
                    idx = self._resolve_agent_index(key)
                    if idx is None:
                        # Also accept pure numeric-string keys ("0", "1", ...)
                        try:
                            idx = int(str(key))
                        except (TypeError, ValueError):
                            idx = None
                    if idx is None:
                        continue
                    agent_key = f"agent_{idx}"
                    if agent_key not in cfg["actor_adapters"]:
                        continue
                    lora_path = value.get("lora_path")
                    if lora_path:
                        cfg["actor_adapters"][agent_key]["lora_path"] = lora_path

            critic_override = override.get("critic_adapter")
            if isinstance(critic_override, dict) and isinstance(cfg.get("critic_adapter"), dict):
                lora_path = critic_override.get("lora_path")
                if lora_path:
                    cfg["critic_adapter"]["lora_path"] = lora_path
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
        result = model.act(chat_prompt, agent_index=agent_index)
        critic_prompt = self._build_critic_prompt(
            agent_index=agent_index,
            messages=messages,
            context=context,
            output_text=result.text,
        )
        critic_input_ids, critic_value = model.evaluate_text_value(critic_prompt)
        print(
            "[MAPPOTrainer] policy_call agent="
            f"{agent_index} generated_tokens={len(result.response_ids)}"
        )
        metadata = {
            "prompt_ids": result.prompt_ids,
            "response_ids": result.response_ids,
            "log_prob": result.log_prob,
            "value": critic_value,
            "entropy": result.entropy,
            "token_count": len(result.response_ids),
            "critic_input_ids": critic_input_ids,
        }
        return result.text, metadata

    # ------------------------------------------------------------------
    def train(self):
        if self.collect_only or self.train_only:
            self.rollout_dir.mkdir(parents=True, exist_ok=True)

        # Collect-only mode: just roll out and save to disk.
        if self.collect_only:
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            update_idx = 1
            self._current_update_idx = update_idx
            self._reset_rollout_stats()
            reused_cache = False
            latest_exists = (
                self.latest_model_path_file is not None
                and self.latest_model_path_file.exists()
            )
            # In 3-stage mode each collect process starts with local update_idx=1.
            # Reuse the initial cache only before the first train/export has produced
            # a latest-model marker; later rounds must collect fresh on-policy data.
            if not latest_exists:
                reused_cache = self._maybe_load_initial_cached_rollout(update_idx)
            if not reused_cache:
                self.collect_rollout()
                self._maybe_save_initial_cached_rollout(update_idx, self.buffer.storage)
                self.log_performance(update_idx)
            else:
                self.accelerator.print(
                    f"[Collect] Reusing cached initial rollout for update {update_idx}."
                )
            self.log_rewards(update_idx, self.buffer.storage)
            self.save_rollout(self.buffer.storage, update_idx)
            self.buffer.clear()
            self.accelerator.wait_for_everyone()
            self.accelerator.print(f"[Collect] saved rollout u{update_idx:05d}")
            return

        # Train-only mode: load rollouts from disk and update policy.
        if self.train_only:
            update_idx = 1
            transitions = self.load_rollouts()
            if not transitions:
                self.accelerator.print("[TrainOnly] No rollouts found; stopping.")
                return
            loss_dict = self.update_policy(transitions)
            self.log_rewards(update_idx, transitions)
            self.log_train_metrics(update_idx, loss_dict, transitions)
            if self.cleanup_rollouts and self.accelerator.is_main_process:
                self._cleanup_rollout_files()
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx} "
                f"loss={loss_dict['loss']:.4f} policy={loss_dict['policy']:.4f} "
                f"value={loss_dict['value']:.4f} entropy={loss_dict['entropy']:.4f}"
            )
            self._maybe_export_latest(update_idx)
            return

        update_idx = 1
        self._current_update_idx = update_idx
        self._reset_rollout_stats()
        reused_cache = self._maybe_load_initial_cached_rollout(update_idx)
        if not reused_cache:
            self.collect_rollout()
            self._maybe_save_initial_cached_rollout(update_idx, self.buffer.storage)
            self.log_performance(update_idx)
        else:
            self.accelerator.print(
                f"[MAPPO] Reusing cached initial rollout for update {update_idx}."
            )
        self.log_rewards(update_idx, self.buffer.storage)
        loss_dict = self.update_policy(self.buffer.storage)
        self.log_train_metrics(update_idx, loss_dict, self.buffer.storage)

        self.accelerator.print(
            f"[MAPPO] Update {update_idx} "
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
        if self.snapshot_records:
            self._collect_snapshot_rollout()
        else:
            self._collect_env_rollout()

    def _append_policy_records(
        self,
        records: List[Any],
        default_reward: float,
        done_flag: bool,
    ) -> int:
        added = 0
        for record in records:
            reward = getattr(record, "reward", default_reward)
            done = float(getattr(record, "done", done_flag))
            meta = record.metadata or {}
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
            communication_reward = float(breakdown.get("communication_reward", 0.0) or 0.0)
            breakdown_total_reward = float(
                raw_entry.get(
                    "total",
                    seq_reward + fmt_reward + validator_reward + communication_reward,
                )
                or (seq_reward + fmt_reward + validator_reward + communication_reward)
            )
            process_reward = seq_reward
            self.buffer.add(
                prompt_ids=meta["prompt_ids"],
                response_ids=meta["response_ids"],
                log_prob=meta["log_prob"],
                value=meta["value"],
                reward=reward,
                done=done,
                agent_index=record.agent_index,
                entropy=meta.get("entropy", 0.0),
                timestep=getattr(record, "timestep", None),
                critic_input_ids=meta.get("critic_input_ids"),
                format_reward=fmt_reward,
                validator_reward=validator_reward,
                process_reward=process_reward,
                sequence_reward=seq_reward,
                communication_reward=communication_reward,
                breakdown_total_reward=breakdown_total_reward,
            )
            added += 1
        return added

    def _collect_env_rollout(self):
        if self.session is None:
            raise RuntimeError("Environment session not initialized.")
        self.buffer.clear()
        step_count = 0
        local_target = self.local_steps_per_update
        while step_count < local_target:
            step_result = self.session.step()
            records = step_result.policy_records or []
            self._update_rollout_stats(
                step_reward=step_result.reward,
                done=bool(step_result.done),
                policy_calls=len(records),
                process_reward=step_result.process_reward,
            )
            if (
                self.max_records_per_step
                and len(records) > self.max_records_per_step
            ):
                if not self._record_cap_warned and self.accelerator.is_main_process:
                    self.accelerator.print(
                        f"[MAPPO] policy_records per step超过{self.max_records_per_step}，将被截断。"
                    )
                    self._record_cap_warned = True
                records = records[: self.max_records_per_step]
            added = self._append_policy_records(
                records, step_result.reward, step_result.done
            )
            if step_result.done:
                self.session.reset()
            if added == 0:
                step_count += 1
            else:
                step_count += added
            if self._rollout_reached_horizon(step_result):
                if not step_result.done:
                    self.session.reset()
                break

    def _collect_snapshot_rollout(self):
        if self.session is None:
            raise RuntimeError("Environment session not initialized.")
        if not self.snapshot_records:
            raise RuntimeError("Snapshot dataset is empty.")
        self.buffer.clear()
        step_count = 0
        local_target = self.local_steps_per_update
        while step_count < local_target:
            snapshot_record = self._next_snapshot_record()
            if snapshot_record is None:
                if self.accelerator.is_main_process:
                    self.accelerator.print("[MAPPO] Snapshot dataset exhausted.")
                break
            try:
                self.session.load_snapshot(snapshot_record.snapshot)
            except Exception as exc:
                if self.accelerator.is_main_process:
                    self.accelerator.print(f"[MAPPO] Failed to load snapshot: {exc}")
                continue
            if self.rollout_horizon is not None:
                snapshot_ts = getattr(self.session.env.state, "timestep", None)
                if snapshot_ts is not None and int(snapshot_ts) >= self.rollout_horizon:
                    break
            if self.session.env.is_done():
                continue
            step_result = self.session.step()
            records = step_result.policy_records or []
            self._update_rollout_stats(
                step_reward=step_result.reward,
                done=bool(step_result.done),
                policy_calls=len(records),
                process_reward=step_result.process_reward,
            )
            if (
                self.max_records_per_step
                and len(records) > self.max_records_per_step
            ):
                if not self._record_cap_warned and self.accelerator.is_main_process:
                    self.accelerator.print(
                        f"[MAPPO] policy_records per step超过{self.max_records_per_step}，将被截断。"
                    )
                    self._record_cap_warned = True
                records = records[: self.max_records_per_step]
            added = self._append_policy_records(
                records, step_result.reward, step_result.done
            )
            if added == 0:
                step_count += 1
            else:
                step_count += added
            if self._rollout_reached_horizon(step_result):
                break

    # ------------------------------------------------------------------
    def compute_advantages(self, transitions: List[TextTransition]):
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
        values = torch.tensor([t.value for t in transitions], dtype=torch.float32)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32)
        advantages = torch.zeros_like(rewards)

        def _compute_agent_advantages(agent_positions: List[int]) -> None:
            if not agent_positions:
                return

            agent_timesteps: List[int] = [
                int(transitions[pos].timestep)
                if transitions[pos].timestep is not None
                else -1
                for pos in agent_positions
            ]
            gammas: List[float] = []
            for local_idx in range(len(agent_positions)):
                if not self.discount_reset_per_timestep:
                    gammas.append(self.gamma)
                    continue
                if local_idx + 1 >= len(agent_positions):
                    gammas.append(self.gamma)
                elif agent_timesteps[local_idx] == agent_timesteps[local_idx + 1]:
                    gammas.append(self.gamma_call)
                else:
                    gammas.append(self.gamma)

            gae = 0.0
            for local_idx in reversed(range(len(agent_positions))):
                pos = agent_positions[local_idx]
                gamma_t = float(gammas[local_idx])
                if local_idx + 1 < len(agent_positions):
                    next_pos = agent_positions[local_idx + 1]
                    next_value = values[next_pos]
                else:
                    next_value = 0.0
                delta = rewards[pos] + gamma_t * next_value * (1 - dones[pos]) - values[pos]
                gae = delta + gamma_t * self.gae_lambda * (1 - dones[pos]) * gae
                advantages[pos] = gae

        agent_positions: Dict[int, List[int]] = {}
        for idx, transition in enumerate(transitions):
            agent_positions.setdefault(int(transition.agent_index), []).append(idx)
        for positions in agent_positions.values():
            _compute_agent_advantages(positions)

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    def update_policy(self, transitions: List[TextTransition]):
        assert self.text_policy is not None
        prompt_tensors = [t.prompt_ids for t in transitions]
        response_tensors = [t.response_ids for t in transitions]
        critic_tensors = [t.critic_input_ids for t in transitions]
        agent_indices = [t.agent_index for t in transitions]
        old_log_probs = torch.tensor(
            [t.log_prob for t in transitions], dtype=torch.float32, device=self.device
        )
        old_values = torch.tensor(
            [t.value for t in transitions], dtype=torch.float32, device=self.device
        )
        advantages, returns = self.compute_advantages(transitions)
        raw_advantages = advantages.clone()
        returns_stats = returns.clone()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        batch_size = max(1, self.train_batch_size)
        num_transitions = len(transitions)
        num_minibatches = math.ceil(num_transitions / batch_size)
        total_optimization_steps = max(1, self.update_epochs * num_minibatches)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_approx_kl = 0.0
        total_value_clipfrac = 0.0
        metric_steps = 0
        actual_optimization_steps = 0
        accum_counter = 0
        stop_early = False

        self.optimizer.zero_grad()
        for _ in range(self.update_epochs):
            if self.shuffle_minibatches and num_transitions > 1:
                perm = torch.randperm(num_transitions)
                ordered_indices = perm.tolist()
            else:
                ordered_indices = list(range(num_transitions))

            for start in range(0, num_transitions, batch_size):
                end = min(start + batch_size, num_transitions)
                batch_indices = ordered_indices[start:end]
                batch_prompts = [prompt_tensors[i] for i in batch_indices]
                batch_responses = [response_tensors[i] for i in batch_indices]
                batch_critic = [critic_tensors[i] for i in batch_indices]
                batch_agent_indices = [agent_indices[i] for i in batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_adv = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                log_probs, entropies, values = self.text_policy(
                    batch_prompts,
                    batch_responses,
                    batch_agent_indices,
                    critic_tensors=batch_critic,
                )
                log_probs = log_probs.to(self.device, dtype=torch.float32)
                entropies = entropies.to(self.device, dtype=torch.float32)
                values = values.to(self.device, dtype=torch.float32)

                ratios = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratios * batch_adv
                clipped_ratios = torch.clamp(
                    ratios, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                surr2 = clipped_ratios * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_pred_clipped = torch.clamp(
                    values,
                    batch_old_values - self.value_clip_coef,
                    batch_old_values + self.value_clip_coef,
                )
                value_losses = (values - batch_returns) ** 2
                value_losses_clipped = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                entropy_loss = -entropies.mean()
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.accelerator.backward(
                    loss / float(self.gradient_accumulation_steps)
                )
                accum_counter += 1
                should_step = (
                    accum_counter >= self.gradient_accumulation_steps
                    or end >= num_transitions
                )
                if should_step:
                    self.accelerator.clip_grad_norm_(
                        self.text_policy.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    actual_optimization_steps += 1
                    accum_counter = 0

                total_loss += loss.item()
                total_policy += policy_loss.item()
                total_value += value_loss.item()
                total_entropy += entropies.mean().item()
                total_clipfrac += (
                    ((ratios - 1.0).abs() > self.clip_coef).float().mean().item()
                )
                total_approx_kl += (batch_old_log_probs - log_probs).mean().item()
                total_value_clipfrac += (
                    (value_losses_clipped > value_losses).float().mean().item()
                )
                metric_steps += 1
                if (
                    should_step
                    and self.target_kl is not None
                    and abs((batch_old_log_probs - log_probs).mean().item()) > self.target_kl
                ):
                    stop_early = True
                    break
            if stop_early:
                break

        metric_denom = metric_steps if metric_steps > 0 else 1
        avg_loss = total_loss / metric_denom if metric_denom > 0 else 0.0
        avg_policy = total_policy / metric_denom if metric_denom > 0 else 0.0
        avg_value = total_value / metric_denom if metric_denom > 0 else 0.0
        avg_entropy = total_entropy / metric_denom if metric_denom > 0 else 0.0
        avg_clipfrac = total_clipfrac / metric_denom if metric_denom > 0 else 0.0
        avg_approx_kl = total_approx_kl / metric_denom if metric_denom > 0 else 0.0
        avg_value_clipfrac = (
            total_value_clipfrac / metric_denom if metric_denom > 0 else 0.0
        )
        reward_mean = (
            float(sum(float(t.reward) for t in transitions)) / float(len(transitions))
            if transitions
            else 0.0
        )
        old_values_cpu = old_values.detach().cpu()
        returns_cpu = returns_stats.detach().cpu()
        returns_var = float(torch.var(returns_cpu, unbiased=False).item())
        if returns_var > 1e-8:
            explained_var = 1.0 - float(
                torch.var(returns_cpu - old_values_cpu, unbiased=False).item()
            ) / returns_var
        else:
            explained_var = 0.0

        return {
            "loss": avg_loss,
            "policy": avg_policy,
            "value": avg_value,
            "entropy": avg_entropy,
            "reward_mean": reward_mean,
            "adv_mean": float(raw_advantages.mean().item()),
            "return_mean": float(returns_cpu.mean().item()),
            "value_mean": float(old_values_cpu.mean().item()),
            "explained_var": explained_var,
            "clipfrac": avg_clipfrac,
            "approx_kl": avg_approx_kl,
            "value_clipfrac": avg_value_clipfrac,
            "optimizer_steps": float(actual_optimization_steps),
            "stopped_early": 1.0 if stop_early else 0.0,
        }

    def log_rewards(self, update_idx: int, transitions: List[TextTransition]):
        if not self.accelerator.is_main_process:
            return
        num = len(transitions)
        agent_stats: Dict[int, Dict[str, float]] = {}
        agent_counts: Dict[int, int] = {}
        for t in transitions:
            idx = int(t.agent_index)
            stats = agent_stats.setdefault(
                idx,
                {
                    "rl": 0.0,
                    "format": 0.0,
                    "validator": 0.0,
                    "sequence": 0.0,
                    "comm": 0.0,
                    "breakdown_total": 0.0,
                    "legacy_process": 0.0,
                },
            )
            agent_counts[idx] = agent_counts.get(idx, 0) + 1
            stats["rl"] += float(getattr(t, "reward", 0.0))
            stats["format"] += float(getattr(t, "format_reward", 0.0))
            stats["validator"] += float(getattr(t, "validator_reward", 0.0))
            stats["sequence"] += float(getattr(t, "sequence_reward", 0.0))
            stats["comm"] += float(getattr(t, "communication_reward", 0.0))
            stats["breakdown_total"] += float(getattr(t, "breakdown_total_reward", 0.0))
            stats["legacy_process"] += float(getattr(t, "process_reward", getattr(t, "reward", 0.0)))
        log_path = self.output_dir / "reward_curve.csv"
        header = (
            "row_idx,update_idx,step,num_transitions,"
            "agent0_n,agent1_n,"
            "agent0_rl_sum,agent0_rl_mean,"
            "agent0_format_sum,agent0_format_mean,"
            "agent0_validator_sum,agent0_validator_mean,"
            "agent0_sequence_sum,agent0_sequence_mean,"
            "agent0_comm_sum,agent0_comm_mean,"
            "agent0_breakdown_total_sum,agent0_breakdown_total_mean,"
            "agent0_legacy_process_sum,agent0_legacy_process_mean,"
            "agent1_rl_sum,agent1_rl_mean,"
            "agent1_format_sum,agent1_format_mean,"
            "agent1_validator_sum,agent1_validator_mean,"
            "agent1_sequence_sum,agent1_sequence_mean,"
            "agent1_comm_sum,agent1_comm_mean,"
            "agent1_breakdown_total_sum,agent1_breakdown_total_mean,"
            "agent1_legacy_process_sum,agent1_legacy_process_mean"
        )
        row_idx, last_row = self._prepare_csv_log(log_path, header)
        prev_step = 0
        if last_row is not None:
            try:
                prev_step = int(float(last_row.get("step", "0") or 0))
            except (TypeError, ValueError):
                prev_step = 0
        step_est = prev_step + num
        a0_stats = agent_stats.get(
            0,
            {
                "rl": 0.0,
                "format": 0.0,
                "validator": 0.0,
                "sequence": 0.0,
                "comm": 0.0,
                "breakdown_total": 0.0,
                "legacy_process": 0.0,
            },
        )
        a1_stats = agent_stats.get(
            1,
            {
                "rl": 0.0,
                "format": 0.0,
                "validator": 0.0,
                "sequence": 0.0,
                "comm": 0.0,
                "breakdown_total": 0.0,
                "legacy_process": 0.0,
            },
        )
        a0_n = int(agent_counts.get(0, 0))
        a1_n = int(agent_counts.get(1, 0))

        def _mean(total: float, n: int) -> float:
            return float(total) / float(n) if n > 0 else 0.0

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{row_idx},{update_idx},{step_est},{num},"
                f"{a0_n},{a1_n},"
                f"{a0_stats['rl']},{_mean(a0_stats['rl'], a0_n)},"
                f"{a0_stats['format']},{_mean(a0_stats['format'], a0_n)},"
                f"{a0_stats['validator']},{_mean(a0_stats['validator'], a0_n)},"
                f"{a0_stats['sequence']},{_mean(a0_stats['sequence'], a0_n)},"
                f"{a0_stats['comm']},{_mean(a0_stats['comm'], a0_n)},"
                f"{a0_stats['breakdown_total']},{_mean(a0_stats['breakdown_total'], a0_n)},"
                f"{a0_stats['legacy_process']},{_mean(a0_stats['legacy_process'], a0_n)},"
                f"{a1_stats['rl']},{_mean(a1_stats['rl'], a1_n)},"
                f"{a1_stats['format']},{_mean(a1_stats['format'], a1_n)},"
                f"{a1_stats['validator']},{_mean(a1_stats['validator'], a1_n)},"
                f"{a1_stats['sequence']},{_mean(a1_stats['sequence'], a1_n)},"
                f"{a1_stats['comm']},{_mean(a1_stats['comm'], a1_n)},"
                f"{a1_stats['breakdown_total']},{_mean(a1_stats['breakdown_total'], a1_n)},"
                f"{a1_stats['legacy_process']},{_mean(a1_stats['legacy_process'], a1_n)}\n"
            )

    def log_train_metrics(
        self,
        update_idx: int,
        loss_dict: Dict[str, float],
        transitions: List[TextTransition],
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        path = self.output_dir / "train_curve.csv"
        header = (
            "row_idx,update_idx,num_transitions,loss,policy_loss,value_loss,entropy,"
            "reward_mean,adv_mean,return_mean,value_mean,explained_var,"
            "clipfrac,approx_kl,value_clipfrac,optimizer_steps,stopped_early"
        )
        row_idx, _ = self._prepare_csv_log(path, header)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{row_idx},{update_idx},{len(transitions)},"
                f"{loss_dict.get('loss', 0.0)},{loss_dict.get('policy', 0.0)},"
                f"{loss_dict.get('value', 0.0)},{loss_dict.get('entropy', 0.0)},"
                f"{loss_dict.get('reward_mean', 0.0)},{loss_dict.get('adv_mean', 0.0)},"
                f"{loss_dict.get('return_mean', 0.0)},{loss_dict.get('value_mean', 0.0)},"
                f"{loss_dict.get('explained_var', 0.0)},"
                f"{loss_dict.get('clipfrac', 0.0)},{loss_dict.get('approx_kl', 0.0)},"
                f"{loss_dict.get('value_clipfrac', 0.0)},{loss_dict.get('optimizer_steps', 0.0)},"
                f"{loss_dict.get('stopped_early', 0.0)}\n"
            )

    def log_performance(self, update_idx: int):
        """Log environment-level performance metrics (traditional RL curves)."""
        stats = dict(self._last_rollout_stats or {})
        env_steps_local = float(stats.get("env_steps", 0) or 0)
        episodes_local = float(stats.get("episodes_completed", 0) or 0)
        successes_local = float(stats.get("success_episodes", 0) or 0)
        env_reward_sum_local = float(stats.get("env_reward_sum", 0.0) or 0.0)
        pos_steps_local = float(stats.get("positive_reward_steps", 0) or 0)
        policy_calls_local = float(stats.get("policy_calls", 0) or 0)
        ep_return_sum_local = float(stats.get("episode_return_sum", 0.0) or 0.0)
        ep_len_sum_local = float(stats.get("episode_lengths_sum", 0) or 0)
        agent0_custom_sum_local = float(stats.get("agent0_custom_return_sum", 0.0) or 0.0)
        agent1_custom_sum_local = float(stats.get("agent1_custom_return_sum", 0.0) or 0.0)
        team_custom_sum_local = float(stats.get("team_custom_return_sum", 0.0) or 0.0)

        # Aggregate across ranks so curves reflect full multi-proc sampling throughput.
        packed = torch.tensor(
            [
                env_steps_local,
                episodes_local,
                successes_local,
                env_reward_sum_local,
                pos_steps_local,
                policy_calls_local,
                ep_return_sum_local,
                ep_len_sum_local,
                agent0_custom_sum_local,
                agent1_custom_sum_local,
                team_custom_sum_local,
            ],
            device=self.device,
            dtype=torch.float64,
        )
        # NOTE: `accelerator.gather` has different shape semantics across versions:
        # - Some return [world_size, N]
        # - Some return [world_size * N] for 1D inputs
        # We normalize to a length-N vector then sum across ranks.
        summed = None
        try:
            # Prefer reduce if available (stable shape, cheaper than gather).
            if hasattr(self.accelerator, "reduce"):
                summed = self.accelerator.reduce(packed, reduction="sum")
        except Exception:
            summed = None
        if summed is None:
            gathered = self.accelerator.gather(packed)
            if gathered.ndim == 2 and gathered.shape[-1] == packed.numel():
                summed = gathered.sum(dim=0)
            elif gathered.ndim == 1 and gathered.numel() == packed.numel():
                summed = gathered
            elif gathered.ndim == 1 and gathered.numel() % packed.numel() == 0:
                world = int(gathered.numel() // packed.numel())
                summed = gathered.view(world, packed.numel()).sum(dim=0)
            else:
                raise ValueError(
                    f"Unexpected gathered stats shape {tuple(gathered.shape)} for packed {tuple(packed.shape)}"
                )
        (
            env_steps_f,
            episodes_f,
            successes_f,
            env_reward_sum,
            pos_steps_f,
            policy_calls_f,
            ep_return_sum,
            ep_len_sum_f,
            agent0_custom_sum,
            agent1_custom_sum,
            team_custom_sum,
        ) = [float(x) for x in summed.tolist()]
        env_steps = int(env_steps_f)
        episodes = int(episodes_f)
        successes = int(successes_f)
        pos_steps = int(pos_steps_f)
        policy_calls = int(policy_calls_f)
        ep_len_sum = int(ep_len_sum_f)

        avg_step_reward = env_reward_sum / env_steps if env_steps > 0 else 0.0
        avg_calls_per_step = policy_calls / env_steps if env_steps > 0 else 0.0
        avg_episode_return = ep_return_sum / episodes if episodes > 0 else 0.0
        avg_episode_len = ep_len_sum / episodes if episodes > 0 else 0.0
        success_rate = successes / episodes if episodes > 0 else 0.0
        avg_agent0_custom_return = agent0_custom_sum / episodes if episodes > 0 else 0.0
        avg_agent1_custom_return = agent1_custom_sum / episodes if episodes > 0 else 0.0
        avg_team_custom_return = team_custom_sum / episodes if episodes > 0 else 0.0

        # If available, attach the training update index of the currently loaded model
        # (useful when evaluation runs are launched separately and local update_idx resets).
        model_update_idx = None
        if self.latest_model_path_file and self.latest_model_path_file.exists():
            try:
                raw = self.latest_model_path_file.read_text(encoding="utf-8").strip()
                parsed = json.loads(raw) if raw else None
                if isinstance(parsed, dict) and parsed.get("update_idx") is not None:
                    model_update_idx = int(parsed["update_idx"])
            except Exception:
                model_update_idx = None
        # All ranks should rendezvous before main writes, to keep logging aligned.
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return

        path = self.output_dir / "performance_curve.csv"
        header = (
            "row_idx,update_idx,model_update_idx,env_steps,episodes_completed,success_episodes,success_rate,"
            "env_reward_sum,avg_step_reward,positive_reward_steps,"
            "policy_calls,avg_calls_per_step,"
            "episode_return_sum,avg_episode_return,avg_episode_len,"
            "agent0_custom_return_sum,avg_agent0_custom_return,"
            "agent1_custom_return_sum,avg_agent1_custom_return,"
            "team_custom_return_sum,avg_team_custom_return"
        )
        row_idx, _ = self._prepare_csv_log(path, header)
        with path.open("a", encoding="utf-8") as f:
            f.write(
                f"{row_idx},{update_idx},{'' if model_update_idx is None else model_update_idx},{env_steps},{episodes},{successes},{success_rate},"
                f"{env_reward_sum},{avg_step_reward},{pos_steps},"
                f"{policy_calls},{avg_calls_per_step},"
                f"{ep_return_sum},{avg_episode_return},{avg_episode_len},"
                f"{agent0_custom_sum},{avg_agent0_custom_return},"
                f"{agent1_custom_sum},{avg_agent1_custom_return},"
                f"{team_custom_sum},{avg_team_custom_return}\n"
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
        unwrapped: QwenLMActorCritic = self.accelerator.unwrap_model(self.text_policy)
        hf_model = unwrapped.model
        tokenizer = unwrapped.tokenizer

        def _safe_name(raw: str) -> str:
            raw = (raw or "").strip() or "adapter"
            cleaned = re.sub(r"[^0-9a-zA-Z_.-]+", "_", raw)
            return cleaned.strip("_") or "adapter"

        def _save_adapter_snapshot(adapter_name: str, out_dir: Path) -> None:
            out_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(hf_model, PeftModel):
                # Prefer saving only the selected adapter when supported (multi-adapter safe).
                try:
                    hf_model.save_pretrained(out_dir, selected_adapters=[adapter_name])
                except TypeError:
                    # Older PEFT: fallback to switching adapter then saving.
                    prev = getattr(hf_model, "active_adapter", None)
                    try:
                        hf_model.set_adapter(adapter_name)
                        hf_model.save_pretrained(out_dir)
                    finally:
                        if prev:
                            try:
                                hf_model.set_adapter(prev)
                            except Exception:
                                pass
            else:
                # Unexpected: is_lora=True but model is not a PeftModel.
                hf_model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

        payload: Dict[str, Any] = {
            "model_path": str(self.model_path),
            "lora_path": None,
            "update_idx": update_idx,
        }

        if unwrapped.is_lora:
            # Export per-agent adapters so the sampler can load different models for each agent.
            exported_actor: Dict[str, Dict[str, Any]] = {}
            for idx, spec in sorted(self.actor_adapters.items()):
                adapter_dir = self.export_latest_dir / f"adapter_{_safe_name(spec.name)}_u{update_idx:05d}"
                self.accelerator.print(
                    f"[Export] Saving adapter[{idx}:{spec.name}] -> {adapter_dir}"
                )
                _save_adapter_snapshot(spec.name, adapter_dir)
                exported_actor[str(idx)] = {
                    "adapter_name": spec.name,
                    "lora_path": str(adapter_dir),
                }
            if exported_actor:
                payload["actor_adapters"] = exported_actor

            if self.critic_adapter is not None:
                spec = self.critic_adapter
                adapter_dir = self.export_latest_dir / f"adapter_{_safe_name(spec.name)}_u{update_idx:05d}"
                self.accelerator.print(
                    f"[Export] Saving adapter[critic:{spec.name}] -> {adapter_dir}"
                )
                _save_adapter_snapshot(spec.name, adapter_dir)
                payload["critic_adapter"] = {
                    "adapter_name": spec.name,
                    "lora_path": str(adapter_dir),
                }

            # Backward-compat: if this is a single-adapter run, keep legacy lora_path field.
            if not self.actor_adapters and self.critic_adapter is None:
                adapter_dir = self.export_latest_dir / f"adapter_u{update_idx:05d}"
                self.accelerator.print(f"[Export] Saving adapter -> {adapter_dir}")
                active = getattr(hf_model, "active_adapter", "default")
                if isinstance(active, (list, tuple)):
                    active_name = str(active[0]) if active else "default"
                else:
                    active_name = str(active) if active else "default"
                _save_adapter_snapshot(active_name, adapter_dir)
                payload["lora_path"] = str(adapter_dir)
        else:
            model_dir = self.export_latest_dir / f"model_u{update_idx:05d}"
            self.accelerator.print(f"[Export] Saving full model -> {model_dir}")
            model_dir.mkdir(parents=True, exist_ok=True)
            hf_model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            payload["model_path"] = str(model_dir)
            payload["lora_path"] = None

        marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.accelerator.print(f"[Export] latest model info -> {marker_path}")

    # ------------------------------------------------------------------
    def _transition_to_dict(self, t: TextTransition):
        def _cpu(x):
            return x.cpu() if isinstance(x, torch.Tensor) else x

        return {
            "prompt_ids": _cpu(t.prompt_ids),
            "response_ids": _cpu(t.response_ids),
            "critic_input_ids": _cpu(t.critic_input_ids),
            "log_prob": t.log_prob,
            "value": t.value,
            "reward": t.reward,
            "done": t.done,
            "agent_index": t.agent_index,
            "entropy": t.entropy,
            "timestep": t.timestep,
            "format_reward": t.format_reward,
            "validator_reward": t.validator_reward,
            "process_reward": t.process_reward,
            "sequence_reward": t.sequence_reward,
            "communication_reward": t.communication_reward,
            "breakdown_total_reward": t.breakdown_total_reward,
        }

    def _dict_to_transition(self, d: Dict[str, Any]) -> TextTransition:
        return TextTransition(
            prompt_ids=d["prompt_ids"],
            response_ids=d["response_ids"],
            critic_input_ids=d.get("critic_input_ids"),
            log_prob=float(d["log_prob"]),
            value=float(d["value"]),
            reward=float(d["reward"]),
            done=float(d["done"]),
            agent_index=int(d["agent_index"]),
            entropy=float(d["entropy"]),
            timestep=int(d["timestep"]) if d.get("timestep") is not None else None,
            format_reward=float(d.get("format_reward", 0.0)),
            validator_reward=float(d.get("validator_reward", 0.0)),
            process_reward=float(d.get("process_reward", d.get("reward", 0.0))),
            sequence_reward=float(d.get("sequence_reward", 0.0)),
            communication_reward=float(d.get("communication_reward", 0.0)),
            breakdown_total_reward=float(
                d.get(
                    "breakdown_total_reward",
                    d.get("raw_total_reward", 0.0),  # legacy fallback if present
                )
            ),
        )

    def save_rollout(self, transitions: List[TextTransition], update_idx: int):
        if not transitions:
            return
        path = self.rollout_dir / f"rollout_rank{self.accelerator.process_index}_u{update_idx:05d}.pt"
        payload = [self._transition_to_dict(t) for t in transitions]
        torch.save(payload, path)

    def _initial_cache_path(self, update_idx: int, rank: Optional[int] = None) -> Path:
        use_rank = self.accelerator.process_index if rank is None else int(rank)
        return (
            self.initial_rollout_cache_dir
            / f"rollout_rank{use_rank}_u{int(update_idx):05d}.pt"
        )

    def _maybe_save_initial_cached_rollout(
        self, update_idx: int, transitions: List[TextTransition]
    ) -> None:
        if self.initial_rollout_cache_updates <= 0:
            return
        if int(update_idx) > self.initial_rollout_cache_updates:
            return
        if not transitions:
            return
        path = self._initial_cache_path(update_idx)
        if path.exists():
            return
        payload = [self._transition_to_dict(t) for t in transitions]
        torch.save(payload, path)
        if self.accelerator.is_main_process:
            self.accelerator.print(
                f"[MAPPO] Saved initial cached rollout -> {path}"
            )

    def _maybe_load_initial_cached_rollout(self, update_idx: int) -> bool:
        if not self.reuse_initial_rollout_cache:
            return False
        if self.initial_rollout_cache_updates <= 0:
            return False
        if int(update_idx) > self.initial_rollout_cache_updates:
            return False
        self.accelerator.wait_for_everyone()
        world_size = max(1, int(self.accelerator.num_processes))
        paths = [
            self._initial_cache_path(update_idx, rank=rank)
            for rank in range(world_size)
        ]
        if not all(path.exists() for path in paths):
            return False
        loaded: List[TextTransition] = []
        for path in paths:
            data = torch.load(path, map_location="cpu")
            loaded.extend([self._dict_to_transition(d) for d in data])
        self.buffer.storage = loaded
        return True

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
