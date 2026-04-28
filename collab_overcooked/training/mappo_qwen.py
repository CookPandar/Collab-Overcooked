"""MAPPO trainer for Collab-Overcooked using the full LLM prompt pipeline."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import dataclass, field
import copy
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import socket
from urllib import error as urllib_error
from urllib import request as urllib_request

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.generation.logits_process import LogitsProcessorList
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
    policy_temperature: Optional[float]
    value: float
    reward: float
    done: float
    agent_index: int
    entropy: float
    timestep: Optional[int] = None
    critic_input_ids: Optional[torch.Tensor] = None
    response_log_probs: Optional[torch.Tensor] = None
    format_reward: float = 0.0
    validator_reward: float = 0.0
    process_reward: float = 0.0
    # Reward breakdown (optional; depends on env/validator providing reward_breakdown)
    # NOTE: `reward` is still the scalar used for RL updates; these fields are only for logging/analysis.
    sequence_reward: float = 0.0
    communication_reward: float = 0.0
    repeat_communication_reward: float = 0.0
    forced_communication_reward: float = 0.0
    paired_comm_reward: float = 0.0
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
        response_log_probs: Optional[torch.Tensor],
        log_prob: float,
        policy_temperature: Optional[float],
        entropy: float,
        value: float,
        critic_input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self.text = text
        self.prompt_ids = prompt_ids
        self.response_ids = response_ids
        self.response_log_probs = response_log_probs
        self.log_prob = log_prob
        self.policy_temperature = policy_temperature
        self.entropy = entropy
        self.value = value
        self.critic_input_ids = critic_input_ids


@dataclass
class PrefixCacheEntry:
    input_ids: torch.Tensor
    past_key_values: Any
    token_count: int


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
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
        }
        # 有 flash-attn2 则启用，否则退回 sdpa，避免缺依赖时报错
        if torch.cuda.is_available():
            try:
                import flash_attn  # type: ignore

                model_kwargs["attn_implementation"] = "flash_attention_2"
            except Exception:
                model_kwargs["attn_implementation"] = "sdpa"
            model_kwargs["device_map"] = {"": str(device)}
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
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        # value head总是训练
        for p in self.value_head.parameters():
            p.requires_grad = True

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.eval_batch_size = max(1, int(eval_batch_size))
        self._prefix_cache: Dict[Tuple[Optional[str], str], PrefixCacheEntry] = {}
        self._prefix_cache_order: List[Tuple[Optional[str], str]] = []
        self._prefix_cache_max_entries = 4
        if not torch.cuda.is_available():
            self.to(device)

    def clear_prefix_cache(self) -> None:
        self._prefix_cache.clear()
        self._prefix_cache_order.clear()

    def _touch_prefix_cache_key(self, key: Tuple[Optional[str], str]) -> None:
        if key in self._prefix_cache_order:
            self._prefix_cache_order.remove(key)
        self._prefix_cache_order.append(key)
        while len(self._prefix_cache_order) > self._prefix_cache_max_entries:
            evicted = self._prefix_cache_order.pop(0)
            self._prefix_cache.pop(evicted, None)

    def _get_prefix_cache_entry(
        self,
        prefix_text: str,
        adapter_name: Optional[str] = None,
    ) -> Optional[PrefixCacheEntry]:
        content = prefix_text or ""
        if not content.strip():
            return None
        key = (adapter_name, content)
        entry = self._prefix_cache.get(key)
        if entry is not None:
            self._touch_prefix_cache_key(key)
            return entry
        inputs = self._prepare_inputs(content)
        with self._use_adapter(adapter_name):
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True,
            )
        entry = PrefixCacheEntry(
            input_ids=inputs["input_ids"].squeeze(0).detach().cpu(),
            past_key_values=self._clone_past_key_values(outputs.past_key_values),
            token_count=int(inputs["input_ids"].shape[1]),
        )
        self._prefix_cache[key] = entry
        self._touch_prefix_cache_key(key)
        return entry

    def _clone_past_key_values(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None
        if isinstance(past_key_values, DynamicCache):
            cloned = DynamicCache(config=self.model.config)
            for src_layer, dst_layer in zip(past_key_values.layers, cloned.layers):
                if not getattr(src_layer, "is_initialized", False):
                    continue
                dst_layer.lazy_initialization(src_layer.keys, src_layer.values)
                dst_layer.keys = src_layer.keys.detach().clone()
                dst_layer.values = src_layer.values.detach().clone()
                dst_layer.is_initialized = True
            return cloned
        if isinstance(past_key_values, torch.Tensor):
            return past_key_values.detach().clone()
        if isinstance(past_key_values, (list, tuple)):
            cloned = [self._clone_past_key_values(item) for item in past_key_values]
            return type(past_key_values)(cloned)
        return copy.deepcopy(past_key_values)

    def _split_full_input_by_prefix(
        self,
        full_text: str,
        prefix_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        full_inputs = self._prepare_inputs(full_text)
        full_ids = full_inputs["input_ids"].squeeze(0)
        if not prefix_text or not prefix_text.strip():
            return full_ids, full_ids.new_empty((0,), dtype=full_ids.dtype)
        prefix_inputs = self._prepare_inputs(prefix_text)
        prefix_ids = prefix_inputs["input_ids"].squeeze(0)
        prefix_len = int(prefix_ids.shape[0])
        if prefix_len <= 0 or prefix_len > int(full_ids.shape[0]):
            raise RuntimeError("Invalid prefix tokenization length for KV cache split.")
        if not torch.equal(full_ids[:prefix_len], prefix_ids):
            raise RuntimeError("Prefix tokens do not align with full prompt tokens.")
        suffix_ids = full_ids[prefix_len:]
        return full_ids, suffix_ids

    def _generate_with_prefix_cache(
        self,
        full_text: str,
        prefix_text: str,
        adapter_name: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prefix_entry = self._get_prefix_cache_entry(prefix_text, adapter_name=adapter_name)
        if prefix_entry is None:
            raise RuntimeError("Prefix cache generation requires a non-empty prefix.")
        full_prompt_ids, suffix_ids_1d = self._split_full_input_by_prefix(full_text, prefix_text)
        response_ids = self._decode_with_generation_processors(
            prompt_ids=full_prompt_ids,
            adapter_name=adapter_name,
            past_key_values=self._clone_past_key_values(prefix_entry.past_key_values),
            processed_prompt_len=prefix_entry.token_count,
        )
        return full_prompt_ids.detach().cpu(), response_ids

    def _decode_with_generation_processors(
        self,
        prompt_ids: torch.Tensor,
        adapter_name: Optional[str],
        past_key_values: Any = None,
        processed_prompt_len: int = 0,
    ) -> torch.Tensor:
        prompt_ids = prompt_ids.to(self.device)
        do_sample = self.temperature is not None and float(self.temperature) > 0.0
        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.do_sample = bool(do_sample)
        generation_config.max_new_tokens = int(self.max_new_tokens)
        # PPO log-prob re-evaluation currently models temperature scaling only.
        # Neutralize inherited sampling warpers from the base model config so the
        # rollout distribution exactly matches `_evaluate_chunk`.
        generation_config.repetition_penalty = 1.0
        for attr in (
            "top_k",
            "top_p",
            "min_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "top_h",
        ):
            if hasattr(generation_config, attr):
                setattr(generation_config, attr, None)
        if do_sample:
            generation_config.temperature = float(self.temperature)
        else:
            generation_config.temperature = None
        logits_processor = self.model._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=int(prompt_ids.shape[0]),
            logits_processor=LogitsProcessorList(),
            device=str(self.device),
        )
        current_ids = prompt_ids.unsqueeze(0)
        with self._use_adapter(adapter_name):
            if processed_prompt_len > 0:
                prefill_ids = prompt_ids[processed_prompt_len:].unsqueeze(0)
                cache_position = torch.arange(
                    processed_prompt_len,
                    int(prompt_ids.shape[0]),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                prefill_ids = prompt_ids.unsqueeze(0)
                cache_position = torch.arange(
                    0, int(prompt_ids.shape[0]), device=self.device, dtype=torch.long
                )
            outputs = self.model(
                input_ids=prefill_ids,
                attention_mask=torch.ones(
                    (1, int(prompt_ids.shape[0])),
                    device=self.device,
                    dtype=torch.long,
                ),
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_ids=cache_position.unsqueeze(0),
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            generated_tokens: List[torch.Tensor] = []
            total_length = int(prompt_ids.shape[0])
            eos_token_id = self.tokenizer.eos_token_id
            for _ in range(self.max_new_tokens):
                processed_scores = logits_processor(current_ids, next_logits.float())
                if do_sample:
                    next_token = torch.multinomial(
                        torch.softmax(processed_scores, dim=-1),
                        num_samples=1,
                    )
                else:
                    next_token = torch.argmax(processed_scores, dim=-1, keepdim=True)
                generated_tokens.append(next_token.squeeze(0))
                current_ids = torch.cat([current_ids, next_token], dim=1)
                if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                    break
                step_position = torch.tensor([total_length], device=self.device, dtype=torch.long)
                total_length += 1
                outputs = self.model(
                    input_ids=next_token,
                    attention_mask=torch.ones(
                        (1, total_length),
                        device=self.device,
                        dtype=torch.long,
                    ),
                    past_key_values=past_key_values,
                    cache_position=step_position,
                    position_ids=step_position.unsqueeze(0),
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]
        return (
            torch.cat(generated_tokens, dim=0).detach().cpu()
            if generated_tokens
            else torch.empty(0, dtype=torch.long)
        )

    def _evaluate_value_text_with_prefix(
        self,
        prefix_text: str,
        full_text: str,
        adapter_name: Optional[str],
    ) -> Tuple[torch.Tensor, float]:
        prefix_entry = self._get_prefix_cache_entry(prefix_text, adapter_name=adapter_name)
        if prefix_entry is None:
            return self.evaluate_text_value(full_text)
        full_prompt_ids, suffix_ids_1d = self._split_full_input_by_prefix(full_text, prefix_text)
        suffix_ids = suffix_ids_1d.unsqueeze(0)
        past_key_values = self._clone_past_key_values(prefix_entry.past_key_values)
        with self._use_adapter(adapter_name):
            outputs = self.model(
                input_ids=suffix_ids,
                attention_mask=torch.ones(
                    (1, prefix_entry.token_count + suffix_ids.shape[1]),
                    device=self.device,
                    dtype=torch.long,
                ),
                past_key_values=past_key_values,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1][:, -1, :]
        value = self.value_head(hidden).squeeze(-1).item()
        return full_prompt_ids.detach().cpu(), value

    def _effective_policy_temperature(self) -> Optional[float]:
        """Return the sampling temperature used to define policy log-probs."""
        if self.temperature is None:
            return None
        temp = float(self.temperature)
        return temp if temp > 0.0 else None

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
        self._enable_all_trainable_adapters()

    def _all_trainable_adapter_names(self) -> List[str]:
        names = [spec.name for _, spec in sorted(self.actor_adapters.items())]
        if self.critic_adapter is not None:
            names.append(self.critic_adapter.name)
        deduped: List[str] = []
        for name in names:
            if name and name not in deduped:
                deduped.append(name)
        return deduped

    def _enable_all_trainable_adapters(self) -> None:
        if not self.is_lora or not isinstance(self.model, PeftModel):
            return
        adapter_names = self._all_trainable_adapter_names()
        if not adapter_names:
            return
        try:
            self.model.set_requires_grad(adapter_names, requires_grad=True)
        except Exception:
            pass

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
            self._enable_all_trainable_adapters()
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

    def _forward_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        base_lm = (
            self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        )
        backbone = getattr(base_lm, "model", None)
        if backbone is None:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states for value evaluation.")
            return hidden_states[-1]

        peft_ctx = (
            self.model._enable_peft_forward_hooks(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            if hasattr(self.model, "_enable_peft_forward_hooks")
            else nullcontext()
        )
        with peft_ctx:
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            if isinstance(outputs, tuple) and outputs:
                last_hidden = outputs[0]
            else:
                raise RuntimeError("Backbone did not return last hidden state.")
        return last_hidden

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
            hidden_states = self._forward_last_hidden(inputs, attention_mask)
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
        prompt_prefix: Optional[str] = None,
    ) -> LMGenerationResult:
        adapter_name = self.actor_adapter_names.get(agent_index)
        critic_input_ids = None
        prefix_text = prompt_prefix or ""
        if prefix_text:
            prompt_ids, response_ids = self._generate_with_prefix_cache(
                full_text=prompt,
                prefix_text=prefix_text,
                adapter_name=adapter_name,
            )
        else:
            inputs = self._prepare_inputs(prompt)
            prompt_ids = inputs["input_ids"].squeeze(0).cpu()
            response_ids = self._decode_with_generation_processors(
                prompt_ids=prompt_ids,
                adapter_name=adapter_name,
                past_key_values=None,
                processed_prompt_len=0,
            )
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        policy_temperature = self._effective_policy_temperature()
        full_seq = torch.cat(
            [prompt_ids.to(self.device), response_ids.to(self.device)], dim=0
        )
        with self._use_adapter(adapter_name):
            log_probs, entropies, token_log_probs = self.evaluate_policy_batch(
                [prompt_ids.to(self.device)],
                [response_ids.to(self.device)],
                [agent_index],
                policy_temperatures=[policy_temperature],
            )
        total_log_prob = float(log_probs[0].item()) if log_probs is not None else 0.0
        avg_entropy = float(entropies[0].item()) if entropies is not None else 0.0
        response_log_probs = (
            token_log_probs[0].detach().cpu() if token_log_probs else None
        )

        value = 0.0
        if critic_prompt:
            critic_inputs = self._prepare_inputs(critic_prompt)
            critic_input_ids = critic_inputs["input_ids"].squeeze(0).cpu()
            value = self._evaluate_value_inputs(
                [critic_input_ids.to(self.device)], use_critic_adapter=True
            )[0].item()
        else:
            seq = full_seq.unsqueeze(0)
            attention_mask = torch.ones_like(seq, dtype=torch.long, device=self.device)
            hidden = self._forward_last_hidden(seq, attention_mask)[:, -1, :]
            value = self.value_head(hidden).squeeze(-1).item()
            if self.critic_adapter_name and self.critic_adapter_name != adapter_name:
                value = self._evaluate_value_single(prompt_ids, response_ids)

        return LMGenerationResult(
            response_text,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_log_probs=response_log_probs,
            log_prob=total_log_prob,
            policy_temperature=policy_temperature,
            entropy=avg_entropy,
            value=value,
            critic_input_ids=critic_input_ids,
        )

    def _evaluate_value_single(self, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> float:
        prompt = prompt_ids.to(self.device)
        response = response_ids.to(self.device)
        with self._use_adapter(self.critic_adapter_name):
            _, _, values, _ = self._evaluate_chunk(
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
        policy_temperatures: Optional[List[Optional[float]]] = None,
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
        if policy_temperatures is None:
            policy_temperatures = [self._effective_policy_temperature()] * num_samples

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
                logp, ent, vals, _ = self._evaluate_chunk(
                    subset_prompts,
                    subset_responses,
                    policy_temperatures=[policy_temperatures[i] for i in sample_indices],
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
                _, _, critic_vals, _ = self._evaluate_chunk(
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

    def evaluate_policy_batch(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        agent_indices: List[int],
        policy_temperatures: Optional[List[Optional[float]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        num_samples = len(prompt_tensors)
        if num_samples == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.float32)
            return empty, empty, []
        prompt_device = [t.to(self.device) for t in prompt_tensors]
        response_device = [t.to(self.device) for t in response_tensors]
        if policy_temperatures is None:
            policy_temperatures = [self._effective_policy_temperature()] * num_samples

        log_probs_buf: List[Optional[torch.Tensor]] = [None] * num_samples
        entropy_buf: List[Optional[torch.Tensor]] = [None] * num_samples
        token_log_probs_buf: List[Optional[torch.Tensor]] = [None] * num_samples
        groups: Dict[Optional[str], List[int]] = {}
        if self.actor_adapters:
            for idx, agent_idx in enumerate(agent_indices):
                adapter_name = self.actor_adapter_names.get(agent_idx)
                groups.setdefault(adapter_name, []).append(idx)
        else:
            groups = {None: list(range(num_samples))}

        for adapter_name, sample_indices in groups.items():
            subset_prompts = [prompt_device[i] for i in sample_indices]
            subset_responses = [response_device[i] for i in sample_indices]
            with self._use_adapter(adapter_name):
                logp, ent, _, token_log_probs = self._evaluate_chunk(
                    subset_prompts,
                    subset_responses,
                    policy_temperatures=[policy_temperatures[i] for i in sample_indices],
                    need_policy=True,
                    need_value=False,
                )
            for local_idx, global_idx in enumerate(sample_indices):
                log_probs_buf[global_idx] = logp[local_idx]
                entropy_buf[global_idx] = ent[local_idx]
                token_log_probs_buf[global_idx] = token_log_probs[local_idx]

        return (
            torch.stack([t for t in log_probs_buf if t is not None]),
            torch.stack([t for t in entropy_buf if t is not None]),
            [
                t
                if t is not None
                else torch.empty(0, device=self.device, dtype=torch.float32)
                for t in token_log_probs_buf
            ],
        )

    def evaluate_values_batch_train(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        agent_indices: List[int],
        critic_tensors: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        num_samples = len(prompt_tensors)
        if num_samples == 0:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        prompt_device = [t.to(self.device) for t in prompt_tensors]
        response_device = [t.to(self.device) for t in response_tensors]
        critic_device = (
            [t.to(self.device) if t is not None else None for t in critic_tensors]
            if critic_tensors is not None
            else [None] * num_samples
        )
        values: List[torch.Tensor] = []
        for idx in range(num_samples):
            critic_tensor = critic_device[idx]
            if critic_tensor is not None:
                values.append(
                    self._evaluate_value_inputs(
                        [critic_tensor], use_critic_adapter=True
                    )[0]
                )
                continue

            agent_idx = agent_indices[idx]
            actor_adapter_name = (
                self.actor_adapter_names.get(agent_idx) if self.actor_adapters else None
            )
            use_actor_value = (
                self.critic_adapter_name is None
                or actor_adapter_name == self.critic_adapter_name
            )
            adapter_ctx = (
                self._use_adapter(actor_adapter_name)
                if use_actor_value
                else self._use_adapter(self.critic_adapter_name)
            )
            with adapter_ctx:
                _, _, vals, _ = self._evaluate_chunk(
                    [prompt_device[idx]],
                    [response_device[idx]],
                    need_policy=False,
                    need_value=True,
                )
            values.append(vals[0])

        return torch.stack(values)

    @torch.no_grad()
    def evaluate_values_batch(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        agent_indices: List[int],
        critic_tensors: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        num_samples = len(prompt_tensors)
        if num_samples == 0:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        prompt_device = [t.to(self.device) for t in prompt_tensors]
        response_device = [t.to(self.device) for t in response_tensors]
        critic_device = (
            [t.to(self.device) if t is not None else None for t in critic_tensors]
            if critic_tensors is not None
            else [None] * num_samples
        )
        values_buf: List[Optional[torch.Tensor]] = [None] * num_samples

        critic_indices = [
            idx for idx, tensor in enumerate(critic_device) if tensor is not None
        ]
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
        if not missing_value_indices:
            return torch.stack([t for t in values_buf if t is not None])

        critic_matches_actor = False
        if self.critic_adapter_name and self.actor_adapter_names:
            critic_matches_actor = all(
                self.actor_adapter_names.get(agent_idx) == self.critic_adapter_name
                for agent_idx in agent_indices
            )
        use_actor_values = self.critic_adapter_name is None or critic_matches_actor
        if use_actor_values:
            groups: Dict[Optional[str], List[int]] = {}
            if self.actor_adapters:
                for idx in missing_value_indices:
                    adapter_name = self.actor_adapter_names.get(agent_indices[idx])
                    groups.setdefault(adapter_name, []).append(idx)
            else:
                groups = {None: list(missing_value_indices)}
            for adapter_name, sample_indices in groups.items():
                with self._use_adapter(adapter_name):
                    _, _, vals, _ = self._evaluate_chunk(
                        [prompt_device[idx] for idx in sample_indices],
                        [response_device[idx] for idx in sample_indices],
                        need_policy=False,
                        need_value=True,
                    )
                for local_idx, global_idx in enumerate(sample_indices):
                    values_buf[global_idx] = vals[local_idx]
        else:
            with self._use_adapter(self.critic_adapter_name):
                _, _, vals, _ = self._evaluate_chunk(
                    [prompt_device[idx] for idx in missing_value_indices],
                    [response_device[idx] for idx in missing_value_indices],
                    need_policy=False,
                    need_value=True,
                )
            for local_idx, global_idx in enumerate(missing_value_indices):
                values_buf[global_idx] = vals[local_idx]

        return torch.stack([t for t in values_buf if t is not None])

    def _evaluate_chunk(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        policy_temperatures: Optional[List[Optional[float]]] = None,
        need_policy: bool = True,
        need_value: bool = True,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[torch.Tensor],
    ]:
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
        base_lm = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        backbone = getattr(base_lm, "model", None)
        lm_head = getattr(base_lm, "lm_head", None)
        if backbone is None or lm_head is None:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits if need_policy else None
        else:
            peft_ctx = (
                self.model._enable_peft_forward_hooks(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                if hasattr(self.model, "_enable_peft_forward_hooks")
                else nullcontext()
            )
            with peft_ctx:
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None:
                if isinstance(outputs, tuple) and outputs:
                    hidden_states = outputs[0]
                else:
                    raise RuntimeError("Backbone did not return last hidden state.")
            logits = None
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        token_log_probs: List[torch.Tensor] = []
        if policy_temperatures is None:
            policy_temperatures = [self._effective_policy_temperature()] * input_ids.size(0)
        token_logits_by_sample: List[Optional[torch.Tensor]] = [None] * input_ids.size(0)
        if need_policy and logits is None:
            gather_hidden: List[torch.Tensor] = []
            gather_meta: List[Tuple[int, int]] = []
            for i in range(input_ids.size(0)):
                p_len = int(prompt_lengths[i].item())
                r_len = int(response_lengths[i].item())
                if r_len <= 0:
                    continue
                start = max(p_len - 1, 0)
                end = start + r_len
                gather_hidden.append(hidden_states[i, start:end, :])
                gather_meta.append((i, r_len))
            if gather_hidden:
                flat_hidden = torch.cat(gather_hidden, dim=0)
                flat_logits = lm_head(flat_hidden)
                offset = 0
                for sample_idx, r_len in gather_meta:
                    token_logits_by_sample[sample_idx] = flat_logits[offset : offset + r_len]
                    offset += r_len
        for i in range(input_ids.size(0)):
            p_len = int(prompt_lengths[i].item())
            r_len = int(response_lengths[i].item())
            if need_policy:
                if r_len == 0:
                    log_probs.append(torch.tensor(0.0, device=device))
                    entropies.append(torch.tensor(0.0, device=device))
                    token_log_probs.append(
                        torch.empty(0, device=device, dtype=torch.float32)
                    )
                else:
                    if logits is not None:
                        start = max(p_len - 1, 0)
                        end = start + r_len
                        token_logits = logits[i, start:end, :]
                    else:
                        token_logits = token_logits_by_sample[i]
                        if token_logits is None:
                            raise RuntimeError("Missing response logits for policy evaluation.")
                    temp = policy_temperatures[i]
                    if temp is not None:
                        # Align PPO re-evaluation with generation-time sampling scores.
                        temp_value = float(temp)
                        if temp_value > 0.0:
                            token_logits = token_logits / temp_value
                    token_ids = input_ids[i, p_len : p_len + r_len]
                    logprob = torch.log_softmax(token_logits, dim=-1)
                    gathered = logprob.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
                    log_probs.append(gathered.sum())
                    token_log_probs.append(gathered)
                    dists = torch.distributions.Categorical(logits=token_logits)
                    entropies.append(dists.entropy().mean())
            if need_value:
                last_index = max(p_len + r_len - 1, 0)
                value_vec = hidden_states[i, last_index, :].unsqueeze(0)
                values.append(self.value_head(value_vec).squeeze(0))
        lp_tensor = torch.stack(log_probs) if need_policy else None
        ent_tensor = torch.stack(entropies) if need_policy else None
        val_tensor = torch.stack(values) if need_value else None
        return lp_tensor, ent_tensor, val_tensor, token_log_probs

    def forward(
        self,
        prompt_tensors: List[torch.Tensor],
        response_tensors: List[torch.Tensor],
        agent_indices: Optional[List[int]] = None,
        critic_tensors: Optional[List[Optional[torch.Tensor]]] = None,
        policy_temperatures: Optional[List[Optional[float]]] = None,
    ):
        """DDP forward pass delegates to evaluate_batch."""
        if agent_indices is None:
            agent_indices = [0] * len(prompt_tensors)
        return self.evaluate_batch(
            prompt_tensors,
            response_tensors,
            agent_indices,
            critic_tensors=critic_tensors,
            policy_temperatures=policy_temperatures,
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
        print(
            "[MAPPO] before Accelerator init "
            f"stage={os.getenv('RL_STAGE_PHASE', '')} "
            f"master_addr={os.getenv('MASTER_ADDR', '')} "
            f"master_port={os.getenv('MASTER_PORT', '')} "
            f"rank={os.getenv('RANK', '')} local_rank={os.getenv('LOCAL_RANK', '')}",
            flush=True,
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=ddp_find_unused)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.device = self.accelerator.device
        print(
            "[MAPPO] after Accelerator init "
            f"process_index={self.accelerator.process_index} "
            f"num_processes={self.accelerator.num_processes} "
            f"device={self.device}",
            flush=True,
        )

        env_latest_file = os.getenv("RL_LATEST_MODEL_FILE", "").strip()
        cfg_latest_file = self.trainer_cfg.get("latest_model_path_file", "")
        latest_file = cfg_latest_file or env_latest_file
        self.latest_model_path_file: Optional[Path] = Path(latest_file) if latest_file else None
        self.export_latest_dir: Optional[Path] = None
        self.export_interval = max(1, int(self.trainer_cfg.get("export_interval", 1)))
        self.collect_only = bool(trainer_cfg.get("collect_only", False))
        self.train_only = bool(trainer_cfg.get("train_only", False))
        self.runtime_stage_phase = self._runtime_stage_phase(
            "train" if self.train_only and not self.collect_only else "collect"
        )
        apply_latest_default = not self.train_only
        self.apply_latest_model_override = bool(
            self.trainer_cfg.get("apply_latest_model_override", apply_latest_default)
        )
        self.rollout_dir = Path(trainer_cfg.get("rollout_dir", "rollouts"))
        self.cleanup_rollouts = bool(trainer_cfg.get("cleanup_rollouts", True))
        self.agents_cfg = full_config.get("agents", {})
        self.agent_roles = self._extract_agent_roles(self.agents_cfg)
        self.critic_role_prompt = self._load_critic_role_prompt()

        if (
            self.apply_latest_model_override
            and self.latest_model_path_file
            and self.latest_model_path_file.exists()
        ):
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
        self.snapshot_paths_configured = bool(snapshot_paths)
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
        self.gradient_checkpointing = bool(
            trainer_cfg.get("gradient_checkpointing", not self.collect_only)
        )
        self.compute_values_in_collect = bool(
            trainer_cfg.get("compute_values_in_collect", False)
        )
        self.collect_value_backend = str(
            trainer_cfg.get(
                "collect_value_backend",
                "vllm" if self.collect_only else "local",
            )
        ).strip().lower()
        self.use_vllm_value_service = bool(
            self.collect_only
            and self.compute_values_in_collect
            and self.collect_value_backend == "vllm"
        )
        self.enable_fresh_value_metrics = bool(
            trainer_cfg.get("enable_fresh_value_metrics", False)
        )
        critic_pretrain_threshold_cfg = trainer_cfg.get(
            "critic_pretrain_value_loss_threshold", None
        )
        self.critic_pretrain_value_loss_threshold = (
            float(critic_pretrain_threshold_cfg)
            if critic_pretrain_threshold_cfg is not None
            else None
        )
        self.load_policy_model = self.train_only or (
            self.collect_only
            and self.compute_values_in_collect
            and not self.use_vllm_value_service
        )
        if self.collect_only and self.collect_value_backend == "vllm":
            # In pure collect mode the worker should behave as an environment client:
            # actor sampling and critic value requests must both be served by vLLM.
            # Keep local model loading hard-disabled here to avoid duplicate GPU residency.
            self.load_policy_model = False
        self.accelerator.print(
            "[MAPPO] runtime flags "
            f"stage={self.runtime_stage_phase} "
            f"collect_only={self.collect_only} train_only={self.train_only} "
            f"compute_values_in_collect={self.compute_values_in_collect} "
            f"collect_value_backend={self.collect_value_backend} "
            f"use_vllm_value_service={self.use_vllm_value_service} "
            f"load_policy_model={self.load_policy_model}"
        )
        target_kl_cfg = trainer_cfg.get("target_kl", None)
        self.target_kl = float(target_kl_cfg) if target_kl_cfg is not None else None
        self.kl_penalty_coef = float(trainer_cfg.get("kl_penalty_coef", 0.0) or 0.0)
        self.entropy_coef = trainer_cfg.get("entropy_coef", 0.01)
        self.value_coef = trainer_cfg.get("value_coef", 0.5)
        self.max_grad_norm = trainer_cfg.get("max_grad_norm", 0.5)
        if self.train_only:
            if "actor_lr" not in trainer_cfg:
                raise ValueError("trainer.actor_lr must be set explicitly.")
            if "critic_lr" not in trainer_cfg:
                raise ValueError("trainer.critic_lr must be set explicitly.")
            if "value_head_lr" not in trainer_cfg:
                raise ValueError("trainer.value_head_lr must be set explicitly.")
            loop_rounds_raw = os.getenv("RL_LOOP_ROUNDS", "").strip()
            if not loop_rounds_raw:
                raise ValueError(
                    "RL_LOOP_ROUNDS must be set by cluster_run_rl.sh for lr scheduling."
                )
            self.actor_lr = float(trainer_cfg["actor_lr"])
            self.critic_lr = float(trainer_cfg["critic_lr"])
            self.value_head_lr = float(trainer_cfg["value_head_lr"])
            self.total_updates = max(1, int(loop_rounds_raw))
        else:
            self.actor_lr = float(trainer_cfg.get("actor_lr", 0.0))
            self.critic_lr = float(trainer_cfg.get("critic_lr", 0.0))
            self.value_head_lr = float(trainer_cfg.get("value_head_lr", 0.0))
            self.total_updates = 1
        self.lr_scheduler_name = str(
            trainer_cfg.get("lr_scheduler", trainer_cfg.get("scheduler", "none"))
        ).strip().lower()
        self.lr_warmup_updates = max(
            0, int(trainer_cfg.get("lr_warmup_updates", trainer_cfg.get("warmup_updates", 0)))
        )
        self.lr_min_ratio = float(trainer_cfg.get("lr_min_ratio", 0.0))
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
            trainer_cfg.get("initial_rollout_cache_updates", 0)
        )
        self.reuse_initial_rollout_cache = bool(
            trainer_cfg.get("reuse_initial_rollout_cache", False)
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

        self.text_policy = None
        self.policy_tokenizer = None
        if self.load_policy_model:
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
            self.accelerator.print(
                "[MAPPO] load_policy_model=true stage="
                f"{self.runtime_stage_phase} gradient_checkpointing={self.gradient_checkpointing} "
                f"critic_adapter={self.critic_adapter.name if self.critic_adapter is not None else 'none'} "
                f"critic_lora={self.critic_adapter.lora_config if self.critic_adapter is not None else None}"
            )
            if self.gradient_checkpointing and hasattr(
                self.text_policy.model, "gradient_checkpointing_enable"
            ):
                self.text_policy.model.gradient_checkpointing_enable()
                if hasattr(self.text_policy.model, "enable_input_require_grads"):
                    self.text_policy.model.enable_input_require_grads()
                if hasattr(self.text_policy.model, "config"):
                    self.text_policy.model.config.use_cache = False
            value_head_path = self.trainer_cfg.get("value_head_path")
            if value_head_path:
                resolved = Path(value_head_path)
                if not resolved.is_absolute():
                    resolved = Path.cwd() / resolved
                if resolved.exists():
                    payload = torch.load(resolved, map_location="cpu")
                    if isinstance(payload, dict) and "state_dict" in payload:
                        payload = payload["state_dict"]
                    self.text_policy.value_head.load_state_dict(payload)
                    self.accelerator.print(f"[MAPPO] Loaded value head from {resolved}")
            self.policy_tokenizer = self.text_policy.tokenizer
        else:
            tokenizer_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
            }
            if trainer_cfg.get("fix_mistral_regex", None) is True or (
                trainer_cfg.get("fix_mistral_regex", None) is None
                and "mistral" in str(self.model_path).lower()
            ):
                tokenizer_kwargs["fix_mistral_regex"] = True
            self.policy_tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                **tokenizer_kwargs,
            )
            if self.policy_tokenizer.pad_token_id is None:
                self.policy_tokenizer.pad_token_id = self.policy_tokenizer.eos_token_id
            self.accelerator.print(
                "[MAPPO] load_policy_model=false stage="
                f"{self.runtime_stage_phase}; collect/eval use vLLM actor with tokenizer-only local state."
            )
        self.accelerator.print(
            "[MAPPO] collect_value_backend="
            f"{self.collect_value_backend} compute_values_in_collect={self.compute_values_in_collect} "
            f"use_vllm_value_service={self.use_vllm_value_service} load_policy_model={self.load_policy_model}"
        )
        self.lr_scheduler = None
        if self.train_only:
            optimizer_groups: List[Dict[str, Any]] = []
            trainable_groups = self._trainable_param_groups()
            actor_params: List[torch.nn.Parameter] = []
            for name, params in trainable_groups.items():
                if name.startswith("actor"):
                    actor_params.extend(params)
            actor_params = self._filter_trainable_params(actor_params)
            critic_params = self._filter_trainable_params(
                trainable_groups.get("critic_adapter", [])
            )
            value_head_params = self._filter_trainable_params(
                trainable_groups.get("value_head", [])
            )
            if actor_params:
                optimizer_groups.append(
                    {"params": actor_params, "lr": self.actor_lr, "name": "actor"}
                )
            if critic_params:
                optimizer_groups.append(
                    {"params": critic_params, "lr": self.critic_lr, "name": "critic_adapter"}
                )
            if value_head_params:
                optimizer_groups.append(
                    {
                        "params": value_head_params,
                        "lr": self.value_head_lr,
                        "name": "value_head",
                    }
                )
            if not optimizer_groups:
                raise ValueError("No trainable parameters found for MAPPO optimizer.")
            self.optimizer = torch.optim.AdamW(optimizer_groups, lr=0.0)
            assert self.text_policy is not None
            self.text_policy, self.optimizer = self.accelerator.prepare(
                self.text_policy, self.optimizer
            )
        else:
            self.optimizer = None
            if self.text_policy is not None:
                self.text_policy = self.accelerator.prepare(self.text_policy)
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
        if (
            self.train_only
            and self.latest_model_path_file is not None
            and self.latest_model_path_file.exists()
            and not self.apply_latest_model_override
        ):
            self.accelerator.print(
                f"[TrainOnly] Ignoring latest model override from {self.latest_model_path_file} "
                "to keep PPO aligned with cached rollout log-probs."
            )

        if self.output_dir:
            self._episode_log_path = (
                self.output_dir
                / f"episode_return_curve_rank{self._runtime_worker_id()}.csv"
            )

    def _current_train_round_idx(self) -> int:
        if self._current_update_idx is not None and int(self._current_update_idx) > 0:
            return int(self._current_update_idx)
        return self._runtime_stage_round_idx()

    def _lr_scheduler_multiplier(self, global_step: int, total_steps: int) -> float:
        schedule = self.lr_scheduler_name
        if schedule in ("", "none", "off", "disabled"):
            return 1.0
        total_steps = max(1, int(total_steps))
        warmup_steps = min(max(0, int(self.lr_warmup_updates)), self.total_updates) * total_steps
        min_ratio = float(self.lr_min_ratio)
        step = max(0, int(global_step))
        if warmup_steps > 0 and step < warmup_steps:
            return max(min_ratio, float(step + 1) / float(warmup_steps))
        if self.total_updates <= self.lr_warmup_updates:
            progress = 1.0
        else:
            decay_steps = max(1, (self.total_updates - self.lr_warmup_updates) * total_steps)
            decay_step = max(0, step - warmup_steps)
            progress = min(1.0, float(decay_step) / float(decay_steps))
        if schedule == "linear":
            return max(min_ratio, 1.0 - (1.0 - min_ratio) * progress)
        if schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_ratio, min_ratio + (1.0 - min_ratio) * cosine)
        if schedule == "constant":
            return 1.0
        raise ValueError(
            f"Unsupported lr_scheduler={self.lr_scheduler_name!r}; expected none|linear|cosine|constant."
        )

    def _build_lr_scheduler(
        self, steps_per_update: int
    ) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
        if self.lr_scheduler_name in ("", "none", "off", "disabled"):
            self.lr_scheduler = None
            return None
        steps_per_update = max(1, int(steps_per_update))
        round_idx = self._current_train_round_idx()
        completed_updates = max(0, round_idx - 1)
        step_offset = completed_updates * steps_per_update
        for group in self.optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda local_step: self._lr_scheduler_multiplier(
                step_offset + int(local_step),
                steps_per_update,
            ),
        )
        for base_lr, group in zip(scheduler.base_lrs, self.optimizer.param_groups):
            group["lr"] = float(base_lr) * self._lr_scheduler_multiplier(
                step_offset, steps_per_update
            )
        self.lr_scheduler = scheduler
        return scheduler

    def _current_group_lrs(self) -> Dict[str, float]:
        current = {
            "actor_lr": 0.0,
            "critic_adapter_lr": 0.0,
            "value_head_lr": 0.0,
        }
        for group in self.optimizer.param_groups:
            name = str(group.get("name", ""))
            lr = float(group.get("lr", 0.0))
            if name == "actor":
                current["actor_lr"] = lr
            elif name == "critic_adapter":
                current["critic_adapter_lr"] = lr
            elif name == "value_head":
                current["value_head_lr"] = lr
        return current

    def _critic_pretrain_enabled(self) -> bool:
        return self.critic_pretrain_value_loss_threshold is not None

    def _critic_pretrain_release_status(
        self,
        epoch_value_loss: float,
    ) -> Tuple[bool, str]:
        if self.critic_pretrain_value_loss_threshold is None:
            return True, "disabled"
        if epoch_value_loss <= float(self.critic_pretrain_value_loss_threshold):
            return True, (
                f"value_loss<={self.critic_pretrain_value_loss_threshold}"
            )
        return False, f"value_loss>{self.critic_pretrain_value_loss_threshold}"

    def _runtime_stage_round_idx(self) -> int:
        raw = os.getenv("RL_STAGE_ROUND_IDX", "").strip()
        if raw:
            try:
                value = int(raw)
                if value > 0:
                    return value
            except ValueError:
                pass
        return 1

    def _runtime_worker_id(self) -> int:
        raw = (
            os.getenv("RL_WORKER_ID")
            or os.getenv("RL_WORKER_RANK")
            or os.getenv("LOCAL_RANK")
            or os.getenv("RANK")
            or ""
        ).strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
        return int(self.accelerator.process_index)

    def _runtime_loop_round_idx(self) -> Optional[int]:
        raw = os.getenv("RL_LOOP_ROUND_IDX", "").strip()
        if raw:
            try:
                value = int(raw)
                if value > 0:
                    return value
            except ValueError:
                pass
        return None

    def _runtime_stage_phase(self, default: str) -> str:
        raw = os.getenv("RL_STAGE_PHASE", "").strip().lower()
        return raw or default

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
            "agent0_comm_repeat": 0.0,
            "agent0_comm_forced": 0.0,
            "agent0_paired_comm": 0.0,
            "agent1_total": 0.0,
            "agent1_sequence": 0.0,
            "agent1_format": 0.0,
            "agent1_validator": 0.0,
            "agent1_comm": 0.0,
            "agent1_comm_repeat": 0.0,
            "agent1_comm_forced": 0.0,
            "agent1_paired_comm": 0.0,
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
            "agent0_custom_return,agent0_sequence_sum,agent0_format_sum,agent0_validator_sum,"
            "agent0_comm_sum,agent0_comm_repeat_sum,agent0_comm_forced_sum,agent0_paired_comm_sum,"
            "agent1_custom_return,agent1_sequence_sum,agent1_format_sum,agent1_validator_sum,"
            "agent1_comm_sum,agent1_comm_repeat_sum,agent1_comm_forced_sum,agent1_paired_comm_sum,"
            "team_custom_return"
        )
        row_idx, _ = self._prepare_csv_log(self._episode_log_path, header)
        update_idx = self._current_update_idx if self._current_update_idx is not None else -1
        rank = self._runtime_worker_id()
        self._episode_counter += 1
        with self._episode_log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{row_idx},{update_idx},{self._episode_counter},{episode_return},"
                f"{episode_len},{1 if had_positive else 0},{rank},"
                f"{custom_stats.get('agent0_total', 0.0)},{custom_stats.get('agent0_sequence', 0.0)},"
                f"{custom_stats.get('agent0_format', 0.0)},{custom_stats.get('agent0_validator', 0.0)},"
                f"{custom_stats.get('agent0_comm', 0.0)},{custom_stats.get('agent0_comm_repeat', 0.0)},"
                f"{custom_stats.get('agent0_comm_forced', 0.0)},{custom_stats.get('agent0_paired_comm', 0.0)},"
                f"{custom_stats.get('agent1_total', 0.0)},{custom_stats.get('agent1_sequence', 0.0)},"
                f"{custom_stats.get('agent1_format', 0.0)},{custom_stats.get('agent1_validator', 0.0)},"
                f"{custom_stats.get('agent1_comm', 0.0)},{custom_stats.get('agent1_comm_repeat', 0.0)},"
                f"{custom_stats.get('agent1_comm_forced', 0.0)},{custom_stats.get('agent1_paired_comm', 0.0)},"
                f"{custom_stats.get('team_total', 0.0)}\n"
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
            "agent0_comm_repeat": 0.0,
            "agent0_comm_forced": 0.0,
            "agent1_total": 0.0,
            "agent1_sequence": 0.0,
            "agent1_format": 0.0,
            "agent1_validator": 0.0,
            "agent1_comm": 0.0,
            "agent1_comm_repeat": 0.0,
            "agent1_comm_forced": 0.0,
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
            comm_repeat = sum(
                float(call.get("repeat_communication_reward", 0.0) or 0.0)
                for call in calls
            )
            comm_forced = sum(
                float(call.get("forced_communication_reward", 0.0) or 0.0)
                for call in calls
            )
            paired_comm = sum(
                float(call.get("paired_comm_reward", 0.0) or 0.0) for call in calls
            )
            total = seq + fmt + validator + comm + paired_comm
            prefix = f"agent{agent_idx}"
            stats[f"{prefix}_total"] = total
            stats[f"{prefix}_sequence"] = seq
            stats[f"{prefix}_format"] = fmt
            stats[f"{prefix}_validator"] = validator
            stats[f"{prefix}_comm"] = comm
            stats[f"{prefix}_comm_repeat"] = comm_repeat
            stats[f"{prefix}_comm_forced"] = comm_forced
            stats[f"{prefix}_paired_comm"] = paired_comm
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
            if abs(float(self._current_episode_return) - 20.0) < 1e-6:
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
        prefix, dynamic = self._build_critic_prompt_parts(
            agent_index=agent_index,
            messages=messages,
            context=context,
            output_text=output_text,
        )
        return prefix + dynamic

    def _build_critic_prompt_parts(
        self,
        agent_index: int,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
        output_text: str,
    ) -> Tuple[str, str]:
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
        prefix_sections = [
            self.critic_role_prompt,
            "[Game Rules]",
            game_rules or "[MISSING]",
            "",
            f"[{role_name} Action Space]",
            acting_actions or "[MISSING]",
            "",
            f"[{teammate_name} Action Space]",
            teammate_actions or "[MISSING]",
            "",
            "[Evaluation Focus]",
            "Estimate the value of this transition for future cumulative return. "
            "Focus on task progress, coordination quality, rule compliance, repeated communication, "
            "format errors, validator errors, and whether the output is an effective embodied action "
            "or an effective communication move under the stated constraints.",
            "",
        ]
        dynamic_sections = [
            "[Global Observation]",
            self._format_global_observation(context.get("global_observation")),
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
        ]
        return "\n".join(prefix_sections), "\n".join(dynamic_sections)

    @staticmethod
    def _split_actor_prompt_prefix(messages: List[Dict[str, str]], chat_prompt: str) -> str:
        latest_user = next(
            (msg.get("content", "") for msg in reversed(messages) if msg.get("role") == "user"),
            "",
        )
        marker = "<input>\n"
        if marker not in latest_user:
            return ""
        fixed_user_prefix, dynamic_user_suffix = latest_user.split(marker, 1)
        fixed_user_prefix += marker
        if not dynamic_user_suffix:
            return ""
        split_pos = chat_prompt.find(fixed_user_prefix)
        if split_pos == -1:
            return ""
        prefix_end = split_pos + len(fixed_user_prefix)
        return chat_prompt[:prefix_end]

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
            "critic_adapter": {"adapter_name": "critic", "lora_path": "..."},
            "value_head_path": "..."
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
            value_head_path = override.get("value_head_path")
            if value_head_path:
                cfg["value_head_path"] = value_head_path
            return
        # fallback：纯字符串表示 model_path
        cfg["model_path"] = content
        cfg["lora_path"] = cfg.get("lora_path", None)

    # ------------------------------------------------------------------
    def _policy_call(self, agent_index: int, messages, context):
        if self.text_policy is None:
            return self._policy_call_without_local_model(agent_index, messages, context)
        return self._policy_call_vllm_actor_local_critic(
            agent_index=agent_index,
            messages=messages,
            context=context,
        )

    def _policy_call_without_local_model(self, agent_index: int, messages, context):
        from ..agents.utils import convert_messages_to_prompt

        assert self.policy_tokenizer is not None
        response_text, actor_metadata = self._query_vllm_actor(
            agent_index=agent_index,
            messages=messages,
            temperature=(context or {}).get("temperature"),
        )
        prompt_text = convert_messages_to_prompt(messages)
        tokenizer = self.policy_tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            chat_prompt = prompt_text
        prompt_ids = tokenizer(
            chat_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0).cpu()
        response_ids = tokenizer(
            response_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0).cpu()
        critic_prefix, critic_dynamic = self._build_critic_prompt_parts(
            agent_index=agent_index,
            messages=messages,
            context=context,
            output_text=response_text,
        )
        critic_prompt = critic_prefix + critic_dynamic
        critic_input_ids = tokenizer(
            critic_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].squeeze(0).cpu()
        critic_value = 0.0
        if self.compute_values_in_collect and self.use_vllm_value_service:
            critic_value = self._query_vllm_value(
                critic_prompt=critic_prompt,
                timeout=float(
                    self.agents_cfg.get(f"agent_{agent_index}", {}).get(
                        "timeout", self.trainer_cfg.get("vllm_request_timeout", 120)
                    )
                ),
            )
        response_log_probs = actor_metadata.get("response_log_probs") or []
        response_log_probs_tensor = torch.tensor(response_log_probs, dtype=torch.float32)
        total_log_prob = actor_metadata.get("log_prob")
        if total_log_prob is None:
            total_log_prob = (
                float(response_log_probs_tensor.sum().item()) if response_log_probs else 0.0
            )
        metadata = {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_log_probs": response_log_probs_tensor,
            "log_prob": float(total_log_prob),
            "policy_temperature": context.get("temperature"),
            "value": float(critic_value),
            "entropy": 0.0,
            "token_count": int(actor_metadata.get("token_count", len(response_ids))),
            "critic_input_ids": critic_input_ids,
            "response_tokens": actor_metadata.get("response_tokens"),
        }
        return response_text, metadata

    def _vllm_endpoint(self, path: str) -> Tuple[int, str]:
        rank_raw = (
            os.environ.get("RL_WORKER_RANK")
            or os.environ.get("LOCAL_RANK")
            or os.environ.get("RANK")
            or "0"
        )
        try:
            rank = int(rank_raw)
        except ValueError:
            rank = 0
        host = os.environ["RL_VLLM_HOST"]
        start_port = int(os.environ["RL_VLLM_START_PORT"])
        return rank, f"http://{host}:{start_port + rank}{path}"

    def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"vLLM endpoint HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"vLLM endpoint unavailable: {url} ({exc})") from exc
        return json.loads(raw)

    def _query_vllm_actor(
        self,
        agent_index: int,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
    ) -> Tuple[str, Dict[str, Any]]:
        model_name = self.agents_cfg.get(f"agent_{agent_index}", {}).get(
            "model", "qwen2.5-7B-instruct"
        )
        actor_adapter_name = None
        actor_spec = self.actor_adapters.get(agent_index)
        if actor_spec is not None and actor_spec.name == model_name:
            actor_adapter_name = actor_spec.name
        timeout = float(
            self.agents_cfg.get(f"agent_{agent_index}", {}).get(
                "timeout", self.trainer_cfg.get("vllm_request_timeout", 120)
            )
        )
        rank, endpoint_url = self._vllm_endpoint("/rl/generate")
        request_started = time.time()
        print(
            "[MAPPOTrainer] vllm request start "
            f"rank={rank} agent={agent_index} model={model_name} "
            f"url={endpoint_url} timeout={timeout} messages={len(messages)}"
        )
        response = self._post_json(
            endpoint_url,
            {
                "messages": messages,
                "adapter_name": actor_adapter_name,
                "temperature": float(temperature if temperature is not None else 0.0),
                "max_tokens": int(self.trainer_cfg.get("max_new_tokens", 512)),
            },
            timeout=timeout,
        )
        elapsed = time.time() - request_started
        response_text = str(response.get("text") or "")
        token_logprobs = [
            float(item) for item in (response.get("response_log_probs") or [])
        ]
        response_tokens = [
            str(item) for item in (response.get("response_tokens") or [])
        ]
        print(
            "[MAPPOTrainer] vllm request done "
            f"rank={rank} agent={agent_index} model={model_name} "
            f"elapsed={elapsed:.2f}s response_tokens={len(response_tokens)}"
        )
        return response_text, {
            "response_log_probs": token_logprobs,
            "response_tokens": response_tokens,
            "log_prob": float(sum(token_logprobs)) if token_logprobs else 0.0,
            "token_count": int(response.get("token_count", len(token_logprobs))),
        }

    def _query_vllm_value(self, critic_prompt: str, timeout: float) -> float:
        rank, endpoint_url = self._vllm_endpoint("/rl/value")
        request_started = time.time()
        print(
            "[MAPPOTrainer] vllm value start "
            f"rank={rank} url={endpoint_url} prompt_chars={len(critic_prompt)}"
        )
        response = self._post_json(
            endpoint_url,
            {
                "critic_text": critic_prompt,
                "adapter_name": self.critic_adapter.name if self.critic_adapter else None,
            },
            timeout=timeout,
        )
        elapsed = time.time() - request_started
        value = float(response.get("value", 0.0))
        print(
            "[MAPPOTrainer] vllm value done "
            f"rank={rank} elapsed={elapsed:.2f}s value={value:.6f}"
        )
        return value

    def _policy_call_vllm_actor_local_critic(self, agent_index: int, messages, context):
        assert self.text_policy is not None
        from ..agents.utils import convert_messages_to_prompt

        response_text, actor_metadata = self._query_vllm_actor(
            agent_index=agent_index,
            messages=messages,
            temperature=(context or {}).get("temperature"),
        )
        prompt_text = convert_messages_to_prompt(messages)
        tokenizer = self.policy_tokenizer
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            chat_prompt = prompt_text

        model = self.accelerator.unwrap_model(self.text_policy)
        prompt_ids = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).cpu()
        response_ids = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).cpu()

        critic_prefix, critic_dynamic = self._build_critic_prompt_parts(
            agent_index=agent_index,
            messages=messages,
            context=context,
            output_text=response_text,
        )
        critic_prompt = critic_prefix + critic_dynamic
        critic_input_ids = tokenizer(critic_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).cpu()
        critic_value = 0.0
        if self.compute_values_in_collect:
            if self.use_vllm_value_service:
                critic_value = self._query_vllm_value(
                    critic_prompt=critic_prompt,
                    timeout=float(
                        self.agents_cfg.get(f"agent_{agent_index}", {}).get(
                            "timeout", self.trainer_cfg.get("vllm_request_timeout", 120)
                        )
                    ),
                )
            else:
                _, critic_value = model._evaluate_value_text_with_prefix(
                    critic_prefix,
                    critic_prompt,
                    adapter_name=model.critic_adapter_name,
                )

        response_log_probs = actor_metadata.get("response_log_probs") or []
        response_log_probs_tensor = torch.tensor(response_log_probs, dtype=torch.float32)
        total_log_prob = actor_metadata.get("log_prob")
        if total_log_prob is None:
            total_log_prob = float(response_log_probs_tensor.sum().item()) if response_log_probs else 0.0
        print(
            "[MAPPOTrainer] vllm actor agent="
            f"{agent_index} response_chars={len(response_text)} response_ids={int(response_ids.numel())} "
            f"logprob_tokens={len(response_log_probs)} total_log_prob={float(total_log_prob):.6f}"
        )

        metadata = {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "response_log_probs": response_log_probs_tensor,
            "log_prob": float(total_log_prob),
            "policy_temperature": context.get("temperature"),
            "value": critic_value,
            "entropy": 0.0,
            "token_count": int(actor_metadata.get("token_count", len(response_ids))),
            "critic_input_ids": critic_input_ids,
            "response_tokens": actor_metadata.get("response_tokens"),
        }
        return response_text, metadata

    # ------------------------------------------------------------------
    def train(self):
        print(
            "[MAPPO] train() enter "
            f"stage={self._runtime_stage_phase('unknown')} "
            f"collect_only={self.collect_only} train_only={self.train_only}",
            flush=True,
        )
        if self.collect_only or self.train_only:
            print(
                "[MAPPO] before rollout_dir.mkdir "
                f"rank={self.accelerator.process_index} path={self.rollout_dir}",
                flush=True,
            )
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            print(
                "[MAPPO] after rollout_dir.mkdir "
                f"rank={self.accelerator.process_index} path={self.rollout_dir}",
                flush=True,
            )

        # Collect-only mode: just roll out and save to disk.
        if self.collect_only:
            self.rollout_dir.mkdir(parents=True, exist_ok=True)
            update_idx = self._runtime_stage_round_idx()
            self._current_update_idx = update_idx
            if self.session is not None:
                phase = self._runtime_stage_phase(
                    "eval" if "eval" in str(self.output_dir).lower() else "collect"
                )
                self.session.set_logging_context(
                    update_idx=update_idx,
                    phase=phase,
                    run_label=self.output_dir.name or self.rollout_dir.name,
                    loop_round_idx=self._runtime_loop_round_idx(),
                    rotate_session=True,
                )
            self._reset_rollout_stats()
            self.collect_rollout()
            self.log_performance(update_idx)
            self.log_rewards(update_idx, self.buffer.storage)
            self.save_rollout(self.buffer.storage, update_idx)
            self.buffer.clear()
            self.accelerator.wait_for_everyone()
            self.accelerator.print(f"[Collect] saved rollout u{update_idx:05d}")
            return

        # Train-only mode: load rollouts from disk and update policy.
        if self.train_only:
            print(
                "[MAPPO] entering train_only branch "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            print(
                "[MAPPO] train_only before update_idx "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            update_idx = self._runtime_stage_round_idx()
            print(
                "[MAPPO] train_only after update_idx "
                f"rank={self.accelerator.process_index} update_idx={update_idx}",
                flush=True,
            )
            self._current_update_idx = update_idx
            print(
                "[MAPPO] train_only before load_rollouts "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx}: begin load_rollouts()"
            )
            transitions = self.load_rollouts()
            print(
                "[MAPPO] train_only after load_rollouts "
                f"rank={self.accelerator.process_index} transitions={len(transitions)}",
                flush=True,
            )
            if not transitions:
                self.accelerator.print("[TrainOnly] No rollouts found; stopping.")
                return
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx}: loaded global transitions={len(transitions)}"
            )
            local_transitions = self._shard_transitions_for_rank(transitions)
            if not local_transitions:
                self.accelerator.print(
                    "[TrainOnly] Local rank received no rollout shard; stopping."
                )
                return
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx}: rank={self.accelerator.process_index} "
                f"local shard transitions={len(local_transitions)}"
            )
            local_transitions_for_logging = local_transitions[
                : getattr(self, "_last_local_shard_valid_count", len(local_transitions))
            ]
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx}: begin materialize_transition_values()"
            )
            self._materialize_transition_values(local_transitions)
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx}: finished materialize_transition_values()"
            )
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx}: begin update_policy()"
            )
            print(
                "[MAPPO] train_only before update_policy "
                f"rank={self.accelerator.process_index} transitions={len(local_transitions)}",
                flush=True,
            )
            loss_dict = self.update_policy(local_transitions)
            print(
                "[MAPPO] train_only after update_policy "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx}: finished update_policy()"
            )
            print(
                "[MAPPO] train_only before log_rewards "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            self.log_rewards(update_idx, local_transitions_for_logging)
            print(
                "[MAPPO] train_only after log_rewards "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            print(
                "[MAPPO] train_only before log_train_metrics "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            self.log_train_metrics(update_idx, loss_dict, local_transitions_for_logging)
            print(
                "[MAPPO] train_only after log_train_metrics "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            if self.cleanup_rollouts and self.accelerator.is_main_process:
                print(
                    "[MAPPO] train_only before cleanup_rollouts "
                    f"rank={self.accelerator.process_index}",
                    flush=True,
                )
                self._cleanup_rollout_files()
                print(
                    "[MAPPO] train_only after cleanup_rollouts "
                    f"rank={self.accelerator.process_index}",
                    flush=True,
                )
            self.accelerator.print(
                f"[TrainOnly] Update {update_idx} "
                f"loss={loss_dict['loss']:.4f} policy={loss_dict['policy']:.4f} "
                f"value={loss_dict['value']:.4f} entropy={loss_dict['entropy']:.4f}"
            )
            print(
                "[MAPPO] train_only before maybe_export_latest "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            self._maybe_export_latest(update_idx)
            print(
                "[MAPPO] train_only after maybe_export_latest "
                f"rank={self.accelerator.process_index}",
                flush=True,
            )
            return

        update_idx = self._runtime_stage_round_idx()
        self._current_update_idx = update_idx
        if self.session is not None:
            self.session.set_logging_context(
                update_idx=update_idx,
                phase=self._runtime_stage_phase("run"),
                run_label=self.output_dir.name or self.rollout_dir.name,
                loop_round_idx=self._runtime_loop_round_idx(),
                rotate_session=True,
            )
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
        if self.snapshot_paths_configured and not self.snapshot_records:
            raise RuntimeError(
                "trainer.off_policy_snapshots was configured, but no snapshot states were loaded. "
                "Please verify the snapshot path exists on the server and is readable."
            )
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
            repeat_communication_reward = float(
                breakdown.get(
                    "repeat_communication_reward",
                    raw_entry.get("repeat_communication_reward", 0.0),
                )
                or 0.0
            )
            forced_communication_reward = float(
                breakdown.get(
                    "forced_communication_reward",
                    raw_entry.get("forced_communication_reward", 0.0),
                )
                or 0.0
            )
            paired_comm_reward = float(breakdown.get("paired_comm_reward", 0.0) or 0.0)
            breakdown_total_reward = float(
                raw_entry.get(
                    "total",
                    seq_reward
                    + fmt_reward
                    + validator_reward
                    + communication_reward
                    + paired_comm_reward,
                )
                or (
                    seq_reward
                    + fmt_reward
                    + validator_reward
                    + communication_reward
                    + paired_comm_reward
                )
            )
            process_reward = seq_reward
            self.buffer.add(
                prompt_ids=meta["prompt_ids"],
                response_ids=meta["response_ids"],
                response_log_probs=meta.get("response_log_probs"),
                log_prob=meta["log_prob"],
                policy_temperature=meta.get("policy_temperature"),
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
                repeat_communication_reward=repeat_communication_reward,
                forced_communication_reward=forced_communication_reward,
                paired_comm_reward=paired_comm_reward,
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

    def _trainable_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        groups: Dict[str, List[torch.nn.Parameter]] = {"value_head": []}
        for agent_idx in sorted(self.actor_adapters.keys()):
            groups[f"actor{int(agent_idx)}"] = []
        if self.critic_adapter is not None:
            groups["critic_adapter"] = []

        for name, param in self.text_policy.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("value_head.") or ".value_head." in name:
                groups["value_head"].append(param)
                continue
            if "lora_" not in name:
                continue
            for agent_idx, spec in sorted(self.actor_adapters.items()):
                if f".{spec.name}." in name:
                    groups[f"actor{int(agent_idx)}"].append(param)
            if self.critic_adapter is not None and f".{self.critic_adapter.name}." in name:
                groups["critic_adapter"].append(param)
        return groups

    @staticmethod
    def _param_group_grad_norm(params: List[torch.nn.Parameter]) -> float:
        total = 0.0
        for param in params:
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total += float(torch.sum(grad * grad).item())
        return math.sqrt(total) if total > 0.0 else 0.0

    @staticmethod
    def _snapshot_param_group(
        params: List[torch.nn.Parameter],
    ) -> List[torch.Tensor]:
        return [param.detach().float().clone() for param in params]

    @staticmethod
    def _param_group_delta_norm(
        params: List[torch.nn.Parameter],
        before: List[torch.Tensor],
    ) -> float:
        total = 0.0
        for param, prev in zip(params, before):
            delta = param.detach().float() - prev
            total += float(torch.sum(delta * delta).item())
        return math.sqrt(total) if total > 0.0 else 0.0

    @staticmethod
    def _filter_trainable_params(
        params: List[torch.nn.Parameter],
    ) -> List[torch.nn.Parameter]:
        seen: set[int] = set()
        filtered: List[torch.nn.Parameter] = []
        for param in params:
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            filtered.append(param)
        return filtered

    def _compute_fresh_value_metrics(
        self,
        transitions: List[TextTransition],
        returns_cpu: torch.Tensor,
        agent_indices: List[int],
    ) -> Dict[str, float]:
        if not transitions:
            return {
                "fresh_value_mean": 0.0,
                "fresh_explained_var": 0.0,
                "agent0_fresh_value_mean": 0.0,
                "agent0_fresh_explained_var": 0.0,
                "agent1_fresh_value_mean": 0.0,
                "agent1_fresh_explained_var": 0.0,
            }

        prompt_tensors = [t.prompt_ids for t in transitions]
        response_tensors = [t.response_ids for t in transitions]
        old_token_log_probs_list: List[torch.Tensor] = []
        critic_tensors = [t.critic_input_ids for t in transitions]
        fresh_values_chunks: List[torch.Tensor] = []
        unwrapped_model: QwenLMActorCritic = self.accelerator.unwrap_model(
            self.text_policy
        )
        batch_span = max(1, int(getattr(unwrapped_model, "eval_batch_size", 1)))
        was_training = self.text_policy.training
        self.text_policy.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(transitions), batch_span):
                    end = min(start + batch_span, len(transitions))
                    fresh_values_chunks.append(
                        unwrapped_model.evaluate_values_batch(
                            prompt_tensors[start:end],
                            response_tensors[start:end],
                            agent_indices[start:end],
                            critic_tensors=critic_tensors[start:end],
                        ).detach().float().cpu()
                    )
        finally:
            if was_training:
                self.text_policy.train()

        fresh_values_cpu = (
            torch.cat(fresh_values_chunks, dim=0)
            if fresh_values_chunks
            else torch.empty(0, dtype=torch.float32)
        )
        metrics: Dict[str, float] = {
            "fresh_value_mean": 0.0,
            "fresh_explained_var": 0.0,
            "agent0_fresh_value_mean": 0.0,
            "agent0_fresh_explained_var": 0.0,
            "agent1_fresh_value_mean": 0.0,
            "agent1_fresh_explained_var": 0.0,
        }
        if fresh_values_cpu.numel() != returns_cpu.numel():
            return metrics

        returns_var = float(torch.var(returns_cpu, unbiased=False).item())
        if returns_var > 1e-8:
            fresh_explained_var = 1.0 - float(
                torch.var(returns_cpu - fresh_values_cpu, unbiased=False).item()
            ) / returns_var
        else:
            fresh_explained_var = 0.0
        metrics["fresh_value_mean"] = float(fresh_values_cpu.mean().item())
        metrics["fresh_explained_var"] = float(fresh_explained_var)

        agent_indices_cpu = torch.tensor(agent_indices, dtype=torch.long)
        for agent_idx in range(2):
            mask = agent_indices_cpu == agent_idx
            if not bool(mask.any().item()):
                continue
            agent_returns = returns_cpu[mask]
            agent_values = fresh_values_cpu[mask]
            agent_returns_var = float(torch.var(agent_returns, unbiased=False).item())
            if agent_returns_var > 1e-8:
                agent_fresh_explained_var = 1.0 - float(
                    torch.var(agent_returns - agent_values, unbiased=False).item()
                ) / agent_returns_var
            else:
                agent_fresh_explained_var = 0.0
            metrics[f"agent{agent_idx}_fresh_value_mean"] = float(
                agent_values.mean().item()
            )
            metrics[f"agent{agent_idx}_fresh_explained_var"] = float(
                agent_fresh_explained_var
            )
        return metrics

    def _materialize_transition_values(self, transitions: List[TextTransition]) -> None:
        assert self.text_policy is not None
        unwrapped_model: QwenLMActorCritic = self.accelerator.unwrap_model(
            self.text_policy
        )
        print(
            "[MAPPO] materialize before "
            f"rank={self.accelerator.process_index} transitions={len(transitions)}",
            flush=True,
        )
        missing = [idx for idx, t in enumerate(transitions) if t.critic_input_ids is not None and float(t.value) == 0.0]
        if not missing:
            print(
                "[MAPPO] materialize skip "
                f"rank={self.accelerator.process_index} missing=0",
                flush=True,
            )
            return
        critic_tensors = [transitions[idx].critic_input_ids for idx in missing]
        batch_span = max(1, int(getattr(unwrapped_model, "eval_batch_size", 1)))
        print(
            "[MAPPO] materialize missing "
            f"rank={self.accelerator.process_index} missing={len(missing)} "
            f"batch_span={batch_span}",
            flush=True,
        )
        was_training = self.text_policy.training
        self.text_policy.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(missing), batch_span):
                    end = min(start + batch_span, len(missing))
                    batch_indices = missing[start:end]
                    print(
                        "[MAPPO] materialize batch "
                        f"rank={self.accelerator.process_index} start={start} end={end}",
                        flush=True,
                    )
                    values = unwrapped_model._evaluate_value_inputs(
                        [transitions[idx].critic_input_ids.to(self.device) for idx in batch_indices],
                        use_critic_adapter=True,
                    ).detach().float().cpu()
                    for local_idx, global_idx in enumerate(batch_indices):
                        transitions[global_idx].value = float(values[local_idx].item())
        finally:
            if was_training:
                self.text_policy.train()
        print(
            "[MAPPO] materialize after "
            f"rank={self.accelerator.process_index}",
            flush=True,
        )

    # ------------------------------------------------------------------
    def update_policy(self, transitions: List[TextTransition]):
        assert self.text_policy is not None
        unwrapped_model: QwenLMActorCritic = self.accelerator.unwrap_model(
            self.text_policy
        )
        rank = self.accelerator.process_index
        critic_pretrain_gate = self._critic_pretrain_enabled()
        prompt_tensors = [t.prompt_ids for t in transitions]
        response_tensors = [t.response_ids for t in transitions]
        critic_tensors = [t.critic_input_ids for t in transitions]
        agent_indices = [t.agent_index for t in transitions]
        policy_temperatures = [t.policy_temperature for t in transitions]
        old_token_log_probs_list: List[torch.Tensor] = []
        for t in transitions:
            if t.response_log_probs is not None:
                old_token_log_probs_list.append(
                    t.response_log_probs.to(self.device, dtype=torch.float32)
                )
                continue
            resp_len = max(1, int(t.response_ids.numel()))
            avg_lp = float(t.log_prob) / float(resp_len)
            old_token_log_probs_list.append(
                torch.full(
                    (resp_len,),
                    avg_lp,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        old_values = torch.tensor(
            [t.value for t in transitions], dtype=torch.float32, device=self.device
        )
        advantages, returns = self.compute_advantages(transitions)
        raw_advantages = advantages.clone()
        returns_stats = returns.clone()
        if advantages.numel() > 1:
            adv_std = advantages.std(unbiased=False).clamp(min=1e-6)
            advantages = (advantages - advantages.mean()) / adv_std
        else:
            advantages = advantages - advantages.mean()
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        batch_size = max(1, self.train_batch_size)
        num_transitions = len(transitions)
        num_minibatches = math.ceil(num_transitions / batch_size)
        total_optimization_steps = max(1, self.update_epochs * num_minibatches)
        optimizer_steps_per_update = max(
            1,
            math.ceil(total_optimization_steps / float(self.gradient_accumulation_steps)),
        )
        print(
            "[MAPPO] update_policy enter "
            f"rank={rank} transitions={num_transitions} batch_size={batch_size} "
            f"epochs={self.update_epochs} minibatches={num_minibatches} "
            f"grad_accum={self.gradient_accumulation_steps} "
            f"optimizer_steps_per_update={optimizer_steps_per_update} "
            f"critic_pretrain_gate={critic_pretrain_gate} "
            f"critic_pretrain_value_loss_threshold={self.critic_pretrain_value_loss_threshold}",
            flush=True,
        )
        scheduler = self._build_lr_scheduler(optimizer_steps_per_update)
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_approx_kl = 0.0
        total_kl_penalty = 0.0
        total_value_clipfrac = 0.0
        total_policy_active_clipfrac = 0.0
        total_policy_active_approx_kl = 0.0
        total_policy_active_token_clipfrac = 0.0
        total_policy_active_token_approx_kl = 0.0
        param_groups = self._trainable_param_groups()
        total_group_grad_norms: Dict[str, float] = {
            name: 0.0 for name in param_groups.keys()
        }
        total_group_param_deltas: Dict[str, float] = {
            name: 0.0 for name in param_groups.keys()
        }
        metric_steps = 0
        policy_active_metric_steps = 0
        actual_optimization_steps = 0
        accum_counter = 0
        stop_early = False
        stop_policy_updates = critic_pretrain_gate
        accum_approx_kl_sum = 0.0
        accum_approx_kl_count = 0

        self.optimizer.zero_grad()
        last_pretrain_value_mean = float("inf")
        for epoch_idx in range(self.update_epochs):
            epoch_value_total = 0.0
            epoch_metric_steps = 0
            print(
                "[MAPPO] update_policy epoch_start "
                f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs}",
                flush=True,
            )
            if self.shuffle_minibatches and num_transitions > 1:
                perm = torch.randperm(num_transitions)
                ordered_indices = perm.tolist()
            else:
                ordered_indices = list(range(num_transitions))

            for minibatch_idx, start in enumerate(
                range(0, num_transitions, batch_size), start=1
            ):
                end = min(start + batch_size, num_transitions)
                batch_indices = ordered_indices[start:end]
                print(
                    "[MAPPO] update_policy minibatch_start "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"minibatch={minibatch_idx}/{num_minibatches} "
                    f"start={start} end={end} size={len(batch_indices)}",
                    flush=True,
                )
                batch_prompts = [prompt_tensors[i] for i in batch_indices]
                batch_responses = [response_tensors[i] for i in batch_indices]
                batch_critic = [critic_tensors[i] for i in batch_indices]
                batch_agent_indices = [agent_indices[i] for i in batch_indices]
                batch_old_values = old_values[batch_indices]
                batch_adv = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                policy_forward_ctx = (
                    nullcontext() if not stop_policy_updates else torch.no_grad()
                )
                print(
                    "[MAPPO] update_policy before policy_forward "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"minibatch={minibatch_idx}/{num_minibatches} "
                    f"stop_policy_updates={stop_policy_updates}",
                    flush=True,
                )
                with policy_forward_ctx:
                    _, entropies, token_log_probs = unwrapped_model.evaluate_policy_batch(
                        batch_prompts,
                        batch_responses,
                        batch_agent_indices,
                        policy_temperatures=[
                            policy_temperatures[i] for i in batch_indices
                        ],
                    )
                print(
                    "[MAPPO] update_policy after policy_forward "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"minibatch={minibatch_idx}/{num_minibatches}",
                    flush=True,
                )
                entropies = entropies.to(self.device, dtype=torch.float32)
                batch_old_token_log_probs = [
                    old_token_log_probs_list[i] for i in batch_indices
                ]
                token_logratio_parts: List[torch.Tensor] = []
                token_policy_loss_parts: List[torch.Tensor] = []
                for sample_idx, (new_lp, old_lp) in enumerate(
                    zip(token_log_probs, batch_old_token_log_probs)
                ):
                    if new_lp.numel() == 0:
                        continue
                    if old_lp.numel() != new_lp.numel():
                        if old_lp.numel() == 0:
                            old_lp = torch.zeros_like(new_lp)
                        else:
                            old_lp = old_lp[: new_lp.numel()]
                            if old_lp.numel() < new_lp.numel():
                                pad = old_lp.new_full(
                                    (new_lp.numel() - old_lp.numel(),),
                                    float(old_lp[-1].item()) if old_lp.numel() > 0 else 0.0,
                                )
                                old_lp = torch.cat([old_lp, pad], dim=0)
                    sample_token_logratio = new_lp - old_lp
                    sample_token_ratios = torch.exp(sample_token_logratio)
                    sample_adv = batch_adv[sample_idx].expand_as(sample_token_logratio)
                    sample_clipped_ratios = torch.clamp(
                        sample_token_ratios,
                        1.0 - self.clip_coef,
                        1.0 + self.clip_coef,
                    )
                    sample_surr1 = sample_token_ratios * sample_adv
                    sample_surr2 = sample_clipped_ratios * sample_adv
                    token_logratio_parts.append(sample_token_logratio)
                    # Keep each sample on the same footing instead of letting long
                    # responses dominate the PPO objective through summed log-probs.
                    token_policy_loss_parts.append(
                        -torch.min(sample_surr1, sample_surr2).mean()
                    )
                token_logratio = (
                    torch.cat(token_logratio_parts, dim=0)
                    if token_logratio_parts
                    else torch.empty(0, dtype=torch.float32, device=self.device)
                )
                token_ratios = (
                    torch.exp(token_logratio)
                    if token_logratio.numel() > 0
                    else token_logratio
                )
                policy_loss_raw = (
                    torch.stack(token_policy_loss_parts).mean()
                    if token_policy_loss_parts
                    else torch.zeros((), dtype=torch.float32, device=self.device)
                )
                # PPO-style non-negative KL approximation under the rollout policy.
                approx_kl = (
                    ((token_ratios - 1.0) - token_logratio).mean().clamp_min(0.0)
                    if token_logratio.numel() > 0
                    else torch.zeros((), dtype=torch.float32, device=self.device)
                )
                token_approx_kl = approx_kl
                kl_penalty_raw = approx_kl * self.kl_penalty_coef
                entropy_loss_raw = -entropies.mean()
                if stop_policy_updates:
                    policy_loss = torch.zeros_like(policy_loss_raw)
                    entropy_loss = torch.zeros_like(entropy_loss_raw)
                    kl_penalty = torch.zeros_like(kl_penalty_raw)
                else:
                    policy_loss = policy_loss_raw
                    entropy_loss = entropy_loss_raw
                    kl_penalty = kl_penalty_raw
                if not stop_policy_updates:
                    policy_objective = (
                        policy_loss_raw
                        + self.entropy_coef * entropy_loss_raw
                        + kl_penalty_raw
                    )
                    print(
                        "[MAPPO] update_policy before policy_backward "
                        f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                        f"minibatch={minibatch_idx}/{num_minibatches}",
                        flush=True,
                    )
                    self.accelerator.backward(
                        policy_objective / float(self.gradient_accumulation_steps)
                    )
                    print(
                        "[MAPPO] update_policy after policy_backward "
                        f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                        f"minibatch={minibatch_idx}/{num_minibatches}",
                        flush=True,
                    )
                print(
                    "[MAPPO] update_policy before value_forward "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"minibatch={minibatch_idx}/{num_minibatches}",
                    flush=True,
                )
                values = unwrapped_model.evaluate_values_batch_train(
                    batch_prompts,
                    batch_responses,
                    batch_agent_indices,
                    critic_tensors=batch_critic,
                ).to(self.device, dtype=torch.float32)
                print(
                    "[MAPPO] update_policy after value_forward "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"minibatch={minibatch_idx}/{num_minibatches}",
                    flush=True,
                )
                value_pred_clipped = torch.clamp(
                    values,
                    batch_old_values - self.value_clip_coef,
                    batch_old_values + self.value_clip_coef,
                )
                value_losses = (values - batch_returns) ** 2
                value_losses_clipped = (value_pred_clipped - batch_returns) ** 2
                value_loss_unclipped = 0.5 * value_losses.mean()
                value_loss_clipped = 0.5 * torch.max(
                    value_losses, value_losses_clipped
                ).mean()
                value_loss = (
                    value_loss_unclipped
                    if critic_pretrain_gate
                    else value_loss_clipped
                )
                batch_value_loss_for_gate = float(
                    value_loss_unclipped.detach().item()
                )
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + kl_penalty
                )
                print(
                    "[MAPPO] update_policy before value_backward "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"minibatch={minibatch_idx}/{num_minibatches}",
                    flush=True,
                )
                released_current_minibatch = False
                if critic_pretrain_gate:
                    last_pretrain_value_mean = batch_value_loss_for_gate
                    can_release, release_reason = self._critic_pretrain_release_status(
                        epoch_value_loss=batch_value_loss_for_gate,
                    )
                    if can_release and not stop_early:
                        critic_pretrain_gate = False
                        stop_policy_updates = False
                        released_current_minibatch = True
                        print(
                            "[MAPPO] critic_pretrain release_actor "
                            f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                            f"minibatch={minibatch_idx}/{num_minibatches} "
                            f"value_loss={batch_value_loss_for_gate:.6f} "
                            f"reason={release_reason}",
                            flush=True,
                        )
                    else:
                        print(
                            "[MAPPO] critic_pretrain keep_actor_blocked "
                            f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                            f"minibatch={minibatch_idx}/{num_minibatches} "
                            f"value_loss={batch_value_loss_for_gate:.6f} "
                            f"reason={release_reason} stop_early={stop_early}",
                            flush=True,
                        )
                self.accelerator.backward(
                    (self.value_coef * value_loss)
                    / float(self.gradient_accumulation_steps)
                )
                print(
                    "[MAPPO] update_policy after value_backward "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"minibatch={minibatch_idx}/{num_minibatches}",
                    flush=True,
                )
                if released_current_minibatch:
                    print(
                        "[MAPPO] critic_pretrain before release_policy_forward "
                        f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                        f"minibatch={minibatch_idx}/{num_minibatches}",
                        flush=True,
                    )
                    _, entropies_release, token_log_probs_release = (
                        unwrapped_model.evaluate_policy_batch(
                            batch_prompts,
                            batch_responses,
                            batch_agent_indices,
                            policy_temperatures=[
                                policy_temperatures[i] for i in batch_indices
                            ],
                        )
                    )
                    token_logratio_parts_release: List[torch.Tensor] = []
                    token_policy_loss_parts_release: List[torch.Tensor] = []
                    for sample_idx, (new_lp, old_lp) in enumerate(
                        zip(token_log_probs_release, batch_old_token_log_probs)
                    ):
                        if new_lp.numel() == 0:
                            continue
                        if old_lp.numel() != new_lp.numel():
                            if old_lp.numel() == 0:
                                old_lp = torch.zeros_like(new_lp)
                            else:
                                old_lp = old_lp[: new_lp.numel()]
                                if old_lp.numel() < new_lp.numel():
                                    pad = old_lp.new_full(
                                        (new_lp.numel() - old_lp.numel(),),
                                        float(old_lp[-1].item()) if old_lp.numel() > 0 else 0.0,
                                    )
                                    old_lp = torch.cat([old_lp, pad], dim=0)
                        sample_token_logratio = new_lp - old_lp
                        sample_token_ratios = torch.exp(sample_token_logratio)
                        sample_adv = batch_adv[sample_idx].expand_as(
                            sample_token_logratio
                        )
                        sample_clipped_ratios = torch.clamp(
                            sample_token_ratios,
                            1.0 - self.clip_coef,
                            1.0 + self.clip_coef,
                        )
                        sample_surr1 = sample_token_ratios * sample_adv
                        sample_surr2 = sample_clipped_ratios * sample_adv
                        token_logratio_parts_release.append(sample_token_logratio)
                        token_policy_loss_parts_release.append(
                            -torch.min(sample_surr1, sample_surr2).mean()
                        )
                    token_logratio_release = (
                        torch.cat(token_logratio_parts_release, dim=0)
                        if token_logratio_parts_release
                        else torch.empty(0, dtype=torch.float32, device=self.device)
                    )
                    token_ratios_release = (
                        torch.exp(token_logratio_release)
                        if token_logratio_release.numel() > 0
                        else token_logratio_release
                    )
                    policy_loss_release = (
                        torch.stack(token_policy_loss_parts_release).mean()
                        if token_policy_loss_parts_release
                        else torch.zeros((), dtype=torch.float32, device=self.device)
                    )
                    approx_kl_release = (
                        ((token_ratios_release - 1.0) - token_logratio_release)
                        .mean()
                        .clamp_min(0.0)
                        if token_logratio_release.numel() > 0
                        else torch.zeros((), dtype=torch.float32, device=self.device)
                    )
                    entropy_loss_release = -entropies_release.to(
                        self.device, dtype=torch.float32
                    ).mean()
                    kl_penalty_release = approx_kl_release * self.kl_penalty_coef
                    policy_objective_release = (
                        policy_loss_release
                        + self.entropy_coef * entropy_loss_release
                        + kl_penalty_release
                    )
                    self.accelerator.backward(
                        policy_objective_release
                        / float(self.gradient_accumulation_steps)
                    )
                    policy_loss_raw = policy_loss_release
                    policy_loss = policy_loss_release
                    entropy_loss_raw = entropy_loss_release
                    entropy_loss = entropy_loss_release
                    kl_penalty_raw = kl_penalty_release
                    kl_penalty = kl_penalty_release
                    approx_kl = approx_kl_release
                    token_approx_kl = approx_kl_release
                    token_ratios = token_ratios_release
                    entropies = entropies_release.to(self.device, dtype=torch.float32)
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        + self.entropy_coef * entropy_loss
                        + kl_penalty
                    )
                    print(
                        "[MAPPO] critic_pretrain after release_policy_backward "
                        f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                        f"minibatch={minibatch_idx}/{num_minibatches}",
                        flush=True,
                    )
                accum_counter += 1
                accum_approx_kl_sum += float(approx_kl.item())
                accum_approx_kl_count += 1
                should_step = (
                    accum_counter >= self.gradient_accumulation_steps
                    or end >= num_transitions
                )
                step_approx_kl = (
                    accum_approx_kl_sum / float(accum_approx_kl_count)
                    if accum_approx_kl_count > 0
                    else float(approx_kl.item())
                )
                if should_step:
                    print(
                        "[MAPPO] update_policy before optimizer_step "
                        f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                        f"minibatch={minibatch_idx}/{num_minibatches} "
                        f"accum_counter={accum_counter}",
                        flush=True,
                    )
                    group_snapshots = {
                        name: self._snapshot_param_group(params)
                        for name, params in param_groups.items()
                    }
                    for name, params in param_groups.items():
                        total_group_grad_norms[name] += self._param_group_grad_norm(
                            params
                        )
                    self.accelerator.clip_grad_norm_(
                        self.text_policy.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.accelerator.unwrap_model(self.text_policy).clear_prefix_cache()
                    for name, params in param_groups.items():
                        total_group_param_deltas[name] += self._param_group_delta_norm(
                            params, group_snapshots[name]
                        )
                    self.optimizer.zero_grad()
                    actual_optimization_steps += 1
                    accum_counter = 0
                    accum_approx_kl_sum = 0.0
                    accum_approx_kl_count = 0
                    print(
                        "[MAPPO] update_policy after optimizer_step "
                        f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                        f"minibatch={minibatch_idx}/{num_minibatches} "
                        f"actual_steps={actual_optimization_steps}",
                        flush=True,
                    )

                total_loss += loss.item()
                total_policy += policy_loss_raw.item()
                total_value += value_loss.item()
                epoch_value_total += float(value_loss.item())
                epoch_metric_steps += 1
                total_entropy += entropies.mean().item()
                total_clipfrac += (
                    ((token_ratios - 1.0).abs() > self.clip_coef).float().mean().item()
                    if token_ratios.numel() > 0
                    else 0.0
                )
                total_approx_kl += approx_kl.item()
                total_kl_penalty += kl_penalty_raw.item()
                total_value_clipfrac += (
                    (value_losses_clipped > value_losses).float().mean().item()
                )
                metric_steps += 1
                if not stop_policy_updates:
                    total_policy_active_clipfrac += (
                        ((token_ratios - 1.0).abs() > self.clip_coef).float().mean().item()
                        if token_ratios.numel() > 0
                        else 0.0
                    )
                    total_policy_active_approx_kl += approx_kl.item()
                    total_policy_active_token_clipfrac += (
                        ((token_ratios - 1.0).abs() > self.clip_coef)
                        .float()
                        .mean()
                        .item()
                    )
                    total_policy_active_token_approx_kl += token_approx_kl.item()
                    policy_active_metric_steps += 1
                if (
                    (not stop_policy_updates)
                    and should_step
                    and self.target_kl is not None
                    and step_approx_kl > self.target_kl
                ):
                    stop_early = True
                    stop_policy_updates = True
                    print(
                        "[MAPPO] update_policy target_kl_reached "
                        f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                        f"minibatch={minibatch_idx}/{num_minibatches} "
                        f"step_approx_kl={step_approx_kl:.6f} target_kl={self.target_kl}",
                        flush=True,
                    )
                print(
                    "[MAPPO] update_policy minibatch_done "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"minibatch={minibatch_idx}/{num_minibatches} "
                    f"should_step={should_step} loss={loss.item():.6f} "
                    f"policy_loss={policy_loss_raw.item():.6f} "
                    f"value_loss={value_loss.item():.6f} approx_kl={approx_kl.item():.6f}",
                    flush=True,
                )
            print(
                "[MAPPO] update_policy epoch_done "
                f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs}",
                flush=True,
            )
            if critic_pretrain_gate and epoch_metric_steps > 0:
                epoch_value_mean = epoch_value_total / float(epoch_metric_steps)
                last_pretrain_value_mean = epoch_value_mean
                print(
                    "[MAPPO] critic_pretrain epoch_blocked "
                    f"rank={rank} epoch={epoch_idx + 1}/{self.update_epochs} "
                    f"epoch_value_loss={epoch_value_mean:.6f}",
                    flush=True,
                )

        metric_denom = metric_steps if metric_steps > 0 else 1
        avg_loss = total_loss / metric_denom if metric_denom > 0 else 0.0
        avg_policy = total_policy / metric_denom if metric_denom > 0 else 0.0
        avg_value = total_value / metric_denom if metric_denom > 0 else 0.0
        avg_entropy = total_entropy / metric_denom if metric_denom > 0 else 0.0
        avg_clipfrac = total_clipfrac / metric_denom if metric_denom > 0 else 0.0
        avg_approx_kl = total_approx_kl / metric_denom if metric_denom > 0 else 0.0
        avg_kl_penalty = total_kl_penalty / metric_denom if metric_denom > 0 else 0.0
        avg_value_clipfrac = (
            total_value_clipfrac / metric_denom if metric_denom > 0 else 0.0
        )
        policy_metric_denom = (
            policy_active_metric_steps if policy_active_metric_steps > 0 else 1
        )
        avg_policy_active_clipfrac = (
            total_policy_active_clipfrac / policy_metric_denom
            if policy_active_metric_steps > 0
            else 0.0
        )
        avg_policy_active_approx_kl = (
            total_policy_active_approx_kl / policy_metric_denom
            if policy_active_metric_steps > 0
            else 0.0
        )
        avg_policy_active_token_clipfrac = (
            total_policy_active_token_clipfrac / policy_metric_denom
            if policy_active_metric_steps > 0
            else 0.0
        )
        avg_policy_active_token_approx_kl = (
            total_policy_active_token_approx_kl / policy_metric_denom
            if policy_active_metric_steps > 0
            else 0.0
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
        per_agent_metrics: Dict[int, Dict[str, float]] = {}
        if transitions:
            agent_indices_cpu = torch.tensor(agent_indices, dtype=torch.long)
            for agent_idx in sorted({int(idx) for idx in agent_indices}):
                mask = agent_indices_cpu == int(agent_idx)
                if not bool(mask.any().item()):
                    continue
                agent_adv = raw_advantages[mask]
                agent_returns = returns_cpu[mask]
                agent_values = old_values_cpu[mask]
                agent_returns_var = float(
                    torch.var(agent_returns, unbiased=False).item()
                )
                if agent_returns_var > 1e-8:
                    agent_explained_var = 1.0 - float(
                        torch.var(
                            agent_returns - agent_values, unbiased=False
                        ).item()
                    ) / agent_returns_var
                else:
                    agent_explained_var = 0.0
                per_agent_metrics[int(agent_idx)] = {
                    "adv_mean": float(agent_adv.mean().item()),
                    "return_mean": float(agent_returns.mean().item()),
                    "value_mean": float(agent_values.mean().item()),
                    "explained_var": agent_explained_var,
                }
        if self.enable_fresh_value_metrics:
            print(
                "[MAPPO] update_policy before fresh_value_metrics "
                f"rank={rank}",
                flush=True,
            )
            fresh_value_metrics = self._compute_fresh_value_metrics(
                transitions=transitions,
                returns_cpu=returns_cpu,
                agent_indices=agent_indices,
            )
            print(
                "[MAPPO] update_policy after fresh_value_metrics "
                f"rank={rank}",
                flush=True,
            )
        else:
            fresh_value_metrics = {
                "fresh_value_mean": 0.0,
                "fresh_explained_var": 0.0,
                "agent0_fresh_value_mean": 0.0,
                "agent0_fresh_explained_var": 0.0,
                "agent1_fresh_value_mean": 0.0,
                "agent1_fresh_explained_var": 0.0,
            }

        result = {
            "loss": avg_loss,
            "policy": avg_policy,
            "value": avg_value,
            "entropy": avg_entropy,
            "reward_mean": reward_mean,
            "adv_mean": float(raw_advantages.mean().item()),
            "return_mean": float(returns_cpu.mean().item()),
            "value_mean": float(old_values_cpu.mean().item()),
            "explained_var": explained_var,
            "fresh_value_mean": float(fresh_value_metrics.get("fresh_value_mean", 0.0)),
            "fresh_explained_var": float(
                fresh_value_metrics.get("fresh_explained_var", 0.0)
            ),
            "clipfrac": avg_clipfrac,
            "approx_kl": avg_approx_kl,
            "policy_active_clipfrac": avg_policy_active_clipfrac,
            "policy_active_approx_kl": avg_policy_active_approx_kl,
            "policy_active_token_clipfrac": avg_policy_active_token_clipfrac,
            "policy_active_token_approx_kl": avg_policy_active_token_approx_kl,
            "kl_penalty": avg_kl_penalty,
            "kl_penalty_coef": float(self.kl_penalty_coef),
            "value_clipfrac": avg_value_clipfrac,
            "optimizer_steps": float(actual_optimization_steps),
            "stopped_early": 1.0 if stop_early else 0.0,
            "critic_pretrain_active_end": 1.0 if critic_pretrain_gate else 0.0,
            "critic_pretrain_released": (
                1.0
                if self._critic_pretrain_enabled() and not critic_pretrain_gate
                else 0.0
            ),
            "critic_pretrain_last_value_loss": (
                0.0
                if not self._critic_pretrain_enabled()
                else float(last_pretrain_value_mean)
            ),
        }
        step_denom = (
            float(actual_optimization_steps) if actual_optimization_steps > 0 else 1.0
        )
        for name in sorted(param_groups.keys()):
            result[f"{name}_grad_norm"] = total_group_grad_norms[name] / step_denom
            result[f"{name}_param_delta"] = total_group_param_deltas[name] / step_denom
        for agent_idx in range(2):
            agent_metrics = per_agent_metrics.get(agent_idx, {})
            result[f"agent{agent_idx}_adv_mean"] = float(
                agent_metrics.get("adv_mean", 0.0)
            )
            result[f"agent{agent_idx}_return_mean"] = float(
                agent_metrics.get("return_mean", 0.0)
            )
            result[f"agent{agent_idx}_value_mean"] = float(
                agent_metrics.get("value_mean", 0.0)
            )
            result[f"agent{agent_idx}_explained_var"] = float(
                agent_metrics.get("explained_var", 0.0)
            )
            result[f"agent{agent_idx}_fresh_value_mean"] = float(
                fresh_value_metrics.get(f"agent{agent_idx}_fresh_value_mean", 0.0)
            )
            result[f"agent{agent_idx}_fresh_explained_var"] = float(
                fresh_value_metrics.get(
                    f"agent{agent_idx}_fresh_explained_var", 0.0
                )
            )
        result.update(self._current_group_lrs())
        print(
            "[MAPPO] update_policy exit "
            f"rank={rank} avg_loss={avg_loss:.6f} avg_policy={avg_policy:.6f} "
            f"avg_value={avg_value:.6f} actual_steps={actual_optimization_steps}",
            flush=True,
        )
        return result

    def log_rewards(self, update_idx: int, transitions: List[TextTransition]):
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
                    "comm_repeat": 0.0,
                    "comm_forced": 0.0,
                    "paired_comm": 0.0,
                    "breakdown_total": 0.0,
                    "legacy_process": 0.0,
                    "rl_nonzero": 0.0,
                    "format_nonzero": 0.0,
                    "validator_nonzero": 0.0,
                    "sequence_nonzero": 0.0,
                    "comm_nonzero": 0.0,
                    "comm_repeat_nonzero": 0.0,
                    "comm_forced_nonzero": 0.0,
                    "paired_comm_nonzero": 0.0,
                    "breakdown_total_nonzero": 0.0,
                    "legacy_process_nonzero": 0.0,
                },
            )
            agent_counts[idx] = agent_counts.get(idx, 0) + 1
            rl_value = float(getattr(t, "reward", 0.0))
            format_value = float(getattr(t, "format_reward", 0.0))
            validator_value = float(getattr(t, "validator_reward", 0.0))
            sequence_value = float(getattr(t, "sequence_reward", 0.0))
            comm_value = float(getattr(t, "communication_reward", 0.0))
            comm_repeat_value = float(
                getattr(t, "repeat_communication_reward", 0.0)
            )
            comm_forced_value = float(
                getattr(t, "forced_communication_reward", 0.0)
            )
            paired_comm_value = float(getattr(t, "paired_comm_reward", 0.0))
            breakdown_total_value = float(getattr(t, "breakdown_total_reward", 0.0))
            legacy_process_value = float(
                getattr(t, "process_reward", getattr(t, "reward", 0.0))
            )
            stats["rl"] += rl_value
            stats["format"] += format_value
            stats["validator"] += validator_value
            stats["sequence"] += sequence_value
            stats["comm"] += comm_value
            stats["comm_repeat"] += comm_repeat_value
            stats["comm_forced"] += comm_forced_value
            stats["paired_comm"] += paired_comm_value
            stats["breakdown_total"] += breakdown_total_value
            stats["legacy_process"] += legacy_process_value
            stats["rl_nonzero"] += 1.0 if rl_value != 0.0 else 0.0
            stats["format_nonzero"] += 1.0 if format_value != 0.0 else 0.0
            stats["validator_nonzero"] += 1.0 if validator_value != 0.0 else 0.0
            stats["sequence_nonzero"] += 1.0 if sequence_value != 0.0 else 0.0
            stats["comm_nonzero"] += 1.0 if comm_value != 0.0 else 0.0
            stats["comm_repeat_nonzero"] += 1.0 if comm_repeat_value != 0.0 else 0.0
            stats["comm_forced_nonzero"] += 1.0 if comm_forced_value != 0.0 else 0.0
            stats["paired_comm_nonzero"] += (
                1.0 if paired_comm_value != 0.0 else 0.0
            )
            stats["breakdown_total_nonzero"] += (
                1.0 if breakdown_total_value != 0.0 else 0.0
            )
            stats["legacy_process_nonzero"] += (
                1.0 if legacy_process_value != 0.0 else 0.0
            )
        a0_stats = agent_stats.get(
            0,
            {
                "rl": 0.0,
                "format": 0.0,
                "validator": 0.0,
                "sequence": 0.0,
                "comm": 0.0,
                "comm_repeat": 0.0,
                "comm_forced": 0.0,
                "paired_comm": 0.0,
                "breakdown_total": 0.0,
                "legacy_process": 0.0,
                "rl_nonzero": 0.0,
                "format_nonzero": 0.0,
                "validator_nonzero": 0.0,
                "sequence_nonzero": 0.0,
                "comm_nonzero": 0.0,
                "comm_repeat_nonzero": 0.0,
                "comm_forced_nonzero": 0.0,
                "paired_comm_nonzero": 0.0,
                "breakdown_total_nonzero": 0.0,
                "legacy_process_nonzero": 0.0,
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
                "comm_repeat": 0.0,
                "comm_forced": 0.0,
                "paired_comm": 0.0,
                "breakdown_total": 0.0,
                "legacy_process": 0.0,
                "rl_nonzero": 0.0,
                "format_nonzero": 0.0,
                "validator_nonzero": 0.0,
                "sequence_nonzero": 0.0,
                "comm_nonzero": 0.0,
                "comm_repeat_nonzero": 0.0,
                "comm_forced_nonzero": 0.0,
                "paired_comm_nonzero": 0.0,
                "breakdown_total_nonzero": 0.0,
                "legacy_process_nonzero": 0.0,
            },
        )
        packed = torch.tensor(
            [
                float(num),
                float(agent_counts.get(0, 0)),
                float(agent_counts.get(1, 0)),
                a0_stats["rl"],
                a0_stats["format"],
                a0_stats["validator"],
                a0_stats["sequence"],
                a0_stats["comm"],
                a0_stats["comm_repeat"],
                a0_stats["comm_forced"],
                a0_stats["paired_comm"],
                a0_stats["breakdown_total"],
                a0_stats["legacy_process"],
                a0_stats["rl_nonzero"],
                a0_stats["format_nonzero"],
                a0_stats["validator_nonzero"],
                a0_stats["sequence_nonzero"],
                a0_stats["comm_nonzero"],
                a0_stats["comm_repeat_nonzero"],
                a0_stats["comm_forced_nonzero"],
                a0_stats["paired_comm_nonzero"],
                a0_stats["breakdown_total_nonzero"],
                a0_stats["legacy_process_nonzero"],
                a1_stats["rl"],
                a1_stats["format"],
                a1_stats["validator"],
                a1_stats["sequence"],
                a1_stats["comm"],
                a1_stats["comm_repeat"],
                a1_stats["comm_forced"],
                a1_stats["paired_comm"],
                a1_stats["breakdown_total"],
                a1_stats["legacy_process"],
                a1_stats["rl_nonzero"],
                a1_stats["format_nonzero"],
                a1_stats["validator_nonzero"],
                a1_stats["sequence_nonzero"],
                a1_stats["comm_nonzero"],
                a1_stats["comm_repeat_nonzero"],
                a1_stats["comm_forced_nonzero"],
                a1_stats["paired_comm_nonzero"],
                a1_stats["breakdown_total_nonzero"],
                a1_stats["legacy_process_nonzero"],
            ],
            device=self.device,
            dtype=torch.float64,
        )
        summed = None
        try:
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
                    f"Unexpected gathered reward stats shape {tuple(gathered.shape)} "
                    f"for packed {tuple(packed.shape)}"
                )
        (
            num_f,
            a0_n_f,
            a1_n_f,
            a0_rl_sum,
            a0_format_sum,
            a0_validator_sum,
            a0_sequence_sum,
            a0_comm_sum,
            a0_comm_repeat_sum,
            a0_comm_forced_sum,
            a0_paired_comm_sum,
            a0_breakdown_total_sum,
            a0_legacy_process_sum,
            a0_rl_nonzero,
            a0_format_nonzero,
            a0_validator_nonzero,
            a0_sequence_nonzero,
            a0_comm_nonzero,
            a0_comm_repeat_nonzero,
            a0_comm_forced_nonzero,
            a0_paired_comm_nonzero,
            a0_breakdown_total_nonzero,
            a0_legacy_process_nonzero,
            a1_rl_sum,
            a1_format_sum,
            a1_validator_sum,
            a1_sequence_sum,
            a1_comm_sum,
            a1_comm_repeat_sum,
            a1_comm_forced_sum,
            a1_paired_comm_sum,
            a1_breakdown_total_sum,
            a1_legacy_process_sum,
            a1_rl_nonzero,
            a1_format_nonzero,
            a1_validator_nonzero,
            a1_sequence_nonzero,
            a1_comm_nonzero,
            a1_comm_repeat_nonzero,
            a1_comm_forced_nonzero,
            a1_paired_comm_nonzero,
            a1_breakdown_total_nonzero,
            a1_legacy_process_nonzero,
        ) = [float(x) for x in summed.tolist()]
        if not self.accelerator.is_main_process:
            return
        log_path = self.output_dir / "reward_curve.csv"
        header = (
            "row_idx,update_idx,step,num_transitions,"
            "agent0_n,agent1_n,"
            "agent0_rl_sum,agent0_rl_mean,"
            "agent0_rl_nonzero_count,agent0_rl_nonzero_ratio,"
            "agent0_format_sum,agent0_format_mean,"
            "agent0_format_nonzero_count,agent0_format_nonzero_ratio,"
            "agent0_validator_sum,agent0_validator_mean,"
            "agent0_validator_nonzero_count,agent0_validator_nonzero_ratio,"
            "agent0_sequence_sum,agent0_sequence_mean,"
            "agent0_sequence_nonzero_count,agent0_sequence_nonzero_ratio,"
            "agent0_comm_sum,agent0_comm_mean,"
            "agent0_comm_nonzero_count,agent0_comm_nonzero_ratio,"
            "agent0_comm_repeat_sum,agent0_comm_repeat_mean,"
            "agent0_comm_repeat_nonzero_count,agent0_comm_repeat_nonzero_ratio,"
            "agent0_comm_forced_sum,agent0_comm_forced_mean,"
            "agent0_comm_forced_nonzero_count,agent0_comm_forced_nonzero_ratio,"
            "agent0_paired_comm_sum,agent0_paired_comm_mean,"
            "agent0_paired_comm_nonzero_count,agent0_paired_comm_nonzero_ratio,"
            "agent0_breakdown_total_sum,agent0_breakdown_total_mean,"
            "agent0_breakdown_total_nonzero_count,agent0_breakdown_total_nonzero_ratio,"
            "agent0_legacy_process_sum,agent0_legacy_process_mean,"
            "agent0_legacy_process_nonzero_count,agent0_legacy_process_nonzero_ratio,"
            "agent1_rl_sum,agent1_rl_mean,"
            "agent1_rl_nonzero_count,agent1_rl_nonzero_ratio,"
            "agent1_format_sum,agent1_format_mean,"
            "agent1_format_nonzero_count,agent1_format_nonzero_ratio,"
            "agent1_validator_sum,agent1_validator_mean,"
            "agent1_validator_nonzero_count,agent1_validator_nonzero_ratio,"
            "agent1_sequence_sum,agent1_sequence_mean,"
            "agent1_sequence_nonzero_count,agent1_sequence_nonzero_ratio,"
            "agent1_comm_sum,agent1_comm_mean,"
            "agent1_comm_nonzero_count,agent1_comm_nonzero_ratio,"
            "agent1_comm_repeat_sum,agent1_comm_repeat_mean,"
            "agent1_comm_repeat_nonzero_count,agent1_comm_repeat_nonzero_ratio,"
            "agent1_comm_forced_sum,agent1_comm_forced_mean,"
            "agent1_comm_forced_nonzero_count,agent1_comm_forced_nonzero_ratio,"
            "agent1_paired_comm_sum,agent1_paired_comm_mean,"
            "agent1_paired_comm_nonzero_count,agent1_paired_comm_nonzero_ratio,"
            "agent1_breakdown_total_sum,agent1_breakdown_total_mean,"
            "agent1_breakdown_total_nonzero_count,agent1_breakdown_total_nonzero_ratio,"
            "agent1_legacy_process_sum,agent1_legacy_process_mean"
            ",agent1_legacy_process_nonzero_count,agent1_legacy_process_nonzero_ratio"
        )
        row_idx, last_row = self._prepare_csv_log(log_path, header)
        prev_step = 0
        if last_row is not None:
            try:
                prev_step = int(float(last_row.get("step", "0") or 0))
            except (TypeError, ValueError):
                prev_step = 0
        num = int(num_f)
        a0_n = int(a0_n_f)
        a1_n = int(a1_n_f)
        step_est = prev_step + num

        def _mean(total: float, n: int) -> float:
            return float(total) / float(n) if n > 0 else 0.0

        with log_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{row_idx},{update_idx},{step_est},{num},"
                f"{a0_n},{a1_n},"
                f"{a0_rl_sum},{_mean(a0_rl_sum, a0_n)},"
                f"{a0_rl_nonzero},{_mean(a0_rl_nonzero, a0_n)},"
                f"{a0_format_sum},{_mean(a0_format_sum, a0_n)},"
                f"{a0_format_nonzero},{_mean(a0_format_nonzero, a0_n)},"
                f"{a0_validator_sum},{_mean(a0_validator_sum, a0_n)},"
                f"{a0_validator_nonzero},{_mean(a0_validator_nonzero, a0_n)},"
                f"{a0_sequence_sum},{_mean(a0_sequence_sum, a0_n)},"
                f"{a0_sequence_nonzero},{_mean(a0_sequence_nonzero, a0_n)},"
                f"{a0_comm_sum},{_mean(a0_comm_sum, a0_n)},"
                f"{a0_comm_nonzero},{_mean(a0_comm_nonzero, a0_n)},"
                f"{a0_comm_repeat_sum},{_mean(a0_comm_repeat_sum, a0_n)},"
                f"{a0_comm_repeat_nonzero},{_mean(a0_comm_repeat_nonzero, a0_n)},"
                f"{a0_comm_forced_sum},{_mean(a0_comm_forced_sum, a0_n)},"
                f"{a0_comm_forced_nonzero},{_mean(a0_comm_forced_nonzero, a0_n)},"
                f"{a0_paired_comm_sum},{_mean(a0_paired_comm_sum, a0_n)},"
                f"{a0_paired_comm_nonzero},{_mean(a0_paired_comm_nonzero, a0_n)},"
                f"{a0_breakdown_total_sum},{_mean(a0_breakdown_total_sum, a0_n)},"
                f"{a0_breakdown_total_nonzero},{_mean(a0_breakdown_total_nonzero, a0_n)},"
                f"{a0_legacy_process_sum},{_mean(a0_legacy_process_sum, a0_n)},"
                f"{a0_legacy_process_nonzero},{_mean(a0_legacy_process_nonzero, a0_n)},"
                f"{a1_rl_sum},{_mean(a1_rl_sum, a1_n)},"
                f"{a1_rl_nonzero},{_mean(a1_rl_nonzero, a1_n)},"
                f"{a1_format_sum},{_mean(a1_format_sum, a1_n)},"
                f"{a1_format_nonzero},{_mean(a1_format_nonzero, a1_n)},"
                f"{a1_validator_sum},{_mean(a1_validator_sum, a1_n)},"
                f"{a1_validator_nonzero},{_mean(a1_validator_nonzero, a1_n)},"
                f"{a1_sequence_sum},{_mean(a1_sequence_sum, a1_n)},"
                f"{a1_sequence_nonzero},{_mean(a1_sequence_nonzero, a1_n)},"
                f"{a1_comm_sum},{_mean(a1_comm_sum, a1_n)},"
                f"{a1_comm_nonzero},{_mean(a1_comm_nonzero, a1_n)},"
                f"{a1_comm_repeat_sum},{_mean(a1_comm_repeat_sum, a1_n)},"
                f"{a1_comm_repeat_nonzero},{_mean(a1_comm_repeat_nonzero, a1_n)},"
                f"{a1_comm_forced_sum},{_mean(a1_comm_forced_sum, a1_n)},"
                f"{a1_comm_forced_nonzero},{_mean(a1_comm_forced_nonzero, a1_n)},"
                f"{a1_paired_comm_sum},{_mean(a1_paired_comm_sum, a1_n)},"
                f"{a1_paired_comm_nonzero},{_mean(a1_paired_comm_nonzero, a1_n)},"
                f"{a1_breakdown_total_sum},{_mean(a1_breakdown_total_sum, a1_n)},"
                f"{a1_breakdown_total_nonzero},{_mean(a1_breakdown_total_nonzero, a1_n)},"
                f"{a1_legacy_process_sum},{_mean(a1_legacy_process_sum, a1_n)},"
                f"{a1_legacy_process_nonzero},{_mean(a1_legacy_process_nonzero, a1_n)}\n"
            )

    def log_train_metrics(
        self,
        update_idx: int,
        loss_dict: Dict[str, float],
        transitions: List[TextTransition],
    ) -> None:
        weight = float(len(transitions))
        metric_names = [
            "loss",
            "policy",
            "value",
            "entropy",
            "reward_mean",
            "adv_mean",
            "return_mean",
            "value_mean",
            "explained_var",
            "fresh_value_mean",
            "fresh_explained_var",
            "agent0_adv_mean",
            "agent0_return_mean",
            "agent0_value_mean",
            "agent0_explained_var",
            "agent0_fresh_value_mean",
            "agent0_fresh_explained_var",
            "agent1_adv_mean",
            "agent1_return_mean",
            "agent1_value_mean",
            "agent1_explained_var",
            "agent1_fresh_value_mean",
            "agent1_fresh_explained_var",
            "clipfrac",
            "approx_kl",
            "policy_active_clipfrac",
            "policy_active_approx_kl",
            "policy_active_token_clipfrac",
            "policy_active_token_approx_kl",
            "kl_penalty",
            "kl_penalty_coef",
            "value_clipfrac",
            "optimizer_steps",
            "stopped_early",
            "critic_pretrain_active_end",
            "critic_pretrain_released",
            "critic_pretrain_last_value_loss",
            "actor_lr",
            "critic_adapter_lr",
            "value_head_lr",
            "actor0_grad_norm",
            "actor0_param_delta",
            "actor1_grad_norm",
            "actor1_param_delta",
            "critic_adapter_grad_norm",
            "critic_adapter_param_delta",
            "value_head_grad_norm",
            "value_head_param_delta",
        ]
        packed = torch.tensor(
            [weight] + [float(loss_dict.get(name, 0.0)) * weight for name in metric_names],
            device=self.device,
            dtype=torch.float64,
        )
        summed = None
        try:
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
                    f"Unexpected gathered train stats shape {tuple(gathered.shape)} "
                    f"for packed {tuple(packed.shape)}"
                )
        total_weight = max(float(summed[0].item()), 1.0)
        aggregated_metrics = {
            name: float(summed[idx + 1].item()) / total_weight
            for idx, name in enumerate(metric_names)
        }
        if not self.accelerator.is_main_process:
            return
        path = self.output_dir / "train_curve.csv"
        header = (
            "row_idx,update_idx,num_transitions,loss,policy_loss,value_loss,entropy,"
            "reward_mean,adv_mean,return_mean,value_mean,explained_var,fresh_value_mean,fresh_explained_var,"
            "agent0_adv_mean,agent0_return_mean,agent0_value_mean,agent0_explained_var,agent0_fresh_value_mean,agent0_fresh_explained_var,"
            "agent1_adv_mean,agent1_return_mean,agent1_value_mean,agent1_explained_var,agent1_fresh_value_mean,agent1_fresh_explained_var,"
            "clipfrac,approx_kl,policy_active_clipfrac,policy_active_approx_kl,"
            "policy_active_token_clipfrac,policy_active_token_approx_kl,"
            "kl_penalty,kl_penalty_coef,value_clipfrac,optimizer_steps,stopped_early,"
            "critic_pretrain_active_end,critic_pretrain_released,critic_pretrain_last_value_loss,"
            "actor_lr,critic_adapter_lr,value_head_lr,"
            "actor0_grad_norm,actor0_param_delta,"
            "actor1_grad_norm,actor1_param_delta,"
            "critic_adapter_grad_norm,critic_adapter_param_delta,"
            "value_head_grad_norm,value_head_param_delta"
        )
        row_idx, _ = self._prepare_csv_log(path, header)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{row_idx},{update_idx},{int(total_weight)},"
                f"{aggregated_metrics.get('loss', 0.0)},{aggregated_metrics.get('policy', 0.0)},"
                f"{aggregated_metrics.get('value', 0.0)},{aggregated_metrics.get('entropy', 0.0)},"
                f"{aggregated_metrics.get('reward_mean', 0.0)},{aggregated_metrics.get('adv_mean', 0.0)},"
                f"{aggregated_metrics.get('return_mean', 0.0)},{aggregated_metrics.get('value_mean', 0.0)},"
                f"{aggregated_metrics.get('explained_var', 0.0)},{aggregated_metrics.get('fresh_value_mean', 0.0)},"
                f"{aggregated_metrics.get('fresh_explained_var', 0.0)},"
                f"{aggregated_metrics.get('agent0_adv_mean', 0.0)},{aggregated_metrics.get('agent0_return_mean', 0.0)},"
                f"{aggregated_metrics.get('agent0_value_mean', 0.0)},{aggregated_metrics.get('agent0_explained_var', 0.0)},"
                f"{aggregated_metrics.get('agent0_fresh_value_mean', 0.0)},{aggregated_metrics.get('agent0_fresh_explained_var', 0.0)},"
                f"{aggregated_metrics.get('agent1_adv_mean', 0.0)},{aggregated_metrics.get('agent1_return_mean', 0.0)},"
                f"{aggregated_metrics.get('agent1_value_mean', 0.0)},{aggregated_metrics.get('agent1_explained_var', 0.0)},"
                f"{aggregated_metrics.get('agent1_fresh_value_mean', 0.0)},{aggregated_metrics.get('agent1_fresh_explained_var', 0.0)},"
                f"{aggregated_metrics.get('clipfrac', 0.0)},{aggregated_metrics.get('approx_kl', 0.0)},"
                f"{aggregated_metrics.get('policy_active_clipfrac', 0.0)},{aggregated_metrics.get('policy_active_approx_kl', 0.0)},"
                f"{aggregated_metrics.get('policy_active_token_clipfrac', 0.0)},{aggregated_metrics.get('policy_active_token_approx_kl', 0.0)},"
                f"{aggregated_metrics.get('kl_penalty', 0.0)},{aggregated_metrics.get('kl_penalty_coef', 0.0)},"
                f"{aggregated_metrics.get('value_clipfrac', 0.0)},{aggregated_metrics.get('optimizer_steps', 0.0)},"
                f"{aggregated_metrics.get('stopped_early', 0.0)},"
                f"{aggregated_metrics.get('critic_pretrain_active_end', 0.0)},{aggregated_metrics.get('critic_pretrain_released', 0.0)},"
                f"{aggregated_metrics.get('critic_pretrain_last_value_loss', 0.0)},"
                f"{aggregated_metrics.get('actor_lr', 0.0)},{aggregated_metrics.get('critic_adapter_lr', 0.0)},{aggregated_metrics.get('value_head_lr', 0.0)},"
                f"{aggregated_metrics.get('actor0_grad_norm', 0.0)},{aggregated_metrics.get('actor0_param_delta', 0.0)},"
                f"{aggregated_metrics.get('actor1_grad_norm', 0.0)},{aggregated_metrics.get('actor1_param_delta', 0.0)},"
                f"{aggregated_metrics.get('critic_adapter_grad_norm', 0.0)},{aggregated_metrics.get('critic_adapter_param_delta', 0.0)},"
                f"{aggregated_metrics.get('value_head_grad_norm', 0.0)},{aggregated_metrics.get('value_head_param_delta', 0.0)}\n"
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
            if out_dir.exists():
                shutil.rmtree(out_dir)
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
            nested_dir = out_dir / adapter_name
            if nested_dir.is_dir() and (nested_dir / "adapter_config.json").exists():
                for child in nested_dir.iterdir():
                    target = out_dir / child.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(child), str(target))
                shutil.rmtree(nested_dir)
            tokenizer.save_pretrained(out_dir)

        payload: Dict[str, Any] = {
            "model_path": str(self.model_path),
            "lora_path": None,
            "update_idx": update_idx,
        }

        value_head_path = self.export_latest_dir / "value_head.pt"
        torch.save(
            {
                "state_dict": unwrapped.value_head.state_dict(),
                "dtype": str(unwrapped.value_head.weight.dtype),
            },
            value_head_path,
        )
        payload["value_head_path"] = str(value_head_path)

        if unwrapped.is_lora:
            # Export per-agent adapters so the sampler can load different models for each agent.
            exported_actor: Dict[str, Dict[str, Any]] = {}
            for idx, spec in sorted(self.actor_adapters.items()):
                adapter_dir = self.export_latest_dir / f"adapter_{_safe_name(spec.name)}"
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
                adapter_dir = self.export_latest_dir / f"adapter_{_safe_name(spec.name)}"
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
                adapter_dir = self.export_latest_dir / "adapter"
                self.accelerator.print(f"[Export] Saving adapter -> {adapter_dir}")
                active = getattr(hf_model, "active_adapter", "default")
                if isinstance(active, (list, tuple)):
                    active_name = str(active[0]) if active else "default"
                else:
                    active_name = str(active) if active else "default"
                _save_adapter_snapshot(active_name, adapter_dir)
                payload["lora_path"] = str(adapter_dir)
        else:
            model_dir = self.export_latest_dir / "model"
            self.accelerator.print(f"[Export] Saving full model -> {model_dir}")
            if model_dir.exists():
                shutil.rmtree(model_dir)
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
            "response_log_probs": _cpu(t.response_log_probs),
            "critic_input_ids": _cpu(t.critic_input_ids),
            "log_prob": t.log_prob,
            "policy_temperature": t.policy_temperature,
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
            "repeat_communication_reward": t.repeat_communication_reward,
            "forced_communication_reward": t.forced_communication_reward,
            "paired_comm_reward": t.paired_comm_reward,
            "breakdown_total_reward": t.breakdown_total_reward,
        }

    def _dict_to_transition(self, d: Dict[str, Any]) -> TextTransition:
        return TextTransition(
            prompt_ids=d["prompt_ids"],
            response_ids=d["response_ids"],
            response_log_probs=d.get("response_log_probs"),
            critic_input_ids=d.get("critic_input_ids"),
            log_prob=float(d["log_prob"]),
            policy_temperature=(
                float(d["policy_temperature"])
                if d.get("policy_temperature") is not None
                else None
            ),
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
            repeat_communication_reward=float(
                d.get("repeat_communication_reward", 0.0)
            ),
            forced_communication_reward=float(
                d.get("forced_communication_reward", 0.0)
            ),
            paired_comm_reward=float(d.get("paired_comm_reward", 0.0)),
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
        path = self.rollout_dir / f"rollout_rank{self._runtime_worker_id()}_u{update_idx:05d}.pt"
        payload = [self._transition_to_dict(t) for t in transitions]
        torch.save(payload, path)

    def _initial_cache_path(self, update_idx: int, rank: Optional[int] = None) -> Path:
        use_rank = self._runtime_worker_id() if rank is None else int(rank)
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
        print(
            "[MAPPO] load_rollouts before barrier "
            f"rank={self.accelerator.process_index}",
            flush=True,
        )
        self.accelerator.wait_for_everyone()
        print(
            "[MAPPO] load_rollouts after barrier "
            f"rank={self.accelerator.process_index}",
            flush=True,
        )
        files = sorted(self.rollout_dir.glob("rollout_rank*_u*.pt"))
        print(
            "[MAPPO] load_rollouts found files "
            f"rank={self.accelerator.process_index} count={len(files)}",
            flush=True,
        )
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

    def _shard_transitions_for_rank(
        self, transitions: List[TextTransition]
    ) -> List[TextTransition]:
        """Shard rollout data across DDP ranks with padding to equal lengths.

        Each rank receives a different, fixed-size shard so that all processes
        execute the same number of optimizer steps. Padding reuses samples from
        the front of the dataset when the total size is not divisible by
        ``world_size``.
        """
        world_size = max(1, int(self.accelerator.num_processes))
        rank = int(self.accelerator.process_index)
        print(
            "[MAPPO] shard before "
            f"rank={rank}/{world_size} total={len(transitions)}",
            flush=True,
        )
        if world_size <= 1 or not transitions:
            return list(transitions)

        total = len(transitions)
        per_rank = int(math.ceil(total / float(world_size)))
        padded_size = per_rank * world_size
        indices = list(range(total))
        if padded_size > total:
            indices.extend(indices[: padded_size - total])

        start = rank * per_rank
        end = start + per_rank
        valid_count = max(0, min(end, total) - start)
        shard_indices = indices[start:end]
        shard = [transitions[idx] for idx in shard_indices]
        self._last_local_shard_valid_count = valid_count
        print(
            "[MAPPO] shard after "
            f"rank={rank}/{world_size} local={len(shard)}",
            flush=True,
        )
        self.accelerator.print(
            "[TrainOnly] rollout shard "
            f"rank={rank}/{world_size} global={total} local={len(shard)} "
            f"padded={padded_size}"
        )
        return shard

    def _cleanup_rollout_files(self):
        if not self.accelerator.is_main_process:
            return
        for p in self.rollout_dir.glob("rollout_rank*_u*.pt"):
            p.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def save_checkpoint(self, tag: str):
        ckpt_dir = self.output_dir / tag
        print(
            "[MAPPO] save_checkpoint begin "
            f"rank={self.accelerator.process_index} tag={tag} path={ckpt_dir}",
            flush=True,
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[MAPPO] save_checkpoint before barrier "
            f"rank={self.accelerator.process_index} tag={tag}",
            flush=True,
        )
        self.accelerator.wait_for_everyone()
        print(
            "[MAPPO] save_checkpoint before save_state "
            f"rank={self.accelerator.process_index} tag={tag}",
            flush=True,
        )
        self.accelerator.save_state(ckpt_dir)
        print(
            "[MAPPO] save_checkpoint after save_state "
            f"rank={self.accelerator.process_index} tag={tag}",
            flush=True,
        )
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
            print(
                "[MAPPO] save_checkpoint wrote metadata "
                f"rank={self.accelerator.process_index} tag={tag} path={meta_path}",
                flush=True,
            )


__all__ = ["MAPPOTrainer"]
