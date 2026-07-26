#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import inspect
import json
import os
import queue
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from uvicorn import run as uvicorn_run

from vllm import LLM, SamplingParams
try:
    from vllm.config import KVTransferConfig
except Exception:  # pragma: no cover
    KVTransferConfig = None  # type: ignore

try:
    from vllm.lora.request import LoRARequest
except Exception:  # pragma: no cover
    from vllm.lora import LoRARequest  # type: ignore


class GenerateRequest(BaseModel):
    messages: List[Dict[str, str]]
    adapter_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    stop: Optional[Union[str, List[str]]] = None


class ValueRequest(BaseModel):
    critic_text: str
    adapter_name: Optional[str] = None


class ReloadAdapterSpec(BaseModel):
    name: str
    path: str
    id: Optional[int] = None


class ReloadAdaptersRequest(BaseModel):
    modules: List[ReloadAdapterSpec]
    value_head_path: Optional[str] = None


class SleepRequest(BaseModel):
    level: int = 1
    mode: str = "abort"


@dataclass
class _GenerateWorkItem:
    prompt: str
    sampling_params: SamplingParams
    lora_request: Optional[LoRARequest]
    event: threading.Event
    outputs: Optional[List[Any]] = None
    error: Optional[BaseException] = None


def _extract_logprob(entry: Any) -> Optional[float]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        value = entry.get("logprob")
        return float(value) if value is not None else None
    value = getattr(entry, "logprob", None)
    return float(value) if value is not None else None


class RLVLLMService:
    def __init__(self, args: argparse.Namespace):
        self.model_path = args.model
        self.served_model_name = args.served_model_name
        self.service_role = str(
            os.environ.get("RL_VLLM_SERVICE_ROLE", "both")
        ).strip().lower()
        if self.service_role not in {"both", "actor", "value"}:
            self.service_role = "both"
        self.hidden_state_root = Path(
            tempfile.gettempdir()
        ) / f"rl_vllm_hidden_states_{args.port}"
        self.hidden_state_root.mkdir(parents=True, exist_ok=True)
        self.generate_lock = threading.Lock()
        requested_serialize_generate = str(
            os.environ.get("RL_VLLM_SERIALIZE_GENERATE", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}
        self.serialize_generate = requested_serialize_generate
        self.parallel_batch_max = max(
            1,
            int(os.environ.get("RL_VLLM_PARALLEL_BATCH_MAX", "8")),
        )
        self.parallel_batch_wait_ms = max(
            0.0,
            float(os.environ.get("RL_VLLM_PARALLEL_BATCH_WAIT_MS", "10")),
        )
        self._generate_queue: "queue.Queue[_GenerateWorkItem]" = queue.Queue()
        self._batcher_thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._sleeping = False

        self.lora_modules = self._load_lora_modules()
        self.max_loras = max(1, int(args.max_loras))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.model_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self.last_hidden_layer = max(0, int(self.model_config.num_hidden_layers) - 1)

        self.value_head: Optional[nn.Linear] = None
        self.value_head_dtype = torch.float32
        self.value_head_path = os.environ.get("RL_VLLM_VALUE_HEAD_PATH", "").strip()
        if self.value_head_path:
            self.value_head = self._load_value_head(Path(self.value_head_path))

        llm_kwargs: Dict[str, Any] = {
            "model": self.model_path,
            "tensor_parallel_size": 1,
            "dtype": "auto",
            "max_model_len": int(args.max_model_len),
            "gpu_memory_utilization": float(args.gpu_memory_utilization),
            "enable_sleep_mode": str(
                os.environ.get("RL_VLLM_ENABLE_SLEEP_MODE", "1")
            ).strip().lower() not in {"0", "false", "no", "off"},
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "disable_log_stats": True,
            "trust_remote_code": True,
            "enforce_eager": bool(args.enforce_eager),
            "enable_lora": bool(self.lora_modules),
        }
        if self.lora_modules:
            llm_kwargs["max_loras"] = self.max_loras
            if int(args.max_lora_rank) > 0:
                llm_kwargs["max_lora_rank"] = int(args.max_lora_rank)
        value_runtime_requested = self.value_head is not None and KVTransferConfig is not None
        if value_runtime_requested:
            llm_kwargs["kv_transfer_config"] = KVTransferConfig(
                kv_connector="ExampleHiddenStatesConnector",
                kv_role="kv_producer",
                kv_connector_extra_config={
                    "shared_storage_path": str(self.hidden_state_root),
                },
            )
            # vLLM's hidden-state extraction API expects a speculative config
            # that routes auxiliary hidden states through a dummy draft model.
            # Older "eagle_*" top-level fields are rejected by recent schemas.
            llm_kwargs["speculative_config"] = {
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": [self.last_hidden_layer],
                    }
                },
            }
        filtered_llm_kwargs = self._filter_llm_kwargs(llm_kwargs)
        requested_value_runtime_enabled = value_runtime_requested and (
            "kv_transfer_config" in filtered_llm_kwargs
            and "speculative_config" in filtered_llm_kwargs
        )
        self.value_runtime_enabled = bool(requested_value_runtime_enabled)
        try:
            self.llm = LLM(**filtered_llm_kwargs)
        except Exception as exc:
            error_text = str(exc)
            if self.value_runtime_enabled and (
                "ExampleHiddenStatesConnector" in error_text
                or "HiddenStatesConnectorV1" in error_text
                or "Unsupported connector type" in error_text
            ):
                print(
                    "[RLVLLMService] hidden-state value runtime unsupported by current vLLM; "
                    "falling back to actor-only mode.",
                    flush=True,
                )
                filtered_llm_kwargs.pop("kv_transfer_config", None)
                filtered_llm_kwargs.pop("speculative_config", None)
                self.value_runtime_enabled = False
                self.llm = LLM(**filtered_llm_kwargs)
            else:
                raise
        allow_unsafe_value_batch = str(
            os.environ.get("RL_VLLM_ALLOW_UNSAFE_VALUE_BATCH", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            not requested_serialize_generate
            and self.value_runtime_enabled
            and not allow_unsafe_value_batch
        ):
            self.serialize_generate = True
            print(
                "[RLVLLMService] forcing serialize_generate=True because "
                "vLLM V1 hidden-state value runtime uses speculative hidden-state "
                "extraction and is not safe to batch with LoRA/Punica in this setup. "
                "Set RL_VLLM_ALLOW_UNSAFE_VALUE_BATCH=1 only for debugging.",
                flush=True,
            )
        print(
            "[RLVLLMService] init "
            f"model={self.model_path} "
            f"service_role={self.service_role} "
            f"lora_modules={sorted(self.lora_modules.keys())} "
            f"enable_lora={bool(filtered_llm_kwargs.get('enable_lora', False))} "
            f"max_loras={filtered_llm_kwargs.get('max_loras')} "
            f"max_lora_rank={filtered_llm_kwargs.get('max_lora_rank')} "
            f"value_head={bool(self.value_head is not None)} "
            f"value_runtime_enabled={self.value_runtime_enabled} "
            f"serialize_generate={self.serialize_generate} "
            f"parallel_batch_max={self.parallel_batch_max} "
            f"parallel_batch_wait_ms={self.parallel_batch_wait_ms}",
            flush=True,
        )
        if not self.serialize_generate:
            self._batcher_thread = threading.Thread(
                target=self._batch_generate_loop,
                name="rl-vllm-generate-batcher",
                daemon=True,
            )
            self._batcher_thread.start()

    @staticmethod
    def _filter_llm_kwargs(llm_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        llm_signature = inspect.signature(LLM.__init__)
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in llm_signature.parameters.values()
        )
        return (
            dict(llm_kwargs)
            if accepts_var_kwargs
            else {
                key: value
                for key, value in llm_kwargs.items()
                if key in llm_signature.parameters
            }
        )

    def _load_lora_modules(self) -> Dict[str, Dict[str, Any]]:
        raw = os.environ.get("RL_VLLM_LORA_MODULES_JSON", "").strip()
        if not raw:
            return {}
        payload = json.loads(raw)
        return self._normalize_lora_modules(payload)

    @classmethod
    def _normalize_lora_modules(
        cls,
        payload: Any,
    ) -> Dict[str, Dict[str, Any]]:
        modules: Dict[str, Dict[str, Any]] = {}
        used_ids: set[int] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            path = str(item.get("path") or "").strip()
            if not name or not path:
                continue
            module_id = cls._coerce_lora_id(item.get("id"))
            if module_id is None:
                module_id = cls._stable_lora_id(name, path)
            while module_id in used_ids:
                module_id += 1
                if module_id > 0x7FFFFFFF:
                    module_id = 1
            used_ids.add(module_id)
            modules[name] = {"id": module_id, "path": path}
        return modules

    @staticmethod
    def _coerce_lora_id(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            module_id = int(value)
        except (TypeError, ValueError):
            return None
        return module_id if module_id > 0 else None

    @staticmethod
    def _stable_lora_id(name: str, path: str) -> int:
        canonical_path = str(Path(path).expanduser().resolve(strict=False))
        key = f"{name}\0{canonical_path}".encode("utf-8")
        digest = hashlib.blake2s(key, digest_size=4).digest()
        return max(1, int.from_bytes(digest, "big") & 0x7FFFFFFF)

    def _lora_module_specs(self) -> List[Dict[str, Any]]:
        return [
            {"name": name, "id": int(spec["id"]), "path": str(spec["path"])}
            for name, spec in sorted(self.lora_modules.items())
        ]

    def _load_value_head(self, path: Path) -> nn.Linear:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            dtype_name = str(payload.get("dtype", "torch.float32"))
        else:
            state_dict = payload
            dtype_name = "torch.float32"
        hidden_size = int(self.model_config.hidden_size)
        linear = nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32)
        linear.load_state_dict(state_dict)
        linear.eval()
        self.value_head_dtype = getattr(torch, dtype_name.split(".")[-1], torch.float32)
        return linear

    def reload_adapters(self, request: ReloadAdaptersRequest) -> Dict[str, Any]:
        self.wake_up()
        modules_payload = [
            {"name": item.name, "path": item.path, "id": item.id}
            for item in request.modules
        ]
        if len(modules_payload) > self.max_loras:
            raise HTTPException(
                status_code=400,
                detail=f"Too many LoRA modules: {len(modules_payload)} > max_loras={self.max_loras}",
            )
        for item in modules_payload:
            path = Path(item["path"])
            if not (path / "adapter_config.json").exists():
                raise HTTPException(
                    status_code=400,
                    detail=f"LoRA adapter_config.json not found: {path}",
                )
        self.lora_modules = self._normalize_lora_modules(modules_payload)
        value_head_path = (request.value_head_path or "").strip()
        if value_head_path:
            self.value_head_path = value_head_path
            self.value_head = self._load_value_head(Path(value_head_path))
        return {
            "ok": True,
            "lora_modules": sorted(self.lora_modules.keys()),
            "lora_module_specs": self._lora_module_specs(),
            "value_head": bool(self.value_head is not None),
        }

    def sleep(self, level: int = 1, mode: str = "abort") -> Dict[str, Any]:
        if not hasattr(self.llm, "sleep"):
            raise HTTPException(status_code=501, detail="vLLM LLM.sleep() is unavailable.")
        sleep_level = int(level)
        if self.lora_modules and sleep_level > 1:
            print(
                "[RLVLLMService] downgrading sleep level to 1 because vLLM "
                "level>1 discards model weights and corrupts LoRA generation "
                "after wake_up in this runtime.",
                flush=True,
            )
            sleep_level = 1
        normalized_mode = str(mode or "abort").strip().lower()
        if normalized_mode not in {"abort", "wait", "keep"}:
            normalized_mode = "abort"
        with self.generate_lock:
            with self._state_lock:
                if self._sleeping:
                    return {"ok": True, "sleeping": True, "already": True}
                self.llm.sleep(level=sleep_level, mode=normalized_mode)
                torch.cuda.empty_cache()
                self._sleeping = True
        return {"ok": True, "sleeping": True, "level": sleep_level, "mode": normalized_mode}

    def wake_up(self) -> Dict[str, Any]:
        if not hasattr(self.llm, "wake_up"):
            raise HTTPException(status_code=501, detail="vLLM LLM.wake_up() is unavailable.")
        with self.generate_lock:
            with self._state_lock:
                if not self._sleeping:
                    return {"ok": True, "sleeping": False, "already": True}
                self.llm.wake_up()
                self._sleeping = False
        return {"ok": True, "sleeping": False}

    def _build_lora_request(
        self,
        adapter_name: Optional[str],
        *,
        strict: bool = False,
    ) -> Optional[LoRARequest]:
        if not adapter_name:
            return None
        spec = self.lora_modules.get(adapter_name)
        if spec is None:
            if strict:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown adapter: {adapter_name}",
                )
            return None
        return LoRARequest(adapter_name, int(spec["id"]), str(spec["path"]))

    @staticmethod
    def _lora_key(lora_request: Optional[LoRARequest]) -> tuple:
        if lora_request is None:
            return ("", 0, "")
        return (
            str(getattr(lora_request, "lora_name", "")),
            int(getattr(lora_request, "lora_int_id", 0)),
            str(getattr(lora_request, "lora_path", "")),
        )

    @staticmethod
    def _sampling_key(sampling_params: SamplingParams) -> tuple:
        stop = getattr(sampling_params, "stop", None)
        if isinstance(stop, str):
            stop_key = (stop,)
        elif stop is None:
            stop_key = ()
        else:
            stop_key = tuple(str(item) for item in stop)
        return (
            float(getattr(sampling_params, "temperature", 0.0)),
            int(getattr(sampling_params, "max_tokens", 0)),
            int(getattr(sampling_params, "logprobs", 0) or 0),
            stop_key,
        )

    @classmethod
    def _batch_key(cls, item: _GenerateWorkItem) -> tuple:
        return (cls._lora_key(item.lora_request), cls._sampling_key(item.sampling_params))

    def _direct_generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams | List[SamplingParams],
        lora_request: Optional[LoRARequest] | List[Optional[LoRARequest]],
    ) -> List[Any]:
        return self.llm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

    def _submit_generate(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest],
    ) -> List[Any]:
        if self.serialize_generate:
            with self.generate_lock:
                return self._direct_generate([prompt], sampling_params, lora_request)

        item = _GenerateWorkItem(
            prompt=prompt,
            sampling_params=sampling_params,
            lora_request=lora_request,
            event=threading.Event(),
        )
        self._generate_queue.put(item)
        item.event.wait()
        if item.error is not None:
            raise item.error
        return item.outputs or []

    def _batch_generate_loop(self) -> None:
        while True:
            first = self._generate_queue.get()
            batch = [first]
            batch_key = self._batch_key(first)
            wait_seconds = self.parallel_batch_wait_ms / 1000.0
            deadline = time.monotonic() + wait_seconds
            while wait_seconds > 0 and time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    item = self._generate_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if self._batch_key(item) == batch_key and len(batch) < self.parallel_batch_max:
                    batch.append(item)
                else:
                    self._generate_queue.put(item)
                    break
            while len(batch) < self.parallel_batch_max:
                try:
                    item = self._generate_queue.get_nowait()
                except queue.Empty:
                    break
                if self._batch_key(item) == batch_key:
                    batch.append(item)
                else:
                    self._generate_queue.put(item)
                    break
            try:
                prompts = [item.prompt for item in batch]
                outputs = self._direct_generate(
                    prompts,
                    sampling_params=first.sampling_params,
                    lora_request=first.lora_request,
                )
                for item, output in zip(batch, outputs):
                    item.outputs = [output]
            except BaseException as exc:
                for item in batch:
                    item.error = exc
            finally:
                for item in batch:
                    item.event.set()

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        parts: List[str] = []
        for item in messages:
            role = str(item.get("role", "user")).strip()
            content = str(item.get("content", ""))
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def generate(self, request: GenerateRequest) -> Dict[str, Any]:
        if self.service_role == "value":
            raise HTTPException(status_code=404, detail="Actor generation is disabled on value service.")
        if self._sleeping:
            self.wake_up()
        prompt = self._apply_chat_template(request.messages)
        stop = request.stop
        if isinstance(stop, str):
            stop = [stop]
        elif stop is not None:
            stop = [str(item) for item in stop if str(item)]
        sampling = SamplingParams(
            temperature=float(request.temperature),
            max_tokens=max(1, int(request.max_tokens)),
            logprobs=1,
            stop=stop,
        )
        outputs = self._submit_generate(
            prompt,
            sampling,
            self._build_lora_request(request.adapter_name, strict=True),
        )
        if not outputs:
            raise HTTPException(status_code=500, detail="Empty vLLM output.")
        output = outputs[0]
        if not output.outputs:
            raise HTTPException(status_code=500, detail="Missing generation candidates.")
        completion = output.outputs[0]
        token_ids = list(getattr(completion, "token_ids", []) or [])
        token_texts = list(getattr(completion, "tokens", []) or [])
        if not token_texts:
            token_texts = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
        raw_logprobs = list(getattr(completion, "logprobs", []) or [])
        token_logprobs: List[float] = []
        for idx, token_id in enumerate(token_ids):
            selected = None
            if idx < len(raw_logprobs):
                per_pos = raw_logprobs[idx]
                if isinstance(per_pos, dict):
                    selected = per_pos.get(token_id)
                    if selected is None:
                        selected = per_pos.get(str(token_id))
                    if selected is None and per_pos:
                        selected = next(iter(per_pos.values()))
                else:
                    selected = per_pos
            logprob = _extract_logprob(selected)
            token_logprobs.append(float(logprob) if logprob is not None else 0.0)
        text = getattr(completion, "text", None)
        if text is None:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return {
            "text": text,
            "response_token_ids": token_ids,
            "response_tokens": token_texts,
            "response_log_probs": token_logprobs,
            "log_prob": float(sum(token_logprobs)),
            "token_count": len(token_ids),
        }

    def _read_hidden_state_file(self, hidden_state_path: Path) -> torch.Tensor:
        with safe_open(str(hidden_state_path), framework="pt") as f:
            if "hidden_states" not in f.keys():
                raise HTTPException(status_code=500, detail="Hidden states missing in safetensors.")
            hidden_states = f.get_tensor("hidden_states")
        if hidden_states.ndim < 2:
            raise HTTPException(status_code=500, detail="Unexpected hidden state shape.")
        if hidden_states.ndim == 2:
            last_hidden = hidden_states[-1]
        else:
            last_hidden = hidden_states[-1, -1]
        return last_hidden.detach().cpu().to(dtype=torch.float32)

    def value(self, request: ValueRequest) -> Dict[str, Any]:
        if self.service_role == "actor":
            return {"value": 0.0, "available": False}
        if self.value_head is None or not self.value_runtime_enabled:
            return {"value": 0.0, "available": False}
        if self._sleeping:
            self.wake_up()
        prompt = request.critic_text
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
        )
        outputs = self._submit_generate(
            prompt,
            sampling,
            self._build_lora_request(request.adapter_name, strict=False),
        )
        if not outputs:
            raise HTTPException(status_code=500, detail="Empty vLLM output for value request.")
        output = outputs[0]
        kv_transfer_params = getattr(output, "kv_transfer_params", None) or {}
        hidden_states_path = kv_transfer_params.get("hidden_states_path")
        if not hidden_states_path:
            return {"value": 0.0, "available": False}
        hidden_state_path = Path(str(hidden_states_path))
        try:
            last_hidden = self._read_hidden_state_file(hidden_state_path)
            with torch.no_grad():
                value = self.value_head(last_hidden.unsqueeze(0)).squeeze(0).item()
            return {"value": float(value), "available": True}
        finally:
            hidden_state_path.unlink(missing_ok=True)


def build_app(service: RLVLLMService) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "model": service.served_model_name,
            "service_role": service.service_role,
            "lora_modules": sorted(service.lora_modules.keys()),
            "lora_module_specs": service._lora_module_specs(),
            "value_head": bool(service.value_head is not None),
            "value_runtime_enabled": bool(service.value_runtime_enabled),
            "serialize_generate": bool(service.serialize_generate),
            "parallel_batch_max": int(service.parallel_batch_max),
            "parallel_batch_wait_ms": float(service.parallel_batch_wait_ms),
            "sleeping": bool(service._sleeping),
            "sleep_supported": bool(hasattr(service.llm, "sleep")),
            "wake_supported": bool(hasattr(service.llm, "wake_up")),
        }

    @app.post("/rl/generate")
    def generate(request: GenerateRequest) -> Dict[str, Any]:
        return service.generate(request)

    @app.post("/rl/value")
    def value(request: ValueRequest) -> Dict[str, Any]:
        return service.value(request)

    @app.post("/rl/reload_adapters")
    def reload_adapters(request: ReloadAdaptersRequest) -> Dict[str, Any]:
        return service.reload_adapters(request)

    @app.post("/rl/sleep")
    def sleep(request: SleepRequest) -> Dict[str, Any]:
        return service.sleep(level=request.level, mode=request.mode)

    @app.post("/rl/wake_up")
    def wake_up() -> Dict[str, Any]:
        return service.wake_up()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--gpu-memory-utilization", required=True)
    parser.add_argument("--max-model-len", required=True)
    parser.add_argument("--api-key", required=False, default="")
    parser.add_argument("--max-loras", type=int, default=1)
    parser.add_argument("--max-lora-rank", type=int, default=0)
    # Backward-compatible no-op: LoRA modules are now passed via
    # RL_VLLM_LORA_MODULES_JSON instead of CLI args.
    parser.add_argument("--lora-modules", nargs="*", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock_path = Path(tempfile.gettempdir()) / f"rl_vllm_port_{args.port}.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            f"[RLVLLMService] port={args.port} is already reserved by another initializing service",
            flush=True,
        )
        return 98
    lock_handle.write(str(os.getpid()))
    lock_handle.flush()
    service = RLVLLMService(args)
    app = build_app(service)
    uvicorn_run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
