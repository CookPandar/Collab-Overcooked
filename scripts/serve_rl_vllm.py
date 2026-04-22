#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class ValueRequest(BaseModel):
    critic_text: str
    adapter_name: Optional[str] = None


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
        self.hidden_state_root = Path(
            tempfile.gettempdir()
        ) / f"rl_vllm_hidden_states_{args.port}"
        self.hidden_state_root.mkdir(parents=True, exist_ok=True)

        self.lora_modules = self._load_lora_modules()
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
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "disable_log_stats": True,
            "trust_remote_code": True,
            "enforce_eager": bool(args.enforce_eager),
            "enable_lora": bool(self.lora_modules),
        }
        if self.lora_modules:
            llm_kwargs["max_loras"] = max(1, int(args.max_loras))
            if int(args.max_lora_rank) > 0:
                llm_kwargs["max_lora_rank"] = int(args.max_lora_rank)
        value_runtime_requested = self.value_head is not None and KVTransferConfig is not None
        if value_runtime_requested:
            llm_kwargs["kv_transfer_config"] = KVTransferConfig(
                kv_connector="HiddenStatesConnectorV1",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "storage_path": str(self.hidden_state_root),
                    "save_hidden_states": True,
                    "load_hidden_states": False,
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
                "HiddenStatesConnectorV1" in error_text
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
        print(
            "[RLVLLMService] init "
            f"model={self.model_path} "
            f"lora_modules={sorted(self.lora_modules.keys())} "
            f"enable_lora={bool(filtered_llm_kwargs.get('enable_lora', False))} "
            f"max_loras={filtered_llm_kwargs.get('max_loras')} "
            f"max_lora_rank={filtered_llm_kwargs.get('max_lora_rank')} "
            f"value_head={bool(self.value_head is not None)} "
            f"value_runtime_enabled={self.value_runtime_enabled}",
            flush=True,
        )

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
        modules: Dict[str, Dict[str, Any]] = {}
        for idx, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            path = str(item.get("path") or "").strip()
            if not name or not path:
                continue
            modules[name] = {"id": idx, "path": path}
        return modules

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
        prompt = self._apply_chat_template(request.messages)
        sampling = SamplingParams(
            temperature=float(request.temperature),
            max_tokens=max(1, int(request.max_tokens)),
            logprobs=1,
        )
        outputs = self.llm.generate(
            [prompt],
            sampling_params=sampling,
            lora_request=self._build_lora_request(request.adapter_name, strict=True),
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
        if self.value_head is None or not self.value_runtime_enabled:
            return {"value": 0.0, "available": False}
        request_id = f"critic-{uuid.uuid4().hex}"
        hidden_state_path = self.hidden_state_root / f"{request_id}.safetensors"
        prompt = request.critic_text
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
        )
        outputs = self.llm.generate(
            [prompt],
            sampling_params=sampling,
            lora_request=self._build_lora_request(request.adapter_name, strict=False),
            kv_transfer_params={"request_id": request_id},
        )
        if not outputs:
            raise HTTPException(status_code=500, detail="Empty vLLM output for value request.")
        if not hidden_state_path.exists():
            return {"value": 0.0, "available": False}
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
            "lora_modules": sorted(service.lora_modules.keys()),
            "value_head": bool(service.value_head is not None),
            "value_runtime_enabled": bool(service.value_runtime_enabled),
        }

    @app.post("/rl/generate")
    def generate(request: GenerateRequest) -> Dict[str, Any]:
        return service.generate(request)

    @app.post("/rl/value")
    def value(request: ValueRequest) -> Dict[str, Any]:
        return service.value(request)

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
    service = RLVLLMService(args)
    app = build_app(service)
    uvicorn_run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
