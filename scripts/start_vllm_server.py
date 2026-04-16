#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - depends on runtime env
    yaml = None


def _resolve_path(raw: Optional[str], base_dir: Path) -> Optional[str]:
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _resolve_lora_dir(raw: Optional[str], base_dir: Path) -> Optional[str]:
    resolved = _resolve_path(raw, base_dir)
    if not resolved:
        return None
    path = Path(resolved)
    if (path / "adapter_config.json").exists():
        return str(path)
    child_dirs = [item for item in path.iterdir()] if path.is_dir() else []
    nested_candidates = [
        item for item in child_dirs
        if item.is_dir() and (item / "adapter_config.json").exists()
    ]
    if len(nested_candidates) == 1:
        return str(nested_candidates[0])
    return str(path)


def _resolve_agent_index(key: Any) -> Optional[int]:
    if isinstance(key, int):
        return key
    text = str(key).strip()
    if not text:
        return None
    if text.startswith("agent_"):
        text = text.split("_", 1)[1]
    try:
        return int(text)
    except ValueError:
        return None


def _apply_latest_override(
    trainer: Dict[str, Any],
    latest_model_path_file: Optional[str],
    base_dir: Path,
) -> None:
    latest_path = _resolve_path(latest_model_path_file, base_dir)
    if not latest_path:
        return
    marker = Path(latest_path)
    if not marker.exists():
        return
    try:
        content = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if not content:
        return
    try:
        payload = json.loads(content)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        model_path = payload.get("model_path")
        if model_path:
            trainer["model_path"] = _resolve_path(str(model_path), base_dir)
        actor_override = payload.get("actor_adapters")
        actor_cfg = trainer.get("actor_adapters")
        if isinstance(actor_override, dict) and isinstance(actor_cfg, dict):
            for key, value in actor_override.items():
                if not isinstance(value, dict):
                    continue
                idx = _resolve_agent_index(key)
                if idx is None:
                    continue
                agent_key = f"agent_{idx}"
                if not isinstance(actor_cfg.get(agent_key), dict):
                    continue
                lora_path = value.get("lora_path")
                if lora_path:
                    actor_cfg[agent_key]["lora_path"] = _resolve_lora_dir(
                        str(lora_path), base_dir
                    )
                adapter_name = value.get("adapter_name")
                if adapter_name:
                    actor_cfg[agent_key]["adapter_name"] = str(adapter_name)
        return
    trainer["model_path"] = _resolve_path(content, base_dir)


def _build_lora_modules(config_path: Path) -> Tuple[str, List[Tuple[str, str]], int]:
    repo_root = Path(os.environ.get("RL_REPO_ROOT", str(Path.cwd()))).resolve()
    if yaml is not None:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        helper_python = os.environ.get("RL_PY_BIN", "python")
        raw = subprocess.check_output(
            [
                helper_python,
                "-c",
                (
                    "import json, sys, yaml; "
                    "print(json.dumps(yaml.safe_load(open(sys.argv[1], encoding='utf-8')) or {}))"
                ),
                str(config_path),
            ],
            text=True,
        )
        data = json.loads(raw)
    trainer = data.setdefault("trainer", {})
    _apply_latest_override(
        trainer,
        trainer.get("latest_model_path_file"),
        repo_root,
    )
    model_path = _resolve_path(trainer.get("model_path"), repo_root)
    if not model_path:
        raise ValueError(f"trainer.model_path missing in {config_path}")
    modules: List[Tuple[str, str]] = []
    max_rank = 0
    actor_adapters = trainer.get("actor_adapters") or {}
    if isinstance(actor_adapters, dict):
        for key, value in sorted(actor_adapters.items()):
            if not isinstance(value, dict):
                continue
            lora_path = _resolve_lora_dir(value.get("lora_path"), repo_root)
            if not lora_path:
                continue
            adapter_name = str(
                value.get("adapter_name")
                or f"agent_{_resolve_agent_index(key) if _resolve_agent_index(key) is not None else key}"
            )
            modules.append((adapter_name, lora_path))
            adapter_cfg = Path(lora_path) / "adapter_config.json"
            if adapter_cfg.exists():
                try:
                    adapter_payload = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                    max_rank = max(max_rank, int(adapter_payload.get("r", 0) or 0))
                except Exception:
                    pass
    return model_path, modules, max_rank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--engine-port', type=int, required=True)
    parser.add_argument('--internal-port-base', type=int, required=False)
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=False)
    parser.add_argument('--served-model-name', required=True)
    parser.add_argument('--gpu-memory-utilization', required=True)
    parser.add_argument('--max-model-len', required=True)
    parser.add_argument('--api-key', required=True)
    parser.add_argument('--max-loras', type=int, default=2)
    parser.add_argument('--max-lora-rank', type=int, default=0)
    parser.add_argument('--enforce-eager', action='store_true')
    args = parser.parse_args()

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    env['VLLM_DP_RANK'] = '0'
    env['VLLM_DP_SIZE'] = '1'
    env['VLLM_DP_MASTER_IP'] = '127.0.0.1'
    env['VLLM_DP_MASTER_PORT'] = str(args.engine_port)
    if args.internal_port_base is not None:
        env['VLLM_PORT'] = str(args.internal_port_base)
    for key in (
        'MASTER_ADDR',
        'MASTER_PORT',
        'WORLD_SIZE',
        'RANK',
        'LOCAL_RANK',
        'NODE_RANK',
        'GROUP_RANK',
        'ROLE_RANK',
        'ROLE_NAME',
    ):
        env.pop(key, None)

    model_path = args.model
    lora_modules: List[Tuple[str, str]] = []
    inferred_max_lora_rank = 0
    if args.config:
        model_path, lora_modules, inferred_max_lora_rank = _build_lora_modules(
            Path(args.config).resolve()
        )

    cmd = [
        sys.executable,
        '-m', 'vllm.entrypoints.openai.api_server',
        '--model', model_path,
        '--served-model-name', args.served_model_name,
        '--host', '127.0.0.1',
        '--port', str(args.port),
        '--dtype', 'auto',
        '--max-model-len', str(args.max_model_len),
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--api-key', args.api_key,
        '--tensor-parallel-size', '1',
        '--enable-prefix-caching',
        '--enable-chunked-prefill',
        '--disable-log-stats',
    ]
    if args.enforce_eager:
        cmd.append('--enforce-eager')
    if lora_modules:
        cmd.extend([
            '--enable-lora',
            '--max-loras', str(max(1, int(args.max_loras))),
        ])
        max_lora_rank = int(args.max_lora_rank) if int(args.max_lora_rank) > 0 else inferred_max_lora_rank
        if max_lora_rank > 0:
            cmd.extend(['--max-lora-rank', str(max_lora_rank)])
        cmd.append('--lora-modules')
        for name, path in lora_modules:
            cmd.append(f'{name}={path}')
    return subprocess.call(cmd, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
