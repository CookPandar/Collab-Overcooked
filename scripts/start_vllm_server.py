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


def _map_generated_path(path_str: str, repo_root: Path, experiment_root: Path) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        return str((experiment_root / path).resolve())
    try:
        rel = path.relative_to(repo_root)
    except ValueError:
        return path_str
    return str((experiment_root / rel).resolve())


def _rewrite_experiment_paths(trainer: Dict[str, Any], repo_root: Path) -> None:
    experiment_root_raw = os.environ.get("RL_EXPERIMENT_ROOT", "").strip()
    if not experiment_root_raw:
        return
    experiment_root = Path(experiment_root_raw).resolve()
    for key in (
        "rollout_dir",
        "output_dir",
        "export_latest_dir",
        "latest_model_path_file",
        "initial_rollout_cache_dir",
    ):
        value = trainer.get(key)
        if isinstance(value, str) and value:
            trainer[key] = _map_generated_path(value, repo_root, experiment_root)
        elif isinstance(value, list):
            trainer[key] = [
                _map_generated_path(item, repo_root, experiment_root)
                if isinstance(item, str)
                else item
                for item in value
            ]


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
        critic_override = payload.get("critic_adapter")
        critic_cfg = trainer.get("critic_adapter")
        if isinstance(critic_override, dict) and isinstance(critic_cfg, dict):
            lora_path = critic_override.get("lora_path")
            if lora_path:
                critic_cfg["lora_path"] = _resolve_lora_dir(str(lora_path), base_dir)
            adapter_name = critic_override.get("adapter_name")
            if adapter_name:
                critic_cfg["adapter_name"] = str(adapter_name)
        value_head_path = payload.get("value_head_path")
        if value_head_path:
            trainer["value_head_path"] = _resolve_path(str(value_head_path), base_dir)
        return
    trainer["model_path"] = _resolve_path(content, base_dir)


def _build_lora_modules(
    config_path: Path,
    service_role: str = "both",
    fallback_model_path: Optional[str] = None,
) -> Tuple[str, List[Tuple[str, str]], int, Optional[str]]:
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
    _rewrite_experiment_paths(trainer, repo_root)
    if str(os.environ.get("RL_COLLECT_VALUE_BACKEND", "")).strip():
        trainer["collect_value_backend"] = str(os.environ["RL_COLLECT_VALUE_BACKEND"]).strip()
    if str(os.environ.get("RL_COMPUTE_VALUES_IN_COLLECT", "")).strip():
        trainer["compute_values_in_collect"] = str(
            os.environ["RL_COMPUTE_VALUES_IN_COLLECT"]
        ).strip().lower() in {"1", "true", "yes", "on"}
    _apply_latest_override(
        trainer,
        trainer.get("latest_model_path_file"),
        repo_root,
    )
    model_path = _resolve_path(trainer.get("model_path"), repo_root)
    fallback_model_path = _resolve_path(fallback_model_path, repo_root)
    if model_path and not Path(model_path).exists() and fallback_model_path:
        print(
            "[start-vllm] resolved trainer.model_path does not exist; "
            f"falling back to --model {fallback_model_path}. missing={model_path}",
            file=sys.stderr,
            flush=True,
        )
        model_path = fallback_model_path
    if not model_path and fallback_model_path:
        model_path = fallback_model_path
    if not model_path:
        raise ValueError(f"trainer.model_path missing in {config_path}")
    modules: List[Tuple[str, str]] = []
    max_rank = 0
    collect_value_backend = str(trainer.get("collect_value_backend", "")).strip().lower()
    compute_values_in_collect = bool(trainer.get("compute_values_in_collect", False))
    value_head_path = (
        _resolve_path(trainer.get("value_head_path"), repo_root)
        if compute_values_in_collect and collect_value_backend == "vllm"
        else None
    )
    include_actor = service_role in {"both", "actor"}
    include_value = service_role in {"both", "value"}
    actor_adapters = trainer.get("actor_adapters") or {}
    if include_actor and isinstance(actor_adapters, dict):
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
    critic_adapter = trainer.get("critic_adapter") or {}
    if include_value and isinstance(critic_adapter, dict):
        critic_lora_path = _resolve_lora_dir(critic_adapter.get("lora_path"), repo_root)
        if critic_lora_path:
            critic_name = str(critic_adapter.get("adapter_name") or "critic")
            modules.append((critic_name, critic_lora_path))
            adapter_cfg = Path(critic_lora_path) / "adapter_config.json"
            if adapter_cfg.exists():
                try:
                    adapter_payload = json.loads(adapter_cfg.read_text(encoding="utf-8"))
                    max_rank = max(max_rank, int(adapter_payload.get("r", 0) or 0))
                except Exception:
                    pass
    if not include_value:
        value_head_path = None
    return model_path, modules, max_rank, value_head_path


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
    parser.add_argument('--service-role', choices=['both', 'actor', 'value'], default='both')
    parser.add_argument('--enforce-eager', action='store_true')
    args = parser.parse_args()

    env = os.environ.copy()
    repo_root = Path(os.environ.get("RL_REPO_ROOT", str(Path.cwd()))).resolve()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_devices:
        visible_ids = [item.strip() for item in visible_devices.split(",") if item.strip()]
        if 0 <= int(args.gpu) < len(visible_ids):
            selected_gpu = visible_ids[int(args.gpu)]
        else:
            selected_gpu = str(args.gpu)
    else:
        selected_gpu = str(args.gpu)
    env['CUDA_VISIBLE_DEVICES'] = selected_gpu
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
    value_head_path: Optional[str] = None
    if args.config:
        model_path, lora_modules, inferred_max_lora_rank, value_head_path = _build_lora_modules(
            Path(args.config).resolve(),
            service_role=args.service_role,
            fallback_model_path=args.model,
        )
    env["RL_VLLM_SERVICE_ROLE"] = args.service_role
    if value_head_path:
        env["RL_VLLM_VALUE_HEAD_PATH"] = value_head_path
    else:
        env.pop("RL_VLLM_VALUE_HEAD_PATH", None)
    if "RL_VLLM_SERIALIZE_GENERATE" in os.environ:
        env["RL_VLLM_SERIALIZE_GENERATE"] = os.environ["RL_VLLM_SERIALIZE_GENERATE"]
    for key in (
        "RL_VLLM_PARALLEL_BATCH_MAX",
        "RL_VLLM_PARALLEL_BATCH_WAIT_MS",
        "RL_VLLM_ALLOW_UNSAFE_VALUE_BATCH",
    ):
        if key in os.environ:
            env[key] = os.environ[key]
    if lora_modules:
        env["RL_VLLM_LORA_MODULES_JSON"] = json.dumps(
            [{"name": name, "path": path} for name, path in lora_modules]
        )

    cmd = [
        sys.executable,
        str(repo_root / 'scripts' / 'serve_rl_vllm.py'),
        '--model', model_path,
        '--served-model-name', args.served_model_name,
        '--host', '127.0.0.1',
        '--port', str(args.port),
        '--max-model-len', str(args.max_model_len),
        '--gpu-memory-utilization', str(args.gpu_memory_utilization),
        '--api-key', args.api_key,
    ]
    if args.enforce_eager:
        cmd.append('--enforce-eager')
    if lora_modules:
        max_loras = max(len(lora_modules), max(1, int(args.max_loras)))
        cmd.extend([
            '--max-loras', str(max_loras),
        ])
        max_lora_rank = int(args.max_lora_rank) if int(args.max_lora_rank) > 0 else inferred_max_lora_rank
        if max_lora_rank > 0:
            cmd.extend(['--max-lora-rank', str(max_lora_rank)])
    return subprocess.call(cmd, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
