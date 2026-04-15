#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def _runtime_worker_rank() -> int:
    raw = (
        os.environ.get("RL_WORKER_RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("RANK")
        or "0"
    )
    try:
        return int(raw)
    except ValueError:
        return 0


def _resolve_agent_index(key):
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


def _resolve_path(raw: str, base_dir: Path) -> str:
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _apply_latest_override(trainer: dict, base_dir: Path) -> None:
    latest_file = trainer.get("latest_model_path_file")
    if not latest_file:
        return
    marker = Path(_resolve_path(str(latest_file), base_dir))
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
    if not isinstance(payload, dict):
        return
    if payload.get("model_path"):
        trainer["model_path"] = _resolve_path(str(payload["model_path"]), base_dir)
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
            if agent_key not in actor_cfg or not isinstance(actor_cfg[agent_key], dict):
                continue
            lora_path = value.get("lora_path")
            if lora_path:
                actor_cfg[agent_key]["lora_path"] = _resolve_path(str(lora_path), base_dir)
            adapter_name = value.get("adapter_name")
            if adapter_name:
                actor_cfg[agent_key]["adapter_name"] = str(adapter_name)


def build_rank_bound_config(src_cfg: Path, stage: str, tmp_dir: Path) -> Path:
    rank = _runtime_worker_rank()
    host = os.environ["RL_VLLM_HOST"]
    start_port = int(os.environ["RL_VLLM_START_PORT"])
    port = start_port + rank
    base_url = f"http://{host}:{port}/v1"
    api_key = os.environ.get("RL_VLLM_API_KEY", "YOUR_API_KEY")
    repo_root = Path(os.environ["RL_REPO_ROOT"])
    model_root = os.environ.get(
        "RL_VLLM_MODEL_PATH",
        "/mnt/volumes/ss-sai-bd-ga/zhangshuwen/models/qwen2.5-7b",
    )

    data = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
    trainer = data.setdefault("trainer", {})
    _apply_latest_override(trainer, repo_root)
    for key in [
        "rollout_dir",
        "output_dir",
        "export_latest_dir",
        "latest_model_path_file",
        "initial_rollout_cache_dir",
    ]:
        value = trainer.get(key)
        if isinstance(value, str) and value and not value.startswith("/"):
            trainer[key] = str(repo_root / value)

    model_path = trainer.get("model_path")
    if not isinstance(model_path, str) or not model_path.startswith("/"):
        trainer["model_path"] = model_root

    if stage == "collect":
        trainer["collect_only"] = True
        trainer["train_only"] = False
    elif stage == "train":
        trainer["collect_only"] = False
        trainer["train_only"] = True
    elif stage == "eval":
        trainer["collect_only"] = True
        trainer["train_only"] = False

    for agent_key, agent in (data.get("agents") or {}).items():
        if not isinstance(agent, dict) or not agent_key.startswith("agent_"):
            continue
        actor_cfg = (trainer.get("actor_adapters") or {}).get(agent_key, {})
        adapter_name = (
            actor_cfg.get("adapter_name")
            if isinstance(actor_cfg, dict) and actor_cfg.get("lora_path")
            else None
        )
        agent["type"] = "vllm"
        agent["api_key"] = api_key
        agent["base_url"] = base_url
        agent["local_model_path"] = trainer["model_path"]
        agent["cuda_visible_devices"] = [rank]
        agent["data_parallel_size"] = 1
        if adapter_name:
            agent["model"] = adapter_name
        agent.pop("model_dirname", None)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f"rl_stage_{stage}_rank{rank}_", suffix=".yaml", dir=str(tmp_dir)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.write_text(yaml.safe_dump(data, allow_unicode=False, sort_keys=False), encoding="utf-8")
    return tmp_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["collect", "train", "eval"])
    parser.add_argument("remainder", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    src_cfg = Path(args.config).resolve()
    py_bin = os.environ.get("RL_PY_BIN") or sys.executable
    with tempfile.TemporaryDirectory(prefix="collab_overcooked_rl_") as tmp_dir_name:
        tmp_cfg = build_rank_bound_config(src_cfg, args.stage, Path(tmp_dir_name))
        cmd = [py_bin, "-m", "collab_overcooked.main_rl", "--config", str(tmp_cfg)]
        remainder = list(args.remainder)
        if remainder and remainder[0] == "--":
            remainder = remainder[1:]
        cmd.extend(remainder)
        return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
