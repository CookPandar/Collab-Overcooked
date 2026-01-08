#!/usr/bin/env python3
"""
Resolve local vLLM server specs from a YAML config.

Prints one server per line:
  served_model_name|local_model_path|host|port|cuda_visible_devices|tensor_parallel_size
"""

from __future__ import annotations

import argparse
import os
from urllib.parse import urlparse

import yaml


def _to_cuda_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(x).strip() for x in value if str(x).strip() != "")
    return str(value).strip()


def _tp_from_cuda(cuda_str: str) -> int:
    if not cuda_str:
        return 1
    return len([x for x in cuda_str.split(",") if x.strip() != ""])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve vLLM server list from YAML config.")
    parser.add_argument("config_path", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config_path))

    servers: dict[tuple, dict] = {}
    for agent_key, agent in (cfg.get("agents") or {}).items():
        if not isinstance(agent, dict):
            continue

        local_path = agent.get("local_model_path") or agent.get("model_dirname") or agent.get("model_path")
        model = agent.get("model")
        base_url = agent.get("base_url")
        if not (local_path and model and base_url):
            continue

        cuda_visible = (
            agent.get("cuda_visible_devices")
            or agent.get("cuda_visible_device")
            or agent.get("visible_gpus")
            or agent.get("gpus")
        )
        cuda_str = _to_cuda_str(cuda_visible)

        tp = agent.get("tensor_parallel_size") or agent.get("tensor_parallel") or agent.get("tp")
        if tp is None:
            tp = _tp_from_cuda(cuda_str)
        try:
            tp = int(tp)
        except Exception as exc:
            raise SystemExit(f"Invalid tensor_parallel_size/tp for agent '{agent_key}': {tp!r}") from exc
        if tp < 1:
            raise SystemExit(
                f"Invalid tensor_parallel_size/tp for agent '{agent_key}': {tp} (must be >= 1)"
            )
        if cuda_str:
            n = _tp_from_cuda(cuda_str)
            if tp > n:
                raise SystemExit(
                    f"Invalid config for agent '{agent_key}': tp={tp} > visible_gpus={n} "
                    f"(cuda_visible_devices={cuda_str!r})"
                )

        parsed = urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        key = (os.path.abspath(local_path), model, host, port, cuda_str, tp)
        servers[key] = {
            "path": os.path.abspath(local_path),
            "model": model,
            "host": host,
            "port": port,
            "cuda": cuda_str,
            "tp": tp,
        }

    if not servers:
        raise SystemExit("No local_model_path/model_path + base_url entries found in config; nothing to serve.")

    for srv in servers.values():
        print(
            f"{srv['model']}|{srv['path']}|{srv['host']}|{srv['port']}|{srv['cuda']}|{srv['tp']}"
        )


if __name__ == "__main__":
    main()

