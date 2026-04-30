#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from start_vllm_server import _build_lora_modules


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _build_payload(config: Path) -> Tuple[List[Dict[str, str]], Optional[str]]:
    _model_path, modules, _max_rank, value_head_path = _build_lora_modules(config)
    return [{"name": name, "path": path} for name, path in modules], value_head_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--start-port", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    modules, value_head_path = _build_payload(Path(args.config).resolve())
    payload: Dict[str, Any] = {"modules": modules, "value_head_path": value_head_path}
    for offset in range(args.count):
        port = args.start_port + offset
        url = f"http://{args.host}:{port}/rl/reload_adapters"
        try:
            result = _post_json(url, payload, timeout=args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            print(f"[reload-vllm] failed port={port} status={exc.code} detail={detail}", file=sys.stderr)
            return 1
        except Exception as exc:
            print(f"[reload-vllm] failed port={port} error={exc}", file=sys.stderr)
            return 1
        print(
            "[reload-vllm] reloaded "
            f"port={port} modules={result.get('lora_modules', [])} "
            f"value_head={result.get('value_head')}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
