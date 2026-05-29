#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any, Dict


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["sleep", "wake"], required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--start-port", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--ports", default="")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--mode", default="abort")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--ignore-unsupported", action="store_true")
    args = parser.parse_args()

    endpoint = "sleep" if args.action == "sleep" else "wake_up"
    payload: Dict[str, Any] = {}
    if args.action == "sleep":
        payload = {"level": args.level, "mode": args.mode}

    if args.ports.strip():
        ports = [
            int(item.strip())
            for item in args.ports.split(",")
            if item.strip()
        ]
    else:
        ports = [args.start_port + offset for offset in range(args.count)]

    for port in ports:
        url = f"http://{args.host}:{port}/rl/{endpoint}"
        try:
            result = _post_json(url, payload, timeout=args.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if args.ignore_unsupported and exc.code in {404, 501}:
                print(
                    f"[vllm-state] skip unsupported action={args.action} port={port} "
                    f"status={exc.code} detail={detail}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            print(
                f"[vllm-state] failed action={args.action} port={port} "
                f"status={exc.code} detail={detail}",
                file=sys.stderr,
            )
            return 1
        except Exception as exc:
            print(
                f"[vllm-state] failed action={args.action} port={port} error={exc}",
                file=sys.stderr,
            )
            return 1
        print(
            f"[vllm-state] action={args.action} port={port} result={result}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
