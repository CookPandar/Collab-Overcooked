#!/usr/bin/env python
"""Audit rollout reward assignment against decoded agent outputs."""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoTokenizer


SECTION_RE = re.compile(
    r"^\s*(Think|Recent Goal|Action)\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)


def as_float(value: Any) -> float:
    try:
        if hasattr(value, "item"):
            return float(value.item())
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def split_action_tokens(action: str) -> List[str]:
    body = re.sub(r"^\s*Action\s*:\s*", "", action or "", flags=re.IGNORECASE)
    body = body.replace("\r\n", "\n").replace("\r", "\n").replace("\n", ";")
    tokens: List[str] = []
    current: List[str] = []
    depth = 0
    for char in body:
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char == ";" and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            continue
        current.append(char)
    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def is_collab_token(token: str) -> bool:
    lowered = (token or "").strip().lower()
    return lowered.startswith("collab(") or lowered.startswith(
        ("request(", "seek(", "ack(", "deny(")
    )


def classify_action(action: str) -> str:
    tokens = split_action_tokens(action)
    if not tokens:
        return "empty"
    collab = [token for token in tokens if is_collab_token(token)]
    embodied = [token for token in tokens if not is_collab_token(token)]
    if collab and embodied:
        return "mixed"
    if collab:
        return "communication"
    if len(embodied) > 1:
        return "multi_embodied"
    return "embodied"


def primary_embodied(action: str) -> str:
    for token in split_action_tokens(action):
        if not is_collab_token(token):
            return token
    return ""


def action_block(text: str) -> str:
    sections = {
        match.group(1).lower(): (match.group(2) or "").strip()
        for match in SECTION_RE.finditer(text or "")
    }
    return sections.get("action", "").strip().strip("`").strip()


def decode_response(tokenizer: Any, record: Dict[str, Any]) -> str:
    response_ids = record.get("response_ids")
    if response_ids is None:
        return ""
    values = response_ids.tolist() if hasattr(response_ids, "tolist") else list(response_ids)
    return tokenizer.decode(values, skip_special_tokens=True)


def iter_rollout_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in sorted(glob.glob(str(path / "rollout_rank*_u*.pt"))):
        yield Path(item)


def rollout_rank(file_name: str) -> Optional[int]:
    match = re.search(r"rollout_rank(\d+)_", file_name)
    if not match:
        return None
    return int(match.group(1))


def load_rows(path: Path, tokenizer: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for file_path in iter_rollout_files(path):
        data = torch.load(file_path, map_location="cpu", weights_only=False)
        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                continue
            text = decode_response(tokenizer, record)
            action = action_block(text)
            rows.append(
                {
                    "file": file_path.name,
                    "idx": idx,
                    "agent": record.get("agent_index"),
                    "ts": record.get("timestep"),
                    "text": text,
                    "action": action,
                    "mode": classify_action(action),
                    "embodied": primary_embodied(action),
                    "reward": as_float(record.get("reward")),
                    "sequence": as_float(record.get("sequence_reward") or record.get("process_reward")),
                    "format": as_float(record.get("format_reward")),
                    "validator": as_float(record.get("validator_reward")),
                }
            )
    return rows


def find_issues(rows: List[Dict[str, Any]]) -> List[tuple[str, Any]]:
    issues: List[tuple[str, Any]] = []
    for row in rows:
        if row["mode"] == "communication" and (
            abs(row["validator"]) > 1e-9 or abs(row["sequence"]) > 1e-9
        ):
            issues.append(("COLLAB_GOT_EXEC_REWARD", row))
        if row["mode"] == "mixed" and row["format"] >= 0.0:
            issues.append(("MIXED_NOT_FORMAT_PENALIZED", row))
        if row["mode"] == "multi_embodied" and row["format"] >= 0.0:
            issues.append(("MULTI_EMBODIED_NOT_FORMAT_PENALIZED", row))

    by_key: Dict[tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[(row["file"], row["agent"], row["ts"])].append(row)
    for key, group in by_key.items():
        group = sorted(group, key=lambda item: int(item["idx"]))
        embodied_no_exec = [
            row
            for row in group
            if row["mode"] == "embodied"
            and row["embodied"]
            and row["validator"] == 0.0
            and row["sequence"] == 0.0
            and row["format"] >= 0.0
        ]
        collab_exec = [
            row
            for row in group
            if row["mode"] == "communication"
            and (row["validator"] > 0.0 or row["sequence"] > 0.0)
        ]
        if embodied_no_exec and collab_exec:
            issues.append(
                (
                    "SAME_TIMESTEP_EXEC_REWARD_DRIFT",
                    {"key": key, "embodied": embodied_no_exec, "collab": collab_exec},
                )
            )
    return issues


def executed_actions_from_log(log_path: Path) -> Dict[Tuple[int, int], set[str]]:
    executed: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    if not log_path.exists():
        return executed
    current_ts: Optional[int] = None
    pending_joint_ts: Optional[int] = None
    joint_re = re.compile(r"\[CollabMainSession\] joint_action=")
    ts_re = re.compile(r"Beginning step at timestep\s+(\d+)")
    list_re = re.compile(r"^\[(.*)\]\s*$")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ts_match = ts_re.search(line)
        if ts_match:
            current_ts = int(ts_match.group(1))
            continue
        if joint_re.search(line):
            pending_joint_ts = current_ts
            continue
        if pending_joint_ts is None:
            continue
        stripped = line.strip()
        match = list_re.match(stripped)
        if not match:
            continue
        items = re.findall(r"'([^']*)'|None", stripped)
        if len(items) < 2:
            continue
        for agent_idx, action in enumerate(items[:2]):
            if not action:
                continue
            executed[(agent_idx, pending_joint_ts)].add(action.replace(" ", ""))
        pending_joint_ts = None
    return executed


def find_missing_executed_rewards(
    rows: List[Dict[str, Any]],
    rollout_path: Path,
    log_dir: Optional[Path],
) -> List[tuple[str, Any]]:
    if log_dir is None:
        return []
    issues: List[tuple[str, Any]] = []
    rows_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_file[row["file"]].append(row)
    for file_name, file_rows in rows_by_file.items():
        rank = rollout_rank(file_name)
        if rank is None:
            continue
        log_path = log_dir / f"collect_worker{rank}_gpu0.log"
        executed = executed_actions_from_log(log_path)
        if not executed:
            continue
        by_agent_ts: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
        for row in file_rows:
            try:
                by_agent_ts[(int(row["agent"]), int(row["ts"]))].append(row)
            except (TypeError, ValueError):
                continue
        for (agent_idx, exec_ts), actions in executed.items():
            candidate_rows: List[Dict[str, Any]] = []
            for source_ts in range(max(0, exec_ts - 8), exec_ts + 1):
                candidate_rows.extend(by_agent_ts.get((agent_idx, source_ts), []))
            for action in actions:
                normalized_action = action.replace(" ", "")
                matching = [
                    row
                    for row in candidate_rows
                    if row["mode"] == "embodied"
                    and str(row["embodied"]).replace(" ", "") == normalized_action
                ]
                if not matching:
                    continue
                if any(row["sequence"] > 0.0 or row["validator"] > 0.0 for row in matching):
                    continue
                issues.append(
                    (
                        "EXECUTED_ACTION_MISSING_REWARD",
                        {
                            "file": file_name,
                            "rank": rank,
                            "log": str(log_path),
                            "agent": agent_idx,
                            "exec_ts": exec_ts,
                            "action": normalized_action,
                            "candidates": matching[:5],
                        },
                    )
                )
    return issues


def print_row(prefix: str, row: Dict[str, Any]) -> None:
    action = str(row["action"]).replace("\n", "\\n")[:500]
    print(
        f"{prefix}{row['file']} idx={row['idx']} agent={row['agent']} ts={row['ts']} "
        f"mode={row['mode']} reward={row['reward']:.4f} seq={row['sequence']:.4f} "
        f"fmt={row['format']:.4f} val={row['validator']:.4f} action={action}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Rollout .pt file or directory containing rollout_rank*_u*.pt")
    parser.add_argument(
        "--model",
        default=os.environ.get("RL_VLLM_MODEL_PATH", "/datacache/LLMs/Qwen2.5-7B-Instruct"),
    )
    parser.add_argument("--max-issues", type=int, default=120)
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Optional rl_workers log dir for executed-action false-negative audit.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rollout_path = Path(args.path)
    rows = load_rows(rollout_path, tokenizer)
    issues = find_issues(rows)
    if args.log_dir:
        issues.extend(find_missing_executed_rewards(rows, rollout_path, Path(args.log_dir)))

    print(f"transitions={len(rows)}")
    print(f"mode_counts={dict(Counter(row['mode'] for row in rows))}")
    print(f"issue_counts={dict(Counter(kind for kind, _ in issues))}")
    for kind, payload in issues[: max(0, args.max_issues)]:
        print(f"\nISSUE {kind}")
        if isinstance(payload, dict) and "key" in payload:
            print(f"key={payload['key']}")
            for row in payload["embodied"][:5]:
                print_row("  embodied ", row)
            for row in payload["collab"][:5]:
                print_row("  collab   ", row)
        elif isinstance(payload, dict) and "candidates" in payload:
            print(
                f"file={payload['file']} rank={payload['rank']} agent={payload['agent']} "
                f"exec_ts={payload['exec_ts']} action={payload['action']} log={payload['log']}"
            )
            for row in payload["candidates"]:
                print_row("  candidate ", row)
        else:
            print_row("  ", payload)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
