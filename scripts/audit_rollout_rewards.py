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


PAIRED_COMM_META_FIELDS = (
    "paired_comm_role",
    "paired_comm_result",
    "paired_comm_target",
    "paired_comm_request_action",
    "paired_comm_request_helpful",
    "paired_comm_consumed_requests",
    "paired_comm_registered_requests",
)

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


def response_has_malformed_code_fence_tail(text: str) -> bool:
    if "```" not in (text or ""):
        return False
    action_match = re.search(r"^\s*Action\s*:", text or "", flags=re.IGNORECASE | re.MULTILINE)
    if action_match is None:
        return True
    after_action = (text or "")[action_match.end() :]
    first_fence = after_action.find("```")
    if first_fence == -1:
        return False
    before_first_fence = after_action[:first_fence].strip()
    if not before_first_fence:
        # The Action content itself is fenced; this is normal.
        closing = after_action.find("```", first_fence + 3)
        if closing == -1:
            return True
        tail = after_action[closing + 3 :].strip()
        return bool(tail)
    tail = after_action[first_fence + 3 :].strip()
    return bool(tail)


def classify_action(action: str, text: str = "") -> str:
    if response_has_malformed_code_fence_tail(text):
        return "malformed"
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


def normalize_action(action: Any) -> str:
    return str(action or "").strip().replace(" ", "")


def unwrap_function_body(text: str) -> Optional[str]:
    stripped = (text or "").strip()
    start = stripped.find("(")
    if start == -1 or not stripped.endswith(")"):
        return None
    return stripped[start + 1 : -1]


def split_first_argument(text: str) -> Tuple[str, str]:
    depth = 0
    for idx, char in enumerate(text or ""):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            return text[:idx], text[idx + 1 :]
    return text, ""


def agent_index_from_label(label: str) -> Optional[int]:
    normalized = (label or "").strip().lower()
    if "assistant" in normalized or "player1" in normalized:
        return 1
    if "chef" in normalized or "player0" in normalized:
        return 0
    return None


def collab_requests(action: str) -> List[Tuple[int, str]]:
    requests: List[Tuple[int, str]] = []
    for token in split_action_tokens(action):
        body = unwrap_function_body(token) if token.lower().startswith("collab(") else token
        if body is None:
            continue
        for segment in split_action_tokens(body):
            stripped = segment.strip()
            if not stripped.lower().startswith("request("):
                continue
            request_body = unwrap_function_body(stripped)
            if request_body is None:
                continue
            target_raw, action_raw = split_first_argument(request_body)
            target_idx = agent_index_from_label(target_raw)
            requested_action = normalize_action(action_raw)
            if target_idx is None or not requested_action:
                continue
            requests.append((target_idx, requested_action))
    return requests


def action_block(text: str) -> str:
    sections = {
        match.group(1).lower(): (match.group(2) or "").strip()
        for match in SECTION_RE.finditer(text or "")
    }
    return strip_action_code_fence(sections.get("action", ""))


def strip_action_code_fence(action: str) -> str:
    stripped = (action or "").strip()
    if not stripped.startswith("```"):
        return stripped.strip("`").strip()
    content = stripped[3:]
    closing = content.rfind("```")
    if closing != -1:
        content = content[:closing]
    content = content.strip()
    if "\n" in content:
        first_line, remainder = content.split("\n", 1)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", first_line.strip()):
            content = remainder
    return content.strip().strip("`").strip()


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


def rollout_update(file_name: str) -> Optional[int]:
    match = re.search(r"_u(\d+)\.pt$", file_name)
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
                    "mode": classify_action(action, text),
                    "embodied": primary_embodied(action),
                    "reward": as_float(record.get("reward")),
                    "breakdown_total": as_float(record.get("breakdown_total_reward")),
                    "sequence": as_float(record.get("sequence_reward") or record.get("process_reward")),
                    "format": as_float(record.get("format_reward")),
                    "validator": as_float(record.get("validator_reward")),
                    "paired_comm": as_float(record.get("paired_comm_reward")),
                    "reward_source_key": record.get("reward_source_key"),
                    **{
                        field: record.get(field)
                        for field in PAIRED_COMM_META_FIELDS
                    },
                    "done": as_float(record.get("done")),
                }
            )
    return rows


def find_issues(rows: List[Dict[str, Any]]) -> List[tuple[str, Any]]:
    issues: List[tuple[str, Any]] = []
    for row in rows:
        if abs(row["reward"] - row["breakdown_total"]) > 1e-6:
            issues.append(("REWARD_BREAKDOWN_MISMATCH", row))
        if row["validator"] < -0.1000001:
            issues.append(("DUPLICATE_VALIDATOR_PENALTY", row))
        if abs(row["paired_comm"]) > 1e-9 and (
            row["format"] < 0.0 or row["validator"] < 0.0
        ):
            issues.append(("ACTION_PENALTY_WITH_PAIR_COMM", row))
        if (
            abs(row["paired_comm"]) > 1e-9
            and row["mode"] == "embodied"
            and normalize_action(row["embodied"]).lower().startswith("wait")
        ):
            issues.append(("WAIT_ACTION_WITH_PAIR_COMM", row))
        if row["mode"] == "communication" and (
            abs(row["validator"]) > 1e-9 or abs(row["sequence"]) > 1e-9
        ):
            issues.append(("COLLAB_GOT_EXEC_REWARD", row))
        if row["mode"] == "mixed" and row["format"] >= 0.0:
            issues.append(("MIXED_NOT_FORMAT_PENALIZED", row))
        if row["mode"] == "multi_embodied" and row["format"] >= 0.0:
            issues.append(("MULTI_EMBODIED_NOT_FORMAT_PENALIZED", row))
        if row["mode"] == "malformed" and row["format"] >= 0.0:
            issues.append(("MALFORMED_NOT_FORMAT_PENALIZED", row))

    issues.extend(find_paired_comm_mismatch_issues(rows))

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


def find_paired_comm_mismatch_issues(rows: List[Dict[str, Any]]) -> List[tuple[str, Any]]:
    issues: List[tuple[str, Any]] = []
    rows_by_file: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_file[row["file"]].append(row)

    for file_name, file_rows in rows_by_file.items():
        pending: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        sorted_rows = sorted(file_rows, key=lambda item: int(item["idx"]))
        for row in sorted_rows:
            for target_idx, requested_action in collab_requests(row["action"]):
                if row["paired_comm"] > 0.0:
                    pending[target_idx].append(
                        {
                            "file": file_name,
                            "request": row,
                            "requested_action": requested_action,
                        }
                    )

            accepted_request_action = _accepted_responder_request_action(row)
            if accepted_request_action and not _has_matching_positive_request(
                sorted_rows, row, accepted_request_action
            ):
                embodied_action = (
                    normalize_action(row.get("embodied"))
                    if row.get("mode") == "embodied"
                    else ""
                )
                if not embodied_action or embodied_action == accepted_request_action:
                    issues.append(
                        (
                            "PAIRED_COMM_ACCEPTED_WITHOUT_REWARDED_REQUEST",
                            {
                                "file": file_name,
                                "response": row,
                                "requested_action": accepted_request_action,
                            },
                        )
                    )

            if row["mode"] != "embodied":
                continue
            agent = row.get("agent")
            if agent is None:
                continue
            action = normalize_action(row["embodied"])
            if not action:
                continue
            matches = [
                item
                for item in pending.get(int(agent), [])
                if item["requested_action"] == action
            ]
            if row["paired_comm"] > 0.0 and not matches:
                if _paired_response_meta_matches(row, action):
                    continue
                if _has_matching_positive_request(sorted_rows, row, action):
                    continue
                issue_kind = (
                    "PAIRED_COMM_POSITIVE_RESPONSE_BAD_META"
                    if _has_paired_response_meta(row)
                    else "PAIRED_COMM_POSITIVE_WITHOUT_MATCHING_REQUEST"
                )
                issues.append(
                    (
                        issue_kind,
                        {
                            "file": file_name,
                            "response": row,
                            "requested_action": action,
                        },
                    )
                )
            if matches and row["paired_comm"] < 0.0:
                issues.append(
                    (
                        "PAIRED_COMM_MATCHED_ACTION_GOT_NEGATIVE",
                        {
                            "file": file_name,
                            "response": row,
                            "request": matches[-1]["request"],
                            "requested_action": action,
                        },
                    )
                )
            if matches and row["paired_comm"] > 0.0:
                removed = False
                remaining = []
                for item in pending[int(agent)]:
                    if not removed and item["requested_action"] == action:
                        removed = True
                        continue
                    remaining.append(item)
                pending[int(agent)] = remaining
    return issues


def _has_paired_response_meta(row: Dict[str, Any]) -> bool:
    return any(row.get(field) is not None for field in PAIRED_COMM_META_FIELDS)


def _paired_response_meta_matches(row: Dict[str, Any], action: str) -> bool:
    role = str(row.get("paired_comm_role") or "").strip().lower()
    if role != "responder":
        return False
    result = str(row.get("paired_comm_result") or "").strip().lower()
    if result and result != "helpful_request_accepted":
        return False
    request_action = normalize_action(row.get("paired_comm_request_action"))
    return bool(request_action and request_action == action)


def _accepted_responder_request_action(row: Dict[str, Any]) -> str:
    if float(row.get("paired_comm") or 0.0) <= 0.0:
        return ""
    role = str(row.get("paired_comm_role") or "").strip().lower()
    if role != "responder":
        return ""
    result = str(row.get("paired_comm_result") or "").strip().lower()
    if result != "helpful_request_accepted":
        return ""
    return normalize_action(row.get("paired_comm_request_action"))


def _has_matching_positive_request(
    rows: List[Dict[str, Any]],
    response_row: Dict[str, Any],
    action: str,
) -> bool:
    """Rollout storage can interleave same-timestep agent calls out of generation order."""
    try:
        response_agent = int(response_row.get("agent"))
        response_ts = int(response_row.get("ts"))
    except (TypeError, ValueError):
        return False
    for request_row in rows:
        if request_row.get("file") != response_row.get("file"):
            continue
        if float(request_row.get("paired_comm") or 0.0) <= 0.0:
            continue
        try:
            request_ts = int(request_row.get("ts"))
        except (TypeError, ValueError):
            continue
        if request_ts > response_ts:
            continue
        for target_idx, requested_action in collab_requests(request_row.get("action", "")):
            if target_idx == response_agent and requested_action == action:
                return True
    return False


def find_partial_success_issues(
    rows: List[Dict[str, Any]],
    *,
    terminal_reward: float,
    agent_index: int,
) -> List[tuple[str, Any]]:
    issues: List[tuple[str, Any]] = []
    if terminal_reward <= 0:
        return issues
    threshold = terminal_reward - 1e-6
    for row in rows:
        if row["reward"] < threshold:
            continue
        if row["agent"] != agent_index:
            issues.append(("PARTIAL_SUCCESS_WRONG_AGENT", row))
        if row["done"] < 0.5:
            issues.append(("PARTIAL_SUCCESS_NOT_TERMINAL", row))
    return issues


def log_lines_for_update(log_path: Path, expected_update: Optional[int]) -> List[str]:
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if expected_update is None:
        return lines

    marker_re = re.compile(r"\[Collect\].*save_rollout.*update=(\d+)")
    markers: List[Tuple[int, int]] = []
    for idx, line in enumerate(lines):
        marker = marker_re.search(line)
        if marker:
            markers.append((idx, int(marker.group(1))))

    if not markers:
        return lines

    matching = [(idx, update) for idx, update in markers if update == expected_update]
    if not matching:
        return []

    # Worker logs are usually overwritten per collection round, but when they
    # are appended this keeps executed-action parsing inside the target update.
    end_idx = matching[0][0]
    previous_markers = [idx for idx, _ in markers if idx < end_idx]
    start_idx = previous_markers[-1] + 1 if previous_markers else 0
    return lines[start_idx : end_idx + 1]


def executed_actions_from_log(
    log_path: Path,
    expected_update: Optional[int] = None,
) -> Dict[Tuple[int, int], set[str]]:
    executed: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    lines = log_lines_for_update(log_path, expected_update)
    if not lines:
        return executed
    current_ts: Optional[int] = None
    pending_joint_ts: Optional[int] = None
    joint_re = re.compile(r"\[CollabMainSession\] joint_action=")
    ts_re = re.compile(r"Beginning step at timestep\s+(\d+)")
    list_re = re.compile(r"^\[(.*)\]\s*$")
    for line in lines:
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
        executed = executed_actions_from_log(log_path, rollout_update(file_name))
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
                if any(
                    row["reward"] > 0.0
                    or row["breakdown_total"] > 0.0
                    or row["sequence"] > 0.0
                    or row["validator"] > 0.0
                    or row["format"] < 0.0
                    or row["validator"] < 0.0
                    for row in matching
                ):
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
    paired_meta = ""
    if _has_paired_response_meta(row):
        paired_meta = (
            f" paired_role={row.get('paired_comm_role')}"
            f" paired_result={row.get('paired_comm_result')}"
            f" paired_request={row.get('paired_comm_request_action')}"
        )
    print(
        f"{prefix}{row['file']} idx={row['idx']} agent={row['agent']} ts={row['ts']} "
        f"mode={row['mode']} reward={row['reward']:.4f} seq={row['sequence']:.4f} "
        f"fmt={row['format']:.4f} val={row['validator']:.4f} "
        f"paired={row['paired_comm']:.4f} source={row.get('reward_source_key')}"
        f"{paired_meta} action={action}"
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
        elif isinstance(payload, dict) and "response" in payload and "request" in payload:
            print(
                f"file={payload.get('file')} requested_action={payload.get('requested_action')}"
            )
            print_row("  request  ", payload["request"])
            print_row("  response ", payload["response"])
        else:
            print_row("  ", payload)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
