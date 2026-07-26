#!/usr/bin/env python
"""Replay saved rollout responses through the current reward/session code."""

from __future__ import annotations

import argparse
import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from collab_overcooked.main import convert_yaml_to_variant, load_config_from_yaml
from collab_overcooked.training.main_session import CollabMainSession


def _decode(tokenizer: Any, ids: Any, *, skip_special_tokens: bool) -> str:
    values = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    return tokenizer.decode(values, skip_special_tokens=skip_special_tokens)


def _action_text(text: str) -> str:
    match = re.search(
        r"^\s*Action\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
        text or "",
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    return (match.group(1) if match else "").strip()


def _build_rows(rollout: Path, tokenizer: Any) -> List[Dict[str, Any]]:
    data = torch.load(rollout, map_location="cpu", weights_only=False)
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        prompt_ids = item.get("prompt_ids")
        response_ids = item.get("response_ids")
        if prompt_ids is None or response_ids is None:
            continue
        prompt = _decode(tokenizer, prompt_ids, skip_special_tokens=False)
        response = _decode(tokenizer, response_ids, skip_special_tokens=True)
        rows.append(
            {
                "idx": idx,
                "agent_index": int(item.get("agent_index", -1)),
                "timestep": item.get("timestep"),
                "prompt": prompt,
                "prompt_compact": re.sub(r"\s+", " ", prompt),
                "response": response,
                "old": item,
                "used": False,
            }
        )
    return rows


def _load_snapshot(path: Path) -> Dict[str, Any]:
    if path.suffix == ".jsonl":
        first = path.read_text(encoding="utf-8").splitlines()[0]
        payload = json.loads(first)
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "snapshot" in payload:
        return payload["snapshot"]
    if isinstance(payload, list) and payload:
        first = payload[0]
        return first.get("snapshot", first)
    return payload


def _chat_prompt(tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages)


class ReplayPolicy:
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any):
        self.rows = rows
        self.tokenizer = tokenizer
        self.matches: List[Dict[str, Any]] = []

    def __call__(self, agent_index: int, messages: List[Dict[str, str]], context: Dict[str, Any]):
        prompt = _chat_prompt(self.tokenizer, messages)
        prompt_compact = re.sub(r"\s+", " ", prompt)
        timestep = context.get("timestep")

        candidates = [
            row
            for row in self.rows
            if not row["used"] and int(row["agent_index"]) == int(agent_index)
        ]
        same_ts = []
        for row in candidates:
            try:
                if timestep is not None and int(row["timestep"]) == int(timestep):
                    same_ts.append(row)
            except (TypeError, ValueError):
                pass
        if same_ts:
            candidates = same_ts
        if not candidates:
            raise RuntimeError(f"No unused replay response for agent={agent_index} timestep={timestep}")

        exact = [row for row in candidates if row["prompt"] == prompt]
        if exact:
            best = exact[0]
            score = 1.0
        else:
            scored = [
                (SequenceMatcher(None, prompt_compact, row["prompt_compact"]).ratio(), row)
                for row in candidates
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            score, best = scored[0]
            if score < 0.88:
                raise RuntimeError(
                    f"Low prompt match score={score:.3f} agent={agent_index} timestep={timestep} "
                    f"best_idx={best['idx']}"
                )
        best["used"] = True
        response = best["response"]
        response_ids = self.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        )["input_ids"].squeeze(0)
        self.matches.append(
            {
                "old_idx": best["idx"],
                "agent_index": agent_index,
                "timestep": timestep,
                "score": score,
                "action": _action_text(response),
            }
        )
        return response, {
            "token_count": int(response_ids.numel()),
            "replay_old_idx": int(best["idx"]),
            "replay_match_score": float(score),
            "replay_action": _action_text(response),
            "replay_prompt": prompt,
            "replay_messages": messages,
            "replay_response": response,
            "response_tokens": [
                self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
                for token_id in response_ids.tolist()
            ],
            "response_log_probs": [0.0 for _ in range(int(response_ids.numel()))],
            "log_prob": 0.0,
        }


def _record_to_row(record: Any, step_result: Any, match: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = dict(record.metadata or {})
    breakdown = dict(meta.get("reward_breakdown") or {})
    raw = dict(breakdown.get("raw") or {})
    context = dict(record.context or {})
    old_idx = meta.get("replay_old_idx")
    match_score = meta.get("replay_match_score")
    action = meta.get("replay_action")
    if match:
        old_idx = match.get("old_idx", old_idx)
        match_score = match.get("score", match_score)
        action = match.get("action", action)
    return {
        "old_idx": old_idx,
        "match_score": match_score,
        "agent_index": int(record.agent_index),
        "timestep": record.timestep,
        "micro_step": record.micro_step,
        "call_index": meta.get("call_index"),
        "call_type": meta.get("call_type") or (record.context or {}).get("call_type"),
        "semantic_call_type": meta.get("semantic_call_type"),
        "action_mode": meta.get("action_mode"),
        "prompt": record.prompt or meta.get("replay_prompt") or "",
        "messages": record.messages or meta.get("replay_messages") or [],
        "context": context,
        "prompt_excerpt": context.get("prompt_excerpt"),
        "observation_excerpt": context.get("observation_excerpt"),
        "response": record.response or meta.get("replay_response") or "",
        "action": action,
        "reward": float(record.reward or 0.0),
        "sequence_reward": float(breakdown.get("sequence_reward", raw.get("sequence_reward", 0.0)) or 0.0),
        "process_reward": float(breakdown.get("sequence_reward", raw.get("sequence_reward", 0.0)) or 0.0),
        "format_reward": float(breakdown.get("format_reward", raw.get("format_reward", 0.0)) or 0.0),
        "validator_reward": float(breakdown.get("validator_reward", raw.get("validator_reward", 0.0)) or 0.0),
        "communication_reward": float(breakdown.get("communication_reward", raw.get("communication_reward", 0.0)) or 0.0),
        "paired_comm_reward": float(breakdown.get("paired_comm_reward", raw.get("paired_comm_reward", 0.0)) or 0.0),
        "breakdown_total_reward": float(raw.get("total", record.reward or 0.0) or 0.0),
        "reward_breakdown": breakdown,
        "process_reward_info": step_result.process_reward,
        "env_info": step_result.env_info,
    }


def _old_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _old_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_old_value(v) for v in value]
    return repr(value)


def _attach_old_rollout_fields(replay_rows: List[Dict[str, Any]], old_rows: List[Dict[str, Any]]) -> None:
    by_idx = {int(row["idx"]): row for row in old_rows}
    for row in replay_rows:
        old_idx = row.get("old_idx")
        if old_idx is None:
            continue
        old_row = by_idx.get(int(old_idx))
        if not old_row:
            continue
        old_payload = old_row.get("old") or {}
        row["old_rollout"] = {
            "idx": old_row.get("idx"),
            "agent_index": old_row.get("agent_index"),
            "timestep": old_row.get("timestep"),
            "prompt": old_row.get("prompt"),
            "response": old_row.get("response"),
            "action": _action_text(old_row.get("response") or ""),
            "reward": _old_value(old_payload.get("reward")),
            "sequence_reward": _old_value(old_payload.get("sequence_reward")),
            "format_reward": _old_value(old_payload.get("format_reward")),
            "validator_reward": _old_value(old_payload.get("validator_reward")),
            "reward_breakdown": _old_value(old_payload.get("reward_breakdown")),
            "advantage": _old_value(old_payload.get("advantage")),
            "return": _old_value(old_payload.get("return")),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--model", default="/datacache/LLMs/Qwen2.5-7B-Instruct")
    parser.add_argument("--out-pt", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--stop-on-exhausted",
        action="store_true",
        help="Save partial replay instead of failing when saved responses are exhausted or no longer match the prompt.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = _build_rows(Path(args.rollout), tokenizer)
    policy = ReplayPolicy(rows, tokenizer)

    config = load_config_from_yaml(args.config)
    variant = convert_yaml_to_variant(config)
    variant["yaml_config"] = config
    session = CollabMainSession(variant, policy)
    session.load_snapshot(_load_snapshot(Path(args.snapshot)))

    replay_rows: List[Dict[str, Any]] = []
    latest_by_old_idx: Dict[int, int] = {}
    step_count = 0
    stopped_reason: Optional[str] = None
    while True:
        if session.env.is_done():
            break
        if int(getattr(session.env.state, "timestep", 0)) >= int(variant.get("horizon", 18)):
            break
        before_matches = len(policy.matches)
        try:
            step_result = session.step()
        except RuntimeError as exc:
            message = str(exc)
            if args.stop_on_exhausted and (
                "No unused replay response" in message
                or "Low prompt match score=" in message
            ):
                stopped_reason = str(exc)
                break
            raise
        new_matches = policy.matches[before_matches:]
        records = step_result.policy_records or []
        records_by_identity = {id(record): record for record in records}
        matched_record_ids = set()
        for record, match in zip(records, new_matches):
            row = _record_to_row(record, step_result, match)
            matched_record_ids.add(id(record))
            old_idx = row.get("old_idx")
            if old_idx is not None:
                latest_by_old_idx[int(old_idx)] = len(replay_rows)
            replay_rows.append(row)
        for record in records_by_identity.values():
            if id(record) in matched_record_ids:
                continue
            row = _record_to_row(record, step_result)
            old_idx = row.get("old_idx")
            if old_idx is not None and int(old_idx) in latest_by_old_idx:
                replay_rows[latest_by_old_idx[int(old_idx)]] = row
            elif old_idx is not None:
                latest_by_old_idx[int(old_idx)] = len(replay_rows)
                replay_rows.append(row)
            else:
                replay_rows.append(row)
        step_count += 1
        if step_result.done:
            break

    _attach_old_rollout_fields(replay_rows, rows)
    out_pt = Path(args.out_pt)
    out_json = Path(args.out_json)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(replay_rows, out_pt)
    out_json.write_text(
        json.dumps(
            {
                "rollout": str(args.rollout),
                "config": str(args.config),
                "snapshot": str(args.snapshot),
                "steps": step_count,
                "stopped_reason": stopped_reason,
                "records": replay_rows,
                "unused_old_indices": [row["idx"] for row in rows if not row["used"]],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved_pt={out_pt}")
    print(f"saved_json={out_json}")
    print(f"steps={step_count} records={len(replay_rows)} unused_old={sum(1 for row in rows if not row['used'])}")
    if stopped_reason:
        print(f"stopped_reason={stopped_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
