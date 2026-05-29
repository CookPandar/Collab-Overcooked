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
from collab_overcooked.reward import ProcessRewardTracker
from collab_overcooked.training.main_session import CollabMainSession
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


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
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: Any, *, force_old_order: bool = False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.matches: List[Dict[str, Any]] = []
        self.force_old_order = force_old_order
        self.next_index = 0

    def __call__(self, agent_index: int, messages: List[Dict[str, str]], context: Dict[str, Any]):
        prompt = _chat_prompt(self.tokenizer, messages)
        prompt_compact = re.sub(r"\s+", " ", prompt)
        timestep = context.get("timestep")

        if self.force_old_order:
            while self.next_index < len(self.rows) and self.rows[self.next_index]["used"]:
                self.next_index += 1
            if self.next_index >= len(self.rows):
                raise RuntimeError(f"No unused replay response for agent={agent_index} timestep={timestep}")
            best = self.rows[self.next_index]
            self.next_index += 1
            score = SequenceMatcher(None, prompt_compact, best["prompt_compact"]).ratio()
            mismatch = (
                int(best["agent_index"]) != int(agent_index)
                or (
                    timestep is not None
                    and best.get("timestep") is not None
                    and int(best["timestep"]) != int(timestep)
                )
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
                    "forced_old_order_mismatch": mismatch,
                    "old_agent_index": best.get("agent_index"),
                    "old_timestep": best.get("timestep"),
                }
            )
            return response, {
                "token_count": int(response_ids.numel()),
                "replay_old_idx": int(best["idx"]),
                "replay_match_score": float(score),
                "replay_action": _action_text(response),
                "replay_forced_old_order_mismatch": bool(mismatch),
                "replay_old_agent_index": best.get("agent_index"),
                "replay_old_timestep": best.get("timestep"),
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
    penalties = list(raw.get("penalties") or [])
    if any(item.get("type") == "format" for item in penalties):
        validator_reward = float(
            breakdown.get("validator_reward", raw.get("validator_reward", 0.0)) or 0.0
        )
        if validator_reward:
            record.reward = float(record.reward or 0.0) - validator_reward
        raw["validator_reward"] = 0.0
        raw["penalties"] = [
            item for item in penalties if item.get("type") != "validator"
        ]
        raw["total"] = float(raw.get("total", record.reward or 0.0) or 0.0) - validator_reward
        breakdown["validator_reward"] = 0.0
        breakdown["raw"] = raw
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
        "collab_reward": float(breakdown.get("collab_reward", raw.get("collab_reward", 0.0)) or 0.0),
        "paired_comm_reward": float(breakdown.get("paired_comm_reward", raw.get("paired_comm_reward", 0.0)) or 0.0),
        "breakdown_total_reward": float(raw.get("total", record.reward or 0.0) or 0.0),
        "reward_breakdown": breakdown,
        "process_reward_info": step_result.process_reward,
        "env_info": step_result.env_info,
    }


def _sync_row_from_raw_reward(row: Dict[str, Any]) -> None:
    breakdown = dict(row.get("reward_breakdown") or {})
    raw = dict(breakdown.get("raw") or {})
    if not raw:
        return
    sequence_reward = float(raw.get("sequence_reward", row.get("sequence_reward", 0.0)) or 0.0)
    validator_reward = float(raw.get("validator_reward", row.get("validator_reward", 0.0)) or 0.0)
    format_reward = float(raw.get("format_reward", row.get("format_reward", 0.0)) or 0.0)
    communication_reward = float(raw.get("communication_reward", row.get("communication_reward", 0.0)) or 0.0)
    collab_reward = float(raw.get("collab_reward", row.get("collab_reward", 0.0)) or 0.0)
    paired_comm_reward = float(raw.get("paired_comm_reward", row.get("paired_comm_reward", 0.0)) or 0.0)
    if sequence_reward <= 0.0 and validator_reward == float(row.get("validator_reward", 0.0) or 0.0):
        return
    row["sequence_reward"] = sequence_reward
    row["process_reward"] = sequence_reward
    row["format_reward"] = format_reward
    row["validator_reward"] = validator_reward
    row["communication_reward"] = communication_reward
    row["collab_reward"] = collab_reward
    row["paired_comm_reward"] = paired_comm_reward
    row["reward"] = (
        sequence_reward
        + format_reward
        + validator_reward
        + communication_reward
        + collab_reward
        + paired_comm_reward
    )
    row["breakdown_total_reward"] = float(raw.get("total", row["reward"]) or row["reward"])
    breakdown["sequence_reward"] = sequence_reward
    breakdown["format_reward"] = format_reward
    breakdown["validator_reward"] = validator_reward
    breakdown["communication_reward"] = communication_reward
    breakdown["collab_reward"] = collab_reward
    breakdown["paired_comm_reward"] = paired_comm_reward
    breakdown["raw"] = raw
    row["reward_breakdown"] = breakdown


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
            "collab_reward": _old_value(old_payload.get("collab_reward")),
            "paired_comm_reward": _old_value(old_payload.get("paired_comm_reward")),
            "reward_breakdown": _old_value(old_payload.get("reward_breakdown")),
            "advantage": _old_value(old_payload.get("advantage")),
            "return": _old_value(old_payload.get("return")),
        }


def _normalize_action_text(action: str) -> str:
    text = (action or "").strip().replace("<|im_end|>", "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    action_match = re.search(
        r"^\s*Action\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
    )
    if action_match:
        text = action_match.group(1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text.replace(" ", "")


def _is_collab_action(action: str) -> bool:
    return bool(
        action
        and action.lower().startswith(("collab(", "request(", "seek(", "ack(", "deny("))
    )


def _apply_offline_comm_rewards(
    replay_rows: List[Dict[str, Any]],
    variant: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Re-score communication rows from saved text after replay alignment.

    Environment/process rewards must stay bound to executed actions. Communication
    process rewards are text-only, so replay can safely recompute them from the
    final row order that ptweb displays.
    """
    reward_settings = config.get("reward") or variant.get("reward") or {}
    if not (
        reward_settings.get("paired_comm_reward_enabled")
        or reward_settings.get("collab_reward_enabled")
    ):
        return
    mdp = OvercookedGridworld.from_layout_name(
        variant.get("layout_name") or variant.get("layout", "cramped_room")
    )
    tracker = ProcessRewardTracker(
        variant.get("order", "baked_bell_pepper"),
        mdp,
        Path("collab_overcooked/prompts/reference"),
        reward_settings,
    )
    rows = sorted(
        replay_rows,
        key=lambda row: (
            int(row.get("timestep") if row.get("timestep") is not None else -1),
            int(row.get("micro_step") if row.get("micro_step") is not None else 0),
            int(row.get("old_idx") if row.get("old_idx") is not None else 10**9),
        ),
    )
    for row in rows:
        action = _normalize_action_text(
            str(row.get("action") or _action_text(row.get("response") or ""))
        )
        if not _is_collab_action(action):
            continue
        agent_index = int(row.get("agent_index", -1))
        if agent_index not in (0, 1):
            continue
        timestep = int(row.get("timestep") if row.get("timestep") is not None else -1)
        paired_reward = 0.0
        paired_meta: Dict[str, Any] = {}
        if getattr(tracker, "enable_paired_comm_reward", False):
            paired_reward, paired_meta = tracker._process_paired_comm_reward(  # noqa: SLF001
                agent_index, timestep, action
            )
            paired_reward = float(paired_reward or 0.0)
        collab_reward = 0.0
        if getattr(tracker, "enable_collab_reward", False):
            collab_reward = float(
                tracker._process_collab_reward(agent_index, action) or 0.0  # noqa: SLF001
            )
        if paired_reward == 0.0 and collab_reward == 0.0:
            continue
        old_paired = float(row.get("paired_comm_reward", 0.0) or 0.0)
        old_collab = float(row.get("collab_reward", 0.0) or 0.0)
        old_reward = float(row.get("reward", 0.0) or 0.0)
        has_execution_reward = bool(
            float(row.get("sequence_reward", 0.0) or 0.0)
            or float(row.get("format_reward", 0.0) or 0.0)
            or float(row.get("validator_reward", 0.0) or 0.0)
        )
        row["paired_comm_reward"] = paired_reward
        row["collab_reward"] = collab_reward
        row["reward"] = old_reward - old_paired - old_collab + paired_reward + collab_reward
        breakdown = dict(row.get("reward_breakdown") or {})
        raw = dict(breakdown.get("raw") or {})
        comm_raw = {
            "timestamp": timestep,
            "source_timestamp": timestep,
            "agent_index": agent_index,
            "call_index": row.get("call_index"),
            "call_type": "communication",
            "action": action,
            "collab_reward": collab_reward,
            "paired_comm_reward": paired_reward,
            "paired_comm_role": paired_meta.get("role"),
            "paired_comm_result": paired_meta.get("result"),
            "paired_comm_target": paired_meta.get("target_agent"),
            "paired_comm_request_action": paired_meta.get("request_action"),
            "paired_comm_request_helpful": paired_meta.get("request_helpful"),
            "total": paired_reward + collab_reward,
        }
        if has_execution_reward:
            raw["offline_comm_reward"] = comm_raw
            raw["paired_comm_reward"] = paired_reward
            raw["collab_reward"] = collab_reward
            raw["total"] = row["reward"]
        else:
            raw.update(comm_raw)
        breakdown.update(
            {
                "collab_reward": collab_reward,
                "paired_comm_reward": paired_reward,
                "call_type": "communication",
                "offline_comm_reward_recomputed": True,
                "raw": raw,
            }
        )
        breakdown.pop("missing_reward_entry", None)
        row["reward_breakdown"] = breakdown


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
    parser.add_argument(
        "--force-old-order",
        action="store_true",
        help="Replay responses strictly by old rollout row order, even if agent/timestep/prompt mismatches current execution.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = _build_rows(Path(args.rollout), tokenizer)
    policy = ReplayPolicy(rows, tokenizer, force_old_order=args.force_old_order)

    config = load_config_from_yaml(args.config)
    variant = convert_yaml_to_variant(config)
    variant["yaml_config"] = config
    session = CollabMainSession(variant, policy)
    session.load_snapshot(_load_snapshot(Path(args.snapshot)))

    replay_rows: List[Dict[str, Any]] = []
    replay_row_by_call: Dict[tuple, int] = {}
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
        records = step_result.policy_records or []
        records_by_identity = {id(record): record for record in records}
        matched_record_ids = set()
        for record in records_by_identity.values():
            if id(record) in matched_record_ids:
                continue
            row = _record_to_row(record, step_result)
            _sync_row_from_raw_reward(row)
            call_key = (
                row.get("agent_index"),
                row.get("timestep"),
                row.get("call_index"),
                row.get("old_idx"),
            )
            if call_key in replay_row_by_call:
                previous = replay_rows[replay_row_by_call[call_key]]
                previous.setdefault("reward_versions", []).append(
                    {
                        "phase": "before_reward_backfill",
                        "reward": previous.get("reward"),
                        "sequence_reward": previous.get("sequence_reward"),
                        "format_reward": previous.get("format_reward"),
                        "validator_reward": previous.get("validator_reward"),
                        "paired_comm_reward": previous.get("paired_comm_reward"),
                        "collab_reward": previous.get("collab_reward"),
                        "reward_breakdown": previous.get("reward_breakdown"),
                    }
                )
                previous["reward_versions"].append(
                    {
                        "phase": "after_reward_backfill",
                        "reward": row.get("reward"),
                        "sequence_reward": row.get("sequence_reward"),
                        "format_reward": row.get("format_reward"),
                        "validator_reward": row.get("validator_reward"),
                        "paired_comm_reward": row.get("paired_comm_reward"),
                        "collab_reward": row.get("collab_reward"),
                        "reward_breakdown": row.get("reward_breakdown"),
                    }
                )
                previous.update(row)
                previous["replay_reward_backfill_merged"] = True
                continue
            replay_row_by_call[call_key] = len(replay_rows)
            row["replay_reward_backfill_merged"] = False
            replay_rows.append(row)
        step_count += 1
        if step_result.done:
            break

    _attach_old_rollout_fields(replay_rows, rows)
    _apply_offline_comm_rewards(replay_rows, variant, config)
    for row in replay_rows:
        _sync_row_from_raw_reward(row)
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
