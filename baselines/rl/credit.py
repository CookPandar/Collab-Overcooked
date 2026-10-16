"""Per-call rewards for RL, computed from the benchmark's agent events.

`CallCredit` is an observer (see docs/extension_api.md). Every LLM call becomes one record:

- format penalty: the call produced a format error, or its reply is mixed / multi-action / malformed;
- validator penalty: the action of the call was rejected by the validator;
- ITES: when the action of a planner_main / validator_correction call passes the validator, the call receives
  TES(H + a) - TES(H) of its agent's validated action history H (best over the reference trajectories);
- terminal reward: added by the rollout to the last call of each agent when the order is delivered.
"""

import glob
import json
import os
import re
from bisect import bisect_right
from collections import defaultdict

from collab_overcooked.agents.collab import LLMAgents
from collab_overcooked.config import REFERENCE_DIR

REWARDABLE_CALL_TYPES = {"planner_main", "validator_correction"}
COLLAB_PREFIXES = ("collab(", "request(", "seek(", "ack(", "deny(")
_ACTION_FIELD = re.compile(r"Action\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
                           re.IGNORECASE | re.DOTALL | re.MULTILINE)


def strip_code_fence(text):
    text = (text or "").strip()
    if not text.startswith("```"):
        return text.strip("`").strip()
    body = text[3:]
    closing = body.rfind("```")
    if closing != -1:
        body = body[:closing]
    body = body.strip()
    if "\n" in body:
        first, rest = body.split("\n", 1)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", first.strip()):
            body = rest
    return body.strip().strip("`").strip()


def action_field(response):
    match = _ACTION_FIELD.search(response or "")
    return strip_code_fence(match.group(1) if match else "")


def normalize_action(action):
    """Canonical form used to compare actions: Action field only, no fences, no spaces."""
    text = (action or "").replace("<|im_end|>", "").strip()
    if not text or text == "[EMPTY]":
        return ""
    match = _ACTION_FIELD.search(text)
    if match:
        text = match.group(1)
    return strip_code_fence(text).strip("` \t\r\n").replace(" ", "")


def is_collab(action):
    return (action or "").strip().lower().startswith(COLLAB_PREFIXES)


def is_wait(action):
    lowered = (action or "").strip().lower()
    return lowered == "wait" or lowered.startswith("wait(")


def split_actions(action):
    """Split an Action field on top-level ';' and newlines."""
    tokens, current, depth = [], [], 0
    for char in (action or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", ";"):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        if char == ";" and depth == 0:
            tokens.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tokens.append("".join(current).strip())
    return [token for token in tokens if token]


def _has_trailing_text(token):
    depth = 0
    for index, char in enumerate(token):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
            if depth == 0:
                return bool(token[index + 1:].strip())
    return False


def _fence_after_action_is_malformed(response):
    if "```" not in response:
        return False
    match = re.search(r"^\s*Action\s*:", response, flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        return True
    after = response[match.end():]
    first = after.find("```")
    if first == -1:
        return False
    if after[:first].strip():
        return bool(after[first + 3:].strip())
    closing = after.find("```", first + 3)
    return closing == -1 or bool(after[closing + 3:].strip())


def response_mode(response):
    """planner_main | communication | wait | empty | mixed | multi_embodied | malformed."""
    response = response or ""
    if LLMAgents._has_malformed_response(response) or _fence_after_action_is_malformed(response):
        return "malformed"
    tokens = split_actions(action_field(response))
    if not tokens:
        return "empty"
    if any("```" in t or re.search(r"(?i)\bpython\b", t) or _has_trailing_text(t) for t in tokens):
        return "malformed"
    collab = [t for t in tokens if is_collab(t)]
    embodied = [t for t in tokens if not is_collab(t)]
    if collab and embodied:
        return "mixed"
    if len(embodied) > 1:
        return "multi_embodied"
    if collab:
        return "communication"
    if embodied and is_wait(embodied[0]):
        return "wait"
    return "planner_main"


def tes(history, reference, beta=0.95):
    """Trajectory efficiency score of an action history against one reference (F-beta of the matched prefix depth)."""
    history = [a for a in history if a]
    reference = [a for a in reference if a]
    if not history or not reference:
        return 0.0
    positions = defaultdict(list)
    for index, action in enumerate(history):
        positions[action].append(index)
    depth = 0
    for start in positions.get(reference[0], []):
        position, current = start, 1
        for action in reference[1:]:
            candidates = positions.get(action)
            if not candidates:
                break
            nxt = bisect_right(candidates, position)
            if nxt >= len(candidates):
                break
            position, current = candidates[nxt], current + 1
        depth = max(depth, current)
    beta_sq = beta * beta
    return (1 + beta_sq) * depth / (len(reference) + beta_sq * len(history))


def load_references(order, reference_dir=REFERENCE_DIR):
    path = sorted(glob.glob(os.path.join(reference_dir, f"*_{order}_ref.*")))[0]
    with open(path) as f:
        content = json.load(f)
    references = {0: [], 1: []}
    for entry in content.values():
        for agent_index in (0, 1):
            sequence = entry.get(f"agent_{agent_index}")
            if sequence:
                references[agent_index].append([normalize_action(a) for a in sequence if a.strip()])
    return references


class CallCredit:
    """Observer that turns agent events into one reward record per LLM call."""

    def __init__(self, order, format_penalty=0.2, validator_penalty=0.1, ites_weight=1.0, tes_beta=0.95):
        self.references = load_references(order)
        self.format_penalty = -abs(format_penalty)
        self.validator_penalty = -abs(validator_penalty)
        self.ites_weight = ites_weight
        self.tes_beta = tes_beta
        self.calls = {}
        self.histories = {0: [], 1: []}
        self.scores = {0: 0.0, 1: 0.0}

    @staticmethod
    def key(source):
        return (int(source["agent_index"]), int(source["timestep"]), int(source["call_index"]))

    def _record(self, source):
        return self.calls.get(self.key(source)) if source else None

    # ---- observer events -------------------------------------------------------------------
    def on_llm_call(self, agent, call, prompt, response, metadata):
        self.calls[self.key(call)] = {
            "agent_index": int(call["agent_index"]),
            "timestep": int(call["timestep"]),
            "call_index": int(call["call_index"]),
            "call_type": call["call_type"],
            "response": response or "",
            "metadata": metadata or {},
            "format_error": False,
            "validator_error": False,
            "validated_action": None,
            "ites": 0.0,
        }

    def on_call_update(self, agent, call, fields):
        record = self._record(call)
        if record is not None:
            record["call_type"] = call["call_type"]

    def on_format_error(self, agent, source, detail):
        record = self._record(source)
        if record is not None:
            record["format_error"] = True

    def on_validation(self, agent, action, source, ok, message, format_error):
        record = self._record(source)
        if record is None:
            return
        if ok:
            self._credit(record, action)
        elif format_error:
            record["format_error"] = True
        elif record["validated_action"] is None:
            record["validator_error"] = True

    def on_action_executed(self, agent, action, source, env_action, interaction_param):
        record = self._record(source)
        if record is not None:
            self._credit(record, action)

    # ---- ITES ------------------------------------------------------------------------------
    def _credit(self, record, action):
        action = normalize_action(action)
        if (record["validated_action"] is not None or not action or is_wait(action) or is_collab(action)
                or record["call_type"] not in REWARDABLE_CALL_TYPES
                or record["format_error"] or record["validator_error"]):
            return
        agent_index = record["agent_index"]
        history = self.histories[agent_index]
        history.append(action)
        after = max((tes(history, ref, self.tes_beta) for ref in self.references[agent_index]), default=0.0)
        record["validated_action"] = action
        record["ites"] = self.ites_weight * (after - self.scores[agent_index])
        self.scores[agent_index] = after

    # ---- final rewards ---------------------------------------------------------------------
    def rewards(self, record):
        """(mode, format_reward, validator_reward, ites_reward) of a call record."""
        mode = response_mode(record["response"])
        if mode in {"mixed", "multi_embodied", "malformed"}:
            return mode, self.format_penalty, 0.0, 0.0
        fmt = self.format_penalty if record["format_error"] else 0.0
        if mode in {"communication", "wait"}:
            return mode, fmt, 0.0, 0.0
        val = self.validator_penalty if record["validator_error"] and fmt == 0.0 else 0.0
        ites = record["ites"] if fmt == 0.0 and val == 0.0 else 0.0
        return mode, fmt, val, ites
