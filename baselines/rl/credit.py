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


ROLE_INDEX = {"chef": 0, "assistant": 1}


def collab_requests(action):
    """[(receiver, normalized action)] of the `request(receiver, action)` primitives of a Collab action."""
    text = (action or "").replace(" ", "")
    found = []
    for match in re.finditer(r"request\((\w+),", text, flags=re.IGNORECASE):
        depth, start = 1, match.end()
        for end in range(start, len(text)):
            depth += {"(": 1, ")": -1}.get(text[end], 0)
            if depth == 0:
                found.append((match.group(1), normalize_action(text[start:end])))
                break
    return found


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


def next_steps(history, references):
    """Next step of each reference with the deepest in-order match of `history` (empty when one is finished)."""
    def depth(reference):
        d = 0
        for action in history:
            if d < len(reference) and action == reference[d]:
                d += 1
        return d
    best = max((depth(r) for r in references), default=0)
    steps = [r[best] for r in references if depth(r) == best and best < len(r)]
    return [] if any(depth(r) == best == len(r) for r in references) else steps


_HOLDS = r"<{role}> holds ([^.]*)\."
_PICKUP = re.compile(r"pickup\((\w+),(\w+)\)")
_UTENSIL_STEP = re.compile(r"\w+\((\w+\d+)\)")
NEEDS_ITEM = ("put_obj_in_utensil(", "place_obj_on_counter(", "deliver_soup(", "fill_dish_with_food(")


def next_step_ready(step, observation, agent_index):
    """Whether the reference step `step` of the agent can be done in the state of `observation` (the prompt):
    pickup(X,counter) needs X on a counter, pickup(X,utensil) needs the utensil to have X ready, a step that puts /
    places / serves needs a held item, a pickup needs empty hands, a utensil step (cut, bake ...) needs the utensil
    to hold all its ingredients; other steps are ready."""
    observation = observation or ""
    observation = observation[max(0, observation.rfind("Kitchen states:")):]  # the current state only
    role = "Chef" if agent_index == 0 else "Assistant"
    held = re.search(_HOLDS.format(role=role), observation or "")
    holding = bool(held) and held.group(1).strip() != "nothing"
    match = _PICKUP.fullmatch(step)
    if match:
        item, place = match.groups()
        if holding:
            return False
        if place == "ingredient_dispenser":
            return True
        if place == "counter":
            states = re.search(r"counters? can be visited by <" + role + r">\. Their states are as follows: ([^\n]*)",
                               observation or "")
            return bool(states and re.search(rf"(?<!\w){re.escape(item)}(?!\w)", states.group(1)))
        return bool(re.search(rf"<{re.escape(place)}> has {re.escape(item)}(?![\w,])", observation or ""))
    if step.startswith(NEEDS_ITEM):
        return holding
    match = _UTENSIL_STEP.fullmatch(step)
    if match:
        return bool(re.search(rf"<{re.escape(match.group(1))}> has [^;]*which are all the ingredients", observation or ""))
    return True


class CallCredit:
    """Observer that turns agent events into one reward record per LLM call."""

    def __init__(self, order, format_penalty=0.2, validator_penalty=0.1, ites_weight=1.0, tes_beta=0.95,
                 collab_weight=0.0, uncoop_penalty=0.0, uncoop_kinds=("repeat", "idle"), collab_mode="promise",
                 collab_window=5):
        self.references = load_references(order)
        # collaboration reward (port of the v1 ProcessRewardTracker option): a Collab request is credited with the
        # TES gain the requested action would give the teammate, once per gain; the teammate is credited the same
        # amount when it executes that action with positive ITES. 0 disables it.
        # collab_mode "uptake": no credit at send time; when the teammate executes a requested action within
        # `collab_window` timesteps of the request with positive ITES, the requesting call gets collab_weight x that
        # ITES (once per request; a request already pending from the same sender adds nothing); the teammate gets
        # nothing extra. collab_mode "promise" (default) is the behaviour above.
        if collab_mode not in ("promise", "uptake"):
            raise ValueError(f"unknown collab_mode {collab_mode}")
        self.collab_weight = collab_weight
        self.collab_mode = collab_mode
        self.collab_window = collab_window
        self.requests = {0: [], 1: []}      # uptake: [request record, action, last timestep] asked of each agent
        self.requested = {0: 0.0, 1: 0.0}  # best teammate TES promised by requests so far
        self.pending = {0: [], 1: []}       # (action, reward) requested from each agent, not executed yet
        # penalty of the collaboration reward on uncooperative calls (see uncooperative()); such a call also gets no
        # share of the delivery reward (compute_advantages). 0 disables it.
        self.uncoop_penalty = -abs(uncoop_penalty)
        self.uncoop_kinds = tuple(uncoop_kinds)  # which of "repeat" / "idle" are penalized
        self.sent = {0: set(), 1: set()}  # each agent's Collab messages since the last executed action
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
            # for the think judge: what the agent saw, and its validated history before this call
            "observation": prompt or "",
            "history_before": list(self.histories[int(call["agent_index"])]),
            "validations": [],
            "collab": 0.0,
            "uncooperative": None,
            "repeat": False,      # a Collab message the agent already sent since the last executed action
            "requests": 0,        # uptake: new requests of this call / how many of them the teammate took up
            "uptaken": 0,
        }
        record = self.calls[self.key(call)]
        if self.collab_weight and self.collab_mode == "uptake":
            self._register_requests(record)
        elif self.collab_weight:
            self._request_credit(record)
        record["repeat"] = self._track_repeat(record)
        if self.uncoop_penalty:
            record["uncooperative"] = self.uncooperative(record)

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
        record["validations"].append((bool(ok), message or ""))
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
        self.sent = {0: set(), 1: set()}  # the kitchen changed: a message may be repeated
        record["ites"] = self.ites_weight * (after - self.scores[agent_index])
        self.scores[agent_index] = after
        if self.collab_weight and self.collab_mode == "uptake":
            self._uptake(record, action)
        elif self.collab_weight and record["ites"] > 0:
            for k, (requested, reward) in enumerate(self.pending[agent_index]):
                if requested == action:
                    record["collab"] += reward
                    del self.pending[agent_index][k]
                    break

    # ---- collaboration reward --------------------------------------------------------------
    def _request_credit(self, record):
        if response_mode(record["response"]) != "communication":
            return
        sender = record["agent_index"]
        target = 1 - sender
        for name, action in collab_requests(action_field(record["response"])):
            if ROLE_INDEX.get(name.lower()) != target or not action:
                continue
            promised = max((tes(self.histories[target] + [action], ref, self.tes_beta)
                            for ref in self.references[target]), default=0.0)
            gain = promised - max(self.scores[target], self.requested[target])
            if gain > 0:
                self.requested[target] = promised
                reward = self.collab_weight * self.ites_weight * gain
                record["collab"] += reward
                self.pending[target].append((action, reward))

    def _register_requests(self, record):
        """uptake mode: remember the requests of a Collab message to the teammate (no credit yet)."""
        if response_mode(record["response"]) != "communication":
            return
        sender, target = record["agent_index"], 1 - record["agent_index"]
        live = [r for r in self.requests[target] if r[2] >= record["timestep"]]
        for name, action in collab_requests(action_field(record["response"])):
            if ROLE_INDEX.get(name.lower()) != target or not action:
                continue
            if any(r[1] == action and r[0]["agent_index"] == sender for r in live):
                continue  # already asked and still pending: a repeated request earns nothing
            entry = [record, action, record["timestep"] + self.collab_window]
            self.requests[target].append(entry)
            live.append(entry)
            record["requests"] += 1

    def _uptake(self, record, action):
        """uptake mode: the agent of `record` executed `action`; credit the earliest live request for it."""
        agent_index = record["agent_index"]
        self.requests[agent_index] = [r for r in self.requests[agent_index] if r[2] >= record["timestep"]]
        matching = [r for r in self.requests[agent_index] if r[1] == action]
        if not matching:
            return
        self.requests[agent_index] = [r for r in self.requests[agent_index] if r[1] != action]
        if record["ites"] > 0:
            request = matching[0][0]
            request["collab"] += self.collab_weight * record["ites"]
            request["uptaken"] += 1

    def _track_repeat(self, record):
        """Whether a Collab message repeats one the agent sent since the last executed action (of either agent;
        so alternating two messages is a repeat too). Tracked for every call (metrics), penalized only through
        uncooperative()."""
        if response_mode(record["response"]) != "communication":
            return False
        message = normalize_action(action_field(record["response"]))
        repeat = message in self.sent[record["agent_index"]]
        self.sent[record["agent_index"]].add(message)
        return repeat

    def uncooperative(self, record):
        """"repeat": a Collab message the agent already sent since the last executed action (of either agent; so
        alternating two messages is a repeat too); "idle": a wait while the agent's next reference step can be done
        now (next_step_ready); else None. Only the kinds in `uncoop_kinds` are returned."""
        agent_index = record["agent_index"]
        action = action_field(record["response"])
        mode = response_mode(record["response"])
        if mode == "communication":
            return "repeat" if record["repeat"] and "repeat" in self.uncoop_kinds else None
        if mode == "wait" and record["call_type"] in REWARDABLE_CALL_TYPES and "idle" in self.uncoop_kinds:
            steps = next_steps(self.histories[agent_index], self.references[agent_index])
            if steps and all(next_step_ready(s, record["observation"], agent_index) for s in steps):
                return "idle"
        return None

    def collab_reward(self, record):
        """Collaboration reward of a call (0 for calls with a format or validator penalty): the request credit
        minus the penalty of an uncooperative call."""
        mode, fmt, val, _ = self.rewards(record)
        if fmt != 0.0 or val != 0.0:
            return 0.0
        return record["collab"] + (self.uncoop_penalty if record["uncooperative"] else 0.0)

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
