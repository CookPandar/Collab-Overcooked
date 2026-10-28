"""Step-level LLM judge of the Think / Recent Goal part of a reply (think_action protocol).

Adapted from the step-level interactivity judge of CollabBench (Qian et al., ICML 2026): a critical judge sees
the observation the agent saw and its full reply, scores one decision on a ten-level scale (0.1 ... 1.0) with
behavioural anchors and returns JSON with the evidence before the score. CollabBench's affective dimensions are
replaced by what makes a reasoning step good in Collab-Overcooked, and the judge is grounded with privileged
facts from the environment that the agent cannot see:

- the recipe of the order (only the Chef sees it in its prompt) and a summary of the game rules;
- the reference action sequences of both roles and the reference steps this agent has already completed;
- what happened to the action of this call: validator verdict and message, and its ITES progress.

Rubric: (1) state grounding, (2) recipe / plan correctness, (3) coordination with the teammate,
(4) think-action consistency. The score is used as a per-call reward of the reasoning tokens (see
`think_advantages` in advantages.py).
"""

import hashlib
import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from collab_overcooked.config import PROMPT_DIR

from .credit import COLLAB_PREFIXES, _ACTION_FIELD, load_references, response_mode

_ACTION_HEADER = re.compile(r"^\s*`*\s*Action\s*:", re.IGNORECASE | re.MULTILINE)
_FIELD = re.compile(r"^\s*(Think|Recent Goal)\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
                    re.IGNORECASE | re.DOTALL | re.MULTILINE)
LEVELS = [round(0.1 * k, 1) for k in range(1, 11)]

RULES = """- Two players in separate spaces: the Chef and the Assistant. Each space lists the utensils / dispensers that
  player can reach (e.g. "Chef space: pot0 oven0 counter"). A player can only use what is in its own space.
- Only the Assistant can take ingredients from the ingredient_dispenser. Items are passed only through the shared
  counter: pick it up, place_obj_on_counter(), and the teammate picks it up from the counter.
- Each player holds at most one item. A utensil must be started after filling it (cut / bake / cook / stir ...).
  Cooked food is taken with pickup(food, utensil) or, if the recipe needs a dish, with fill_dish_with_food(utensil)
  while holding a dish. deliver_soup() serves the finished food.
- Only the Chef knows the recipe; the Assistant must ask (Collab(seek(...))) when it does not know the procedure.
- Collab(request / seek / ack / deny ...) is communication; it costs no time but the Action field of one reply
  must contain either one Collab(...) or exactly one embodied action, never both.
- The reply has three fields: Think (reasoning), Recent Goal (the agent's short plan) and Action."""

SYSTEM = ("You are a critical, state-aware evaluator of the reasoning of an AI agent playing a two-player "
          "cooperative cooking game (Collab-Overcooked). You evaluate the quality of the agent's REASONING "
          "(its Think and Recent Goal fields) for one decision. You must actively look for factual mistakes "
          "about the kitchen state, wrong recipe steps, ignored teammate messages and mismatches between the "
          "reasoning and the chosen Action. Reply with JSON only.")

TEMPLATE = """## Game rules (summary)
{rules}

## Order: {order}
Recipe (privileged; only the Chef sees it in its prompt):
{recipe}

## Privileged reference solution (one valid way to finish the order)
Chef (agent 0) reference actions: {ref_chef}
Assistant (agent 1) reference actions: {ref_assistant}
Division of labour in the reference: {labour}
Reference steps already completed by the {role} before this decision: {done}
Next reference step(s) of the {role}: {next_steps}

## The agent being evaluated: the {role}
Call type: {call_type} ({call_hint})

<|The Start of the Observation Given to the Agent|>
{observation}
<|The End of the Observation Given to the Agent|>

<|The Start of the Agent's Reply|>
{response}
<|The End of the Agent's Reply|>

## What happened to the Action (privileged, from the environment)
{outcome}

## How to evaluate
Judge the reasoning, not the luck of the outcome. Use the observation and the privileged facts as evidence.
Examine four aspects together:
1. State grounding: are the statements about what each player holds, utensil contents, counters, the current
   actions and the conversation correct according to the observation? Invented or contradicted facts are severe.
2. Plan correctness: does the reasoning identify the correct next step(s) of the recipe given the progress so far,
   respect who can reach what and who may use the ingredient dispenser, and is the Recent Goal a correct, ordered,
   up-to-date plan? Compare with the reference solution (other valid orders of independent steps are fine).
3. Coordination: does it use the teammate's messages and state (answer a seek / request, avoid duplicate work,
   ask or instruct only when needed, avoid redundant chatter and needless waiting)?
4. Consistency: does the Action follow from the reasoning and the first unfinished step of the Recent Goal?
   implied_action is what the Think itself says the agent will do NOW (e.g. "I will wait" -> wait; "I will ask
   the Chef ..." -> Collab(seek(...)); "I will place the slices on the counter" -> place_obj_on_counter()). Do NOT
   put your own idea of the best action there: whether the decision is good is aspect 2, not aspect 4.
   action_follows = false only if the Action does something different from implied_action (a message when the
   Think decided to act, a different object / utensil, acting when it decided to wait, ...).
Length and polish do not matter; a short reasoning that is correct is better than a long one with errors.

## Scoring (choose ONE score from {levels})
- 0.9-1.0 Excellent: grounded in the observation without errors, identifies the right next step for this role,
  handles the teammate correctly, Recent Goal is correct and the Action is exactly what the reasoning concludes.
- 0.6-0.8 Good: correct overall with minor issues (generic or partly outdated Recent Goal, slightly inefficient
  but valid step, unnecessary message, small omission).
- 0.3-0.5 Weak: at least one substantive problem: a wrong recipe step or role assignment, ignores a pending
  teammate message, the Action does not follow from the reasoning, or the plan would stall the team.
- 0.1-0.2 Poor: hallucinates the kitchen state, reasons about a different order, contradicts the observation,
  is empty / copied boilerplate, or the Action contradicts the reasoning.

## Output format (JSON only)
{{"analysis": "<brief evidence for each aspect, at most 120 words>", "implied_action": "<the action the Think / Recent Goal conclude, in the game's action syntax>", "action_follows": <true or false>, "grounding": <1-5>, "plan": <1-5>, "coordination": <1-5>, "consistency": <1-5>, "score": <one of {levels}>}}
If action_follows is false the score must be at most 0.4."""

CALL_HINTS = {
    "planner_main": "the regular decision of a timestep",
    "validator_correction": "a new decision after the previous Action of this timestep was rejected; the "
                            "observation ends with the error message",
    "communication": "a reply inside a dialogue with the teammate",
    "format_correction": "a re-generation after a malformed reply",
    "rethink": "a reflection after a correction",
}


def split_response(response):
    """(reasoning text, action text): the reasoning is everything before the Action field."""
    response = response or ""
    match = _ACTION_HEADER.search(response)
    if match is None:
        return response, ""
    return response[:match.start()], response[match.start():]


def reasoning_fields(response):
    fields = {m.group(1).lower(): m.group(2).strip().strip("`").strip() for m in _FIELD.finditer(response or "")}
    return fields.get("think", ""), fields.get("recent goal", "")


def action_start_token(tokenizer, response_ids, response):
    """Index of the first response token of the Action field (len(response_ids) if there is none)."""
    match = _ACTION_HEADER.search(response or "")
    if match is None:
        return len(response_ids)
    # the header may start after leading whitespace; the split goes at the start of the word "Action"
    target = response.index("Action", match.start()) if "Action" in response[match.start():match.end()] \
        else match.start()
    lo, hi = 0, len(response_ids)
    while lo < hi:  # smallest k whose decoded prefix reaches past the start of the header
        mid = (lo + hi) // 2
        if len(tokenizer.decode(response_ids[:mid + 1], skip_special_tokens=True)) > target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _recipe(order):
    for name in os.listdir(os.path.join(PROMPT_DIR, "recipe")):
        if name.endswith(f"_{order}.txt") and name.split("_", 1)[1] == f"{order}.txt":
            with open(os.path.join(PROMPT_DIR, "recipe", name)) as f:
                return f.read().strip()
    return "(unknown)"


def outcome_text(record):
    mode = response_mode(record["response"])
    lines = [f"Reply type: {mode}."]
    if record.get("format_error") or mode in {"mixed", "multi_embodied", "malformed", "empty"}:
        lines.append("The reply had a FORMAT ERROR (missing field, or not exactly one action / one Collab).")
    for ok, message in record.get("validations", []):
        lines.append("Validator: accepted." if ok else f"Validator: REJECTED: {message.strip()}")
    if record.get("validated_action"):
        progress = record.get("ites", 0.0)
        verdict = ("advanced the reference solution" if progress > 1e-6 else
                   "did not match the next reference steps" if progress < -1e-6 else "made no reference progress")
        lines.append(f"Executed action {record['validated_action']}: {verdict} (TES change {progress:+.3f}).")
    if mode == "communication":
        lines.append("The Action was a message to the teammate (no kitchen action).")
    if mode == "wait":
        lines.append("The Action was wait.")
    return "\n".join(lines)


def division_of_labour(refs):
    """e.g. "the Assistant does cut(chopping_board0); the Chef does bake(oven0), deliver_soup()"."""
    parts = []
    for i, name in ((1, "Assistant"), (0, "Chef")):
        steps = [a for a in refs[i] if not a.startswith(("pickup(", "place_obj_on_counter", "put_obj_in_utensil"))]
        parts.append(f"the {name} does " + (", ".join(steps) if steps else "only pick / place / put steps"))
    return "; ".join(parts) + ". Each player only operates the utensils in its own space."


def build_prompt(record, order, references):
    role = "Chef" if record["agent_index"] == 0 else "Assistant"
    refs = {i: (references[i][0] if references[i] else []) for i in (0, 1)}
    own = refs[record["agent_index"]]
    done = record.get("history_before")
    if done is None:  # offline calibration data without an action history
        done_text, next_text = "(unknown)", "(unknown)"
    else:
        remaining = [a for a in own if a not in done]
        done_text = ", ".join(done) or "(none)"
        next_text = ", ".join(remaining[:2]) or "(none: its part is finished)"
    return TEMPLATE.format(
        labour=division_of_labour(refs),
        rules=RULES, order=order, recipe=_recipe(order), role=role,
        ref_chef=" -> ".join(refs[0]) or "(none)", ref_assistant=" -> ".join(refs[1]) or "(none)",
        done=done_text, next_steps=next_text,
        call_type=record["call_type"], call_hint=CALL_HINTS.get(record["call_type"], ""),
        observation=(record.get("observation") or "").strip(), response=(record["response"] or "").strip(),
        outcome=outcome_text(record), levels=", ".join(str(x) for x in LEVELS))


def parse_judgement(text):
    """{score, grounding, plan, coordination, consistency, analysis} or None."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        match = re.search(r'"score"\s*:\s*([0-9.]+)', text)
        if not match:
            return None
        data = {"score": float(match.group(1))}
    try:
        score = float(data["score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    follows = data.get("action_follows")
    if isinstance(follows, str):
        follows = follows.strip().lower() not in ("false", "no", "0")
    if follows is False:  # enforce the rubric's cap
        score = min(score, 0.4)
    result = {"score": min(LEVELS, key=lambda x: abs(x - score)), "analysis": str(data.get("analysis", ""))[:2000],
              "implied_action": str(data.get("implied_action", ""))[:300], "action_follows": follows}
    for key in ("grounding", "plan", "coordination", "consistency"):
        try:
            result[key] = float(data[key])
        except (KeyError, TypeError, ValueError):
            result[key] = None
    return result


def read_env_file(path):
    values = {}
    for line in open(os.path.expanduser(path)):
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


class ThinkJudge:
    """Scores calls with an OpenAI-compatible chat API: a hosted model (e.g. gpt-5.5 through the shuai API, with
    `env_file` holding OPENAI_BASE_URL / OPENAI_API_KEY) or a local vLLM server (`url`)."""

    def __init__(self, url=None, model="judge", workers=32, max_tokens=None, temperature=None, timeout=300,
                 retries=3, env_file=None, extra=None, log_path=None):
        values = read_env_file(env_file) if env_file else {}
        self.base_url = (url or os.getenv("JUDGE_BASE_URL") or values.get("OPENAI_BASE_URL")).rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.api_key = os.getenv("JUDGE_API_KEY") or values.get("OPENAI_API_KEY")
        self.model = model
        self.workers = workers
        self.max_tokens = max_tokens
        self.temperature = temperature  # None: the provider default (reasoning models reject other values)
        self.timeout = timeout
        self.retries = retries
        self.extra = extra or {}
        self.cache = {}
        self.lock = threading.Lock()
        self._local = threading.local()
        self.usage = {"calls": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0}
        # every request with its full input and raw output is appended here (data for training a judge model)
        self.log_path = os.path.expanduser(log_path) if log_path else None
        self.context = {}  # extra fields written with every logged request (e.g. run, update)

    def log(self, kind, system, prompt, raw, parsed, usage=None, error=None, **fields):
        if not self.log_path:
            return
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "model": self.model,
                 "temperature": self.temperature, **self.context, **fields,
                 "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                 "raw_output": raw, "parsed": parsed, "usage": usage, "error": error}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self.lock:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(line)

    def _request(self, prompt, temperature, system=SYSTEM):
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                   **self.extra}
        if temperature is not None:
            payload["temperature"] = temperature
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                                         headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode()
        if body.lstrip().startswith("data:"):  # some gateways stream even without stream=true
            parts, usage = [], {}
            for line in body.splitlines():
                line = line.strip()
                if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                    continue
                event = json.loads(line[5:])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    parts.append((choice.get("delta") or {}).get("content") or "")
            content = "".join(parts)
        else:
            data = json.loads(body)
            usage = data.get("usage") or {}
            content = data["choices"][0]["message"]["content"] or ""
        self._local.usage = usage
        with self.lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        return content.split("</think>")[-1]

    def judge_prompt(self, prompt):
        key = hashlib.sha1(prompt.encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        result, raw = None, ""
        for attempt in range(self.retries + 1):
            try:
                raw = self._request(prompt, self.temperature)
            except Exception as error:  # API hiccup: back off and retry, then give up (no think reward)
                raw = f"[request failed: {error}]"
                with self.lock:
                    self.usage["failures"] += 1
                self.log("call", SYSTEM, prompt, None, None, error=str(error)[:500], attempt=attempt)
                time.sleep(2 * (attempt + 1))
                continue
            result = parse_judgement(raw)
            self.log("call", SYSTEM, prompt, raw, result, usage=getattr(self._local, "usage", None),
                     attempt=attempt)
            if result is not None:
                break
        result = dict(result or {"score": None}, raw=raw)
        with self.lock:
            self.cache[key] = result
        return result

    def judge_records(self, records, order):
        """Judge a list of call records (see rollout.py) of one episode of `order`."""
        references = load_references(order)
        prompts = [build_prompt(r, order, references) for r in records]
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.judge_prompt, prompts))


def judgeable(transition):
    """Only well-formed replies with a non-empty Think are judged; the others keep their format penalty."""
    if transition["format_reward"] < 0:
        return False
    think, _ = reasoning_fields(transition["response"])
    return bool(think) and _ACTION_FIELD.search(transition["response"] or "") is not None


__all__ = ["ThinkJudge", "build_prompt", "parse_judgement", "split_response", "action_start_token", "judgeable",
           "reasoning_fields", "COLLAB_PREFIXES"]


# ---- episode-level judging: one request scores every judged call of an episode ---------------------------------
# The fixed part (rules, recipe, reference, rubric) is sent once per episode instead of once per call, the
# observations are compressed (the scene layout is stated once) and the judge returns a short verdict per call.
EPISODE_SYSTEM = ("You are a critical, state-aware evaluator of the reasoning of two AI agents playing a cooperative "
                  "cooking game (Collab-Overcooked). You score the REASONING (Think and Recent Goal) of every listed "
                  "decision. Reply with JSON only.")

EPISODE_TEMPLATE = """## Game rules (summary)
{rules}
Spaces: {spaces}

## Order: {order}
Recipe (privileged; only the Chef sees it): {recipe}
Reference solution (privileged, one valid way): Chef: {ref_chef} | Assistant: {ref_assistant}
Division of labour: {labour}

## Rubric for each decision (score ONE value from {levels})
Judge the reasoning, not luck. Check: (1) grounding: statements about holdings, utensils, counters, current actions
and messages match that step's observation; invented facts are severe; (2) plan: the right next recipe step for this
role given progress, respecting who can reach what; Recent Goal correct and current; (3) coordination: answers
teammate messages, no duplicate work, no needless chatter or waiting; (4) consistency: the Action does what the Think
says it will do now (judge this against the Think's own conclusion, not against your preferred action).
0.9-1.0 excellent, no errors | 0.6-0.8 correct with minor issues | 0.3-0.5 a substantive error (wrong step / role,
ignored message, plan would stall) | 0.1-0.2 hallucinated state, wrong order, contradiction or empty boilerplate.
If the Action does not follow the Think (follows=false), the score is at most 0.4. Length and polish do not matter.
Separately rate "grounded" (one value from {levels}): only the factual correctness of what the Think / Recent Goal
state or presuppose about the CURRENT state: what each player holds, utensil contents and whether they are empty,
ready to start, processing or done, counters, which utensils each player can reach, whether a step can be done right
now (e.g. "I must wait" when nothing blocks the step, or "I will put X in pot0" when the agent cannot reach pot0), the
teammate's current action and its messages. 1.0 = every such statement is correct (or none is made), 0.6-0.8 = one
minor slip, 0.3-0.5 = a wrong fact that matters for the decision, 0.1-0.2 = several wrong or invented facts.

## Decisions of one episode in time order
Each decision: the observation the agent saw (scene layout omitted), its reply, and what happened (privileged).
{steps}

## Output (JSON only, one entry per decision id, in order)
{{"steps": [{{"id": <id>, "follows": <true|false>, "score": <score>, "grounded": <score>, "why": "<at most 15 words>"}}, ...]}}"""

_STATIC_LINES = re.compile(r"^\s*(Scene \d+:|Order:|Chef space:|Assistant space:).*$\n?", re.MULTILINE)


def compress_observation(observation):
    text = _STATIC_LINES.sub("", observation or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _spaces(observation):
    found = re.findall(r"^\s*((?:Chef|Assistant) space:.*)$", observation or "", flags=re.MULTILINE)
    return "; ".join(" ".join(line.split()) for line in found) or "(see observations)"


def short_outcome(record):
    mode = response_mode(record["response"])
    parts = [mode]
    for ok, message in record.get("validations", []):
        parts.append("validator ok" if ok else f"validator REJECTED: {' '.join(message.split())[:160]}")
    if record.get("validated_action"):
        progress = record.get("ites", 0.0)
        parts.append(f"executed {record['validated_action']} "
                     + ("(advanced the reference)" if progress > 1e-6 else
                        "(off the reference)" if progress < -1e-6 else "(no reference progress)"))
    done = record.get("history_before")
    if done is not None:
        parts.append(f"reference steps done by this agent before: {len(done)}")
    return "; ".join(parts)


def build_episode_prompt(records, order, references):
    """`records`: judge records (see rollout.py) in time order, each with `id`."""
    refs = {i: (references[i][0] if references[i] else []) for i in (0, 1)}
    steps = []
    for r in records:
        role = "Chef" if r["agent_index"] == 0 else "Assistant"
        steps.append(f"### id {r['id']} | {role} | {r['call_type']}\n[observation]\n{compress_observation(r['observation'])}"
                     f"\n[reply]\n{(r['response'] or '').strip().strip('`').strip()}\n[outcome] {short_outcome(r)}")
    return EPISODE_TEMPLATE.format(
        rules=RULES, spaces=_spaces(records[0]["observation"] if records else ""), order=order,
        recipe=" ".join(_recipe(order).split()), ref_chef=" -> ".join(refs[0]) or "(none)",
        ref_assistant=" -> ".join(refs[1]) or "(none)", labour=division_of_labour(refs),
        levels=", ".join(str(x) for x in LEVELS), steps="\n\n".join(steps))


def parse_episode_judgement(text, ids):
    """{id: {score, follows, why}} for the ids present and valid in the reply."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    entries = []
    try:
        entries = json.loads(text[start:end + 1])["steps"]
    except (ValueError, KeyError, TypeError):  # fall back to the individual entries
        for match in re.finditer(r"\{[^{}]*\"id\"[^{}]*\}", text):
            try:
                entries.append(json.loads(match.group(0)))
            except ValueError:
                pass
    result = {}
    for entry in entries if isinstance(entries, list) else []:
        try:
            key, score = int(entry["id"]), float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if key not in ids or not 0.0 <= score <= 1.0:
            continue
        follows = entry.get("follows")
        if isinstance(follows, str):
            follows = follows.strip().lower() not in ("false", "no", "0")
        if follows is False:
            score = min(score, 0.4)
        try:  # state grounding of the reasoning, a separate reward term (think_reward.grounding_weight)
            grounded = float(entry["grounded"])
            grounded = min(LEVELS, key=lambda x: abs(x - grounded)) if 0.0 <= grounded <= 1.0 else None
        except (KeyError, TypeError, ValueError):
            grounded = None
        result[key] = {"score": min(LEVELS, key=lambda x: abs(x - score)), "action_follows": follows,
                       "grounding": grounded, "analysis": str(entry.get("why", ""))[:300]}
    return result


def judge_episode(judge, records, order, chunk=40, meta=None):
    """Scores of `records` (one episode, time order): a list of result dicts or None. Long episodes are split into
    chunks of `chunk` decisions (each chunk repeats the fixed header)."""
    references = load_references(order)
    results = [None] * len(records)
    for start in range(0, len(records), chunk):
        part = [dict(r, id=start + k + 1) for k, r in enumerate(records[start:start + chunk])]
        ids = {r["id"] for r in part}
        prompt = build_episode_prompt(part, order, references)
        parsed = {}
        for attempt in range(judge.retries + 1):
            try:
                raw = judge._request(prompt, judge.temperature, system=EPISODE_SYSTEM)
            except Exception as error:
                with judge.lock:
                    judge.usage["failures"] += 1
                judge.log("episode", EPISODE_SYSTEM, prompt, None, None, error=str(error)[:500], attempt=attempt,
                          order=order, **(meta or {}))
                time.sleep(5 * (attempt + 1))
                continue
            new = parse_episode_judgement(raw, ids - set(parsed))
            judge.log("episode", EPISODE_SYSTEM, prompt, raw, {str(k): v for k, v in new.items()},
                      usage=getattr(judge._local, "usage", None), attempt=attempt, order=order,
                      decisions=[{"id": r["id"], "agent_index": r["agent_index"], "call_type": r["call_type"],
                                  "action_mode": response_mode(r["response"])} for r in part], **(meta or {}))
            parsed.update(new)
            if len(parsed) == len(ids):
                break
        for r in part:
            results[r["id"] - 1] = parsed.get(r["id"])
    return results
