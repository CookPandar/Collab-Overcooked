"""Rule-verified rewards of the reasoning (Think / Recent Goal) of a think_action reply, on the original task.

An LLM only translates the free-text reasoning into canonical claims (each with a verbatim quote, dropped when the
quote is not in the reply); whether a claim is true is decided by rules on the observation, the reference plans and
the actual trajectory of the episode. Three rewards per call, used alone or summed (think_reward.rules):

  ground  first view: coverage of the key facts of the agent's next due reference step (what it holds, the state
          of the utensil of that step, whether the step can be done now), stated correctly, minus 2 x the share of
          wrong claims about itself / the kitchen; waiting while every due step is ready counts as a wrong claim.
          Saying nothing covers nothing, so silence is not a safe answer.
  mate    teammate: claims about what the teammate holds / is doing / can reach / can do now
          (+1 right, -1 wrong), the first step the plan gives the teammate (+1 if the teammate executes it within
          k timesteps, else -1) and every request(teammate, X) of the Action (X reachable by the teammate, X among the
          teammate's next two reference steps, X executed by the teammate within k timesteps), averaged.
  mate_v2 mate without the dilution exploit (agents learned to pad the reasoning with the teammate's state, copied
          from the observation, to average away wrong predictions): the claims about the teammate's current state
          count as ONE item (their mean, -1..1) of weight 0.25; the plan's first teammate step (+-1) and every request
          (2 s / 3 - 1) have weight 1; the weighted items are summed and clipped to [-1, 1].
  plan    multi-step plan (Recent Goal): per role, the longest common subsequence of the planned actions with the
          role's next H remaining reference steps over sum(min(remaining, H)), minus the share of planned steps
          given to the wrong role or not remaining for anyone (done / impossible).
"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .credit import is_collab, is_wait, load_references, next_step_ready, normalize_action, response_mode
from .think_judge import reasoning_fields

ROLE_NAMES = ("Chef", "Assistant")
STATUSES = ("empty", "ready_to_start", "processing", "finished")
WITHIN = 5    # timesteps in which a requested / predicted teammate action must happen
HORIZON = 3   # remaining reference steps per role a plan is compared with
_UTENSIL = re.compile(r"\((\w+\d+)\)")
_PICKUP = re.compile(r"pickup\((\w+),(\w+)\)")
_REQUEST = re.compile(r"request\(\s*(Chef|Assistant)\s*,\s*([a-z_]+\([^()]*\))\s*\)")
_KNOWN = re.compile(r"(pickup|put_obj_in_utensil|place_obj_on_counter|fill_dish_with_food|deliver_soup|"
                    r"[a-z_]+\(\w+\d+\))")

EXTRACT_SYSTEM = ("You convert the free-text reasoning of agents in a two-player cooking game into canonical claims. "
                  "You translate literally: never correct, complete or judge what the agent wrote. Reply with JSON only.")

EXTRACT_TEMPLATE = """Players: Chef, Assistant. {spaces}
Action syntax: pickup(<item>,<place>) where place is a utensil, counter or a dispenser; put_obj_in_utensil(<utensil>);
place_obj_on_counter(); <verb>(<utensil>) to start a utensil, e.g. cut(chopping_board0), cook(pot0), bake(oven0),
stir(blender0); fill_dish_with_food(<utensil>); deliver_soup(). Items are snake_case, e.g. pumpkin_slices, potato.

For every reply below, list the facts its Think and Recent Goal state EXPLICITLY about the current moment, as claims
of these types ("I" / "my" means the replying player). Do not infer: a plan, intention or next step ("the next step is
to put X in pot0", "I will cut") is NOT a claim (plans go to "plan"); skip vague or hypothetical statements:
  {{"type": "hold", "player": P, "item": <item or "nothing">}}
  {{"type": "utensil", "utensil": U, "status": <empty|ready_to_start|processing|finished or null>, "contents": <item or null>}}
  {{"type": "can_operate", "player": P, "utensil": U, "value": true|false}}
  {{"type": "ready", "player": P, "action": A, "value": true|false}}   only when the text says P can / cannot do A
                                                                     right now, or must wait before A (value false)
  {{"type": "doing", "player": P, "action": <A or "nothing">}}          (what P is executing / waiting at this moment)
Every claim has "quote": the exact words of the reply it comes from (copied verbatim, at most 8 words).
"plan": the Recent Goal as an ordered list of {{"player": P, "action": A, "quote": ...}} (a request(P, A) in the
Recent Goal is a step of P; skip wait and messages); [] when the Recent Goal is empty or has no concrete step.

{calls}

Output: {{"calls": [{{"id": <id>, "claims": [...], "plan": [...]}}, ...]}}"""


def _spaces(observation):
    return {r: (re.search(rf"^{r} space:(.*)$", observation, re.M) or re.search("()", "")).group(1).split()
            for r in ROLE_NAMES}


def _kitchen(observation):
    """{utensil: (status, contents)} of the current kitchen state."""
    text = observation[observation.rfind("Kitchen states:"):].split("\n")[0][len("Kitchen states: "):]
    out = {}
    for part in text.split(";"):
        m = re.match(r"<(\w+)> (.*)", part.strip())
        if not m:
            continue
        name, rest = m.groups()
        if rest.startswith("is empty"):
            out[name] = ("empty", "nothing")
        elif rest.startswith("starts processing"):
            out[name] = ("processing", re.match(r"starts processing ([^,]+)", rest).group(1).strip())
        elif rest.startswith("has"):
            contents = re.split(r",\s*which", rest[4:])[0].strip()
            out[name] = ("ready_to_start" if "all the ingredients you need" in rest else "finished", contents)
    return out


def _holds(observation, role):
    m = re.search(rf"<{role}> holds ([^.]*)\.", observation[observation.rfind("Agent State:"):])
    return m.group(1).strip() if m else None


def _doing(observation, role):
    m = re.search(rf"current action being executed by {role} is \[(.*?)\](?:\]|\s|$)", observation)
    if not m:
        return None
    action = m.group(1).strip("[]")
    return "nothing" if action in ("", "EMPTY") or is_wait(normalize_action(action)) else normalize_action(action)


def _item_names(text):
    """Names an item description can be referred to by: 'a dish with x and needs ...' -> {dish, x, dish_with_x}."""
    text = re.sub(r"\s+and needs.*$", "", (text or "").strip().lower())
    text = re.sub(r"^(a|an|the|one|some)\s+", "", text).replace(" ", "_")
    if text in ("nothing", "none", "empty", "empty_hands"):
        return {"nothing"}
    names = {text}
    m = re.match(r"dish_with_(\w+)", text)
    if m:
        names |= {"dish", m.group(1)}
    return names | {n[:-1] for n in names if n.endswith("s")} | {n + "s" for n in names}


def _same_item(claimed, truth):
    return bool(_item_names(claimed) & _item_names(truth))


def _reachable(action, space):
    """Whether every utensil / dispenser the action uses is in `space`."""
    places = _UTENSIL.findall(action)
    m = _PICKUP.fullmatch(action)
    if m:
        places = [m.group(2)]
    return all(p == "counter" or p in space for p in places)


def _remaining(history, references, horizon=None):
    """Remaining steps of the references with the deepest in-order match of `history` (the first on ties)."""
    def depth(ref):
        d = 0
        for a in history:
            if d < len(ref) and a == ref[d]:
                d += 1
        return d
    if not references:
        return []
    best = max(depth(r) for r in references)
    ref = next(r for r in references if depth(r) == best)
    return ref[best:best + horizon] if horizon else ref[best:]


def _lcs(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            table[i + 1][j + 1] = table[i][j] + 1 if x == y else max(table[i][j + 1], table[i + 1][j])
    return table[-1][-1]


def executed_actions(records):
    """[(index, agent_index, timestep, action)] of the kitchen actions that were executed, in time order. Uses
    `validated_action` (training records); saved rollouts without it fall back to a well-formed kitchen action
    whose last validation passed."""
    out = []
    for i, r in enumerate(records):
        if "validated_action" in r:
            action = r["validated_action"]
        else:
            ok = not r.get("validations") or r["validations"][-1][0]
            action = normalize_action(r["response"]) if response_mode(r["response"]) == "planner_main" and ok else None
        if action and not is_wait(action) and not is_collab(action):
            out.append((i, r["agent_index"], r["timestep"], action))
    return out


# ---- extraction ---------------------------------------------------------------------------------------------------

def _reply_text(response):
    think, goal = reasoning_fields(response)
    return f"Think: {think}\nRecent Goal: {goal}"


def build_extract_prompt(records, ids):
    calls = "\n\n".join(f"### id {k} | {ROLE_NAMES[r['agent_index']]}\n{_reply_text(r['response'])}"
                        for k, r in zip(ids, records))
    spaces = " ".join(f"{role} can reach: {' '.join(s)}." for role, s in _spaces(records[0]["observation"]).items())
    return EXTRACT_TEMPLATE.format(spaces=spaces, calls=calls)


def _quoted(entry, text):
    quote = " ".join(str(entry.get("quote", "")).lower().split()).strip(" .\"'")
    return bool(quote) and quote in " ".join(text.lower().split())


def parse_extraction(raw, records, ids):
    """{id: {"claims": [...], "plan": [...]}} keeping only entries whose quote is in that reply."""
    text = re.sub(r"^```(?:json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    try:
        calls = json.loads(text[text.find("{"):text.rfind("}") + 1])["calls"]
    except (ValueError, KeyError, TypeError):
        return {}
    by_id = dict(zip(ids, records))
    out = {}
    for c in calls if isinstance(calls, list) else []:
        try:
            k = int(c["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if k not in by_id:
            continue
        reply = _reply_text(by_id[k]["response"])
        keep = lambda xs: [x for x in xs if isinstance(x, dict) and _quoted(x, reply)] if isinstance(xs, list) else []
        out[k] = {"claims": keep(c.get("claims")), "plan": keep(c.get("plan")),
                  "dropped": len(c.get("claims") or []) + len(c.get("plan") or [])}
        out[k]["dropped"] -= len(out[k]["claims"]) + len(out[k]["plan"])
    return out


_SLOTS = {}  # judge -> semaphore: at most judge.workers extraction requests in flight over all episodes


def extract(judge, records, chunk=8):
    """Extraction of every record (one episode): a list of {"claims", "plan"} or None when it failed."""
    import threading
    slots = _SLOTS.setdefault(id(judge), threading.BoundedSemaphore(judge.workers))
    results = [None] * len(records)
    parts = [list(range(s, min(s + chunk, len(records)))) for s in range(0, len(records), chunk)]

    def one(part):
        ids = [k + 1 for k in part]
        prompt = build_extract_prompt([records[k] for k in part], ids)
        parsed = {}
        for attempt in range(judge.retries + 1):
            try:
                with slots:
                    raw = judge._request(prompt, judge.temperature, system=EXTRACT_SYSTEM)
            except Exception as error:
                with judge.lock:
                    judge.usage["failures"] += 1
                judge.log("extract", EXTRACT_SYSTEM, prompt, None, None, error=str(error)[:500], attempt=attempt)
                time.sleep(5 * (attempt + 1))
                continue
            new = parse_extraction(raw, [records[k] for k in part], ids)
            judge.log("extract", EXTRACT_SYSTEM, prompt, raw, {str(k): v for k, v in new.items()},
                      usage=getattr(judge._local, "usage", None), attempt=attempt)
            parsed.update({k: v for k, v in new.items() if k not in parsed})
            if len(parsed) == len(ids):
                break
        for k in part:
            results[k] = parsed.get(k + 1)

    with ThreadPoolExecutor(max(1, len(parts))) as pool:
        list(pool.map(one, parts))
    return results


# ---- verification -------------------------------------------------------------------------------------------------

def _canon(action):
    return normalize_action(str(action or ""))


def check_claim(claim, observation, me):
    """(view, correct) of one claim: view 'F' (the agent / kitchen) or 'S' (the teammate); None if unverifiable."""
    player = claim.get("player")
    agent = ROLE_NAMES.index(player) if player in ROLE_NAMES else None
    kind = claim.get("type")
    view = "F" if agent in (None, me) else "S"
    if kind == "hold" and agent is not None:
        truth = _holds(observation, player)
        return (view, _same_item(claim.get("item"), truth)) if truth is not None else None
    if kind == "utensil":
        state = _kitchen(observation).get(claim.get("utensil"))
        if state is None:
            return None
        checks = []
        if claim.get("status") in STATUSES:
            checks.append(claim["status"] == state[0])
        if claim.get("contents"):
            checks.append(_same_item(claim["contents"], state[1]))
        return ("F", all(checks)) if checks else None
    if kind == "can_operate" and agent is not None and isinstance(claim.get("value"), bool):
        space = _spaces(observation)[player]
        utensil = claim.get("utensil")
        if not utensil or utensil == "counter":
            return None
        return view, (utensil in space) == claim["value"]
    if kind == "ready" and agent is not None and isinstance(claim.get("value"), bool):
        action = _canon(claim.get("action"))
        if not _KNOWN.match(action) or is_wait(action) or is_collab(action):
            return None
        m = _PICKUP.fullmatch(action)
        if agent != me and m and m.group(2) == "counter":  # the teammate's counters are not in this observation
            return None
        ready = _reachable(action, _spaces(observation)[player]) and next_step_ready(action, observation, agent)
        return view, ready == claim["value"]
    if kind == "doing" and agent is not None:
        truth = _doing(observation, player)
        if truth is None:
            return None
        claimed = _canon(claim.get("action"))
        if claimed.lower() in ("nothing", "none", "") or is_wait(claimed):
            claimed = "nothing"
        return view, claimed == truth
    return None


def rewards(records, extractions, order):
    """Per record: {"ground", "mate", "plan"} (None where not defined) and the checked items, for one episode."""
    refs = load_references(order)
    executed = executed_actions(records)
    last_t = max((r["timestep"] for r in records), default=0)
    out = []
    for i, (r, ex) in enumerate(zip(records, extractions)):
        if ex is None:
            out.append(None)
            continue
        me, mate = r["agent_index"], 1 - r["agent_index"]
        obs, t = r["observation"], r["timestep"]
        history = {a: [x for j, ag, _, x in executed if j < i and ag == a] for a in (0, 1)}
        future = [x for j, ag, ts, x in executed if j > i and ag == mate and ts <= t + WITHIN]
        settled = t + WITHIN <= last_t  # the window is complete (else misses are not counted)
        items = {"F": [], "S": []}
        for claim in ex["claims"]:
            checked = check_claim(claim, obs, me)
            if checked:
                items[checked[0]].append((claim, checked[1]))
        steps = [(ROLE_NAMES.index(s["player"]), _canon(s.get("action"))) for s in ex["plan"]
                 if s.get("player") in ROLE_NAMES and not is_wait(_canon(s.get("action")))]
        predicted = next((x for a, x in steps if a == mate), None)  # what the plan expects of the teammate next
        if predicted and (predicted in future or settled):
            items["S"].append(({"type": "plan_mate", "player": ROLE_NAMES[mate], "action": predicted},
                               predicted in future))
        # ground: coverage of the key facts of the due steps
        due = sorted(set(_remaining(history[me], refs[me], 1)))
        keys = {("hold",)} | {("utensil", u) for s in due for u in _UTENSIL.findall(s)} | {("ready", s) for s in due}
        covered = set()
        for claim, ok in items["F"]:
            if not ok:
                continue
            if claim["type"] == "hold" and claim.get("player") == ROLE_NAMES[me]:
                covered.add(("hold",))
            elif claim["type"] == "utensil":
                covered.add(("utensil", claim.get("utensil")))
            elif claim["type"] == "ready" and claim.get("player") == ROLE_NAMES[me]:
                covered.add(("ready", _canon(claim.get("action"))))
        wrong = sum(not ok for _, ok in items["F"])
        n = len(items["F"])
        waited = response_mode(r["response"]) == "wait"
        if waited and due and all(_reachable(s, _spaces(obs)[ROLE_NAMES[me]]) and next_step_ready(s, obs, me) for s in due):
            wrong, n = wrong + 1, n + 1
        ground = len(covered & keys) / len(keys) - 2 * wrong / max(1, n)
        # mate: claims about the teammate and the requests of the Action
        scores = [1.0 if ok else -1.0 for _, ok in items["S"]]
        requests = []
        mate_next = _remaining(history[mate], refs[mate], 2)
        for who, action in _REQUEST.findall(normalize_action(r["response"]).replace(" ", "")):
            if who != ROLE_NAMES[mate]:
                continue
            action = _canon(action)
            done = action in future
            if not done and not settled:
                continue
            parts = (_reachable(action, _spaces(obs)[ROLE_NAMES[mate]]), action in mate_next, done)
            requests.append((action, parts))
            scores.append(2 * sum(parts) / 3 - 1)
        mate_reward = sum(scores) / len(scores) if scores else 0.0
        state = [1.0 if ok else -1.0 for c, ok in items["S"] if c["type"] != "plan_mate"]
        forward = [1.0 if ok else -1.0 for c, ok in items["S"] if c["type"] == "plan_mate"]
        forward += [2 * sum(p) / 3 - 1 for _, p in requests]
        mate_v2 = (0.25 * sum(state) / len(state) if state else 0.0) + sum(forward)
        mate_v2 = max(-1.0, min(1.0, mate_v2))
        # plan: the Recent Goal against the remaining reference steps of both roles
        remaining = {a: _remaining(history[a], refs[a]) for a in (0, 1)}
        if any(remaining.values()):
            cover = sum(_lcs([x for a, x in steps if a == p], remaining[p][:HORIZON]) for p in (0, 1))
            cover /= sum(min(len(remaining[p]), HORIZON) for p in (0, 1))
            bad = sum(x not in remaining[a] for a, x in steps)
            plan_reward = cover - bad / max(1, len(steps))
        else:
            plan_reward = None
        out.append({"ground": ground, "mate": mate_reward, "mate_v2": mate_v2, "plan": plan_reward,
                    "detail": {"due": due, "keys": sorted(map(list, keys)), "covered": sorted(map(list, covered & keys)),
                               "F": [(c, ok) for c, ok in items["F"]], "S": [(c, ok) for c, ok in items["S"]],
                               "requests": requests, "waited": waited, "plan_steps": steps,
                               "remaining": {ROLE_NAMES[a]: remaining[a][:HORIZON] for a in (0, 1)}}})
    return out


def rule_signal(result, weights, combine="sum"):
    """Weighted sum of the rewards in `weights` ({"ground": w, ...}); None when the extraction failed.
    combine "mean": the weighted mean instead (sum / total weight of the enabled rules; full_v2 -> about [-1, 1])."""
    if result is None:
        return None
    total = sum(w * (result[k] or 0.0) for k, w in weights.items() if w)
    if combine == "mean":
        norm = sum(abs(w) for w in weights.values() if w)
        return total / norm if norm else 0.0
    if combine != "sum":
        raise ValueError(f"unknown rules_combine {combine}")
    return total
