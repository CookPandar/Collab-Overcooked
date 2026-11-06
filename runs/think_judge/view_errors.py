"""Failure attribution of Think-protocol episodes along two views, with an LLM judge and a fixed, task-specific
error taxonomy (see TAXONOMY):

- view 1 (the agent itself): is the Action faithful to its own Think, and is its belief about the objective state
  and about the task right?
- view 2 (the teammate): does it model the teammate correctly: fixed profile (space, skills), current state /
  knowledge, and next behaviour?

Pass 1 labels every call (both roles interleaved in time order, chunks of --chunk calls) with zero or more error
codes. Pass 2 names one primary cause per failed episode from the stall point (reference depth, next due step,
last progress of each role) and the labelled errors. Inputs are RL rollouts (rl/<run>/rollouts/uNNNNN.jsonl.gz),
which keep the observation of each call.
usage (from ~/Collab-Overcooked-private):
  view_errors.py --run l5_none --updates 16-20 46-50 --episodes 12 --out eval/view_errors_l5.jsonl
"""
import argparse
import gzip
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from baselines.rl.credit import load_references, next_steps, normalize_action
from baselines.rl.think_judge import RULES, ThinkJudge, _recipe, _spaces, compress_observation, division_of_labour

TJ = Path("runs/think_judge")
TAXONOMY = """View 1 - the agent itself
- F1 THINK_ACTION: the Action does not carry out the decision of its own Think / Recent Goal (other action, object
  or utensil; or it waits / chats although its Think decided to act).
- F2 STATE: the Think states or relies on a false fact about the current objective state shown in the observation:
  what the agent holds, what a utensil / counter contains, whether an item is raw / cut / cooked / ready, which
  utensils and dispensers are in its own space.
- F3 TASK: the state is read correctly but the task is misunderstood: recipe ingredients or order of processing,
  what an ingredient still needs, game rules (cut before cooking, a dish is needed, ...), or the next plan step.
View 2 - the teammate
- S1 PROFILE: wrong fixed model of the teammate: asks or expects it to use a utensil outside its space or a skill it
  does not have, or gives it work only the agent itself can do.
- S2 BELIEF: wrong belief about the teammate's current state or knowledge: assumes it holds / placed / finished
  something it has not, assumes it knows something never communicated, or ignores / misreads its latest message.
- S3 PREDICTION: wrong prediction of the teammate's next behaviour: waits for an action the teammate will not take
  (e.g. both wait for each other), keeps repeating a request the teammate is not acting on instead of acting or
  re-planning, or acts so that it duplicates or blocks the teammate's work.
- OTHER: a format error or rejected action not explained by any code above."""
CODES = ["F1", "F2", "F3", "S1", "S2", "S3", "OTHER"]
ROOT_CODES = CODES[:-1] + ["PACE", "OTHER"]

SYSTEM = ("You are a strict analyst of failures of two LLM agents (Chef and Assistant) cooperating in a text "
          "version of Overcooked. You label decisions with a fixed error taxonomy and cite evidence. Reply with JSON only.")
CALL_TEMPLATE = """## Game rules (summary)
{rules}

## This episode
Spaces: {spaces}
Order: {order}. Recipe: {recipe}
Reference plan (one correct solution). Chef: {ref_chef}. Assistant: {ref_assistant}. Division of labour: {labour}

## Error taxonomy
{taxonomy}

## Decisions (both agents, in time order)
Each block: what the agent saw, its reply (Think / Recent Goal / Action) and what happened. The observation is the
objective state at that moment; use the other agent's blocks to know what the teammate really held, did and said.

{steps}

## Task
For EVERY decision id above, list the error codes that apply (empty list if the decision is fine; at most 2 codes,
most important first) and one short piece of evidence. Judge only what is visible here. Reply exactly:
{{"steps": [{{"id": 1, "codes": ["F2"], "why": "..."}}, ...]}}"""
ROOT_TEMPLATE = """## Error taxonomy
{taxonomy}
- PACE: no decisive error; the agents follow a correct plan but are too slow to finish in time.

## Failed episode
Order: {order}. Horizon {steps} steps; the episode ended without delivering the order.
Reference plan. Chef: {ref_chef}. Assistant: {ref_assistant}.
Where each role stopped (reference steps reached / total, next due step, timestep of its last progress):
{stall}

Labelled errors (timestep, role, codes, evidence), in time order:
{errors}

## Task
Name the ONE primary cause of the failure: the error type that, if fixed, would most likely have let the team finish
(usually the error that stopped progress of the role that blocks the other). Optionally one secondary cause.
Reply exactly: {{"primary": "S3", "secondary": "F2", "why": "..."}}"""


def outcome(call):
    parts = [call.get("action_mode") or "?"]
    for ok, message in call.get("validations") or []:
        parts.append("validator ok" if ok else f"validator REJECTED: {' '.join(str(message).split())[:140]}")
    ites = call.get("ites") or 0.0
    if abs(ites) > 1e-9:
        parts.append("advanced the reference plan" if ites > 0 else "executed, off the reference plan")
    return "; ".join(parts)


def load_episodes(run, updates):
    episodes = {}
    for u in updates:
        path = TJ / "rl" / run / "rollouts" / f"u{u:05d}.jsonl.gz"
        summaries, calls = {}, {}
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            if "episode_summary" in r:
                summaries[r["episode_summary"]] = r
            else:
                calls.setdefault(r["episode"], []).append(r)
        for e, rows in calls.items():
            if e in summaries and not summaries[e]["success"]:
                rows.sort(key=lambda c: (c["timestep"], c["call_index"], c["agent_index"]))
                episodes[(u, e)] = {"update": u, "episode": e, "order": rows[0]["order"], "calls": rows,
                                    "steps": summaries[e]["steps"]}
    return episodes


def header(order):
    refs = load_references(order)
    first = {i: refs[i][0] if refs[i] else [] for i in (0, 1)}
    return refs, first


def label_calls(judge, ep, chunk):
    refs, first = header(ep["order"])
    calls = ep["calls"]
    prompts = []
    for start in range(0, len(calls), chunk):
        steps = []
        for k, c in enumerate(calls[start:start + chunk]):
            role = "Chef" if c["agent_index"] == 0 else "Assistant"
            steps.append(f"### id {k + 1} | t={c['timestep']} | {role}\n[observation]\n"
                         f"{compress_observation(c['observation'])}\n[reply]\n{(c['response'] or '').strip()}\n"
                         f"[outcome] {outcome(c)}")
        prompts.append((start, CALL_TEMPLATE.format(
            rules=RULES, spaces=_spaces(calls[0]["observation"]), order=ep["order"],
            recipe=" ".join(_recipe(ep["order"]).split()), ref_chef=" -> ".join(first[0]),
            ref_assistant=" -> ".join(first[1]), labour=division_of_labour(first), taxonomy=TAXONOMY,
            steps="\n\n".join(steps))))
    def request(prompt):
        for attempt in range(3):
            try:
                raw = judge._request(prompt, None, system=SYSTEM)
                text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
                return json.loads(text[text.find("{"):text.rfind("}") + 1])["steps"]
            except Exception as error:  # API hiccup or bad JSON: retry, then leave the chunk unlabelled
                print(f"chunk attempt {attempt}: {type(error).__name__}: {str(error)[:200]}", flush=True)
        return []

    labels = [None] * len(calls)
    with ThreadPoolExecutor(4) as pool:  # the chunks of an episode are independent requests
        answers = list(pool.map(request, [prompt for _, prompt in prompts]))
    for (start, _), entries in zip(prompts, answers):
        n = min(chunk, len(calls) - start)
        for entry in entries:
            try:
                k = int(entry["id"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= k < n:
                codes = [c for c in entry.get("codes") or [] if c in CODES][:2]
                labels[start + k] = {"codes": codes, "why": str(entry.get("why", ""))[:240]}
    return labels


def stall_text(ep):
    refs, first = header(ep["order"])
    lines = []
    for agent, role in ((0, "Chef"), (1, "Assistant")):
        history, last = [], None
        for c in ep["calls"]:
            if c["agent_index"] == agent and abs(c.get("ites") or 0) > 1e-9:
                history.append(normalize_action(c["response"]))
                if c["ites"] > 0:
                    last = c["timestep"]
        depth = max(sum(1 for _ in _matched(history, r)) for r in refs[agent])
        due = next_steps(history, refs[agent])
        lines.append(f"- {role}: {depth}/{min(len(r) for r in refs[agent])}, next due {due[0] if due else 'DONE'}, "
                     f"last progress at t={last}")
    return "\n".join(lines)


def _matched(history, reference):
    d = 0
    for action in history:
        if d < len(reference) and action == reference[d]:
            d += 1
            yield action


def root_cause(judge, ep, labels):
    _, first = header(ep["order"])
    errors = []
    for c, lab in zip(ep["calls"], labels):
        if lab and lab["codes"]:
            role = "Chef" if c["agent_index"] == 0 else "Asst"
            errors.append(f"t={c['timestep']} {role} {','.join(lab['codes'])}: {lab['why']}")
    if len(errors) > 220:  # keep the prompt bounded: start and end of the episode
        errors = errors[:60] + ["..."] + errors[-160:]
    prompt = ROOT_TEMPLATE.format(taxonomy=TAXONOMY, order=ep["order"], steps=ep["steps"],
                                  ref_chef=" -> ".join(first[0]), ref_assistant=" -> ".join(first[1]),
                                  stall=stall_text(ep), errors="\n".join(errors) or "(none)")
    for attempt in range(3):
        try:
            raw = judge._request(prompt, None, system=SYSTEM)
            out = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
            if out.get("primary") in ROOT_CODES:
                return {"primary": out["primary"],
                        "secondary": out.get("secondary") if out.get("secondary") in ROOT_CODES else None,
                        "why": str(out.get("why", ""))[:400], "stall": stall_text(ep)}
        except Exception:
            pass
    return {"primary": None, "secondary": None, "why": "judge failed", "stall": stall_text(ep)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--updates", nargs="+", required=True, help="ranges like 16-20")
    parser.add_argument("--episodes", type=int, default=12, help="failed episodes per update range")
    parser.add_argument("--chunk", type=int, default=40)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--env-file", default="~/ParaCook/.env.shuaiapi.local")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    judge = ThinkJudge(model=args.model, env_file=args.env_file)
    out = TJ / args.out
    done = {(r["group"], r["update"], r["episode"]) for r in map(json.loads, open(out))} if out.exists() else set()
    jobs = []
    for spec in args.updates:
        lo, hi = map(int, spec.split("-"))
        episodes = load_episodes(args.run, range(lo, hi + 1))
        rng = random.Random(f"{args.run}/{spec}")
        keys = sorted(episodes)
        rng.shuffle(keys)
        by_order = {}
        for k in keys:  # round-robin over orders so each group covers all dishes
            by_order.setdefault(episodes[k]["order"], []).append(k)
        picked = []
        while len(picked) < args.episodes and any(by_order.values()):
            for order in sorted(by_order):
                if by_order[order] and len(picked) < args.episodes:
                    picked.append(by_order[order].pop(0))
        jobs += [(spec, episodes[k]) for k in picked if (spec, k[0], k[1]) not in done]

    def run(job):
        spec, ep = job
        labels = label_calls(judge, ep, args.chunk)
        root = root_cause(judge, ep, labels)
        return {"group": spec, "update": ep["update"], "episode": ep["episode"], "order": ep["order"],
                "steps": ep["steps"], "root": root,
                "calls": [{"agent_index": c["agent_index"], "timestep": c["timestep"],
                           "action_mode": c.get("action_mode"), "labels": lab}
                          for c, lab in zip(ep["calls"], labels)]}

    with ThreadPoolExecutor(args.workers) as pool, open(out, "a") as f:
        for result in pool.map(run, jobs):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{result['group']} u{result['update']} ep{result['episode']}: {result['root']['primary']} "
                  f"usage {judge.usage}", flush=True)


if __name__ == "__main__":
    main()
