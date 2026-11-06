"""Probe questions on the two views of an agent, built from successful terra teacher episodes (all splits) with
rule-derived answers, so any checkpoint can be scored without an LLM judge.

The question is appended to the exact prompt the agent sees at that call (its role, rules, observation, past
conversation); answers come from the observation, the spaces, the teacher's own future actions and the teammate's
calls in the same episode. View F (the agent itself, objective state and task) and view S (the teammate):

  F-hold     what you hold                         S-hold     what the teammate holds
  F-content  what a utensil contains               S-doing    what the teammate is executing now
  F-status   state of a utensil (empty / ready to  S-profile  can the teammate operate U (yes / no)
             start / processing / finished)        S-next     the teammate's next kitchen action
  F-space    can you operate U (yes / no)           S-wants    what the teammate's latest request asks of you
  F-next     your next kitchen action (the task)   S-who      which role performs a reference step
  F-feasible can you do action X right now          S-canreq   could the teammate do X right now if asked

View P (multi-step planning against the reference plan; progress = the in-order matched prefix of each role's
executed kitchen actions so far, as in ITES):
  P-due        your next due reference step        P-mate-due   the teammate's next due reference step
  P-before     must step X come before Y (yes/no)  P-remaining  how many steps of your plan remain

usage (from ~/Collab-Overcooked-private): build_probes.py --per-type 30 --out eval/probes/probes.jsonl
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from baselines.rl.credit import is_collab, is_wait, load_references, next_step_ready, next_steps, normalize_action

DATA = Path("runs/teacher_l456/data/think")
ROLES = {"chef": ("Chef", "Assistant"), "assistant": ("Assistant", "Chef")}
LETTERS = "ABCD"
HEADER = "\n\n[Probe question - answer from the observation above; do not take an action]\n"


def observation(row):
    text = row["messages"][-1]["content"]
    return text[text.find("Current Observation:"):]


def spaces(obs):
    return {role: re.search(rf"^{role} space:(.*)$", obs, re.M).group(1).split() for role in ("Chef", "Assistant")}


def kitchen(obs):
    """{utensil: (status, contents)}; status in empty / ready_to_start / processing / finished."""
    line = obs[obs.find("Kitchen states:"):].split("\n")[0][len("Kitchen states: "):]
    out = {}
    for part in line.split(";"):
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


def holds(obs, role):
    m = re.search(rf"<{role}> holds ([^.]*)\.", obs)
    return m.group(1).strip() if m else None


def doing(obs, role):
    m = re.search(rf"current action being executed by {role} is \[(.*?)\](?:\]|\s|$)", obs)
    if not m:
        return None
    action = m.group(1).strip("[]")
    return "nothing" if action in ("", "EMPTY") else normalize_action(action)


def request_to(obs, me, mate):
    """The action of the teammate's latest request to `me` in the past conversation, if any."""
    end = obs.find("Recent Goal:")
    block = obs[obs.find("Past conversation:"):end if end > 0 else None]
    said = "\n".join(l for l in block.split("\n") if l.startswith(f"{mate}:"))
    found = re.findall(rf"request\(\s*{me}\s*,\s*([a-z_]+\([^()]*\))\s*\)", said)
    return normalize_action(found[-1]) if found else None


def physical(action):
    return bool(action) and not is_wait(action) and not is_collab(action)


def plan_depth(history, references):
    """(depth, reference) of the reference with the deepest in-order match of `history`; None when ambiguous."""
    def depth(reference):
        d = 0
        for action in history:
            if d < len(reference) and action == reference[d]:
                d += 1
        return d
    if not references:
        return None
    best = max(depth(r) for r in references)
    deepest = [r for r in references if depth(r) == best]
    return (best, deepest[0]) if all(r == deepest[0] for r in deepest) else None


def make(rng, qtype, question, truth, pool=None, fixed=None):
    """A multiple-choice probe: `fixed` gives the full option list (binary / two-role questions), else the truth
    plus 3 distractors drawn from `pool`."""
    if truth is None:
        return None
    if fixed:
        opts = list(fixed)
    else:
        others = sorted({p for p in pool if p and p != truth})
        if len(others) < 3:
            return None
        opts = rng.sample(others, 3) + [truth]
        rng.shuffle(opts)
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
    return {"qtype": qtype, "view": qtype.split("-")[0], "question": question, "options": opts,
            "answer": opts.index(truth),
            "prompt_suffix": f"{HEADER}{question}\n{lines}\nReply with the letter of the correct option."}


STATUS = {"empty": "it is empty", "ready_to_start": "it holds all its ingredients and has not started",
          "processing": "it is processing and not ready yet", "finished": "its contents are done / ready to take"}


def episode_probes(rng, level, rows_by_role, split):
    """All candidate probes of one teacher episode: {qtype: [probe, ...]}."""
    out = defaultdict(list)
    order = next(iter(rows_by_role.values()))[0]["order"]
    refs = load_references(order)
    plan = {i: refs[i][0] for i in (0, 1) if refs[i]}
    utensil_steps = [s for s in sum(plan.values(), []) if re.search(r"\((\w+\d+)\)", s)]
    for role, rows in rows_by_role.items():
        me, mate = ROLES[role]
        agent = 0 if role == "chef" else 1
        mate_rows = rows_by_role.get("assistant" if role == "chef" else "chef", [])
        own_actions = [normalize_action(r["response"]) for r in rows]
        mate_actions = [normalize_action(r["response"]) for r in mate_rows]
        action_pool = {a for a in own_actions + mate_actions if physical(a)} | set(sum(plan.values(), []))
        episode_items = set()  # everything held or in a utensil during the episode: distractors for item questions
        for r in sum(rows_by_role.values(), []):
            o = observation(r)
            episode_items |= {c for _, c in kitchen(o).values()} | {holds(o, x) for x in ("Chef", "Assistant")}
        for i, row in enumerate(rows):
            obs = observation(row)
            sp, kit = spaces(obs), kitchen(obs)
            items = (episode_items | {"nothing", "one dish"}) - {None}
            base = {"level": level, "split": split, "order": order, "role": role, "episode": row["episode"],
                    "timestep": row["timestep"], "messages": row["messages"]}
            t = row["timestep"]
            found = []
            found.append(make(rng, "F-hold", "What are you holding right now?", holds(obs, me), items))
            found.append(make(rng, "S-hold", f"What is the {mate} holding right now?", holds(obs, mate), items))
            busy = [u for u, (s, _) in kit.items() if s != "empty"]
            if busy:
                u = rng.choice(busy)
                found.append(make(rng, "F-content", f"What is in {u} right now?", kit[u][1], items))
            if kit:
                u = rng.choice(busy) if busy and rng.random() < 0.7 else rng.choice(sorted(kit))
                found.append(make(rng, "F-status", f"Which statement about {u} is true right now?",
                                  STATUS[kit[u][0]], fixed=list(STATUS.values())))
            only_me = [u for u in sp[me] if u not in sp[mate] and u != "counter"]
            only_mate = [u for u in sp[mate] if u not in sp[me] and u != "counter"]
            exclusive = only_me + only_mate  # yes / no, so that both roles get the question (the Chef has 2 own utensils)
            if only_me and only_mate:
                u = rng.choice(exclusive)
                found.append(make(rng, "F-space", f"Can YOU operate {u} yourself?", "yes" if u in only_me else "no",
                                  fixed=["yes", "no"]))
                u = rng.choice(exclusive)
                found.append(make(rng, "S-profile", f"Can the {mate} operate {u}?", "yes" if u in only_mate else "no",
                                  fixed=["yes", "no"]))
            mine_next = next((a for a in own_actions[i:] if physical(a)), None)
            found.append(make(rng, "F-next", "Which kitchen action should you perform next to progress the order?",
                              mine_next, action_pool))
            now = doing(obs, mate)
            mate_next = next((a for r, a in zip(mate_rows, mate_actions)
                              if r["timestep"] > t and physical(a) and a != now), None)
            found.append(make(rng, "S-next", f"What will the {mate} most likely do as its next kitchen action?",
                              mate_next, action_pool))
            found.append(make(rng, "S-doing", f"What is the {mate} executing right now?", now, action_pool | {"nothing"}))
            found.append(make(rng, "S-wants", f"What did the {mate} most recently ask you to do?",
                              request_to(obs, me, mate), action_pool))
            if plan.get(agent):
                step = rng.choice(plan[agent])
                found.append(make(rng, "F-feasible", f"Can you execute {step} right now (all preconditions met)?",
                                  "yes" if next_step_ready(step, obs, agent) else "no", fixed=["yes", "no"]))
            if utensil_steps:
                step = rng.choice(utensil_steps)
                u = re.search(r"\((\w+\d+)\)", step).group(1)
                who = "Chef" if u in sp["Chef"] else "Assistant" if u in sp["Assistant"] else None
                found.append(make(rng, "S-who", f"Who has to perform the step {step}?", who, fixed=["Chef", "Assistant"]))
            if plan.get(1 - agent):
                mate_obs = next((observation(r) for r in reversed(mate_rows) if r["timestep"] <= t), None)
                if mate_obs:
                    step = rng.choice(plan[1 - agent])
                    found.append(make(rng, "S-canreq", f"If you asked the {mate} to do {step} now, could it do it right away?",
                                      "yes" if next_step_ready(step, mate_obs, 1 - agent) else "no", fixed=["yes", "no"]))
            own_history = [a for a in own_actions[:i] if physical(a)]
            mate_history = [a for r, a in zip(mate_rows, mate_actions) if r["timestep"] < t and physical(a)]
            all_steps = set(sum(plan.values(), []))
            due = next_steps(own_history, refs[agent])
            if len(set(due)) == 1:
                found.append(make(rng, "P-due", "Following the recipe plan, which step is the next one YOU still have to "
                                  "complete?", due[0], all_steps))
            mate_due = next_steps(mate_history, refs[1 - agent])
            if len(set(mate_due)) == 1:
                found.append(make(rng, "P-mate-due", f"Following the recipe plan, which step is the next one the {mate} "
                                  "still has to complete?", mate_due[0], all_steps))
            matched = plan_depth(own_history, refs[agent])
            if matched:
                d, ref = matched
                remaining = len(ref) - d
                unique = [x for x in ref[d:] if ref[d:].count(x) == 1]  # yes / no on the order of two remaining steps
                if len(unique) >= 2:
                    x, y = rng.sample(unique, 2)
                    found.append(make(rng, "P-before", f"In your recipe plan, do you have to do {x} before {y}?",
                                      "yes" if ref.index(x, d) < ref.index(y, d) else "no", fixed=["yes", "no"]))
                lo = max(0, remaining - rng.randrange(4))
                found.append(make(rng, "P-remaining", "Counting from now, how many steps of your own recipe plan are "
                                  "still left to do (including the next one)?", str(remaining),
                                  fixed=[str(n) for n in range(lo, lo + 4)]))
            for p in found:
                if p:
                    out[p["qtype"]].append(dict(base, **p))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--per-type", type=int, default=30, help="probes per (level, role, qtype); both roles get the same number")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    probes = []
    for level in args.levels:
        pool = defaultdict(list)  # (role, qtype) -> probes
        for split in ("train", "dev", "test"):
            episodes = defaultdict(lambda: defaultdict(list))
            for role in ROLES:
                path = DATA / f"l{level}_{split}_{role}.jsonl"
                if path.exists():
                    for line in open(path):
                        row = json.loads(line)
                        episodes[(row["order"], row["episode"])][role].append(row)
            for key in sorted(episodes):
                for rows in episodes[key].values():
                    rows.sort(key=lambda r: (r["timestep"], r.get("call_type", "")))
                for qtype, items in episode_probes(rng, level, episodes[key], split).items():
                    for p in items:
                        pool[(p["role"], qtype)].append(p)
        # the same number of probes for both roles in every qtype (and as many yes as no for yes / no questions)
        for qtype in sorted({q for _, q in pool}):
            by_role = {role: pool.get((role, qtype), []) for role in ROLES}
            if all(p["options"] == ["yes", "no"] for items in by_role.values() for p in items):
                split = {role: ([p for p in items if p["answer"] == 0], [p for p in items if p["answer"] == 1])
                         for role, items in by_role.items()}
                n = min([args.per_type // 2] + [len(part) for parts in split.values() for part in parts])
                for yes, no in split.values():
                    probes += rng.sample(yes, n) + rng.sample(no, n)
            else:
                n = min([args.per_type] + [len(items) for items in by_role.values()])
                for items in by_role.values():
                    probes += rng.sample(items, n)
    out = Path("runs/think_judge") / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for k, p in enumerate(probes):
            f.write(json.dumps(dict(p, id=k), ensure_ascii=False) + "\n")
    counts = defaultdict(int)
    for p in probes:
        counts[(p["level"], p["qtype"])] += 1
    print(len(probes), "probes")
    for level in args.levels:
        print(level, {q: counts[(level, q)] for q in sorted({p["qtype"] for p in probes})})


if __name__ == "__main__":
    main()
