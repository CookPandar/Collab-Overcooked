"""Where along the reference plan each role stops in failed eval episodes.

The executed actions of a role are the calls with ITES != 0 (credit.py credits exactly the validated, non-wait,
non-Collab actions). For each failed episode and role: the matched reference depth, and the reference step that
was due next (credit.next_steps), generalized by masking ingredient names so steps of different orders pool.
usage (from ~/Collab-Overcooked-private): stall_points.py NAME=EPISODES_JSONL ...
"""
import json
import re
import sys
from collections import Counter

from baselines.rl.credit import load_references, next_steps, normalize_action

PLACES = re.compile(r"^(ingredient_dispenser|dish_dispenser|counter|serving_table|\w+\d+)$")
ROLES = {0: "Chef", 1: "Asst"}


def generalize(step):
    match = re.fullmatch(r"(\w+)\((.*)\)", step)
    if not match:
        return step
    args = [a if PLACES.match(a) else "X" for a in match.group(2).split(",") if a]
    return f"{match.group(1)}({','.join(args)})"


def stalls(episode):
    refs = load_references(episode["order"])
    out = {}
    for agent in (0, 1):
        history = [normalize_action(c["response"]) for c in episode["calls"]
                   if c["agent_index"] == agent and abs(c["ites"]) > 1e-9]
        due = next_steps(history, refs[agent])
        depth = max(sum(1 for _ in _matched(history, r)) for r in refs[agent])
        out[agent] = (depth, min(len(r) for r in refs[agent]), generalize(due[0]) if due else "DONE")
    return out


def _matched(history, reference):
    d = 0
    for action in history:
        if d < len(reference) and action == reference[d]:
            d += 1
            yield action


def main():
    for pair in sys.argv[1:]:
        name, path = pair.split("=", 1)
        episodes = [json.loads(line) for line in open(path)]
        failed = [e for e in episodes if not e["success"]]
        print(f"== {name}: {len(failed)}/{len(episodes)} failed")
        if not failed:
            continue
        results = [stalls(e) for e in failed]
        for agent in (0, 1):
            steps = Counter(r[agent][2] for r in results)
            depths = Counter(f"{r[agent][0]}/{r[agent][1]}" for r in results)
            top = ", ".join(f"{s} {n}" for s, n in steps.most_common(4))
            print(f"  {ROLES[agent]} due next: {top}")
            print(f"  {ROLES[agent]} depth/ref: {', '.join(f'{d}:{n}' for d, n in sorted(depths.items(), key=lambda x: -x[1])[:6])}")


if __name__ == "__main__":
    main()
