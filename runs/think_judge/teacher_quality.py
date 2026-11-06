"""Quality of the terra teacher demonstrations per level and role (train split of runs/teacher_l456/data):
- plan efficiency: matched reference depth / physical actions of each teacher episode (1.0 = no detours),
  plus the share of episodes whose physical actions exceed the reference length by > 50%;
- label consistency: among rows with an identical observation, the share whose action differs from the
  majority action for that observation (a lower bound on label noise; history is disabled, so the observation
  is all the model sees).
usage (from ~/Collab-Overcooked-private): teacher_quality.py [PROTOCOL]"""
import json
import sys
from collections import Counter, defaultdict
from statistics import mean, median

from baselines.rl.credit import is_collab, is_wait, load_references, normalize_action

PROTOCOL = sys.argv[1] if len(sys.argv) > 1 else "action_only"


def depth(history, reference):
    d = 0
    for action in history:
        if d < len(reference) and action == reference[d]:
            d += 1
    return d


for level in (2, 3, 4, 5, 6):
    for role, agent in (("chef", 0), ("assistant", 1)):
        rows = [json.loads(l) for l in open(f"runs/teacher_l456/data/{PROTOCOL}/l{level}_train_{role}.jsonl")]
        episodes = defaultdict(list)
        for r in rows:
            episodes[(r["order"], r["episode"])].append(r)
        eff, ratio = [], []
        for (order, _), eps in episodes.items():
            refs = load_references(order)[agent]
            phys = [a for a in (normalize_action(r["response"]) for r in sorted(eps, key=lambda r: r["timestep"]))
                    if a and not is_wait(a) and not is_collab(a)]
            best = max(refs, key=lambda ref: depth(phys, ref))
            eff.append(depth(phys, best) / max(1, len(phys)))
            ratio.append(len(phys) / len(best))
        groups = defaultdict(list)
        for r in rows:
            groups[r["messages"][-1]["content"].split("<input>")[-1]].append(normalize_action(r["response"]))
        dup = [g for g in groups.values() if len(g) > 1]
        dup_rows = sum(len(g) for g in dup)
        conflict = sum(len(g) - Counter(g).most_common(1)[0][1] for g in dup)
        print(f"L{level} {role:9s} eps {len(episodes):2d} | plan eff mean {mean(eff):.2f} min {min(eff):.2f} | "
              f"phys/ref median {median(ratio):.2f} >1.5x {sum(x > 1.5 for x in ratio) / len(ratio):.2f} | "
              f"rows in repeated states {dup_rows / len(rows):.2f}, of which off-majority {conflict / max(1, dup_rows):.2f}")
