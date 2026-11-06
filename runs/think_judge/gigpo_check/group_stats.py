"""Offline check for GiGPO-style step groups in Collab-Overcooked rollouts: calls are grouped by an anchor-state key
(order, role, kitchen states, agent states; the conversation and the agent's own Recent Goal are excluded) over the
episodes of a group (the episodes of `merge` consecutive updates with the same order). Reports how many calls fall
in groups of size >= 2, how many groups span >= 2 episodes, and how often a group mixes call kinds (msg / wait /
progress / other / rejected) so that a step-level advantage can contrast them."""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parent.parent / "rl"


def key(row):
    obs = row["observation"]
    kitchen = re.search(r"Kitchen states:(.*)", obs)
    agents = re.search(r"Agent State:(.*)", obs)
    return (row["order"], row["agent_index"], kitchen.group(1).strip() if kitchen else "",
            agents.group(1).strip() if agents else "")


def kind(row):
    action = row["response"].split("Action:")[-1].strip().lower()
    if row["format_reward"] < 0:
        return "format"
    if action.startswith(("collab(", "request(")):
        return "msg"
    if action.startswith("wait"):
        return "wait"
    if row["validator_reward"] < 0:
        return "rejected"
    return "progress" if row["ites"] > 1e-9 else "other"


def load(run, update):
    path = RUNS / run / "rollouts" / f"u{update:05d}.jsonl.gz"
    if not path.exists():
        return []
    return [d for d in map(json.loads, gzip.open(path, "rt")) if "episode_summary" not in d]


def stats(run, updates, merge):
    calls = in_group = cross = mixed_calls = 0
    sizes = Counter()
    for start in range(0, len(updates) - merge + 1, merge):
        rows = [r for u in updates[start:start + merge] for r in load(run, u)]
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        for g in groups.values():
            calls += len(g)
            sizes[min(len(g), 10)] += 1
            if len(g) >= 2:
                in_group += len(g)
                cross += len(g) * (len({(r["episode"]) for r in g}) >= 2)
                kinds = {kind(r) for r in g}
                mixed_calls += len(g) * (len(kinds - {"format"}) >= 2 and bool(kinds & {"msg", "wait"})
                                         and bool(kinds & {"progress", "other"}))
    return {"run": run, "merge_updates": merge, "calls": calls, "in_group>=2": round(in_group / calls, 3),
            "in_cross_episode_group": round(cross / calls, 3),
            "in_group_mixing_talk/wait_and_action": round(mixed_calls / calls, 3)}


if __name__ == "__main__":
    for run in sys.argv[1:] or ["l2r_none", "l2t_sum_matchfloor_g1"]:
        updates = list(range(1, 51))
        for merge in (1, 2, 4):
            print(json.dumps(stats(run, updates, merge)))
