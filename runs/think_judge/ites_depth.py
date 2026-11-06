"""How far each model gets along the reference plan, per agent, from the eval episode logs.

A call with ITES > 0 extends the matched in-order prefix of the best reference by one step (tes() in credit.py), so
their count is the reference depth reached; an executed non-wait, non-Collab action has ITES != 0. Per episode and
agent we take depth, executed actions, all calls, and the shortest reference length of the order.
usage: ites_depth.py NAME=EPISODES_JSONL ...   (run from ~/Collab-Overcooked-private)
"""
import json
import sys
from statistics import mean

from baselines.rl.credit import load_references

ROLES = {0: "Chef", 1: "Asst"}


def episode_stats(episode):
    refs = load_references(episode["order"])
    out = {}
    for agent in (0, 1):
        calls = [c for c in episode["calls"] if c["agent_index"] == agent]
        depth = sum(c["ites"] > 1e-9 for c in calls)
        executed = sum(abs(c["ites"]) > 1e-9 for c in calls)
        out[agent] = {"depth": depth, "executed": executed, "calls": len(calls),
                      "ref": min(len(r) for r in refs[agent])}
    return out


def main():
    print(f"{'model':26s} {'role':4s} {'ref':>4s} {'depth mean/max':>14s} {'done% mean/max':>15s} "
          f"{'eff/exec':>8s} {'eff/calls':>9s} {'exec':>5s} {'calls':>6s} {'succ':>5s}")
    for pair in sys.argv[1:]:
        name, path = pair.split("=", 1)
        episodes = [json.loads(line) for line in open(path)]
        stats = [episode_stats(e) for e in episodes]
        for agent in (0, 1):
            s = [x[agent] for x in stats]
            done = [x["depth"] / x["ref"] for x in s]
            print(f"{name:26s} {ROLES[agent]:4s} {mean(x['ref'] for x in s):4.1f} "
                  f"{mean(x['depth'] for x in s):7.1f}/{max(x['depth'] for x in s):<6d} "
                  f"{100 * mean(done):7.0f}/{100 * max(done):<6.0f}  "
                  f"{sum(x['depth'] for x in s) / max(1, sum(x['executed'] for x in s)):8.2f} "
                  f"{sum(x['depth'] for x in s) / max(1, sum(x['calls'] for x in s)):9.2f} "
                  f"{mean(x['executed'] for x in s):5.0f} {mean(x['calls'] for x in s):6.0f} "
                  f"{mean(e['success'] for e in episodes):5.2f}")


if __name__ == "__main__":
    main()
