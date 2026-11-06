"""Step-level timeline of the two roles against the reference trajectories of baselines/rl/credit.py.

For every call the role's matched depth in its reference (longest in-order prefix of the reference found in the role's
validated actions, best reference) gives the role's next reference step. The Chef's steps `pickup(X,counter)` need
the Assistant to have put X on a counter first, so a Chef call that waits or talks is
- blocked: the next step is pickup(X,counter), X is on no counter the Chef can reach and the Chef does not hold it;
- idle: the next step is possible now (X is on a counter, or a step on the Chef's own utensils).
Per (checkpoint, level): each role's final depth, where each role stops (its next reference step at the end of
failed episodes), the Chef's wait / talk calls split into blocked and idle, how many timesteps the Chef takes to pick
up an item after it appears on a counter, and the Assistant's invalid actions.

    python runs/think_judge/role_timeline.py eval/trace/*_l3.jsonl.episodes.jsonl
    python runs/think_judge/role_timeline.py --rollouts l2_sum_w0.5 41 50      # L2 training rollouts (no validated
                                                                              # action saved: the parsed valid action)
"""
import collections
import gzip
import json
import os
import re
import sys

sys.path[:0] = [os.path.expanduser("~/Collab-Overcooked-private"), os.path.expanduser("~/Collab-Overcooked-private/lib/overcooked_ai")]
from baselines.rl.credit import load_references, normalize_action  # noqa: E402

ACTION = re.compile(r"Action\s*:\s*(.*)", re.S)
HOLD = re.compile(r"<Chef> holds ([^.]*)\.")
COUNTERS = re.compile(r"counters? can be visited by <Chef>\. Their states are as follows: (.*?)(?:\n|Agent State)", re.S)
PICKUP_COUNTER = re.compile(r"pickup\((\w+),\s*counter\)")
_refs = {}


def references(order):
    if order not in _refs:
        _refs[order] = load_references(order)
    return _refs[order]


def action_of(call):
    if call.get("validated_action"):
        return normalize_action(call["validated_action"])
    m = ACTION.search(call.get("response") or "")
    return normalize_action(" ".join(m.group(1).split()).strip("` ")) if m else ""


def depth(history, reference):
    d = 0
    for a in history:
        if d < len(reference) and a == reference[d]:
            d += 1
    return d


def best(history, refs):
    """(depth, reference) with the deepest match."""
    return max(((depth(history, r), r) for r in refs), key=lambda x: (x[0], -len(x[1])))


def on_counter(observation, item):
    m = COUNTERS.search(observation or "")
    return bool(m and re.search(rf"(?<![\w]){re.escape(item)}(?![\w])", m.group(1)))


def chef_holds(observation, item):
    m = HOLD.search(observation or "")
    return bool(m and re.search(rf"(?<![\w]){re.escape(item)}(?![\w])", m.group(1)))


def analyse(episodes):
    out = collections.Counter()
    stops = {0: collections.Counter(), 1: collections.Counter()}
    latency, depths, handover = [], {0: [], 1: []}, []
    invalid_at = collections.Counter()
    for e in episodes:
        refs = references(e["order"])
        history = {0: [], 1: []}
        appeared = {}  # item -> first timestep seen on a Chef counter
        for c in sorted(e["calls"], key=lambda c: (c["timestep"], c["agent_index"])):
            role, obs = c["agent_index"], c.get("observation") or ""
            d, ref = best(history[role], refs[role])
            nxt = ref[d] if d < len(ref) else "done"
            act = action_of(c)
            if role == 0:
                for m in re.finditer(r"(\w+)", (COUNTERS.search(obs) or [None, ""])[1] if COUNTERS.search(obs) else ""):
                    appeared.setdefault(m.group(1), c["timestep"])
            idle_call = act.startswith("wait") or act.startswith("collab") or act.startswith("Collab")
            if c["validator_reward"] < 0:
                out[f"invalid_{role}"] += 1
                invalid_at[(role, nxt)] += 1
            if role == 0 and idle_call:
                m = PICKUP_COUNTER.match(nxt)
                if m and not on_counter(obs, m.group(1)) and not chef_holds(obs, m.group(1)):
                    out["chef_blocked"] += 1
                elif nxt == "done":
                    out["chef_done_idle"] += 1
                else:
                    out["chef_idle"] += 1
                    out[f"chef_idle_next:{nxt.split('(')[0]}"] += 1
            if role == 1 and idle_call:
                out["asst_idle"] += 1
            out[f"calls_{role}"] += 1
            if c["validator_reward"] >= 0 and c["format_reward"] >= 0 and not idle_call and act:
                before = d
                history[role].append(act)
                if role == 0 and best(history[0], refs[0])[0] > before:
                    m = PICKUP_COUNTER.match(nxt)
                    if m and m.group(1) in appeared:
                        latency.append(c["timestep"] - appeared[m.group(1)])
        first = PICKUP_COUNTER.match(refs[0][0][0])
        handover.append(appeared.get(first.group(1)) if first else None)
        for role in (0, 1):
            d, ref = best(history[role], refs[role])
            depths[role].append(d / len(ref))
            if not e["success"]:
                stops[role][ref[d].split(",")[0] + ")" if d < len(ref) else "done"] += 1
        out["episodes"] += 1
        out["success"] += e["success"]
    n = max(1, out["episodes"])
    chef_wait = out["chef_blocked"] + out["chef_idle"] + out["chef_done_idle"]
    return {"episodes": out["episodes"], "success": out["success"] / n,
            "chef_depth": sum(depths[0]) / n, "asst_depth": sum(depths[1]) / n,
            "chef_wait_talk_per_ep": chef_wait / n,
            "chef_blocked_share": out["chef_blocked"] / max(1, chef_wait),
            "chef_idle_share": out["chef_idle"] / max(1, chef_wait),
            "chef_idle_next": {k.split(":")[1]: out[k] / max(1, out["chef_idle"]) for k in out if k.startswith("chef_idle_next:")},
            "chef_pickup_latency": sum(latency) / len(latency) if latency else None, "latency_n": len(latency),
            "first_handover_t": (sum(h for h in handover if h is not None) / max(1, sum(h is not None for h in handover))),
            "no_handover": sum(h is None for h in handover) / n,
            "asst_wait_talk_per_ep": out["asst_idle"] / n,
            "invalid_per_ep": {"chef": out["invalid_0"] / n, "assistant": out["invalid_1"] / n},
            "chef_stops": dict(stops[0].most_common(4)), "asst_stops": dict(stops[1].most_common(4)),
            "asst_invalid_at": {k[1]: v for k, v in invalid_at.most_common(8) if k[0] == 1}}


def load_rollouts(run, first, last):
    episodes = []
    for u in range(first, last + 1):
        rows = [json.loads(l) for l in gzip.open(f"rl/{run}/rollouts/u{u:05d}.jsonl.gz", "rt")]
        success = {r["episode_summary"]: r["success"] for r in rows if "episode_summary" in r}
        calls = collections.defaultdict(list)
        for r in rows:
            if "call_index" in r:
                calls[r["episode"]].append(r)
        episodes += [{"order": cs[0]["order"], "success": success.get(ep, False), "calls": cs} for ep, cs in calls.items()]
    return episodes


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if sys.argv[1] == "--rollouts":
        result = {f"{sys.argv[2]} u{sys.argv[3]}-{sys.argv[4]}": analyse(load_rollouts(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))}
    else:
        result = {}
        for path in sys.argv[1:]:
            groups = collections.defaultdict(list)
            for e in map(json.loads, open(path)):
                groups[e["checkpoint"]].append(e)
            level = re.search(r"_l(\d)\.jsonl", path).group(1)
            for ck, eps in groups.items():
                result[f"{ck.rstrip('/').split('/')[-1] if 'checkpoints' not in ck else ck.split('/')[-3]} L{level}"] = analyse(eps)
    print(json.dumps(result, indent=1, ensure_ascii=False))
