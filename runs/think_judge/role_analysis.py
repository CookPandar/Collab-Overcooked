"""Chef vs Assistant: who is the bottleneck. The ITES of a role summed over an episode is the completed fraction of
that role's reference sub-sequence (0-1; 1.0 for the Chef in every successful episode), so the two roles' episode
ITES can be compared directly.
- training (rollouts of the RL runs): per update and role, the ITES, the error rates, the wait / Collab share of the
  calls, repeats and the mean action advantage -> eval/role_counts.json;
- zero-shot levels 1-6 (eval/all_levels): per checkpoint, level and role the ITES, plus the failed episodes split by
  which role did not finish its part -> eval/role_levels.json.
    python runs/think_judge/role_analysis.py
"""
import collections
import glob
import gzip
import json
import os
import re

os.chdir(os.path.dirname(os.path.abspath(__file__)))
RUNS = ["l2_none", "l2_split_w0.5", "l2_sum_w0.5", "l2_decoupled_w0.5", "l2_gated_w0.5", "l2_gated_w0.5_collab",
        "l2_none_redund", "l2_action_w0.5", "l2_none_grpo", "l2_none_grpo_agentgym", "l2_none_ppo_token",
        "l2_none_grpo_gated", "l2_none_ppo_token_gated"]
DONE = 0.999  # a role finished its part


def action(response):
    m = re.search(r"Action\s*:\s*(.*)", response or "", re.S)
    return " ".join(m.group(1).split()).strip("` ") if m else ""


def role_stats(episodes):
    """episodes: lists of calls (agent_index, response, ites, format_reward, validator_reward[, adv_action, judge_score]).
    Returns {role: metrics per episode}."""
    out = {}
    for role, name in ((0, "chef"), (1, "assistant")):
        agg = collections.Counter()
        judge = []
        for calls in episodes:
            last = None
            for c in calls:
                if c["agent_index"] != role:
                    continue
                a = action(c.get("response"))
                agg["calls"] += 1
                agg["ites"] += c["ites"]
                agg["format_err"] += c["format_reward"] < 0
                agg["validator_err"] += c["validator_reward"] < 0
                agg["wait"] += a.startswith("wait")
                agg["collab"] += a.startswith("Collab")
                agg["repeat"] += a.startswith("Collab") and a == last
                agg["adv"] += c.get("adv_action") or 0.0
                if c.get("judge_score") is not None:
                    judge.append(c["judge_score"])
                last = a
        n, calls = max(1, len(episodes)), max(1, agg["calls"])
        out[name] = {"ites": agg["ites"] / n, "calls": agg["calls"] / n, "repeat": agg["repeat"] / n,
                     "format_err": agg["format_err"] / calls, "validator_err": agg["validator_err"] / calls,
                     "wait_share": agg["wait"] / calls, "collab_share": agg["collab"] / calls,
                     "adv": agg["adv"] / calls, "judge": sum(judge) / len(judge) if judge else None}
    return out


def bottleneck(episodes):
    """Failed episodes by which role did not finish its part (fractions of all episodes)."""
    count = collections.Counter()
    for success, calls in episodes:
        ites = [sum(c["ites"] for c in calls if c["agent_index"] == r) for r in (0, 1)]
        if success:
            count["success"] += 1
        elif ites[0] < DONE and ites[1] >= DONE:
            count["chef_only"] += 1
        elif ites[0] >= DONE and ites[1] < DONE:
            count["assistant_only"] += 1
        elif ites[0] < DONE and ites[1] < DONE:
            count["both"] += 1
        else:
            count["neither"] += 1  # both parts done, the order still failed (delivery / timing)
    n = max(1, len(episodes))
    return {k: count[k] / n for k in ("success", "chef_only", "assistant_only", "both", "neither")}


training = {}
for run in RUNS:
    rows = []
    for path in sorted(glob.glob(f"rl/{run}/rollouts/u*.jsonl.gz")):
        update = int(os.path.basename(path)[1:6])
        records = [json.loads(l) for l in gzip.open(path, "rt")]
        success = {r["episode_summary"]: r["success"] for r in records if "episode_summary" in r}
        by_episode = collections.defaultdict(list)
        for r in records:
            if "call_index" in r:
                by_episode[r["episode"]].append(r)
        episodes = list(by_episode.values())
        rows.append(dict(update=update, **{f"{role}/{k}": v for role, m in role_stats(episodes).items()
                                         for k, v in m.items()},
                         **{f"bottleneck/{k}": v for k, v in bottleneck(
                             [(success.get(e, False), cs) for e, cs in by_episode.items()]).items()}))
    if rows:
        training[run] = rows
        print(run, len(rows), flush=True)
json.dump(training, open("eval/role_counts.json", "w"))

levels = {}
for path in sorted(glob.glob("eval/all_levels/*_l[1-6].jsonl.episodes.jsonl")):
    name, level = re.match(r"(.*)_l(\d)\.jsonl", os.path.basename(path)).groups()
    episodes = [json.loads(l) for l in open(path)]
    levels.setdefault(name, {})[level] = {**{f"{role}/{k}": v for role, m in role_stats([e["calls"] for e in episodes]).items()
                                            for k, v in m.items()},
                                         **{f"bottleneck/{k}": v for k, v in bottleneck(
                                             [(e["success"], e["calls"]) for e in episodes]).items()}}
json.dump(levels, open("eval/role_levels.json", "w"))
print("levels:", len(levels))
