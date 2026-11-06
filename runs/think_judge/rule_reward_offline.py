"""Offline check of the rule-verified think rewards (baselines/rl/rule_rewards.py) on saved rollouts.

usage (from ~/Collab-Overcooked-private):
  rule_reward_offline.py --out eval/rule_rewards/offline.jsonl [--episodes N] LABEL=RUN_DIR:U1,U2 ...
Every judgeable call of the first N episodes of each listed update is extracted (gpt-5.6-terra) and scored; one
line per call is written (already scored labels are skipped), then means per label and role are printed.
"""
import argparse
import gzip
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from baselines.rl.rule_rewards import extract, rewards
from baselines.rl.think_judge import ThinkJudge, judgeable

TJ = Path("runs/think_judge")


def episodes(run, update):
    rows = [json.loads(l) for l in gzip.open(TJ / "rl" / run / "rollouts" / f"u{update:05d}.jsonl.gz", "rt")]
    success = {r["episode_summary"]: r["success"] for r in rows if "episode_summary" in r}
    calls = defaultdict(list)
    for r in rows:
        if "episode_summary" not in r and r.get("call_type"):
            calls[r["episode"]].append(r)
    return [(e, calls[e], success.get(e)) for e in sorted(calls)]


def score_episode(judge, records):
    chosen = [k for k, r in enumerate(records) if judgeable(r)]
    found = extract(judge, [records[k] for k in chosen])
    aligned = [None] * len(records)
    for k, x in zip(chosen, found):
        aligned[k] = x
    return aligned, rewards(records, aligned, records[0]["order"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("specs", nargs="+")
    args = parser.parse_args()
    out = TJ / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["label"] for l in open(out)} if out.exists() else set()
    judge = ThinkJudge(env_file="~/ParaCook/.env.shuaiapi.local", model=args.model, workers=8, extra={"reasoning_effort": "low"},
                       log_path=str(out.with_suffix(".log.jsonl")))
    jobs = []
    for spec in args.specs:
        label, rest = spec.split("=", 1)
        run, updates = rest.split(":")
        if label in done:
            continue
        for u in map(int, updates.split(",")):
            for e, records, success in episodes(run, u)[:args.episodes]:
                jobs.append((label, u, e, records, success))
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda j: score_episode(judge, j[3]), jobs))
    with open(out, "a") as f:
        for (label, u, e, records, success), (found, scores) in zip(jobs, results):
            for r, x, s in zip(records, found, scores):
                if x is None and s is None:
                    continue
                f.write(json.dumps({"label": label, "update": u, "episode": e, "success": success,
                                    "agent_index": r["agent_index"], "timestep": r["timestep"],
                                    "response": r["response"], "observation": r["observation"],
                                    "judge_grounding": (r.get("judge_dims") or {}).get("grounding"),
                                    "extraction": x, **({k: s[k] for k in ("ground", "mate", "plan")} if s else {}),
                                    "detail": s["detail"] if s else None}, default=str) + "\n")
    rows = [json.loads(l) for l in open(out)]
    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("ground", "mate", "plan"):
            if r.get(k) is not None:
                table[(r["label"], r["agent_index"])][k].append(r[k])
        table[(r["label"], r["agent_index"])]["failed"].append(r["extraction"] is None)
    for (label, a), d in sorted(table.items()):
        print(f"{label:14s} {'Chef' if a == 0 else 'Assistant':9s} n={len(d['failed']):4d} failed={sum(d['failed'])} "
              + " ".join(f"{k}={sum(v) / len(v):+.3f}" for k, v in d.items() if k != "failed" and v))
    print("usage", judge.usage)


if __name__ == "__main__":
    main()
