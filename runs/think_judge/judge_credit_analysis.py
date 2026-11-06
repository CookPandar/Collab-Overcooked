"""Offline decomposition of the LLM-judge think reward (l2t_*): which call categories it rewards or punishes,
how much of it a few rule features explain, and what it adds to the environment advantage in batches with and
without a success.  usage: judge_credit_analysis.py RUN [RUN ...]   (prints tables; writes eval/judge_credit/RUN.json)"""
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TJ = Path(__file__).parent
ACTION = re.compile(r"Action:\s*(.*)")


def action_of(response):
    found = ACTION.findall(response or "")
    return found[-1].strip() if found else ""


def categorize(rows):
    """Adds kind / cat / pos / repeat to the calls of one update (sorted by episode, timestep, call_index)."""
    rows.sort(key=lambda d: (d["episode"], d["timestep"], d["call_index"]))
    since, sent = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(set))
    for d in rows:
        e, i, a = d["episode"], d["agent_index"], action_of(d["response"])
        low = a.lower()
        if d["format_reward"] < 0:
            d["kind"], d["cat"] = "format", "format_error"
        elif low.startswith("collab"):
            since[e][i] += 1
            d["kind"] = "msg"
            d["repeat"] = a in sent[e][i]
            sent[e][i].add(a)
            d["cat"] = "msg_repeat" if d["repeat"] else f"msg_pos{min(since[e][i], 3)}"
        elif low.startswith("wait"):
            d["kind"], d["cat"] = "wait", "wait"
        else:
            d["kind"] = "act"
            if d["validator_reward"] < 0:
                d["cat"] = "act_rejected"
            else:
                d["cat"] = "act_progress" if d["ites"] > 1e-9 else "act_regress" if d["ites"] < -1e-9 else "act_noprogress"
                since[e] = defaultdict(int)  # an executed action resets both agents' counters
                sent[e] = defaultdict(set)


def load(run):
    calls = []
    for path in sorted((TJ / "rl" / run / "rollouts").glob("u*.jsonl.gz")):
        u = int(path.name[1:6])
        rows, success = [], {}
        for line in gzip.open(path, "rt"):
            d = json.loads(line)
            if "episode_summary" in d:
                success[d["episode_summary"]] = d["success"]
            else:
                rows.append(d)
        categorize(rows)
        batch_success = any(success.values())
        for d in rows:
            d["u"], d["success"], d["batch_success"] = u, success.get(d["episode"], False), batch_success
        calls += rows
    return calls


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def r2(calls, key_fn):
    ys = [c["judge_score"] for c in calls]
    cells = defaultdict(list)
    for c in calls:
        cells[key_fn(c)].append(c["judge_score"])
    m, total = mean(ys), sum((y - mean(ys)) ** 2 for y in ys)
    resid = sum((c["judge_score"] - mean(cells[key_fn(c)])) ** 2 for c in calls)
    return 1 - resid / total if total else float("nan")


ORDER = ["act_progress", "act_noprogress", "act_regress", "act_rejected", "msg_pos1", "msg_pos2", "msg_pos3",
         "msg_repeat", "wait", "format_error"]
for run in sys.argv[1:]:
    calls = load(run)
    judged = [c for c in calls if c.get("judge_score") is not None]
    sum_mode = "sum" in run
    out = {"run": run, "calls": len(calls), "judged": len(judged), "tables": {}}
    print(f"\n===== {run}: {len(calls)} calls, {len(judged)} judged")
    for role in (0, 1):
        print(f"-- {'Chef' if role == 0 else 'Assistant'}")
        print(f"{'category':15s} {'share':>6s} {'judge':>6s} {'wj all':>7s} {'wj fail':>8s} {'wj succ':>8s} {'A fail':>7s} {'A succ':>7s}")
        mine = [c for c in calls if c["agent_index"] == role]
        for cat in ORDER:
            cs = [c for c in mine if c["cat"] == cat]
            if not cs:
                continue
            wj = lambda c: c["think_reward"] or 0.0
            env = lambda c: (c["adv_action"] - (c["think_reward"] or 0.0)) if sum_mode else c["adv_action"]
            fb = [c for c in cs if not c["batch_success"]]
            sb = [c for c in cs if c["batch_success"]]
            row = dict(share=len(cs) / len(mine), judge=mean([c["judge_score"] for c in cs if c.get("judge_score") is not None]),
                       wj=mean([wj(c) for c in cs]), wj_fail=mean([wj(c) for c in fb]), wj_succ=mean([wj(c) for c in sb]),
                       A_fail=mean([env(c) for c in fb]), A_succ=mean([env(c) for c in sb]))
            out["tables"][f"{role}/{cat}"] = row
            print(f"{cat:15s} {row['share']:6.2f} {row['judge']:6.2f} {row['wj']:+7.2f} {row['wj_fail']:+8.2f} "
                  f"{row['wj_succ']:+8.2f} {row['A_fail']:+7.2f} {row['A_succ']:+7.2f}")
    feats = {"role×category": lambda c: (c["agent_index"], c["cat"]),
             "+episode success": lambda c: (c["agent_index"], c["cat"], c["success"]),
             "+timestep bin": lambda c: (c["agent_index"], c["cat"], c["success"], c["timestep"] // 10)}
    out["r2"] = {k: r2(judged, f) for k, f in feats.items()}
    print("R² of the judge score explained by rule features:", {k: round(v, 3) for k, v in out["r2"].items()})
    fail_batches = len({c["u"] for c in calls if not c["batch_success"]})
    print(f"updates without any success: {fail_batches}/{len({c['u'] for c in calls})}")
    (TJ / "eval" / "judge_credit").mkdir(parents=True, exist_ok=True)
    (TJ / "eval" / "judge_credit" / f"{run}.json").write_text(json.dumps(out, indent=1))
