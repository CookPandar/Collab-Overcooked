"""E1 consequence check: does "Think says it will request X, the sent request asks for Y" become more frequent during
training with the judge reward, compared with the outcome-only run (l2r_none, same config without judge)?

Uses every decision in the rollouts (not only the judged first chunks). The intent / request parser is the F1c one in
e1_facts.py (precision 130/141 after an independent manual check).

usage: python runs/think_judge/e1_trend.py --out DIR [--runs l2t_sum_matchfloor_g1 l2r_none ...]
"""
import argparse
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_facts as F  # noqa: E402

RL = os.path.join(F.ROOT, "runs/think_judge/rl")


def classify(response):
    """None if the decision has no single 'request the teammate to X' intent + single request; else True if X != Y."""
    think = response.split("Recent Goal", 1)[0].split("Think:", 1)[-1]
    act = re.search(r"Action:\s*(.+)", response)
    if not act:
        return None
    action = act.group(1).strip().strip("`").strip()
    kind, sent = F.stated_now(think)
    if kind != "communicate" or F.action_type(action) != "communicate" or ";" in action:
        return None
    m, a = F.ASKED.search(sent), F.SENT.match(action)
    if not m or not a:
        return None
    said = F.clause_kind(m.group(4).lower())
    if said is None:
        return None
    sent_kind = {"slice": "cut", "chop": "cut"}.get(a.group(1), a.group(1))
    # literal comparison only: the cut == put-on-board equivalence was found to hide real mismatches
    return said != sent_kind


def run_stats(run):
    rows = []
    for name in sorted(os.listdir(os.path.join(RL, run, "rollouts"))):
        u = int(name[1:6])
        n_dec = n_req = n_mis = 0
        with gzip.open(os.path.join(RL, run, "rollouts", name), "rt") as f:
            for line in f:
                d = json.loads(line)
                if not d.get("response"):
                    continue
                n_dec += 1
                c = classify(d["response"])
                if c is not None:
                    n_req += 1
                    n_mis += c
        rows.append(dict(run=run, update=u, decisions=n_dec, request_intent=n_req, content_mismatch=n_mis))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", nargs="+", default=["l2t_sum_matchfloor_g1", "l2t_gated_matchfloor_g1", "l2r_none",
                                                  "l2r_ground", "l2r_mate", "l2r_plan"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    lines = []
    with open(os.path.join(args.out, "per_update.jsonl"), "w") as f:
        for run in args.runs:
            rows = run_stats(run)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            parts = []
            for lo in range(1, 51, 10):
                g = [r for r in rows if lo <= r["update"] < lo + 10]
                req, mis, dec = (sum(r[k] for r in g) for k in ("request_intent", "content_mismatch", "decisions"))
                parts.append(f"u{lo}-{lo + 9}: {mis}/{req}={mis / req:.3f}" if req else f"u{lo}-{lo + 9}: -")
            lines.append(f"{run}: " + " | ".join(parts))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
