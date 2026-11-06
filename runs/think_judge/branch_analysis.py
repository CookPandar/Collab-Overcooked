"""Analysis of branch_value.py: does the judge score of a Think rank the Thinks of one state by their value?

    python runs/think_judge/branch_analysis.py sum_u50 none_u50

Per state: Q_A(k,m) = U, Q_T(k) = mean_m U, V = mean_k Q_T. Reports, per run and pooled:
- variance of U split into between-Think (Var_k Q_T) and within-Think (mean_k Var_m U) parts;
- within-state Spearman correlation of each predictor with Q_T (Think level) and with U (branch level), averaged
  over states with a non-constant target;
- pairwise accuracy on Think pairs whose Q_T differ by at least 0.5 (ties of the predictor count one half).
Predictors: judge score with / without the call outcome, the immediate ITES of the call, the Think length.
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def ranks(x):
    x = np.asarray(x, float)
    order = x.argsort()
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ranks of ties
        r[x == v] = r[x == v].mean()
    return r


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


PREDICTORS = {"judge (with outcome)": "judge_with_outcome", "judge (no outcome)": "judge_no_outcome",
              "immediate ITES": "ites", "Think length": "think_tokens"}


def analyse(rows):
    states = collections.defaultdict(list)
    for r in rows:
        if r["diverged"] or not r["judgeable"]:
            continue
        r = dict(r, ites=r["target"]["sequence_reward"])
        states[r["state"]].append(r)
    report = {"states": 0, "branches": 0, "var_between": [], "var_within": []}
    corr_t = collections.defaultdict(list)
    corr_b = collections.defaultdict(list)
    pairs = collections.defaultdict(lambda: [0.0, 0])
    for branches in states.values():
        thinks = collections.defaultdict(list)
        for b in branches:
            thinks[b["k"]].append(b)
        if len(thinks) < 2:
            continue
        report["states"] += 1
        report["branches"] += len(branches)
        q_t = {k: np.mean([b["U"] for b in bs]) for k, bs in thinks.items()}
        report["var_between"].append(np.var(list(q_t.values())))
        report["var_within"].append(np.mean([np.var([b["U"] for b in bs]) for bs in thinks.values()]))
        for name, key in PREDICTORS.items():
            values = {k: [b.get(key) for b in bs if b.get(key) is not None] for k, bs in thinks.items()}
            ks = [k for k in thinks if values[k]]
            pred = {k: np.mean(values[k]) for k in ks}
            c = spearman([pred[k] for k in ks], [q_t[k] for k in ks])
            if c is not None:
                corr_t[name].append(c)
            usable = [b for b in branches if b.get(key) is not None]
            c = spearman([b[key] for b in usable], [b["U"] for b in usable])
            if c is not None:
                corr_b[name].append(c)
            for i, a in enumerate(ks):
                for b in ks[i + 1:]:
                    if abs(q_t[a] - q_t[b]) < 0.5:
                        continue
                    better, worse = (a, b) if q_t[a] > q_t[b] else (b, a)
                    hit = 1.0 if pred[better] > pred[worse] else 0.5 if pred[better] == pred[worse] else 0.0
                    pairs[name][0] += hit
                    pairs[name][1] += 1
    vb, vw = np.mean(report["var_between"] or [0]), np.mean(report["var_within"] or [0])
    lines = [f"states {report['states']}, branches {report['branches']}; "
             f"Var(U) between Thinks {vb:.3f}, within Thinks {vw:.3f} (between share {vb / max(vb + vw, 1e-9):.2f})",
             f"{'predictor':22s} {'rho(Q_T)':>9s} {'n':>4s} {'rho(U)':>8s} {'n':>4s} {'pair acc':>9s} {'pairs':>6s}"]
    for name in PREDICTORS:
        t, b = corr_t[name], corr_b[name]
        hit, n = pairs[name]
        lines.append(f"{name:22s} {np.mean(t) if t else float('nan'):9.3f} {len(t):4d} "
                     f"{np.mean(b) if b else float('nan'):8.3f} {len(b):4d} {hit / n if n else float('nan'):9.3f} {n:6d}")
    return "\n".join(lines)


def main():
    pooled = []
    for name in sys.argv[1:]:
        rows = [dict(r, state=f"{name}/{r['state']}")
                for r in map(json.loads, open(ROOT / "eval" / "branch_value" / f"{name}.jsonl"))]
        pooled += rows
        print(f"== {name}: {len(rows)} branches, {sum(r['diverged'] for r in rows)} diverged replays")
        print(analyse(rows))
    if len(sys.argv) > 2:
        print("== pooled")
        print(analyse(pooled))


if __name__ == "__main__":
    main()
