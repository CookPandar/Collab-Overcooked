"""Success per level (L1-L6) + L2 train vs held-out dishes for the L2 rule-reward arms (eval_all_levels outputs).
usage: python3 eval_table_l2r.py [ARM ...]   (run from runs/think_judge; an ARM starting with "l2" is a full run name,
e.g. l2s_s0 or l2t_sum_matchfloor_g1, else the l2r_ARM rule arm)"""
import json, sys
from pathlib import Path
HELD = {"boiled_corn_slices", "baked_potato_slices"}
arms = sys.argv[1:] or ["none", "ground", "mate", "mate_v2", "plan", "full", "ground_split", "full_v2_split"]
print("| arm | " + " | ".join(f"L{l}" for l in range(1, 7)) + " | L2 train | L2 held-out |")
print("|---" * 9 + "|")
for a in arms:
    row = []
    run = a if a.startswith("l2") else f"l2r_{a}"
    for l in range(1, 7):
        p = Path(f"eval/all_levels/think_{run}_u50_l{l}.jsonl")
        if p.exists() and p.stat().st_size:
            r = json.loads(p.read_text().splitlines()[-1]); row.append(f"{r['success_rate']:.2f}")
        else:
            row.append("—")
    ep = Path(f"eval/all_levels/think_{run}_u50_l2.jsonl.episodes.jsonl")
    tr = ho = "—"
    if ep.exists() and Path(str(ep).replace(".episodes.jsonl", "")).stat().st_size:
        eps = [json.loads(x) for x in open(ep)]
        t = [e["success"] for e in eps if e["order"] not in HELD]; h = [e["success"] for e in eps if e["order"] in HELD]
        tr = f"{sum(t)/len(t):.2f} ({sum(t)}/{len(t)})"; ho = f"{sum(h)/len(h):.2f} ({sum(h)}/{len(h)})"
    print(f"| {a} | " + " | ".join(row) + f" | {tr} | {ho} |")
