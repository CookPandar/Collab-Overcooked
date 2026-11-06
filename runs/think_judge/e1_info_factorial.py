"""E1 pilot: does the episode-level think judge's score of a decision depend on information the agent could not see?

2x2 per target decision, on the logged training requests of l2t_sum_matchfloor_g1 (first chunk of each episode):
  F+/F-  later decisions of the episode visible / replaced by id-only slots ("content withheld")
  P+/P-  privileged facts visible / withheld: reference solution, division of labour, reference-progress notes in
         [outcome]; for Assistant targets also the recipe (the Chef sees the recipe, so it stays for Chef targets)
F+P+ is the original training request re-sent. Every condition is requested `--repeats` times (judge noise).
Plan: Scheduler-GRPOinMulti-Agent-Collaboration/research_plans/grounded_collaboration_rewards_20261028/
motivation_validation_20261031/plan_zh.md, E1.

usage: python runs/think_judge/e1_info_factorial.py --out DIR [--episodes 24 --targets 4 --repeats 2] [--dry]
       python runs/think_judge/e1_info_factorial.py --out DIR --analyze
"""
import argparse
import gzip
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from baselines.rl.think_judge import EPISODE_SYSTEM, ThinkJudge, parse_episode_judgement  # noqa: E402

RUN = os.path.join(ROOT, "runs/think_judge/rl/l2t_sum_matchfloor_g1")
STEP = re.compile(r"^### id (\d+) \| (Chef|Assistant) \| .*$", re.MULTILINE)
PROGRESS = re.compile(r" \((advanced the reference|off the reference|no reference progress)\)")
DONE = re.compile(r"; reference steps done by this agent before: \d+")
WITHHELD = "(withheld)"


def load_requests():
    """One logged first-chunk request per (update, episode)."""
    seen = {}
    with open(os.path.join(RUN, "judge_annotations/rank0.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d.get("kind") != "episode" or not d.get("decisions") or d["decisions"][0]["id"] != 1:
                continue
            seen.setdefault((d["update"], int(d["episode"])), d)
    return seen


def load_success():
    success = {}
    for name in sorted(os.listdir(os.path.join(RUN, "rollouts"))):
        update = int(name[1:6])
        with gzip.open(os.path.join(RUN, "rollouts", name), "rt") as f:
            for line in f:
                if '"episode_summary"' in line:
                    d = json.loads(line)
                    success[(update, int(d["episode_summary"]))] = bool(d["success"])
    return success


def split_steps(prompt):
    """(head, [(id, role, text)], tail): the prompt around the per-decision blocks."""
    matches = list(STEP.finditer(prompt))
    tail_at = prompt.index("\n\n## Output")
    steps = []
    for k, m in enumerate(matches):
        end = matches[k + 1].start() if k + 1 < len(matches) else tail_at
        steps.append((int(m.group(1)), m.group(2), prompt[m.start():end].rstrip("\n")))
    return prompt[:matches[0].start()], steps, prompt[tail_at:]


def progress_note(text):
    m = PROGRESS.search(text.split("[outcome]")[-1])
    return m.group(1) if m else None


def redact_head(head, role):
    head = re.sub(r"^Reference solution \(privileged, one valid way\):.*$", f"Reference solution: {WITHHELD}", head,
                  flags=re.MULTILINE)
    head = re.sub(r"^Division of labour:.*$", f"Division of labour: {WITHHELD}", head, flags=re.MULTILINE)
    if role == "Assistant":
        head = re.sub(r"^Recipe \(privileged; only the Chef sees it\):.*$", f"Recipe: {WITHHELD}", head,
                      flags=re.MULTILINE)
    return head


def redact_step(text):
    body, _, outcome = text.rpartition("[outcome]")
    return body + "[outcome]" + DONE.sub("", PROGRESS.sub("", outcome))


def build(prompt, target, role, future, privileged):
    head, steps, tail = split_steps(prompt)
    if not privileged:
        head = redact_head(head, role)
    blocks = []
    for sid, srole, text in steps:
        if sid > target and not future:
            blocks.append(f"### id {sid} | {srole} | (content withheld)")
            continue
        blocks.append(text if privileged else redact_step(text))
    return head + "\n\n".join(blocks) + tail


def sample(requests, success, n_episodes, n_targets, seed):
    rng = random.Random(seed)
    pools = {True: [], False: []}
    for key, d in requests.items():
        if key[0] >= 10 and key in success and len(d["decisions"]) >= 12:
            pools[success[key]].append(key)
    chosen = []
    for ok in (True, False):
        pool = sorted(pools[ok])
        chosen += rng.sample(pool, min(n_episodes // 2, len(pool)))
    plan = []
    for key in chosen:
        _, steps, _ = split_steps(requests[key]["messages"][1]["content"])
        n = len(steps)
        for q in range(n_targets):  # spread over the chunk, avoiding the very first and last decision
            lo, hi = 1 + q * (n - 2) // n_targets, 1 + (q + 1) * (n - 2) // n_targets
            sid, role, text = steps[rng.randrange(lo, max(lo + 1, hi))]
            plan.append({"update": key[0], "episode": key[1], "success": success[key], "target": sid, "role": role,
                         "n_steps": n, "progress": progress_note(text),
                         "future_progress": sum(progress_note(t) == "advanced the reference"
                                                for i, _, t in steps if i > sid),
                         "train_score": None})
    return chosen, plan


def jobs_for(requests, chosen, plan, repeats):
    """Each job: one request; F+ requests are shared by all targets of the same episode and role-version."""
    jobs = {}
    for t in plan:
        prompt = requests[(t["update"], t["episode"])]["messages"][1]["content"]
        for future in (True, False):
            for privileged in (True, False):
                # F+: one request per episode (P+) or per episode x role (P-); F-: one per target
                key = (t["update"], t["episode"], "F+" if future else f"F-{t['target']}",
                       "P+" if privileged else f"P-{t['role']}")
                for r in range(repeats):
                    jobs.setdefault(key + (r,), build(prompt, t["target"], t["role"], future, privileged))
    return jobs


def run(args):
    os.makedirs(args.out, exist_ok=True)
    requests, success = load_requests(), load_success()
    chosen, plan = sample(requests, success, args.episodes, args.targets, args.seed)
    for t in plan:
        parsed = requests[(t["update"], t["episode"])].get("parsed") or {}
        t["train_score"] = (parsed.get(str(t["target"])) or {}).get("score")
    with open(os.path.join(args.out, "plan.json"), "w") as f:
        json.dump({"episodes": chosen, "targets": plan, "args": vars(args)}, f, indent=1)
    jobs = jobs_for(requests, chosen, plan, args.repeats)
    chars = sum(len(p) for p in jobs.values())
    print(f"{len(chosen)} episodes, {len(plan)} targets, {len(jobs)} requests, ~{chars / 3.5 / 1e6:.1f}M prompt tokens")
    if args.dry:
        key = next(k for k in jobs if k[2] != "F+" and k[3] != "P+")
        with open(os.path.join(args.out, "example_F-P-.txt"), "w") as f:
            f.write(jobs[key])
        return
    path = os.path.join(args.out, "responses.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {tuple(d["key"]) for d in map(json.loads, f) if d["raw"] is not None}  # failed ones are retried
    judge = ThinkJudge(model=args.model, env_file=args.env_file, workers=args.workers,
                       log_path=os.path.join(args.out, "judge_log.jsonl"))
    todo = [(k, p) for k, p in jobs.items() if tuple(k) not in done]

    def one(item):
        key, prompt = item
        for attempt in range(4):
            try:
                raw = judge._request(prompt, judge.temperature, system=EPISODE_SYSTEM)
                return key, raw, getattr(judge._local, "usage", None)
            except Exception as error:  # noqa: BLE001
                err = str(error)[:300]
        return key, None, {"error": err}

    with ThreadPoolExecutor(args.workers) as pool, open(path, "a") as f:
        for n, (key, raw, usage) in enumerate(pool.map(one, todo), 1):
            f.write(json.dumps({"key": list(key), "raw": raw, "usage": usage}) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("usage", judge.usage)


def analyze(args):
    import numpy as np
    with open(os.path.join(args.out, "plan.json")) as f:
        plan = json.load(f)["targets"]
    responses = {}
    with open(os.path.join(args.out, "responses.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            if d["raw"] is not None or tuple(d["key"]) not in responses:
                responses[tuple(d["key"])] = d["raw"]
    rows = []
    for t in plan:
        cell = {}
        for fk, pk in (("F+", "P+"), ("F+", "P-"), ("F-", "P+"), ("F-", "P-")):
            fkey = "F+" if fk == "F+" else f"F-{t['target']}"
            pkey = "P+" if pk == "P+" else f"P-{t['role']}"
            vals = []
            for r in range(args.repeats):
                raw = responses.get((t["update"], t["episode"], fkey, pkey, r))
                got = parse_episode_judgement(raw, {t["target"]}).get(t["target"]) if raw else None
                vals.append(None if got is None else (got["score"], got["grounding"]))
            cell[fk + pk] = vals
        rows.append(dict(t, cells=cell))
    with open(os.path.join(args.out, "scores.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def mean_score(row, cell, dim=0):
        vals = [v[dim] for v in row["cells"][cell] if v is not None and v[dim] is not None]
        return float(np.mean(vals)) if vals else None

    def boot(diffs, groups, n=5000, seed=0):
        """Mean and 95% interval, resampling episodes (targets of one episode are not independent)."""
        rng = np.random.default_rng(seed)
        by = {}
        for d, g in zip(diffs, groups):
            by.setdefault(g, []).append(d)
        keys = list(by)
        stats = [np.mean(np.concatenate([by[keys[i]] for i in rng.integers(0, len(keys), len(keys))]))
                 for _ in range(n)]
        return float(np.mean(diffs)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

    out = []
    for dim, name in ((0, "score"), (1, "grounded")):
        complete = [r for r in rows if all(mean_score(r, c, dim) is not None for c in r["cells"])]
        groups = [(r["update"], r["episode"]) for r in complete]
        m = {c: np.array([mean_score(r, c, dim) for r in complete]) for c in ("F+P+", "F+P-", "F-P+", "F-P-")}
        out.append(f"## {name}: {len(complete)}/{len(rows)} targets with all four cells")
        out.append("cell means: " + ", ".join(f"{c} {v.mean():.3f}" for c, v in m.items()))
        fut = (m["F+P+"] + m["F+P-"] - m["F-P+"] - m["F-P-"]) / 2
        priv = (m["F+P+"] + m["F-P+"] - m["F+P-"] - m["F-P-"]) / 2
        inter = (m["F+P+"] - m["F-P+"]) - (m["F+P-"] - m["F-P-"])
        for label, d in (("future effect (F+ - F-)", fut), ("privileged effect (P+ - P-)", priv),
                         ("interaction", inter)):
            mu, lo, hi = boot(d, groups)
            out.append(f"{label}: {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]; |diff|>0 in {np.mean(np.abs(d) > 1e-9):.0%}")
        for ok in (True, False):  # hindsight: does seeing the future move scores toward the episode outcome?
            sel = [i for i, r in enumerate(complete) if r["success"] == ok]
            if sel:
                mu, lo, hi = boot(fut[sel], [groups[i] for i in sel])
                out.append(f"future effect | episode {'success' if ok else 'failure'} (n={len(sel)}): "
                           f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        for note in ("advanced the reference", "off the reference", "no reference progress", None):
            sel = [i for i, r in enumerate(complete) if r["progress"] == note]
            if sel:
                mu, lo, hi = boot(priv[sel], [groups[i] for i in sel])
                out.append(f"privileged effect | target {note or 'no executed action'} (n={len(sel)}): "
                           f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        noise = [abs(r["cells"][c][0][dim] - r["cells"][c][1][dim]) for r in complete for c in r["cells"]
                 if len(r["cells"][c]) > 1 and all(v is not None and v[dim] is not None for v in r["cells"][c][:2])]
        if noise:
            out.append(f"repeat noise: mean |rep1 - rep2| {np.mean(noise):.3f}, identical {np.mean(np.array(noise) < 1e-9):.0%}")
        if dim == 0:
            paired = [(r["train_score"], mean_score(r, "F+P+")) for r in complete if r["train_score"] is not None]
            if paired:
                a, b = np.array(paired).T
                out.append(f"training score vs re-sent F+P+: mean |diff| {np.mean(np.abs(a - b)):.3f}, "
                           f"corr {np.corrcoef(a, b)[0, 1]:.2f}")
    text = "\n".join(out)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--targets", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--env_file", default="~/ParaCook/.env.shuaiapi.local")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
