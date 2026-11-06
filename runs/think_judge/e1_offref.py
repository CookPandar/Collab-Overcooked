"""E1 follow-up: does showing the reference solution lower the judge's score of off-reference actions?

The pilot (e1_info_factorial.py) found P+ < P- only for actions marked "off the reference" (n=8). Here: every
off-reference executed action in the first chunk of the l2t_sum_matchfloor_g1 episodes (update >= 10), plus the
reference-advancing actions of the same episodes as controls; future visible (F+), P+ vs P-, `--repeats` calls each.
F+ requests are shared by all targets of an episode (P-: one version per target role).

usage: python runs/think_judge/e1_offref.py --out DIR [--repeats 3] [--dry] | --analyze
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_info_factorial as E  # noqa: E402

CATS = ("off the reference", "advanced the reference")


def targets(requests, success, max_episodes, seed):
    import random
    out, keys = [], []
    for key in sorted(requests):
        if key[0] < 10 or key not in success:
            continue
        _, steps, _ = E.split_steps(requests[key]["messages"][1]["content"])
        notes = [(sid, role, E.progress_note(text)) for sid, role, text in steps]
        if any(n == CATS[0] for _, _, n in notes):
            keys.append(key)
    keys = sorted(random.Random(seed).sample(keys, min(max_episodes, len(keys))))
    for key in keys:
        _, steps, _ = E.split_steps(requests[key]["messages"][1]["content"])
        notes = [(sid, role, E.progress_note(text)) for sid, role, text in steps]
        parsed = requests[key].get("parsed") or {}
        for sid, role, note in notes:
            if note in CATS:
                out.append({"update": key[0], "episode": key[1], "success": success[key], "target": sid,
                            "role": role, "progress": note, "train_score": (parsed.get(str(sid)) or {}).get("score")})
    return out


def run(args):
    os.makedirs(args.out, exist_ok=True)
    requests, success = E.load_requests(), E.load_success()
    plan = targets(requests, success, args.max_episodes, args.seed)
    with open(os.path.join(args.out, "plan.json"), "w") as f:
        json.dump({"targets": plan, "args": vars(args)}, f, indent=1)
    jobs = {}
    for t in plan:
        prompt = requests[(t["update"], t["episode"])]["messages"][1]["content"]
        for privileged in (True, False):
            for r in range(args.repeats):
                key = (t["update"], t["episode"], "P+" if privileged else f"P-{t['role']}", r)
                jobs.setdefault(key, E.build(prompt, t["target"], t["role"], True, privileged))
    n_off = sum(t["progress"] == CATS[0] for t in plan)
    print(f"{len({(t['update'], t['episode']) for t in plan})} episodes, {n_off} off-reference + "
          f"{len(plan) - n_off} advancing targets, {len(jobs)} requests, "
          f"~{sum(map(len, jobs.values())) / 3.5 / 1e6:.1f}M prompt tokens")
    if args.dry:
        return
    path = os.path.join(args.out, "responses.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {tuple(d["key"]) for d in map(json.loads, f) if d["raw"] is not None}
    judge = E.ThinkJudge(model=args.model, env_file=args.env_file, workers=args.workers)

    def one(item):
        key, prompt = item
        err = None
        for _ in range(4):
            try:
                return key, judge._request(prompt, judge.temperature, system=E.EPISODE_SYSTEM), \
                    getattr(judge._local, "usage", None)
            except Exception as error:  # noqa: BLE001
                err = str(error)[:300]
        return key, None, {"error": err}

    todo = [(k, p) for k, p in jobs.items() if k not in done]
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
        for d in map(json.loads, f):
            if d["raw"] is not None or tuple(d["key"]) not in responses:
                responses[tuple(d["key"])] = d["raw"]
    lines = []
    for dim, name in ((0, "score"), (1, "grounded")):
        rows = []
        for t in plan:
            cell = {}
            for pk in ("P+", f"P-{t['role']}"):
                vals = []
                for r in range(args.repeats):
                    raw = responses.get((t["update"], t["episode"], pk, r))
                    got = E.parse_episode_judgement(raw, {t["target"]}).get(t["target"]) if raw else None
                    v = None if got is None else (got["score"], got["grounding"])[dim]
                    if v is not None:
                        vals.append(v)
                cell[pk[:2]] = vals
            if cell["P+"] and cell["P-"]:
                rows.append((t, np.mean(cell["P+"]), np.mean(cell["P-"]), cell))
        lines.append(f"## {name}: {len(rows)}/{len(plan)} targets scored in both cells")
        diffs = {}
        for cat in CATS:
            sel = [(t, a, b) for t, a, b, _ in rows if t["progress"] == cat]
            d = np.array([a - b for _, a, b in sel])
            g = [(t["update"], t["episode"]) for t, _, _ in sel]
            diffs[cat] = (d, g)
            mu, lo, hi = boot(d, g)
            lines.append(f"{cat} (n={len(sel)}): P+ {np.mean([a for _, a, _ in sel]):.3f}, "
                         f"P- {np.mean([b for _, _, b in sel]):.3f}, P+ - P- {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        # difference-in-differences: does the reference penalise off-reference actions more than advancing ones?
        (d1, g1), (d0, g0) = diffs[CATS[0]], diffs[CATS[1]]
        mu, lo, hi = boot_did(d1, g1, d0, g0)
        lines.append(f"off - advancing difference of (P+ - P-): {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        noise = [abs(x - y) for _, _, _, c in rows for v in c.values() for x, y in zip(v, v[1:])]
        lines.append(f"repeat noise: mean |consecutive repeats| {np.mean(noise):.3f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")


def _clusters(d, g):
    by = {}
    for x, k in zip(d, g):
        by.setdefault(k, []).append(x)
    return by


def boot(d, g, n=5000, seed=0):
    import numpy as np
    rng, by = np.random.default_rng(seed), _clusters(d, g)
    keys = list(by)
    stats = [np.mean(np.concatenate([by[keys[i]] for i in rng.integers(0, len(keys), len(keys))])) for _ in range(n)]
    return float(np.mean(d)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def boot_did(d1, g1, d0, g0, n=5000, seed=0):
    """Episodes resampled jointly for both categories."""
    import numpy as np
    rng = np.random.default_rng(seed)
    b1, b0 = _clusters(d1, g1), _clusters(d0, g0)
    keys = sorted(set(b1) | set(b0))
    stats = []
    for _ in range(n):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        x1 = [v for k in pick for v in b1.get(k, [])]
        x0 = [v for k in pick for v in b0.get(k, [])]
        if x1 and x0:
            stats.append(np.mean(x1) - np.mean(x0))
    return float(np.mean(d1) - np.mean(d0)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max_episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--env_file", default="~/ParaCook/.env.shuaiapi.local")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
