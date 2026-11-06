"""Offline gate for advantage estimators: recompute per-call advantages with the trainer's advantage code on saved
rollouts and check that they rank calls sensibly in FAILED episodes (no success signal: only progress and penalties).

    python runs/think_judge/adv_gate.py --set advantage_estimator=grpo group_size=8 grpo_norm=none ... \
        --rollouts runs/think_judge/rl/l2p_ppo_warm/rollouts/u0000{1,25}.jsonl.gz --pool 1
    python runs/think_judge/adv_gate.py --config runs/think_judge/configs/l2g_drgrpo_fix.yaml ...

Checks (failed episodes only):
  (a) mean A(kitchen action with ITES > 0) > mean A(message or wait) > mean A(rejected kitchen action);
  (b) |corr(A, timestep)| < --max-corr (default 0.15);
  (d) over (failed episode, role) trajectories: corr(mean A of the trajectory's messages / waits, number of its
      kitchen actions) > --min-act-corr (default -0.3): the trajectory-level credit shared by all calls must not
      reward "act less, talk / wait more" (the GRPO collapse; nan = no shared term = pass);
  (c) reported only: fraction of GRPO (group, role) score groups with ~zero spread.
Grouping: as the trainer, a GRPO group is `group_size` episodes of the same order played by the same policy; saved
rollouts of non-GRPO runs interleave orders, so episodes are grouped by (pool of --pool consecutive files, order),
in chunks of group_size. The gae estimator is evaluated with V = 0 (the call-level value head starts at zero; the
token-level critic of l2p_ppo_warm had held-out EV ~0); --logged uses the advantages saved in the rollouts.
Exit code 0 = pass, 1 = fail.
"""
import argparse
import gzip
import json
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import yaml  # noqa: E402

from baselines.rl.advantages import (compute_advantages, gae_advantages, grpo_advantages,  # noqa: E402
                                     local_penalty, normalize_advantages)

# the trainer's defaults of the keys used here (baselines/rl/train.py DEFAULTS; importing train needs torch)
DEFAULTS = {"advantage_estimator": "reinforce", "group_size": 4, "grpo_gates": [], "grpo_norm": "std",
            "grpo_score": "all", "grpo_min_spread": None, "grpo_step_weight": 0.0, "grpo_step_gamma": 0.95, "grpo_step_norm": None, "gamma": 0.99, "advantage_clip": 10.0,
            "intra_gamma": 1.0, "advantage_norm": None, "advantage_std_floor": 0.0,
            "critic": {"gamma": 1.0, "lam": 1.0, "level": "call", "gates": [], "local_penalties": False,
                       "std_floor": 0.0}}


def load(files, pool):
    """Transitions as the trainer builds them (reward = format + validator + ITES + collab + terminal + redundancy)."""
    transitions, summaries = [], {}
    for k, path in enumerate(files):
        for line in gzip.open(path, "rt"):
            r = json.loads(line)
            if "episode_summary" in r:
                summaries[(path, r["episode_summary"])] = dict(r, pool=k // pool)
                continue
            r["sequence_reward"] = r["ites"]
            r["collab_reward"] = r.get("collab_reward") or 0.0
            r["redundancy_reward"] = r.get("redundancy_reward") or 0.0
            r["reward"] = (r["format_reward"] + r["validator_reward"] + r["ites"] + r["collab_reward"]
                           + r["terminal_reward"] + r["redundancy_reward"])
            r["file"] = path
            transitions.append(r)
    episodes = {}  # (file, episode id) -> new unique id
    for key in summaries:
        episodes[key] = len(episodes)
    chunks = defaultdict(list)
    for key in sorted(summaries, key=lambda k: episodes[k]):
        s = summaries[key]
        chunks[(s["pool"], s["order"])].append(key)
    return transitions, summaries, episodes, chunks


def assign_groups(transitions, summaries, episodes, chunks, group_size):
    group = {}
    for (pool, order), keys in chunks.items():
        for i, key in enumerate(keys):
            group[key] = (pool, order, i // group_size)
    for t in transitions:
        key = (t["file"], t["episode"])
        t["episode"] = episodes[key]
        t["group"] = group[key]
        t["success"] = summaries[key]["success"]
    return transitions


def advantages(cfg, transitions, logged=False, stats=None):
    """Per-call advantages as baselines/rl/train.py computes them (before the think-reward mixing)."""
    if logged:
        return [t["adv_action"] for t in transitions]
    estimator, clip = cfg["advantage_estimator"], cfg["advantage_clip"]
    if estimator == "grpo":
        adv = grpo_advantages(transitions, gates=cfg["grpo_gates"], norm=cfg["grpo_norm"], score=cfg["grpo_score"],
                              min_spread=cfg["grpo_min_spread"], stats=stats,
                              step_weight=cfg["grpo_step_weight"], step_gamma=cfg["grpo_step_gamma"],
                              step_norm=cfg["grpo_step_norm"])
    elif estimator == "gae":
        critic = cfg["critic"]
        if critic.get("level", "call") != "call":
            print("note: token-level critic evaluated as call level with V = 0")
        adv, _ = gae_advantages(transitions, [0.0] * len(transitions), critic["gamma"], critic["lam"],
                                local_penalties=critic.get("local_penalties", False),
                                std_floor=critic.get("std_floor", 0.0))
    else:
        adv = compute_advantages(transitions, cfg["gamma"], clip, cfg["intra_gamma"])
    adv = [max(-clip, min(clip, a)) for a in adv]
    if cfg["advantage_norm"] in ("role", "role_sign"):
        adv = normalize_advantages(transitions, adv, clip, keep_sign=cfg["advantage_norm"] == "role_sign",
                                   std_floor=cfg["advantage_std_floor"])
    return adv


def category(t):
    if t["format_reward"] < 0:
        return "format_err"
    mode = t["action_mode"]
    if mode in ("communication", "wait"):
        return "msg_wait"
    if mode == "planner_main":
        if t["validator_reward"] < 0:
            return "rejected"
        return "progress" if t["ites"] > 1e-9 else "regress" if t["ites"] < -1e-9 else "noprogress"
    return "other"


def corr(a, b):
    if len(a) < 2:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    sa = math.sqrt(sum((x - ma) ** 2 for x in a) / len(a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b) / len(b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a) / (sa * sb) if sa * sb else float("nan")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def zero_spread_frac(transitions, cfg, eps=1e-6):
    """Fraction of (group, role) score groups (size > 1) whose GRPO scores span <= eps, for the configured score."""
    local = local_penalty if cfg["grpo_score"] == "outcome" else (lambda t: 0.0)
    score = defaultdict(float)
    for t in transitions:
        score[(t["episode"], t["agent_index"])] += t["reward"] - local(t)
    members = defaultdict(list)
    group = {(t["episode"], t["agent_index"]): (t["group"], t["agent_index"]) for t in transitions}
    for key, value in score.items():
        members[group[key]].append(value)
    sized = [v for v in members.values() if len(v) > 1]
    return sum(max(v) - min(v) <= eps for v in sized) / max(1, len(sized)), len(sized)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="run config (yaml); only the advantage keys are used")
    p.add_argument("--set", nargs="*", default=[], help="key=value overrides (yaml values)")
    p.add_argument("--rollouts", nargs="+", required=True)
    p.add_argument("--pool", type=int, default=1, help="consecutive rollout files pooled for grouping")
    p.add_argument("--logged", action="store_true", help="use the advantages saved in the rollouts")
    p.add_argument("--max-corr", type=float, default=0.15)
    p.add_argument("--min-act-corr", type=float, default=-0.3)
    p.add_argument("--name", default="")
    args = p.parse_args(argv)
    cfg = json.loads(json.dumps(DEFAULTS))
    if args.config:
        with open(args.config) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in loaded.items() if k in DEFAULTS and k != "critic"})
        cfg["critic"].update(loaded.get("critic") or {})
    for item in args.set:
        key, value = item.split("=", 1)
        value = yaml.safe_load(value)
        if key == "critic":
            cfg["critic"].update(value)
        else:
            cfg[key] = value
    transitions, summaries, episodes, chunks = load(args.rollouts, args.pool)
    transitions = assign_groups(transitions, summaries, episodes, chunks, cfg["group_size"])
    stats = {}
    adv = advantages(cfg, transitions, args.logged, stats)
    fail = [i for i, t in enumerate(transitions) if not t["success"]]
    by_cat = defaultdict(list)
    by_role_cat = defaultdict(list)
    for i in fail:
        c = category(transitions[i])
        by_cat[c].append(adv[i])
        by_role_cat[(transitions[i]["agent_index"], c)].append(adv[i])
    n_ep = len(summaries)
    n_succ = sum(s["success"] for s in summaries.values())
    est = cfg["advantage_estimator"]
    print(f"== {args.name or est}  estimator={est}" + (" (logged)" if args.logged else "")
          + (f" G={cfg['group_size']} norm={cfg['grpo_norm']} score={cfg['grpo_score']} "
             f"min_spread={cfg['grpo_min_spread']} step_weight={cfg['grpo_step_weight']} step_norm={cfg['grpo_step_norm']}" if est == "grpo" else "")
          + (f" critic={cfg['critic']}" if est == "gae" else ""))
    print(f"   episodes {n_ep} (success {n_succ}), calls {len(transitions)}, calls in failed episodes {len(fail)}")
    for c in ("progress", "noprogress", "regress", "msg_wait", "rejected", "format_err", "other"):
        if by_cat[c]:
            roles = "  ".join(f"r{r}={mean(by_role_cat[(r, c)]):+.3f}(n={len(by_role_cat[(r, c)])})"
                              for r in (0, 1) if by_role_cat[(r, c)])
            print(f"   fail {c:<11} n={len(by_cat[c]):5d} mean A={mean(by_cat[c]):+.3f}   {roles}")
    a_prog, a_msg, a_rej = mean(by_cat["progress"]), mean(by_cat["msg_wait"]), mean(by_cat["rejected"])
    gate_a = a_prog > a_msg > a_rej
    c_t = corr([adv[i] for i in fail], [transitions[i]["timestep"] for i in fail])
    gate_b = abs(c_t) < args.max_corr
    c_ites = corr([adv[i] for i in fail], [transitions[i]["ites"] for i in fail])
    succ = [i for i, t in enumerate(transitions) if t["success"]]
    traj = defaultdict(list)
    for i in fail:
        traj[(transitions[i]["episode"], transitions[i]["agent_index"])].append(i)
    # trajectory-level (shared) part of the credit: the mean A of the trajectory's messages / waits (calls without
    # a local reward term), against the number of kitchen actions of the trajectory
    rows = [(mean([adv[i] for i in idx if category(transitions[i]) == "msg_wait"]),
             sum(transitions[i]["action_mode"] == "planner_main" for i in idx)) for idx in traj.values()]
    rows = [r for r in rows if r[0] == r[0]]
    c_act = corr([r[0] for r in rows], [r[1] for r in rows])
    gate_d = not c_act < args.min_act_corr  # nan (no trajectory-level term, e.g. reinforce) passes
    zfrac, ngroups = zero_spread_frac(transitions, cfg)
    print(f"   (a) progress {a_prog:+.3f} > msg/wait {a_msg:+.3f} > rejected {a_rej:+.3f}: "
          f"{'PASS' if gate_a else 'FAIL'}")
    print(f"   (b) failed eps corr(A, timestep) = {c_t:+.3f} (|.| < {args.max_corr}): {'PASS' if gate_b else 'FAIL'}"
          f"   [corr(A, ITES) = {c_ites:+.3f}]")
    print(f"   (d) failed trajectories corr(msg/wait A, #kitchen actions) = {c_act:+.3f} (> {args.min_act_corr}): "
          f"{'PASS' if gate_d else 'FAIL'}")
    print(f"   (c) zero-spread score groups: {zfrac:.3f} of {ngroups} (group, role) groups"
          + (f"; dropped by grpo_min_spread: {stats.get('dropped_frac', 0):.3f}"
             f"; calls in step groups >= 2: {stats.get('step_grouped_frac', 0):.3f}" if est == "grpo" else ""))
    if succ:
        print(f"   success eps mean A={mean([adv[i] for i in succ]):+.3f} vs failed {mean([adv[i] for i in fail]):+.3f}")
    ok = gate_a and gate_b and gate_d
    print(f"   GATE: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
