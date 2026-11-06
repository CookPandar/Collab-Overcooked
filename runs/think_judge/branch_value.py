"""Does the think judge track the value of a Think? Branch real continuations from sampled decision states.

    python runs/think_judge/branch_value.py --checkpoint runs/think_judge/rl/l2_sum_w0.5/checkpoints/u00050 \
        --name sum_u50 --gpu 0 --port 18310

1. Base episodes (Level 2, every order) are played with the checkpoint; every LLM call is recorded. The numpy /
   random state is seeded per episode, so replaying the recorded replies reproduces the episode exactly.
2. Decision states: judgeable calls of the base episodes (spread over the episode, before the last 10 steps).
3. At a state h, K replies are sampled (Think z_k with its Action a_k1); for each z_k, M-1 more Actions are sampled
   with z_k fixed as a prefix (a_k2 ...). Every (z_k, a_km) is played to the end of the episode once: the prefix
   replays the recorded replies, call n is forced to the branch reply, later calls are sampled.
4. U = team reward of the calls from n on (format + validator + ITES + terminal of both agents). Then
   Q_A(k,m) = U_km, Q_T(k) = mean_m U_km, V = mean_k Q_T(k).
5. The think judge (gpt-5.6-terra, step-level prompt) scores every branch call twice: as in training (it sees the
   validator verdict and the TES change of the call) and without that outcome. All judge requests are logged.

Writes eval/branch_value/<name>.jsonl (one row per branch) and <name>_base.jsonl; analyse with
branch_analysis.py.
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
SEED_MODULUS = 2**31 - 1
_TOKENIZER = None


def complete(url, model, prompt_ids, seed, max_tokens, temperature=0.7):
    request = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens, "temperature": temperature,
               "seed": seed % SEED_MODULUS, "return_token_ids": True, "skip_special_tokens": True}
    http = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(request).encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(http, timeout=600) as response:
        choice = json.load(response)["choices"][0]
    return choice["text"], [int(i) for i in choice["token_ids"]]


class BranchPolicy:
    """Numbers its calls; calls in `forced` return the given reply, the others are sampled with seed + index.
    A forced call whose prompt differs from the recorded one marks the replay as diverged."""

    def __init__(self, url, tokenizer, served, seed, forced=None, max_tokens=384, temperature=0.7):
        self.url, self.tokenizer, self.served = url, tokenizer, served
        self.seed, self.forced = seed, forced or {}
        self.max_tokens, self.temperature = max_tokens, temperature
        self.index, self.diverged, self.record = 0, False, []

    def generate(self, agent_index, messages, seed):
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                 enable_thinking=False)
        prompt_ids = [int(i) for i in (ids["input_ids"] if hasattr(ids, "keys") else ids)]
        index = self.index
        self.index += 1
        if index in self.forced:
            forced = self.forced[index]
            if forced.get("prompt_ids") is not None and forced["prompt_ids"] != prompt_ids:
                self.diverged = True
            text, response_ids = forced["text"], forced["response_ids"]
        else:
            text, response_ids = complete(self.url, self.served[agent_index], prompt_ids, self.seed + index,
                                          self.max_tokens, self.temperature)
        self.record.append({"agent_index": agent_index, "prompt_ids": prompt_ids, "text": text,
                            "response_ids": response_ids})
        return text, {"prompt_ids": prompt_ids, "response_ids": response_ids, "token_count": len(response_ids),
                      "global_index": index}


def _init_worker():
    global _TOKENIZER
    sys.stdout = open(os.devnull, "w")  # the benchmark agents print their dialogues
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL)


def play(job):
    """One episode with seeded numpy / random; returns the transitions and the policy's call record."""
    import numpy as np
    from baselines.rl.rollout import run_episode
    np.random.seed(job["env_seed"])
    random.seed(job["env_seed"])
    policy = BranchPolicy(job["url"], _TOKENIZER, job["served"], job["sample_seed"],
                          {int(k): v for k, v in job.get("forced", {}).items()})
    episode = run_episode(policy, job["order"], job["horizon"], seed=0, response_protocol="think_action",
                          history_window=0, reflection=False,
                          reward={"format_penalty": 0.2, "validator_penalty": 0.1, "ites_weight": 1.0,
                                  "tes_beta": 0.95}, terminal_reward=2.0)
    for t in episode["transitions"]:
        for key in ("prompt_ids", "response_ids", "messages"):
            t.pop(key, None)
    return dict(job, success=episode["success"], steps=episode["steps"], transitions=episode["transitions"],
                calls=policy.record, diverged=policy.diverged)


def strip_job(result):
    return {k: v for k, v in result.items() if k not in ("forced", "calls", "url", "served")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=18310)
    parser.add_argument("--episodes-per-order", type=int, default=2)
    parser.add_argument("--states-per-episode", type=int, default=6)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--judge-model", default="gpt-5.6-terra")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from baselines.rl.serving import VLLMServer
    from baselines.rl.think_judge import ThinkJudge, action_start_token, judgeable
    from collab_overcooked.config import TASK_LEVELS

    out = ROOT / "eval" / "branch_value"
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rng = random.Random(args.seed)
    server = VLLMServer(str(Path.home() / "vllm/bin/python"), MODEL, args.port, args.gpu, 6, 8192,
                        log_path=out / f"{args.name}.vllm.log")
    try:
        ck = Path(args.checkpoint)
        served = server.load_adapters({0: ck / "Chef", 1: ck / "Assistant"}, "branch")
        pool = multiprocessing.get_context("spawn").Pool(args.workers, _init_worker)
        common = {"url": server.url, "served": served, "horizon": args.horizon}

        # 1. base episodes
        jobs = [dict(common, order=order, base=e, env_seed=args.seed + 1000 * e + j, sample_seed=args.seed * 7 + 1000 * e + j)
                for e in range(args.episodes_per_order) for j, order in enumerate(TASK_LEVELS["level_2"])]
        bases = pool.map(play, jobs)
        with open(out / f"{args.name}_base.jsonl", "w") as f:
            for b in bases:
                f.write(json.dumps(strip_job(b)) + "\n")
        print(f"base episodes: {sum(b['success'] for b in bases)}/{len(bases)} successful", file=sys.stderr, flush=True)

        # 2. decision states and 3. branch replies
        branches = []
        for b_index, base in enumerate(bases):
            candidates = [t for t in base["transitions"] if judgeable(t) and t["timestep"] < args.horizon - 10
                          and t["policy_call"] is not None]
            if not candidates:
                continue
            candidates.sort(key=lambda t: t["policy_call"])
            picks = sorted(rng.sample(range(len(candidates)), min(args.states_per_episode, len(candidates))))
            for p in picks:
                n = candidates[p]["policy_call"]
                call = base["calls"][n]
                model = served[call["agent_index"]]
                for k in range(args.k):
                    text, ids = complete(server.url, model, call["prompt_ids"], rng.randrange(SEED_MODULUS), 384)
                    replies = [(text, ids)]
                    cut = action_start_token(tokenizer, ids, text)
                    for _ in range(args.m - 1):
                        more, more_ids = complete(server.url, model, call["prompt_ids"] + ids[:cut],
                                                  rng.randrange(SEED_MODULUS), max(64, 384 - cut))
                        full = ids[:cut] + more_ids
                        replies.append((tokenizer.decode(full, skip_special_tokens=True), full))
                    for m, (reply, reply_ids) in enumerate(replies):
                        forced = {i: {"text": c["text"], "response_ids": c["response_ids"], "prompt_ids": c["prompt_ids"]}
                                  for i, c in enumerate(base["calls"][:n])}
                        forced[n] = {"text": reply, "response_ids": reply_ids, "prompt_ids": None}
                        branches.append(dict(common, order=base["order"], base=b_index, env_seed=base["env_seed"],
                                             sample_seed=rng.randrange(SEED_MODULUS), state=f"{b_index}:{n}",
                                             call=n, k=k, m=m, think_tokens=cut, forced=forced))
        print(f"{len(branches)} branches from {len(branches) // (args.k * args.m)} states", file=sys.stderr, flush=True)
        results = pool.map(play, branches, chunksize=1)
        pool.close()
    finally:
        server.close()

    # 4. returns from the branch call on, 5. judge scores with and without the outcome of the call
    judge = ThinkJudge(model=args.judge_model, workers=16, env_file="~/ParaCook/.env.shuaiapi.local",
                       log_path=ROOT / "judge_annotations" / f"branch_{args.name}.jsonl")
    judge.context = {"run": args.name, "branch_value": True}
    rows, records = [], []
    for r in results:
        own = [t for t in r["transitions"] if t["policy_call"] is not None]
        target = next((t for t in own if t["policy_call"] == r["call"]), None)
        future = sum(t["reward"] for t in own if t["policy_call"] >= r["call"])
        record = target["judge_record"] if target else None
        rows.append(dict(strip_job(r), transitions=None, U=future, target=target and {
            k: target[k] for k in ("agent_index", "timestep", "call_type", "response", "format_reward",
                                   "validator_reward", "sequence_reward", "reward")},
            judgeable=bool(target and judgeable(target))))
        records.append(record if rows[-1]["judgeable"] else None)
    for variant in ("with_outcome", "no_outcome"):
        jobs = [(i, dict(rec, validations=[], validated_action=None) if variant == "no_outcome" else rec)
                for i, rec in enumerate(records) if rec is not None]
        by_order = {}
        for i, rec in jobs:
            by_order.setdefault(rows[i]["order"], []).append((i, rec))
        for order, items in by_order.items():
            scores = judge.judge_records([rec for _, rec in items], order)
            for (i, _), s in zip(items, scores):
                rows[i][f"judge_{variant}"] = s.get("score")
                rows[i][f"follows_{variant}"] = s.get("action_follows")
    with open(out / f"{args.name}.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps({"branches": len(rows), "diverged": sum(r["diverged"] for r in rows),
                      "judge_usage": judge.usage}), flush=True)


if __name__ == "__main__":
    main()
