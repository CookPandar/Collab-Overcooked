"""Teacher-forced step accuracy of single-level SFT models: on teacher states (runs/teacher_l456/data), does the
model's greedy action equal the teacher's? Separates "cannot fit / imitate the teacher" from "imitates each step
but compounding errors in rollouts".

Per (model, role, split) samples up to --rows calls (fixed seed); split "held" = dev + test orders. Each record
keeps the relative position of the call in its teacher episode, so accuracy can be read by early / late steps.
usage (from ~/Collab-Overcooked-private): imitation_acc.py --gpu G --port P --out OUT.jsonl NAME=SFT_DIR:LEVEL:PROTOCOL ...
"""
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from baselines.rl.credit import is_collab, is_wait, normalize_action
from baselines.rl.rollout import VLLMPolicy
from baselines.rl.serving import VLLMServer

DATA = Path("runs/teacher_l456/data")
MODEL = "/datacache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
ROLES = {"chef": 0, "assistant": 1}


def load_rows(protocol, level, role, splits, n, rng):
    rows = []
    for split in splits:
        path = DATA / protocol / f"l{level}_{split}_{role}.jsonl"
        if path.exists():
            rows += [json.loads(line) for line in open(path)]
    last = defaultdict(int)  # (order, episode) -> last timestep of the teacher episode
    for row in rows:
        key = (row["order"], row["episode"])
        last[key] = max(last[key], row["timestep"])
    for row in rows:
        row["position"] = row["timestep"] / max(1, last[(row["order"], row["episode"])])
    return rng.sample(rows, min(n, len(rows)))


def kind(action):
    return "wait" if is_wait(action) else "collab" if is_collab(action) else "physical"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    server = VLLMServer("/home/zhangshuwen/vllm/bin/python", MODEL, args.port, args.gpu, 8, 8192,
                        log_path=args.out + ".vllm.log")
    done = set()
    if Path(args.out).exists():
        done = {json.loads(line)["model"] for line in open(args.out)}
    try:
        for k, spec in enumerate(args.models):
            name, rest = spec.split("=", 1)
            sft, level, protocol = rest.rsplit(":", 2)
            if name in done:
                continue
            served = server.load_adapters({0: Path(sft) / "Chef", 1: Path(sft) / "Assistant"}, f"m{k}")
            policy = VLLMPolicy(server.url, tokenizer, served, temperature=0.0,
                                max_tokens=128 if protocol == "action_only" else 384)
            jobs = []
            for role, agent in ROLES.items():
                for split, splits in (("train", ["train"]), ("held", ["dev", "test"])):
                    rng = random.Random(f"{name}/{role}/{split}")
                    jobs += [(agent, split, row) for row in load_rows(protocol, int(level), role, splits, args.rows, rng)]

            def run(job):
                agent, split, row = job
                text, _ = policy.generate(agent, row["messages"], seed=0)
                target, predicted = normalize_action(row["response"]), normalize_action(text)
                return {"model": name, "level": int(level), "role": "chef" if agent == 0 else "assistant",
                        "split": split, "order": row["order"], "position": row["position"],
                        "call_type": row["call_type"], "kind": kind(target), "target": target,
                        "predicted": predicted, "match": target == predicted}

            with ThreadPoolExecutor(args.workers) as pool:
                results = list(pool.map(run, jobs))
            with open(args.out, "a") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
            acc = sum(r["match"] for r in results) / len(results)
            print(f"{name}: {len(results)} calls, exact match {acc:.3f}", flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
