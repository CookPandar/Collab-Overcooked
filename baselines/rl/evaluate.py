"""Evaluate RL checkpoints (or the SFT adapters) on the tasks of a level with the training rollout code.

    python -m baselines.rl.evaluate --model QWEN3_1_7B --checkpoints runs/x/checkpoints/u00010 runs/x/checkpoints/u00050 \
        --level 3 --episodes-per-task 4 --horizon 50 --response-protocol think_action --output runs/x/eval_l3.jsonl

A checkpoint is a directory with Chef/ and Assistant/ adapters. Appends one JSON line per checkpoint (success
rate per task, mean steps / ITES / error rates) and one per episode (to `<output>.episodes.jsonl`).
"""

import argparse
import json
import multiprocessing
import time
from pathlib import Path

from collab_overcooked.config import TASK_LEVELS

from .serving import VLLMServer
from .train import _init_worker, _run_episode, summarize


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--episodes-per-task", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--response-protocol", default="think_action")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=18900)
    parser.add_argument("--kv-cache-gib", type=float, default=12)
    parser.add_argument("--vllm-python", default="python")
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--save-trace", action="store_true",
                        help="also write each call's observation, validated action and validator messages")
    args = parser.parse_args(argv)

    tasks = TASK_LEVELS[f"level_{args.level}"]
    cfg = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
           "response_protocol": args.response_protocol, "history_window": 0, "reflection": False,
           "reward": {"format_penalty": 0.2, "validator_penalty": 0.1, "ites_weight": 1.0, "tes_beta": 0.95},
           "terminal_reward": 2.0}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    server = VLLMServer(args.vllm_python, args.model, args.port, args.gpu, args.kv_cache_gib, 8192,
                        log_path=output.with_suffix(".vllm.log"))
    try:
        pool = multiprocessing.get_context("spawn").Pool(args.workers, _init_worker, (args.model,))
        for k, checkpoint in enumerate(args.checkpoints):
            started = time.time()
            served = server.load_adapters({0: Path(checkpoint) / "Chef", 1: Path(checkpoint) / "Assistant"}, f"e{k}")
            jobs = [{"cfg": cfg, "url": server.url, "served": served, "horizon": args.horizon, "order": order,
                     "episode_id": i * len(tasks) + j, "seed": args.seed + (i * len(tasks) + j) * 100003}
                    for i in range(args.episodes_per_task) for j, order in enumerate(tasks)]
            episodes = pool.map(_run_episode, jobs)
            per_task = {order: sum(e["success"] for e in episodes if e["order"] == order) / args.episodes_per_task
                        for order in tasks}
            record = {"checkpoint": str(checkpoint), "level": args.level, "episodes": len(episodes),
                      "temperature": args.temperature, "horizon": args.horizon, **summarize(episodes),
                      "success_per_task": per_task, "seconds": time.time() - started}
            with output.open("a") as f:
                f.write(json.dumps(record) + "\n")
            with Path(str(output) + ".episodes.jsonl").open("a") as f:
                for e in episodes:
                    f.write(json.dumps({"checkpoint": str(checkpoint), "order": e["order"], "success": e["success"],
                                        "steps": e["steps"],
                                        "calls": [{"agent_index": t["agent_index"], "timestep": t["timestep"],
                                                   "call_type": t["call_type"], "response": t["response"],
                                                   "format_reward": t["format_reward"],
                                                   "validator_reward": t["validator_reward"],
                                                   "ites": t["sequence_reward"],
                                                   **({k: t["judge_record"][k] for k in
                                                       ("observation", "validated_action", "validations")}
                                                      if args.save_trace else {})} for t in e["transitions"]]})
                            + "\n")
            print(json.dumps({k: v for k, v in record.items() if k != "success_per_task"}), flush=True)
        pool.close()
    finally:
        server.close()


if __name__ == "__main__":
    main()
