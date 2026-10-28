"""Collect SFT data: a teacher model plays the benchmark and its clean LLM calls become training examples.

    python -m baselines.sft.collect --teacher gpt-4o --protocol think_action --level 1 2 --episodes 10 \
        --output data/sft/gpt4o_think.jsonl

The teacher is any OpenAI-compatible chat model (`--base-url` for a local server). Every call of an episode is
one example {role, order, messages, response}; the prompts are exactly the ones the RL policy sees with the
same `--protocol`. Kept calls: well-formed replies with no format or validator error, from successful
episodes unless --keep-failed. With `think_action` the responses contain Think / Recent Goal / Action
(SFT with reasoning); with `action_only` a single Action field.
"""

import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

from collab_overcooked.config import LOCAL_SERVER_API_KEY, TASK_LEVELS, get_openai_api_key

ROLES = ("chef", "assistant")
BAD_MODES = {"mixed", "multi_embodied", "malformed"}


class TeacherPolicy:
    def __init__(self, model, base_url=None, temperature=0.7, max_tokens=512, reasoning_effort=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=LOCAL_SERVER_API_KEY if base_url else get_openai_api_key(), base_url=base_url,
                             max_retries=5, timeout=300)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.reasoning_effort = reasoning_effort
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def generate(self, agent_index, messages, seed):
        kwargs = {}  # reasoning models reject temperature/max_tokens, so both are optional
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens:
            kwargs["max_completion_tokens"] = self.max_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        response = self.client.chat.completions.create(model=self.model, messages=messages, seed=seed, **kwargs)
        text = response.choices[0].message.content or ""
        self.usage["calls"] += 1
        if response.usage:
            self.usage["prompt_tokens"] += response.usage.prompt_tokens or 0
            self.usage["completion_tokens"] += response.usage.completion_tokens or 0
        return text, {"messages": messages, "prompt_ids": [], "response_ids": []}


def _episode(job):
    from baselines.rl.rollout import run_episode
    sys.stdout = open(os.devnull, "w")  # the benchmark agents print their dialogues
    args = job["args"]
    if args.env_file:
        from baselines.rl.think_judge import read_env_file
        os.environ.update(read_env_file(args.env_file))
    policy = TeacherPolicy(args.teacher, args.base_url, args.temperature, args.max_tokens, args.reasoning_effort)
    episode = run_episode(policy, job["order"], args.horizon, seed=job["seed"], response_protocol=args.protocol)
    examples = [{"role": ROLES[t["agent_index"]], "order": job["order"], "episode": job["episode"],
                 "timestep": t["timestep"], "call_type": t["call_type"], "messages": t["messages"],
                 "response": t["response"]}
                for t in episode["transitions"]
                if t["action_mode"] not in BAD_MODES and t["format_reward"] == 0 and t["validator_reward"] == 0]
    return {"order": job["order"], "episode": job["episode"], "success": episode["success"],
            "steps": episode["steps"], "calls": len(episode["transitions"]), "usage": policy.usage,
            "examples": examples}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", required=True, help="chat model name, e.g. gpt-4o")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible server of a local teacher")
    parser.add_argument("--protocol", default="action_only", choices=["action_only", "think_action"])
    parser.add_argument("--level", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--tasks", nargs="*", default=None, help="orders (default: all tasks of --level)")
    parser.add_argument("--episodes", type=int, default=10, help="episodes per task")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7, help="negative: omit (reasoning models)")
    parser.add_argument("--max-tokens", type=int, default=512, help="0: omit")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--env-file", default=None, help="file with OPENAI_BASE_URL / OPENAI_API_KEY")
    parser.add_argument("--keep-failed", action="store_true", help="also keep clean calls of failed episodes")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.temperature < 0:
        args.temperature = None
    tasks = args.tasks or [t for level in args.level for t in TASK_LEVELS[f"level_{level}"]]
    jobs = [{"args": args, "order": order, "episode": e, "seed": args.seed + 100003 * (i * args.episodes + e)}
            for i, order in enumerate(tasks) for e in range(args.episodes)]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kept, summary = 0, []
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool, output.open("w") as f:
        for result in pool.imap_unordered(_episode, jobs):
            if result["success"] or args.keep_failed:
                for example in result["examples"]:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                kept += len(result["examples"])
            summary.append({k: v for k, v in result.items() if k != "examples"})
            print(f"{result['order']} #{result['episode']}: success={result['success']} steps={result['steps']} "
                  f"clean calls {len(result['examples'])}/{result['calls']} tokens {result['usage']}", flush=True)
    successes = sum(r["success"] for r in summary)
    usage = {k: sum(r["usage"][k] for r in summary) for k in ("calls", "prompt_tokens", "completion_tokens")}
    print(f"{successes}/{len(summary)} successful episodes, {kept} examples -> {output}; usage {usage}")
    Path(str(output) + ".episodes.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
