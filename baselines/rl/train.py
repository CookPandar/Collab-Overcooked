"""RL training of the Chef and Assistant LoRA adapters on Collab-Overcooked.

    python -m baselines.rl.train --config baselines/configs/rl/level2_agentgym.yaml
    torchrun --nproc_per_node 2 -m baselines.rl.train --config ...      # data parallel over 2 GPUs

Every update: collect `episodes_per_update` episodes with the current adapters (vLLM), compute per-call
rewards and advantages, run one PPO update (agentgym or marshal objective), then serve the new adapters.
The vLLM server and the trained model stay alive for the whole run. The run resumes from `output_dir`.
"""

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml

from collab_overcooked.config import TASK_LEVELS

from .advantages import compute_advantages
from .policy import ROLE_ADAPTERS, LoraPolicy
from .serving import VLLMServer
from .update import next_lr_scale, ppo_update

DEFAULTS = {
    "model": None,                     # base model directory (Qwen3-1.7B)
    "adapters": None,                  # {chef: dir, assistant: dir}: initial LoRA adapters, also the KL reference
    "output_dir": None,
    "tasks": None,                     # list of orders; default: all tasks of `level`
    "level": 2,
    "updates": 50,
    "episodes_per_update": 8,
    "horizon": 50,
    "horizon_schedule": [],            # [[last_update, horizon], ...], e.g. [[5, 20], [10, 35]]
    "response_protocol": "action_only",  # or think_action (needs adapters trained on that format)
    "history_window": 0,
    "reflection": False,
    "temperature": 0.7,
    "max_new_tokens": 128,
    "objective": "agentgym",           # agentgym | marshal
    "lr": 1e-5,
    "kl_coef": 0.02,
    "adaptive_lr": None,               # e.g. {kl_threshold: 0.05, decay: 0.5, min_scale: 0.1}
    "gamma": 0.99,
    "advantage_clip": 10.0,
    "reward": {"format_penalty": 0.2, "validator_penalty": 0.1, "ites_weight": 1.0, "tes_beta": 0.95},
    "terminal_reward": 2.0,
    "seed": 261008,
    "checkpoint_every": 20,            # keep these updates; the latest two are always kept for recovery
    "rollout_workers": 8,
    "tokens_per_batch": 8192,          # tokens per forward pass of the update (lower it if out of memory)
    "vllm_python": "python",           # interpreter of the environment where vLLM is installed
    "vllm_port": 18000,
    "vllm_kv_cache_gib": 8,
    "max_model_len": 8192,
}
ROLES = {0: "chef", 1: "assistant"}


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = dict(DEFAULTS, **(yaml.safe_load(f) or {}))
    for item in overrides:
        key, value = item.split("=", 1)
        cfg[key] = yaml.safe_load(value)
    cfg["tasks"] = cfg["tasks"] or TASK_LEVELS[f"level_{cfg['level']}"]
    for key in ("model", "adapters", "output_dir"):
        if not cfg[key]:
            raise ValueError(f"config needs `{key}`")
    return cfg


def horizon_at(cfg, update):
    for last, horizon in sorted(cfg["horizon_schedule"]):
        if update <= last:
            return horizon
    return cfg["horizon"]


# ---- rollout workers (separate processes: the benchmark agents are pure Python) ---------------------
_TOKENIZER = None


def _init_worker(model):
    global _TOKENIZER
    sys.stdout = open(os.devnull, "w")  # the benchmark agents print their dialogues
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(model)


def _run_episode(job):
    from .rollout import VLLMPolicy, run_episode
    cfg = job["cfg"]
    policy = VLLMPolicy(job["url"], _TOKENIZER, job["served"], cfg["temperature"], cfg["max_new_tokens"])
    started = time.time()
    episode = run_episode(policy, job["order"], job["horizon"], seed=job["seed"],
                          response_protocol=cfg["response_protocol"], history_window=cfg["history_window"],
                          reflection=cfg["reflection"], reward=cfg["reward"], terminal_reward=cfg["terminal_reward"])
    episode.update(episode_id=job["episode_id"], seconds=time.time() - started)
    return episode


# ---- data parallel helpers ----------------------------------------------------------------------------
def _dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _gather(obj, world):
    if world == 1:
        return [obj]
    result = [None] * world
    torch.distributed.all_gather_object(result, obj)
    return result


def _barrier(world):
    if world > 1:
        torch.distributed.barrier()


# ---- training loop -------------------------------------------------------------------------------------
def summarize(episodes):
    calls = [t for e in episodes for t in e["transitions"]]
    n = max(1, len(episodes))
    return {
        "success_rate": sum(e["success"] for e in episodes) / n,
        "steps": sum(e["steps"] for e in episodes) / n,
        "episode_return": sum(t["reward"] for t in calls) / n,
        "ites_per_episode": sum(t["sequence_reward"] for t in calls) / n,
        "calls_per_episode": len(calls) / n,
        "format_error_rate": sum(t["format_reward"] < 0 for t in calls) / max(1, len(calls)),
        "validator_error_rate": sum(t["validator_reward"] < 0 for t in calls) / max(1, len(calls)),
    }


def save_adapters(policy, directory):
    tmp = Path(str(directory) + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for agent_index in policy.trainable:
        policy.save(agent_index, tmp)  # PEFT writes <tmp>/<adapter name>/
    shutil.rmtree(directory, ignore_errors=True)
    tmp.replace(directory)
    return {i: Path(directory) / ROLE_ADAPTERS[i] for i in policy.trainable}


def prune_checkpoints(root, keep_every, current):
    for path in Path(root).glob("u[0-9]*"):
        update = int(path.name[1:])
        if update < current - 1 and update % keep_every:
            shutil.rmtree(path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value overrides of the config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    rank, world = _dist()
    local_gpu = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    server_gpu = visible[local_gpu] if visible != [""] else str(local_gpu)
    device = f"cuda:{local_gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_gpu)

    out = Path(cfg["output_dir"])
    checkpoints = out / "checkpoints"
    if rank == 0:
        checkpoints.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _barrier(world)
    initial = {i: Path(cfg["adapters"][ROLES[i]]) for i in ROLES}
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"update": 0, "lr_scale": 1.0}
    current = ({i: checkpoints / f"u{state['update']:05d}" / ROLE_ADAPTERS[i] for i in ROLES}
               if state["update"] else initial)

    server = VLLMServer(cfg["vllm_python"], cfg["model"], cfg["vllm_port"] + local_gpu, server_gpu,
                        cfg["vllm_kv_cache_gib"], cfg["max_model_len"], log_path=out / f"vllm_rank{rank}.log")
    try:
        served = server.load_adapters(current, f"u{state['update']}")
        policy = LoraPolicy(cfg["model"], current, initial, device=device)
        pool = multiprocessing.get_context("spawn").Pool(cfg["rollout_workers"], _init_worker, (cfg["model"],))
        n_tasks, per_update = len(cfg["tasks"]), cfg["episodes_per_update"]
        for update in range(state["update"] + 1, cfg["updates"] + 1):
            started = time.time()
            horizon = horizon_at(cfg, update)
            jobs = []
            for e in range(rank, per_update, world):
                episode_id = (update - 1) * per_update + e
                jobs.append({"cfg": cfg, "url": server.url, "served": served, "horizon": horizon,
                             "order": cfg["tasks"][episode_id % n_tasks], "episode_id": episode_id,
                             "seed": cfg["seed"] + episode_id * 100003})
            episodes = sorted((e for part in _gather(pool.map(_run_episode, jobs), world) for e in part),
                              key=lambda e: e["episode_id"])
            collect_seconds = time.time() - started

            transitions = [dict(t, episode=e["episode_id"]) for e in episodes for t in e["transitions"]]
            advantages = compute_advantages(transitions, cfg["gamma"], cfg["advantage_clip"])
            samples = [{"agent_index": t["agent_index"], "advantage": a,
                        "prompt_ids": torch.tensor(t["prompt_ids"]), "response_ids": torch.tensor(t["response_ids"])}
                       for t, a in zip(transitions, advantages)]
            update_started = time.time()
            server.sleep()  # the GPU memory of vLLM goes to the update
            lr = cfg["lr"] * state["lr_scale"]
            metrics = ppo_update(policy, samples, objective=cfg["objective"], temperature=cfg["temperature"], lr=lr,
                                 tokens_per_batch=cfg["tokens_per_batch"], kl_coef=cfg["kl_coef"],
                                 seed=cfg["seed"] + update, rank=rank, world=world)
            torch.cuda.empty_cache()
            server.wake_up()
            update_seconds = time.time() - update_started
            state = {"update": update, "lr_scale": next_lr_scale(cfg["adaptive_lr"], state["lr_scale"],
                                                                 metrics["update_kl"])}

            directory = checkpoints / f"u{update:05d}"
            if rank == 0:
                save_adapters(policy, directory)
            _barrier(world)
            served = server.load_adapters({i: directory / ROLE_ADAPTERS[i] for i in ROLES}, f"u{update}")
            if rank == 0:
                record = {"update": update, "horizon": horizon, "lr": lr, **summarize(episodes),
                          "orders": [e["order"] for e in episodes],
                          **{f"update/{k}": v for k, v in metrics.items()},
                          "time/collect": collect_seconds, "time/update": update_seconds,
                          "time/total": time.time() - started}
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                state_path.write_text(json.dumps(state))
                prune_checkpoints(checkpoints, cfg["checkpoint_every"], update)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()
                                  if k != "orders"}), flush=True)
        pool.close()
    finally:
        server.close()


if __name__ == "__main__":
    main()
