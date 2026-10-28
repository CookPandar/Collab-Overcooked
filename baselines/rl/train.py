"""RL training of the Chef and Assistant LoRA adapters on Collab-Overcooked.

    python -m baselines.rl.train --config baselines/configs/rl/level2_agentgym.yaml
    torchrun --nproc_per_node 2 -m baselines.rl.train --config ...      # data parallel over 2 GPUs

Every update: collect `episodes_per_update` episodes with the current adapters (vLLM), compute per-call
rewards and advantages, run one PPO update (agentgym or marshal objective), then serve the new adapters.
The vLLM server and the trained model stay alive for the whole run. The run resumes from `output_dir`.
"""

import argparse
import gzip
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

from .advantages import (centered_judge_scores, check_gates, compute_advantages, gae_advantages, gated,
                         grpo_advantages, normalize_advantages, ratio_scale, redundant_calls, think_advantages,
                         think_weight, token_gae_advantages)
from .rule_rewards import extract, rewards, rule_signal
from .policy import CRITIC_ADAPTERS, ROLE_ADAPTERS, LoraPolicy
from .serving import VLLMServer
from .think_judge import ThinkJudge, judge_episode, judgeable
from .update import critic_update, critic_values, next_lr_scale, ppo_update

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
    # progress-gated horizon (replaces horizon_schedule), e.g. {stages: [60, 90, 120, 150], threshold: 0.4, window: 5,
    # min_updates: 3, max_updates: 15}: the next stage starts once the mean TES (ites_per_episode / 2, in [0, 1]) of
    # the last `window` updates of the stage reaches `threshold`, or after max_updates updates in the stage
    "horizon_adaptive": None,
    "response_protocol": "action_only",  # or think_action (needs adapters trained on that format)
    "history_window": 0,
    "reflection": False,
    "temperature": 0.7,
    "max_new_tokens": 128,
    "objective": "agentgym",           # agentgym | marshal
    "lr": 1e-5,
    "kl_coef": 0.02,
    "entropy_coef": 0.0,               # verl actor entropy_coeff (AgentGym-RL: 0.001)
    "adaptive_lr": None,               # e.g. {kl_threshold: 0.05, decay: 0.5, min_scale: 0.1}
    # KL anomaly guard, e.g. {min_token_logprob: -9.21, max_update_kl: 0.05, max_ref_kl: 1.0, lr_decay: 0.5,
    # min_lr_scale: 0.1}: tokens with old log-prob < min_token_logprob are masked out of the loss; an update whose
    # update KL (k3, policy after vs before) or clamped token KL to the reference exceeds its threshold is undone
    # (the adapters before the update are restored) and the learning rate scale is multiplied by lr_decay
    "kl_guard": None,
    "gamma": 0.99,
    "advantage_clip": 10.0,
    # reinforce: the advantages above (no baseline). The AgentGym-RL estimators (verl core_algos):
    # grpo: each role's episode return normalized within a group of `group_size` episodes of the same order;
    # gae: PPO with a critic per role (LoRA + value head, `critic` settings), GAE whitened per role; critic.level
    # "call": one value per call (the prompt), "token": one value per response token as verl, the calls of a role
    # chained as one multi-turn sequence (AgentGym-RL); critic.warmup: the first updates train the critic only.
    "advantage_estimator": "reinforce",
    "group_size": 4,
    "grpo_gates": [],                  # our GRPO variant: "invalid" and / or "failure" (see grpo_advantages)
    "critic": {"lr": 1e-5, "cliprange_value": 0.5, "gamma": 1.0, "lam": 1.0, "level": "call", "warmup": 0,
               "gates": []},          # gates: as grpo_gates, applied to the (per-token) GAE advantages
    "advantage_norm": None,            # "role": (A - mean) / std over the calls of each role in the update batch;
                                       # "role_sign": the same, clipped so that no advantage changes sign
    "reward": {"format_penalty": 0.2, "validator_penalty": 0.1, "ites_weight": 1.0, "tes_beta": 0.95},
    "terminal_reward": 2.0,
    "intra_gamma": 1.0,                # < 1: the k-th call of an agent in a timestep gets intra_gamma^k of the
                                       # delivery share (reinforce estimator; fewer rounds of talk per timestep)
    "redundancy_penalty": 0.0,         # > 0: rule-based penalty on repeated Collab calls and Chef stalls
                                       # (advantages.redundant_calls), a judge-free ablation of the think reward
    # think reward (think_action protocol): an LLM judge scores the Think / Recent Goal of every well-formed call
    # (baselines/rl/think_judge.py); `mode` decides which tokens it credits (advantages.think_advantages).
    # mode none with a judge url still judges the calls (for monitoring) but does not use the score.
    # grounding_weight: weight of the judge's state-grounding score added to its overall score; scale "match":
    # the think signal is rescaled per role to (at least) the std of the action advantages of the batch, so weight 1
    # puts the think term on the level of the action term or above
    # rules: rule-verified rewards instead of the judge's score (baselines/rl/rule_rewards.py), e.g.
    # {ground: 1, mate: 1, plan: 1}; the judge model then only extracts the claims of the reasoning
    # rules_combine "mean": the weighted mean of the rules instead of the sum; scale "ratio": per role, the centered
    # think signal is rescaled to std = `ratio` x std(A) of the batch (advantages.ratio_scale); center "role_type":
    # centered within (role, action kind) groups (message / wait / kitchen action) instead of per role
    "think_reward": {"mode": "none", "weight": 0.5, "anneal_to": None, "anneal": None, "grounding_weight": 0.0,
                     "scale": None, "rules": None, "rules_combine": "sum", "ratio": 0.25, "center": "role"},
    "judge": None,                     # ThinkJudge kwargs, e.g. {env_file: ~/ParaCook/.env.shuaiapi.local, model: gpt-5.5}
    "save_rollouts": True,             # rollouts/uNNNNN.jsonl.gz: every call with its rewards and judgement
    "save_every": 25,                  # rollouts and judge annotations only of update 1, every save_every-th and the last
    "seed": 261008,
    "checkpoint_every": 25,            # keep these updates; the latest two are always kept for recovery
    "rollout_workers": 8,
    "tokens_per_batch": 6144,          # tokens per forward pass of the update (lower it if out of memory)
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


def next_stage(adaptive, stage, progress):
    """Stage after an update; progress: mean TES of each update of the current stage so far."""
    window = progress[-adaptive["window"]:]
    ready = len(progress) >= adaptive["min_updates"] and sum(window) / len(window) >= adaptive["threshold"]
    if stage + 1 < len(adaptive["stages"]) and (ready or len(progress) >= adaptive["max_updates"]):
        return stage + 1
    return stage


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
def judge_episodes(judge, episodes):
    """Adds `judge` (score dict or None) to every transition; one judge request per episode (think_judge)."""
    jobs = []
    for e in episodes:
        for t in e["transitions"]:
            t["judge"] = None
        judged = [t for t in e["transitions"] if judgeable(t)]
        if judged:
            jobs.append((judged, e["order"], e["episode_id"]))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(judge.workers) as pool:
        results = pool.map(lambda job: judge_episode(judge, [t["judge_record"] for t in job[0]], job[1],
                                                     meta={"episode": job[2]}), jobs)
        for (judged, _, _), scores in zip(jobs, results):
            for t, score in zip(judged, scores):
                t["judge"] = score if score is not None else {"score": None}
    return sum(len(j) for j, _, _ in jobs)


def rule_episodes(judge, episodes, weights, combine="sum"):
    """Rule-verified think rewards: `judge` (score = weighted sum, rules = the three rewards) on every judgeable
    transition; one extraction per episode."""
    def one(e):
        records = [dict(t["judge_record"], timestep=t["timestep"]) for t in e["transitions"]]
        chosen = [k for k, t in enumerate(e["transitions"]) if judgeable(t)]
        found = extract(judge, [records[k] for k in chosen])
        aligned = [None] * len(records)
        for k, x in zip(chosen, found):
            aligned[k] = x
        return chosen, rewards(records, aligned, e["order"])

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(judge.workers) as pool:
        results = list(pool.map(one, episodes))
    n = 0
    for e, (chosen, scores) in zip(episodes, results):
        for t in e["transitions"]:
            t["judge"] = None
        for k in chosen:
            result = scores[k]
            e["transitions"][k]["judge"] = ({"score": None} if result is None else
                                            {"score": rule_signal(result, weights, combine),
                                             "rules": {x: result[x] for x in ("ground", "mate", "mate_v2", "plan")},
                                             "rule_detail": result["detail"]})
        n += len(chosen)
    return n


def judge_metrics(transitions):
    judged = [t for t in transitions if t.get("judge") is not None]
    scored = [t for t in judged if t["judge"].get("score") is not None]
    if not judged:
        return {}
    result = {"judge/judged_frac": len(judged) / max(1, len(transitions)),
              "judge/failure_rate": 1 - len(scored) / len(judged)}
    for role, name in ROLES.items():
        values = [t["judge"]["score"] for t in scored if t["agent_index"] == role]
        result[f"judge/score_{name}"] = sum(values) / len(values) if values else 0.0
        for key in ("ground", "mate", "mate_v2", "plan"):
            values = [t["judge"]["rules"][key] for t in scored if t["agent_index"] == role
                      and t["judge"].get("rules") and t["judge"]["rules"].get(key) is not None]
            if values:
                result[f"rules/{key}_{name}"] = sum(values) / len(values)
    if scored:
        result["judge/score"] = sum(t["judge"]["score"] for t in scored) / len(scored)
        follows = [t["judge"]["action_follows"] for t in scored if t["judge"].get("action_follows") is not None]
        if follows:
            result["judge/action_follows_rate"] = sum(bool(f) for f in follows) / len(follows)
        valid = [t["judge"]["score"] for t in scored if t["validator_reward"] == 0 and t["sequence_reward"] > 0]
        other = [t["judge"]["score"] for t in scored if not (t["validator_reward"] == 0 and t["sequence_reward"] > 0)]
        if valid and other:  # does the judged reasoning quality track the action's progress?
            result["judge/score_progress_minus_other"] = sum(valid) / len(valid) - sum(other) / len(other)
    return result


def kl_anomaly(guard, metrics):
    """Whether an update must be undone: its update KL or its clamped token KL to the reference is too large."""
    return (metrics.get("update_kl", 0.0) > guard.get("max_update_kl", float("inf"))
            or metrics.get("kl_clamped", 0.0) > guard.get("max_ref_kl", float("inf")))


def think_signal(judge, grounding_weight):
    """The judge's overall score plus `grounding_weight` x its state-grounding score (None when not scored; when the
    grounding score is missing, the overall score stands in for it)."""
    if not judge or judge.get("score") is None:
        return None
    grounding = judge.get("grounding")
    if not grounding_weight or grounding is None:
        return judge["score"] * (1 + grounding_weight) if grounding_weight else judge["score"]
    return judge["score"] + grounding_weight * grounding


def match_scale(transitions, advantages, think, raw):
    """Rescale the centered think signal per role to the std of the action advantages of the batch, never below
    its own scale (factor >= 1: early batches without successes have tiny action advantages); returns (scaled think
    signal, factor applied to role 0's signal)."""
    def std(values):
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5 if values else 0.0
    target = std(advantages) if advantages else 0.0
    factors = {}
    for role in (0, 1):
        values = [j for t, j, r in zip(transitions, think, raw) if t["agent_index"] == role and r is not None]
        spread = std(values) if values else 0.0
        factors[role] = max(1.0, target / spread) if spread > 1e-6 else 0.0
    return [j * factors[t["agent_index"]] for t, j in zip(transitions, think)], factors[0]


def save_rollouts(path, episodes, transitions, scores, pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for t, score, (adv_think, adv_action) in zip(transitions, scores, pairs):
            judge = t.get("judge") or {}
            f.write(json.dumps({
                "episode": t["episode"], "order": t["order"], "agent_index": t["agent_index"],
                "timestep": t["timestep"], "call_index": t["call_index"], "call_type": t["call_type"],
                "action_mode": t["action_mode"], "observation": t["judge_record"]["observation"],
                "response": t["response"], "validations": t["judge_record"]["validations"],
                "format_reward": t["format_reward"], "validator_reward": t["validator_reward"],
                "ites": t["sequence_reward"], "collab_reward": t.get("collab_reward", 0.0),
                "terminal_reward": t["terminal_reward"], "uncooperative": t.get("uncooperative"),
                "judge_score": judge.get("score"), "judge_dims": {k: judge.get(k) for k in
                                                                  ("grounding", "plan", "coordination", "consistency")},
                "judge_analysis": judge.get("analysis"), "rules": judge.get("rules"),
                "rule_detail": judge.get("rule_detail"), "think_reward": score,
                "adv_think": adv_think, "adv_action": adv_action, "action_start": t.get("action_start"),
                "value": t["value"][0] if isinstance(t.get("value"), list) and t["value"] else t.get("value"),
                "redundancy_reward": t.get("redundancy_reward", 0.0), "repeat": t.get("repeat"),
                "collab_requests": t.get("collab_requests"), "uptaken": t.get("uptaken"),
                "response_tokens": len(t["response_ids"])}) + "\n")
        for e in episodes:
            f.write(json.dumps({"episode_summary": e["episode_id"], "order": e["order"], "success": e["success"],
                                "steps": e["steps"]}) + "\n")


def summarize(episodes):
    calls = [t for e in episodes for t in e["transitions"]]
    n = max(1, len(episodes))
    return {
        "success_rate": sum(e["success"] for e in episodes) / n,
        "steps": sum(e["steps"] for e in episodes) / n,
        "episode_return": sum(t["reward"] for t in calls) / n,
        "ites_per_episode": sum(t["sequence_reward"] for t in calls) / n,
        "collab_per_episode": sum(t.get("collab_reward", 0.0) for t in calls) / n,
        "calls_per_episode": len(calls) / n,
        "repeat_per_episode": sum(t.get("uncooperative") == "repeat" for t in calls) / n,
        "idle_per_episode": sum(t.get("uncooperative") == "idle" for t in calls) / n,
        "format_error_rate": sum(t["format_reward"] < 0 for t in calls) / max(1, len(calls)),
        "validator_error_rate": sum(t["validator_reward"] < 0 for t in calls) / max(1, len(calls)),
        **comm_metrics([dict(t, episode=k) for k, e in enumerate(episodes) for t in e["transitions"]]),
    }


def comm_metrics(calls):
    """Communication metrics of an update: repeat share (repeated Collab messages / all Collab messages, repeat as
    in CallCredit: already sent by the agent since the last executed action), calls per (episode, agent, timestep)
    with a call, the share of messages / waits / kitchen actions among each role's calls, and the uptake rate
    (collab_mode uptake: requests taken up / new requests)."""
    messages = [t for t in calls if t.get("action_mode") == "communication"]
    steps = {(t.get("episode"), t["agent_index"], t["timestep"]) for t in calls}
    result = {"comm/repeat_share": sum(bool(t.get("repeat")) for t in messages) / max(1, len(messages)),
              "comm/calls_per_agent_step": len(calls) / max(1, len(steps))}
    for role, name in ROLES.items():
        mine = [t for t in calls if t["agent_index"] == role]
        n = max(1, len(mine))
        result[f"comm/msg_share_{name}"] = sum(t.get("action_mode") == "communication" for t in mine) / n
        result[f"comm/wait_share_{name}"] = sum(t.get("action_mode") == "wait" for t in mine) / n
        result[f"comm/action_share_{name}"] = sum(t.get("action_mode") == "planner_main" for t in mine) / n
    requests = sum(t.get("collab_requests", 0) for t in calls)
    if requests:
        result["comm/uptake_rate"] = sum(t.get("uptaken", 0) for t in calls) / requests
    return result


def save_adapters(policy, directory):
    tmp = Path(str(directory) + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for agent_index in policy.trainable:
        policy.save(agent_index, tmp)  # PEFT writes <tmp>/<adapter name>/
    for agent_index in policy.critics:
        policy.save_critic(agent_index, tmp)
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
        estimator = cfg["advantage_estimator"]
        if estimator not in ("reinforce", "grpo", "gae"):
            raise ValueError(f"unknown advantage_estimator {estimator}")
        if estimator != "reinforce" and cfg["advantage_norm"]:
            raise ValueError("advantage_norm only applies to the reinforce estimator")
        if estimator == "grpo" and cfg["episodes_per_update"] % cfg["group_size"]:
            raise ValueError("episodes_per_update must be a multiple of group_size")
        if estimator == "gae":
            if world > 1:
                raise ValueError("the gae estimator runs on one GPU")
            critic_cfg = dict(DEFAULTS["critic"], **(cfg.get("critic") or {}))
            check_gates(critic_cfg["gates"])
            if critic_cfg["level"] not in ("call", "token"):
                raise ValueError(f"unknown critic level {critic_cfg['level']}")
            if state["update"]:  # resume: the critic saved with the checkpoint
                saved = {i: checkpoints / f"u{state['update']:05d}" / CRITIC_ADAPTERS[i] for i in ROLES}
                policy.add_critics(saved, {i: saved[i] / "value_head.pt" for i in ROLES})
            else:  # verl initializes the critic from the actor; the value heads start at zero
                policy.add_critics(initial)
        pool = multiprocessing.get_context("spawn").Pool(cfg["rollout_workers"], _init_worker, (cfg["model"],))
        n_tasks, per_update = len(cfg["tasks"]), cfg["episodes_per_update"]
        judge_log = str(out / "judge_annotations" / f"rank{rank}.jsonl")
        judge = ThinkJudge(**dict({"log_path": judge_log}, **cfg["judge"])) if cfg.get("judge") else None
        think_cfg = dict(DEFAULTS["think_reward"], **(cfg.get("think_reward") or {}))
        if estimator == "gae" and critic_cfg["level"] == "token" and think_cfg["mode"] != "none":
            raise ValueError("the token-level critic only runs with think mode none")
        if think_cfg["mode"] != "none" and judge is None:
            raise ValueError("think_reward.mode needs a `judge`")
        for update in range(state["update"] + 1, cfg["updates"] + 1):
            started = time.time()
            adaptive = cfg["horizon_adaptive"]
            horizon = adaptive["stages"][state.get("stage", 0)] if adaptive else horizon_at(cfg, update)
            keep = update == 1 or update % cfg["save_every"] == 0 or update == cfg["updates"]  # storage is limited
            if judge is not None:
                judge.log_path = judge_log if keep else None
            jobs = []
            for e in range(rank, per_update, world):
                episode_id = (update - 1) * per_update + e
                # grpo: `group_size` consecutive episodes play the same order (one group)
                slot = episode_id // cfg["group_size"] if estimator == "grpo" else episode_id
                jobs.append({"cfg": cfg, "url": server.url, "served": served, "horizon": horizon,
                             "order": cfg["tasks"][slot % n_tasks], "episode_id": episode_id,
                             "seed": cfg["seed"] + episode_id * 100003})
            local = pool.map(_run_episode, jobs)
            collect_seconds = time.time() - started
            judge_started = time.time()
            if judge is not None:
                judge.context = {"run": str(out), "update": update, "think_mode": think_cfg["mode"]}
                n_judged = (rule_episodes(judge, local, think_cfg["rules"], think_cfg["rules_combine"])
                            if think_cfg["rules"]
                            else judge_episodes(judge, local))
                n_failed = sum(1 for e in local for t in e["transitions"]
                               if t.get("judge") is not None and t["judge"].get("score") is None)
                if n_judged and n_failed > 0.5 * n_judged:  # e.g. the API balance ran out: stop, resume later
                    raise RuntimeError(f"judge failed on {n_failed}/{n_judged} calls in update {update}; "
                                       "the run stops before this update (resume after fixing the judge)")
            judge_seconds = time.time() - judge_started
            episodes = sorted((e for part in _gather(local, world) for e in part), key=lambda e: e["episode_id"])

            transitions = [dict(t, episode=e["episode_id"], order=e["order"], group=e["episode_id"] // cfg["group_size"],
                                success=e["success"])
                           for e in episodes for t in e["transitions"]]
            critic_metrics, asleep = {}, False
            if cfg["redundancy_penalty"]:
                for t, flag in zip(transitions, redundant_calls(transitions)):
                    t["redundancy_reward"] = -cfg["redundancy_penalty"] if flag else 0.0
                    t["reward"] += t["redundancy_reward"]
            if estimator == "grpo":
                advantages = grpo_advantages(transitions, gates=cfg["grpo_gates"])
            elif estimator == "gae":
                server.sleep()  # the critic forward passes need the GPU memory of vLLM
                asleep = True
                values = critic_values(policy, [{"agent_index": t["agent_index"], "prompt_ids": torch.tensor(t["prompt_ids"]),
                                                 "response_ids": torch.tensor(t["response_ids"])} for t in transitions],
                                       cfg["tokens_per_batch"], token_level=critic_cfg["level"] == "token")
                if critic_cfg["level"] == "token":
                    token_advantages, returns = token_gae_advantages(transitions, values, critic_cfg["gamma"],
                                                                     critic_cfg["lam"])
                    token_advantages = [[max(-cfg["advantage_clip"], min(cfg["advantage_clip"], a)) for a in adv]
                                        for adv in token_advantages]
                    token_advantages = [[min(a, 0.0) for a in adv] if gated(t, critic_cfg["gates"]) else adv
                                        for t, adv in zip(transitions, token_advantages)]
                    advantages = [sum(a) / len(a) if a else 0.0 for a in token_advantages]
                else:
                    advantages, returns = gae_advantages(transitions, values, critic_cfg["gamma"], critic_cfg["lam"])
                    advantages = [min(a, 0.0) if gated(t, critic_cfg["gates"]) else a
                                  for t, a in zip(transitions, advantages)]
                for t, v in zip(transitions, values):
                    t["value"] = v
            else:
                advantages = compute_advantages(transitions, cfg["gamma"], cfg["advantage_clip"], cfg["intra_gamma"])
            advantages = [max(-cfg["advantage_clip"], min(cfg["advantage_clip"], a)) for a in advantages]
            if cfg["advantage_norm"] in ("role", "role_sign"):
                advantages = normalize_advantages(transitions, advantages, cfg["advantage_clip"],
                                                  keep_sign=cfg["advantage_norm"] == "role_sign")
            elif cfg["advantage_norm"]:
                raise ValueError(f"unknown advantage_norm {cfg['advantage_norm']}")
            raw = [think_signal(t.get("judge"), think_cfg["grounding_weight"]) for t in transitions]
            think = centered_judge_scores(transitions, raw, think_cfg["center"])
            think_scale = think_scale_assistant = 1.0
            if think_cfg["scale"] == "match":
                think, think_scale = match_scale(transitions, advantages, think, raw)
            elif think_cfg["scale"] == "ratio":
                think, factors = ratio_scale(transitions, advantages, think, raw, think_cfg["ratio"])
                think_scale, think_scale_assistant = factors[0], factors[1]
            elif think_cfg["scale"]:
                raise ValueError(f"unknown think_reward.scale {think_cfg['scale']}")
            weight = think_weight(think_cfg, update)
            pairs = think_advantages(transitions, advantages, think, think_cfg["mode"], weight, cfg["advantage_clip"])
            samples = []
            for t, a, (adv_think, adv_action) in zip(transitions, advantages, pairs):
                sample = {"agent_index": t["agent_index"], "advantage": a,
                          "prompt_ids": torch.tensor(t["prompt_ids"]), "response_ids": torch.tensor(t["response_ids"])}
                if think_cfg["mode"] != "none":
                    token_advantage = torch.full((len(t["response_ids"]),), float(adv_action))
                    split = t.get("action_start")
                    token_advantage[:len(t["response_ids"]) if split is None else split] = float(adv_think)
                    sample["token_advantage"] = token_advantage
                if estimator == "gae" and critic_cfg["level"] == "token":
                    sample["token_advantage"] = torch.tensor(token_advantages[len(samples)], dtype=torch.float32)
                samples.append(sample)
            if rank == 0 and cfg["save_rollouts"] and keep:
                save_rollouts(out / "rollouts" / f"u{update:05d}.jsonl.gz", episodes, transitions,
                              [weight * j for j in think], pairs)
            update_started = time.time()
            if not asleep:
                server.sleep()  # the GPU memory of vLLM goes to the update
            if estimator == "gae":  # verl updates the critic, then the actor
                critic_metrics = critic_update(policy, samples, returns, values, cfg["tokens_per_batch"],
                                               critic_cfg["lr"], critic_cfg["cliprange_value"])
                flat_r, flat_v = returns, values
                if critic_cfg["level"] == "token":
                    flat_r, flat_v = [r for rs in returns for r in rs], [v for vs in values for v in vs]
                critic_metrics["value_mean"] = sum(flat_v) / max(1, len(flat_v))
                critic_metrics["return_mean"] = sum(flat_r) / max(1, len(flat_r))
                mean_r = critic_metrics["return_mean"]
                var_r = sum((r - mean_r) ** 2 for r in flat_r) / max(1, len(flat_r))
                var_e = sum((r - v) ** 2 for r, v in zip(flat_r, flat_v)) / max(1, len(flat_r))
                critic_metrics["explained_var"] = 1 - var_e / var_r if var_r > 0 else 0.0
            lr = cfg["lr"] * state["lr_scale"]
            if estimator == "gae" and update <= critic_cfg["warmup"]:  # verl trainer.critic_warmup
                metrics = {"pg_loss": 0.0, "kl": 0.0, "ppo_kl": 0.0, "clipfrac": 0.0, "grad_norm": 0.0,
                           "update_kl": 0.0, "critic_warmup": 1}
            else:
                guard = cfg["kl_guard"]
                snapshot = [p.detach().to("cpu", copy=True) for p in policy.parameters()] if guard else None
                metrics = ppo_update(policy, samples, objective=cfg["objective"], temperature=cfg["temperature"],
                                     lr=lr, tokens_per_batch=cfg["tokens_per_batch"], kl_coef=cfg["kl_coef"],
                                     entropy_coef=cfg["entropy_coef"],
                                     seed=cfg["seed"] + update, rank=rank, world=world,
                                     min_token_logprob=(guard or {}).get("min_token_logprob"))
                if guard:
                    metrics["reverted"] = int(kl_anomaly(guard, metrics))
                    if metrics["reverted"]:
                        with torch.no_grad():
                            for p, saved in zip(policy.parameters(), snapshot):
                                p.copy_(saved.to(p.device))
                    del snapshot
            torch.cuda.empty_cache()
            server.wake_up()
            update_seconds = time.time() - update_started
            lr_scale = (next_lr_scale(cfg["adaptive_lr"], state["lr_scale"], metrics["update_kl"])
                        if cfg["adaptive_lr"] else state["lr_scale"])  # keep a decay set by the KL guard
            if metrics.get("reverted"):
                lr_scale = max(cfg["kl_guard"].get("min_lr_scale", 0.1),
                               lr_scale * cfg["kl_guard"].get("lr_decay", 0.5))
            stage, progress = state.get("stage", 0), state.get("stage_progress", [])
            if adaptive:  # episodes are gathered on every rank, so all ranks agree on the stage
                progress = progress + [summarize(episodes)["ites_per_episode"] / 2]
                stage, progress = (stage + 1, []) if next_stage(adaptive, stage, progress) > stage else (stage, progress)
            state = {"update": update, "lr_scale": lr_scale, "stage": stage, "stage_progress": progress}

            directory = checkpoints / f"u{update:05d}"
            if rank == 0:
                save_adapters(policy, directory)
            _barrier(world)
            served = server.load_adapters({i: directory / ROLE_ADAPTERS[i] for i in ROLES}, f"u{update}")
            if rank == 0:
                think_tokens = [t["action_start"] for t in transitions if t.get("action_start") is not None]
                record = {"update": update, "horizon": horizon, "lr": lr, **summarize(episodes),
                          **judge_metrics(transitions), "think/weight": weight, "think/scale": think_scale,
                          "think/scale_assistant": think_scale_assistant,
                          "think/reasoning_tokens": sum(think_tokens) / max(1, len(think_tokens)),
                          "think/response_tokens": sum(len(t["response_ids"]) for t in transitions)
                          / max(1, len(transitions)),
                          "think/adv_think": sum(p[0] for p in pairs) / max(1, len(pairs)),
                          "think/adv_action": sum(p[1] for p in pairs) / max(1, len(pairs)),
                          "redundant_per_episode": sum(t.get("redundancy_reward", 0.0) < 0 for t in transitions)
                          / max(1, len(episodes)),
                          "time/judge": judge_seconds, **({f"judge/api_{k}": v for k, v in judge.usage.items()}
                                                          if judge is not None else {}),
                          "orders": [e["order"] for e in episodes],
                          **{f"update/{k}": v for k, v in metrics.items()},
                          **{f"critic/{k}": v for k, v in critic_metrics.items()},
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
