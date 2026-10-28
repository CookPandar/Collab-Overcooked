"""Advantages: immediate local reward + discounted terminal reward (no baseline, no normalization).

For a call of agent i at timestep t of an episode,
    A = r_local + keep * sum_{t' >= t} gamma^(t' - t) * r_terminal(i, t')
where r_local = ITES + format + validator (+ collaboration) rewards of the call and keep = 0 for calls with a
format or validator penalty or flagged uncooperative by the collaboration reward (they never share the delivery
reward). A call with a format penalty never gets a
positive advantage. Advantages are clipped to [-clip, clip].
intra_gamma < 1 discounts the delivery share inside a timestep: the k-th call (k = 0, 1, ...) of an agent in a
timestep gets intra_gamma^k of it, so every extra round of communication in a timestep is worth less; the
discount restarts at every timestep (it does not reach the calls of earlier timesteps).
"""

import re
from collections import defaultdict


def compute_advantages(transitions, gamma=0.99, clip=10.0, intra_gamma=1.0):
    """`transitions`: dicts with episode, agent_index, timestep, reward, terminal_reward, format_reward,
    validator_reward. Returns a list of advantages."""
    terminal_by_step = defaultdict(lambda: defaultdict(float))
    for t in transitions:
        terminal_by_step[(t["episode"], t["agent_index"])][t["timestep"]] += t["terminal_reward"]
    terminal_return = {}
    for key, by_step in terminal_by_step.items():
        future, future_step = 0.0, None
        for step in sorted(by_step, reverse=True):
            discount = 1.0 if future_step is None else gamma ** (future_step - step)
            future, future_step = by_step[step] + discount * future, step
            terminal_return[key + (step,)] = future
    rank = {}  # position of each call among its agent's calls of the same timestep
    in_step = defaultdict(list)
    for i, t in enumerate(transitions):
        in_step[(t["episode"], t["agent_index"], t["timestep"])].append(i)
    for index in in_step.values():
        for k, i in enumerate(sorted(index, key=lambda i: transitions[i].get("call_index", i))):
            rank[i] = k
    advantages = []
    for i, t in enumerate(transitions):
        local = t["reward"] - t["terminal_reward"]
        invalid = t["format_reward"] < 0 or t["validator_reward"] < 0 or bool(t.get("uncooperative"))
        share = intra_gamma ** rank[i] * terminal_return[(t["episode"], t["agent_index"], t["timestep"])]
        value = local + (0.0 if invalid else share)
        if t["format_reward"] < 0 and value > 0:
            value = 0.0
        advantages.append(max(-clip, min(clip, value)))
    return advantages


def normalize_advantages(transitions, advantages, clip=10.0, eps=1e-6, keep_sign=False):
    """Group-relative advantages: (A - mean) / std over the calls of the same role in the batch, so that the calls
    of a failed episode that made no progress (A ~ 0) get a negative advantage. A call with a format penalty still
    never gets a positive advantage. With keep_sign the normalized value cannot cross the sign of the raw one:
    A > 0 stays >= 0 and A <= 0 stays <= 0 (A = 0 counts as non-positive, as the v1 positive-return gate)."""
    result = list(advantages)
    for role in (0, 1):
        index = [i for i, t in enumerate(transitions) if t["agent_index"] == role]
        if len(index) < 2:
            continue
        values = [advantages[i] for i in index]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        for i in index:
            value = (advantages[i] - mean) / (std + eps)
            if keep_sign:
                value = max(value, 0.0) if advantages[i] > 0 else min(value, 0.0)
            if transitions[i]["format_reward"] < 0 and value > 0:
                value = 0.0
            result[i] = max(-clip, min(clip, value))
    return result


# ---- AgentGym-RL estimators (verl core_algos): GRPO outcome advantage and PPO with a critic (GAE) ------
def _role_episodes(transitions):
    """{(episode, role): [transition index, ...]} in call order."""
    groups = defaultdict(list)
    for i, t in enumerate(transitions):
        groups[(t["episode"], t["agent_index"])].append(i)
    for index in groups.values():
        index.sort(key=lambda i: (transitions[i]["timestep"], transitions[i]["call_index"]))
    return groups


def grpo_advantages(transitions, eps=1e-6, gates=()):
    """verl compute_grpo_outcome_advantage: the score of a role's trajectory is the sum of its call rewards
    (format + validator + ITES + terminal); it is normalized by the mean and the unbiased std over the episodes of
    the same group (`t["group"]`: same order in the same update) and given to every call of the trajectory.
    A group of one episode keeps its raw score.
    `gates` (our variant, not in verl): "invalid": a call with a format or validator penalty gets min(A, 0), as in
    compute_advantages; "failure": every call of a failed episode (`t["success"]` false) gets min(A, 0)."""
    check_gates(gates)
    scores = {key: sum(transitions[i]["reward"] for i in index) for key, index in _role_episodes(transitions).items()}
    group_of = {(t["episode"], t["agent_index"]): (t["group"], t["agent_index"]) for t in transitions}
    members = defaultdict(list)
    for key, score in scores.items():
        members[group_of[key]].append(score)
    normalized = {}
    for key, score in scores.items():
        values = members[group_of[key]]
        if len(values) == 1:
            normalized[key] = score
            continue
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        normalized[key] = (score - mean) / (std + eps)
    return [min(normalized[(t["episode"], t["agent_index"])], 0.0) if gated(t, gates)
            else normalized[(t["episode"], t["agent_index"])] for t in transitions]


def check_gates(gates):
    unknown = set(gates) - {"invalid", "failure"}
    if unknown:
        raise ValueError(f"unknown advantage gates {sorted(unknown)}")


def gated(t, gates):
    """Whether the advantage of call `t` is capped at 0 (min(A, 0)) under `gates`."""
    return ("invalid" in gates and (t["format_reward"] < 0 or t["validator_reward"] < 0)
            or "failure" in gates and not t["success"])


def gae_advantages(transitions, values, gamma=1.0, lam=1.0):
    """verl compute_gae_advantage_return at the call level: each role's calls of an episode form one trajectory
    with reward r_i (the call reward) and value V_i (the critic at the call's prompt); V after the last call is 0.
        delta_i = r_i + gamma V_{i+1} - V_i,   A_i = sum_k (gamma lam)^k delta_{i+k},   R_i = A_i + V_i
    Returns (advantages whitened over the calls of each role in the batch as verl masked_whiten, critic returns)."""
    advantages, returns = [0.0] * len(transitions), [0.0] * len(transitions)
    for index in _role_episodes(transitions).values():
        running, next_value = 0.0, 0.0
        for i in reversed(index):
            delta = transitions[i]["reward"] + gamma * next_value - values[i]
            running = delta + gamma * lam * running
            advantages[i], returns[i], next_value = running, running + values[i], values[i]
    whitened = list(advantages)
    for role in (0, 1):
        index = [i for i, t in enumerate(transitions) if t["agent_index"] == role]
        if len(index) < 2:
            continue
        mean = sum(advantages[i] for i in index) / len(index)
        var = sum((advantages[i] - mean) ** 2 for i in index) / (len(index) - 1)
        for i in index:
            whitened[i] = (advantages[i] - mean) / (var + 1e-8) ** 0.5
    return whitened, returns


def token_gae_advantages(transitions, values, gamma=1.0, lam=1.0):
    """verl compute_gae_advantage_return at the token level, as AgentGym-RL on a multi-turn trajectory: the response
    tokens of a role's calls of an episode form one sequence (the prompt tokens between calls are masked: the GAE
    passes through them), the reward of a call is on its last token, `values[i]` holds the critic value before
    each response token of call i. A call without response tokens adds its reward to the previous token.
    Returns (per-call lists of advantages whitened over all response tokens of each role as verl masked_whiten,
    per-call lists of returns)."""
    advantages = [[0.0] * len(v) for v in values]
    returns = [[0.0] * len(v) for v in values]
    for index in _role_episodes(transitions).values():
        running, next_value, pending = 0.0, 0.0, 0.0
        for i in reversed(index):
            if not len(values[i]):
                pending += transitions[i]["reward"]
                continue
            for k in reversed(range(len(values[i]))):
                reward = transitions[i]["reward"] + pending if k == len(values[i]) - 1 else 0.0
                delta = reward + gamma * next_value - values[i][k]
                running = delta + gamma * lam * running
                advantages[i][k], returns[i][k], next_value = running, running + values[i][k], values[i][k]
            pending = 0.0
    whitened = [list(a) for a in advantages]
    for role in (0, 1):
        flat = [a for i, t in enumerate(transitions) if t["agent_index"] == role for a in advantages[i]]
        if len(flat) < 2:
            continue
        mean = sum(flat) / len(flat)
        std = (sum((a - mean) ** 2 for a in flat) / (len(flat) - 1) + 1e-8) ** 0.5
        for i, t in enumerate(transitions):
            if t["agent_index"] == role:
                whitened[i] = [(a - mean) / std for a in advantages[i]]
    return whitened, returns


# ---- rule-based redundancy penalty (ablation of the think judge, no LLM) --------------------------------
_ACTION = re.compile(r"Action\s*:\s*(.*)", re.S)
_CHEF_HOLDS = re.compile(r"<Chef> holds ([^.]*)\.")


def redundant_calls(transitions):
    """Flags the calls that runs/think_judge/count_behavior.py counts as repeats or stalls:
    - repeat: a Collab action identical to the same agent's previous action in the episode;
    - stall: the Chef waits or talks while holding raw `*_slices` (not yet baked or boiled; Level-2 recipes).
    Returns a list of bools in the order of `transitions`."""
    flags = [False] * len(transitions)
    for index in _role_episodes(transitions).values():
        last = None
        for i in index:
            t = transitions[i]
            match = _ACTION.search(t["response"] or "")
            action = " ".join(match.group(1).split()).strip("` ") if match else ""
            talk = action.startswith("Collab")
            if talk and action == last:
                flags[i] = True
            last = action
            if t["agent_index"] == 0 and (talk or action.startswith("wait")):
                held = _CHEF_HOLDS.search(t["judge_record"]["observation"] or "")
                held = held.group(1) if held else ""
                if held.endswith("_slices") and not held.startswith(("baked", "boiled")):
                    flags[i] = True
    return flags


# ---- think reward (think_action protocol) ------------------------------------------------------------
THINK_MODES = ("none", "sum", "split", "decoupled", "gated", "action")


def think_weight(cfg, update):
    """Weight of the think reward at `update`: `weight`, linearly annealed to `anneal_to` over `anneal` updates."""
    weight = float(cfg.get("weight", 0.0))
    if cfg.get("anneal_to") is None or not cfg.get("anneal"):
        return weight
    start, end = cfg["anneal"]
    if update <= start:
        return weight
    if update >= end:
        return float(cfg["anneal_to"])
    return weight + (float(cfg["anneal_to"]) - weight) * (update - start) / (end - start)


ACTION_KINDS = {"communication": "message", "wait": "wait", "planner_main": "action"}


def action_kind(t):
    """message | wait | action (a kitchen action) | other (malformed, mixed, empty ...) from the call's action mode
    (credit.response_mode)."""
    return ACTION_KINDS.get(t.get("action_mode"), "other")


def centered_judge_scores(transitions, scores, center="role"):
    """Judge score minus the mean judge score of the same role in the batch (step-level relative reward, as the
    step-level group of GiGPO). Calls without a score (not judged, judge failure) get 0.
    center "role_type": the mean of the same (role, action kind) group instead (action_kind), so the think signal
    is neutral to the choice between messages, waits and kitchen actions."""
    if center not in ("role", "role_type"):
        raise ValueError(f"unknown think_reward.center {center}")
    group = ((lambda t: t["agent_index"]) if center == "role"
             else (lambda t: (t["agent_index"], action_kind(t))))
    sums = defaultdict(lambda: [0.0, 0])
    for t, s in zip(transitions, scores):
        if s is not None:
            sums[group(t)][0] += s
            sums[group(t)][1] += 1
    means = {k: total / n for k, (total, n) in sums.items()}
    return [0.0 if s is None else s - means[group(t)] for t, s in zip(transitions, scores)]


def ratio_scale(transitions, advantages, think, raw, ratio):
    """Fixed-ratio scale of the centered think signal: per role, factor = ratio x std(A of the role) / std(think of
    the role's scored calls), so the think term has std = ratio x std(A) in the batch (0 when either std is 0).
    Returns (scaled think signal, {role: factor})."""
    def std(values):
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    factors = {}
    for role in (0, 1):
        spread = std([j for t, j, r in zip(transitions, think, raw) if t["agent_index"] == role and r is not None])
        target = std([a for t, a in zip(transitions, advantages) if t["agent_index"] == role])
        factors[role] = ratio * target / spread if spread > 1e-6 else 0.0
    return [j * factors[t["agent_index"]] for t, j in zip(transitions, think)], factors


def think_advantages(transitions, advantages, think, mode, weight, clip=10.0):
    """(reasoning-token advantage, action-token advantage) of every call.

    A is the action advantage of compute_advantages, j the centered judge score, w the weight.
        none       A          / A        (the reward of the earlier pipeline on every token)
        sum        A + w j    / A + w j  (one scalar per call, as CollabBench's A^T + w A^S)
        split      A + w j    / A        (the judge only credits the reasoning tokens)
        decoupled  w j        / A        (reasoning tokens only see the judge, action tokens only the environment)
        gated      split, but a call whose action got a format or validator penalty cannot gain from the judge
        action     A          / A + w j  (mirror of split: the judge of the reasoning only credits the action tokens)
    """
    if mode not in THINK_MODES:
        raise ValueError(f"unknown think reward mode {mode}")
    result = []
    for t, a, j in zip(transitions, advantages, think):
        bonus = weight * j
        if mode == "gated" and (t["format_reward"] < 0 or t["validator_reward"] < 0):
            bonus = min(bonus, 0.0)
        if mode == "none":
            pair = (a, a)
        elif mode == "sum":
            pair = (a + bonus, a + bonus)
        elif mode == "decoupled":
            pair = (bonus, a)
        elif mode == "action":
            pair = (a, a + bonus)
        else:  # split, gated
            pair = (a + bonus, a)
        result.append(tuple(max(-clip, min(clip, v)) for v in pair))
    return result
