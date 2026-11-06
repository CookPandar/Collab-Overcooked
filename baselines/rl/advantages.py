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

import bisect
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


def normalize_advantages(transitions, advantages, clip=10.0, eps=1e-6, keep_sign=False, std_floor=0.0):
    """Group-relative advantages: (A - mean) / std over the calls of the same role in the batch, so that the calls
    of a failed episode that made no progress (A ~ 0) get a negative advantage. A call with a format penalty still
    never gets a positive advantage. With keep_sign the normalized value cannot cross the sign of the raw one:
    A > 0 stays >= 0 and A <= 0 stays <= 0 (A = 0 counts as non-positive, as the v1 positive-return gate).
    std_floor: divide by max(std, std_floor), so a batch with little spread (e.g. no success: only the small
    penalties and ITES) is only centered, not blown up to unit std."""
    result = list(advantages)
    for role in (0, 1):
        index = [i for i, t in enumerate(transitions) if t["agent_index"] == role]
        if len(index) < 2:
            continue
        values = [advantages[i] for i in index]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        for i in index:
            value = (advantages[i] - mean) / (max(std, std_floor) + eps)
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


def local_penalty(t):
    """Per-call format + validator penalty (<= 0): the part of the call reward that is about the call itself, not
    about task progress or outcome (+ the uniform call cost of the team scheme, absent otherwise)."""
    return t["format_reward"] + t["validator_reward"] + t.get("call_cost", 0.0)


GRPO_NORMS = ("std", "none")
GRPO_SCORES = ("all", "outcome")


def grpo_advantages(transitions, eps=1e-6, gates=(), norm="std", score="all", min_spread=None, stats=None,
                    step_weight=0.0, step_gamma=0.95, step_norm=None):
    """verl compute_grpo_outcome_advantage: the score of a role's trajectory is the sum of its call rewards
    (format + validator + ITES + terminal); it is normalized by the mean and the unbiased std over the episodes of
    the same group (`t["group"]`: same order in the same update) and given to every call of the trajectory.
    A group of one episode keeps its raw score.
    `gates` (our variant, not in verl): "invalid": a call with a format or validator penalty gets min(A, 0), as in
    compute_advantages; "failure": every call of a failed episode (`t["success"]` false) gets min(A, 0).
    Options (defaults = the verl behaviour above):
      norm "none": Dr.GRPO, score - group mean without the division by the std;
      score "outcome": the trajectory score leaves out the per-call format / validator penalties (local_penalty);
          each call gets its own penalty added to its advantage instead, so the group comparison is only about
          task progress (ITES) and outcome (terminal reward);
      min_spread: DAPO dynamic sampling, a group whose scores span <= min_spread (max - min) is dropped: its
          trajectories get group advantage 0 (no policy-gradient term; local penalties still apply);
      step_weight > 0: GiGPO, adds step_weight x the step-level relative advantage (step_advantages, discount
          step_gamma, normalized by `step_norm`, default `norm`) of each call.
    `stats` (dict, optional) receives groups, dropped_groups and dropped_frac (per (group, role))."""
    check_gates(gates)
    if norm not in GRPO_NORMS:
        raise ValueError(f"unknown grpo_norm {norm}")
    if score not in GRPO_SCORES:
        raise ValueError(f"unknown grpo_score {score}")
    local = (lambda t: local_penalty(t)) if score == "outcome" else (lambda t: 0.0)
    min_spread = None if min_spread is None else float(min_spread)  # yaml reads 1e-6 as a string
    scores = {key: sum(transitions[i]["reward"] - local(transitions[i]) for i in index)
              for key, index in _role_episodes(transitions).items()}
    group_of = {(t["episode"], t["agent_index"]): (t["group"], t["agent_index"]) for t in transitions}
    members = defaultdict(list)
    for key, value in scores.items():
        members[group_of[key]].append(value)
    dropped = {g for g, values in members.items()
               if min_spread is not None and len(values) > 1 and max(values) - min(values) <= min_spread}
    if stats is not None:
        stats.update(groups=len(members), dropped_groups=len(dropped),
                     dropped_frac=len(dropped) / max(1, len(members)))
    normalized = {}
    for key, value in scores.items():
        values = members[group_of[key]]
        if group_of[key] in dropped:
            normalized[key] = 0.0
            continue
        if len(values) == 1:
            normalized[key] = value
            continue
        mean = sum(values) / len(values)
        if norm == "none":
            normalized[key] = value - mean
            continue
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        normalized[key] = (value - mean) / (std + eps)
    step = [0.0] * len(transitions)
    if step_weight:
        step = step_advantages(transitions, step_norm or norm, step_gamma, eps, local, stats)
    result = []
    for t, s in zip(transitions, step):
        a = normalized[(t["episode"], t["agent_index"])] + step_weight * s + local(t)
        result.append(min(a, 0.0) if gated(t, gates) else a)
    return result


_KITCHEN = re.compile(r"Kitchen states:(.*)")
_AGENTS = re.compile(r"Agent State:(.*)")


def anchor_key(t):
    """GiGPO anchor state of a call (runs/think_judge/gigpo_check/group_stats.py): order, role, kitchen and agent
    states of the observation; the conversation and the agent's own Recent Goal are left out."""
    obs = t.get("observation")
    if obs is None:
        obs = (t.get("judge_record") or {}).get("observation") or ""
    kitchen, agents = _KITCHEN.search(obs), _AGENTS.search(obs)
    return (t.get("order"), t["agent_index"], kitchen.group(1).strip() if kitchen else obs,
            agents.group(1).strip() if agents else "")


def step_advantages(transitions, norm="none", gamma=0.95, eps=1e-6, local=lambda t: 0.0, stats=None):
    """GiGPO step-level relative advantage: the discounted return-to-go of each call (over its role's calls of the
    episode, rewards without the `local` part) minus the mean return of the calls with the same anchor state
    (anchor_key) in the same group (`t["group"]`, both roles kept apart), divided by the group's std with norm
    "std". Calls alone in their step group get 0."""
    returns = [0.0] * len(transitions)
    for index in _role_episodes(transitions).values():
        future = 0.0
        for i in reversed(index):
            future = transitions[i]["reward"] - local(transitions[i]) + gamma * future
            returns[i] = future
    groups = defaultdict(list)
    for i, t in enumerate(transitions):
        groups[(t["group"],) + anchor_key(t)].append(i)
    result = [0.0] * len(transitions)
    grouped = 0
    for index in groups.values():
        if len(index) < 2:
            continue
        grouped += len(index)
        values = [returns[i] for i in index]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if norm == "std" else 1.0
        for i in index:
            result[i] = (returns[i] - mean) / (std + eps) if norm == "std" else returns[i] - mean
    if stats is not None:
        stats["step_grouped_frac"] = grouped / max(1, len(transitions))
    return result


def check_gates(gates):
    unknown = set(gates) - {"invalid", "failure", "format"}
    if unknown:
        raise ValueError(f"unknown advantage gates {sorted(unknown)}")


def gated(t, gates):
    """Whether the advantage of call `t` is capped at 0 (min(A, 0)) under `gates`."""
    return ("invalid" in gates and (t["format_reward"] < 0 or t["validator_reward"] < 0)
            or "format" in gates and t["format_reward"] < 0  # invalid output only (the team scheme)
            or "failure" in gates and not t["success"])


def gae_advantages(transitions, values, gamma=1.0, lam=1.0, local_penalties=False, std_floor=0.0):
    """verl compute_gae_advantage_return at the call level: each role's calls of an episode form one trajectory
    with reward r_i (the call reward) and value V_i (the critic at the call's prompt); V after the last call is 0.
        delta_i = r_i + gamma V_{i+1} - V_i,   A_i = sum_k (gamma lam)^k delta_{i+k},   R_i = A_i + V_i
    Returns (advantages whitened over the calls of each role in the batch as verl masked_whiten, critic returns).
    local_penalties: the per-call format / validator penalty (local_penalty) is left out of r_i (so it is not
    accumulated into the returns of earlier calls, nor learned by the critic) and added to the call's own raw
    advantage before whitening. std_floor: whiten with max(std, std_floor) (0: plain std, as verl)."""
    advantages, returns = [0.0] * len(transitions), [0.0] * len(transitions)
    local = (lambda t: local_penalty(t)) if local_penalties else (lambda t: 0.0)
    for index in _role_episodes(transitions).values():
        running, next_value = 0.0, 0.0
        for i in reversed(index):
            delta = transitions[i]["reward"] - local(transitions[i]) + gamma * next_value - values[i]
            running = delta + gamma * lam * running
            advantages[i], returns[i], next_value = running, running + values[i], values[i]
    if local_penalties:
        advantages = [a + local(t) for t, a in zip(transitions, advantages)]
    whitened = list(advantages)
    for role in (0, 1):
        index = [i for i, t in enumerate(transitions) if t["agent_index"] == role]
        if len(index) < 2:
            continue
        mean = sum(advantages[i] for i in index) / len(index)
        var = sum((advantages[i] - mean) ** 2 for i in index) / (len(index) - 1)
        std = max((var + 1e-8) ** 0.5, std_floor)
        for i in index:
            whitened[i] = (advantages[i] - mean) / std
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


CATEGORIES = ("act_progress", "act_noprogress", "act_regress", "act_rejected", "msg_pos1", "msg_pos2", "msg_pos3",
              "msg_repeat", "wait")


def call_categories(transitions):
    """Rule category of every call (None for format errors and other modes), as runs/think_judge/
    judge_credit_analysis.py: a kitchen action is act_progress / act_noprogress / act_regress by the sign of its ITES
    or act_rejected; a message is msg_repeat when its agent already sent it since the last executed action (of
    either agent), else msg_pos1 / msg_pos2 / msg_pos3 (3rd and later message since that action); wait."""
    from .credit import action_field, normalize_action
    order = sorted(range(len(transitions)), key=lambda i: (transitions[i]["episode"], transitions[i]["timestep"],
                                                           transitions[i].get("call_index", i)))
    since, sent = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(set))
    result = [None] * len(transitions)
    for i in order:
        t = transitions[i]
        e, agent, kind = t["episode"], t["agent_index"], action_kind(t)
        if t["format_reward"] < 0 or kind == "other":
            continue
        if kind == "message":
            message = normalize_action(action_field(t["response"]))
            since[e][agent] += 1
            result[i] = "msg_repeat" if message in sent[e][agent] else f"msg_pos{min(since[e][agent], 3)}"
            sent[e][agent].add(message)
        elif kind == "wait":
            result[i] = "wait"
        elif t["validator_reward"] < 0:
            result[i] = "act_rejected"
        else:
            ites = t["sequence_reward"]
            result[i] = "act_progress" if ites > 1e-9 else "act_regress" if ites < -1e-9 else "act_noprogress"
            since[e], sent[e] = defaultdict(int), defaultdict(set)  # an executed action resets both agents
    return result


def category_scores(transitions, table):
    """(score, category) of every call from the rule critic `table` ({category: score}; unknown categories and
    format errors get None, i.e. no think signal)."""
    unknown = set(table) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"unknown think_reward.category keys {sorted(unknown)}")
    cats = call_categories(transitions)
    return [(table.get(c) if c else None, c) for c in cats]


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


# ---- monitoring --------------------------------------------------------------------------------------
def advantage_metrics(transitions, advantages):
    """Mean advantage per call kind in failed episodes (progress: kitchen action with ITES > 0; msg_wait; rejected:
    validator penalty) and corr(A, timestep) over failed-episode calls, as the offline gate
    runs/think_judge/adv_gate.py checks (progress > msg_wait > rejected, |corr| small)."""
    kinds = defaultdict(list)
    fail = [(t, a) for t, a in zip(transitions, advantages) if not t.get("success")]
    for t, a in fail:
        if t["format_reward"] < 0:
            continue
        if t.get("action_mode") in ("communication", "wait"):
            kinds["msg_wait"].append(a)
        elif t.get("action_mode") == "planner_main":
            if t["validator_reward"] < 0:
                kinds["rejected"].append(a)
            elif t["sequence_reward"] > 1e-9:
                kinds["progress"].append(a)
    result = {f"adv/fail_{k}": sum(v) / len(v) for k, v in kinds.items() if v}
    result["adv/mean"] = sum(advantages) / max(1, len(advantages))
    if len(fail) > 1:
        xs, ys = [a for _, a in fail], [t["timestep"] for t, _ in fail]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sx = (sum((x - mx) ** 2 for x in xs) / len(xs)) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys) / len(ys)) ** 0.5
        if sx * sy > 0:
            result["adv/fail_corr_timestep"] = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / len(xs) / (sx * sy)
    return result


# ---- team-grounded return (2026-11-03, docs/reward_redesign.md) ----------------------------------------
# reward.scheme "team": both agents share one team reward per environment timestep,
#     R_t = progress_weight * (gamma^dt * Phi_t - Phi_prev) + success_reward * [delivered at t],
# Phi_t = team reference progress after the calls of timestep t (team_progress), dt the gap to the previous
# timestep with calls; a potential-based shaping (Ng et al. 1999): its sum over an episode telescopes to
# gamma^(T) Phi_T - Phi_0, so it cannot be farmed and does not change the optimal policy. Messages, waits and
# kitchen actions are scored by the same thing, the team's progress after them; a rejected action is a no-op
# (no validator penalty). Local terms, kept out of the returns: the format penalty of an invalid output and a
# uniform cost per LLM call (`call_cost`, the same for every kind of call).

def reference_depth(history, reference):
    """Length of the in-order prefix of `reference` matched by `history` (monotone in the history)."""
    d = 0
    for action in history:
        if d < len(reference) and action == reference[d]:
            d += 1
    return d


def team_progress(transitions, references_of):
    """Team reference progress after every call, in [0, 1]: (d_chef + d_assistant) / (len_chef + len_assistant),
    d the deepest in-order match of the agent's validated kitchen actions so far over its references (the best
    reference per agent). `references_of(order)` -> {agent_index: [reference, ...]}. Transitions need episode,
    order, agent_index, timestep, call_index and validated_action (None if the call executed no kitchen action)."""
    progress = [0.0] * len(transitions)
    by_episode = defaultdict(list)
    for i, t in enumerate(transitions):
        by_episode[t["episode"]].append(i)
    for index in by_episode.values():
        index.sort(key=lambda i: (transitions[i]["timestep"], transitions[i]["call_index"], transitions[i]["agent_index"]))
        refs = references_of(transitions[index[0]]["order"])
        history = {0: [], 1: []}
        best = {0: (0, max(1, min((len(r) for r in refs[0]), default=1))),
                1: (0, max(1, min((len(r) for r in refs[1]), default=1)))}
        for i in index:
            t = transitions[i]
            if t.get("validated_action"):
                agent = t["agent_index"]
                history[agent].append(t["validated_action"])
                best[agent] = max(((reference_depth(history[agent], r), len(r)) for r in refs[agent]),
                                  key=lambda x: x[0] / max(1, x[1]), default=best[agent])
            progress[i] = (best[0][0] + best[1][0]) / (best[0][1] + best[1][1])
    return progress


def team_returns(transitions, progress, gamma=0.99, progress_weight=1.0, success_reward=1.0, shaping="delta"):
    """Adds to every transition: team_reward (the R_t above, given to each role's last call at or before t, so
    that each role's call chain carries every team reward once, for the critic) and team_return (sum over
    t' >= t of gamma^(t' - t) R_t', t the call's timestep; the same for both roles). Also returns
    {episode: (timesteps, returns)} for time-aligned baselines (team_return_at)."""
    by_episode = defaultdict(list)
    for i, t in enumerate(transitions):
        by_episode[t["episode"]].append(i)
    curves = {}
    for e, index in by_episode.items():
        steps = sorted({transitions[i]["timestep"] for i in index})
        phi = {s: max(progress[i] for i in index if transitions[i]["timestep"] <= s) for s in steps}
        success = bool(transitions[index[0]].get("success"))
        reward, prev_phi, prev_step = {}, 0.0, None
        for s in steps:
            dt = 0 if prev_step is None else s - prev_step
            # delta: the progress increment (>= 0, sums to Phi_T <= 1: cannot be farmed); pbrs: gamma^dt Phi_t -
            # Phi_prev, policy-invariant but a holding cost (1 - gamma^dt) Phi on every timestep without progress
            reward[s] = progress_weight * ((gamma ** dt if shaping == "pbrs" else 1.0) * phi[s] - prev_phi)
            prev_phi, prev_step = phi[s], s
        if success:
            reward[steps[-1]] += success_reward
        ret, future, next_step = {}, 0.0, None
        for s in reversed(steps):
            future = reward[s] + (0.0 if next_step is None else gamma ** (next_step - s) * future)
            ret[s], next_step = future, s
        curves[e] = (steps, [ret[s] for s in steps])
        for role in (0, 1):
            calls = sorted((i for i in index if transitions[i]["agent_index"] == role),
                           key=lambda i: (transitions[i]["timestep"], transitions[i]["call_index"]))
            for i in calls:
                transitions[i]["team_reward"] = 0.0
            for s in steps:  # the role's last call at or before s carries R_s
                owner = [i for i in calls if transitions[i]["timestep"] <= s]
                if owner:
                    transitions[owner[-1]]["team_reward"] += reward[s]
        for i in index:
            transitions[i]["team_return"] = ret[transitions[i]["timestep"]]
    return curves


def team_return_at(curve, step, gamma=0.99):
    """Team return of an episode from timestep `step` on: the return of its first timestep >= step, discounted
    to `step`; 0 after its last timestep (the episode is over: nothing left to gain)."""
    steps, returns = curve
    k = bisect.bisect_left(steps, step)
    if k == len(steps):
        return 0.0
    return gamma ** (steps[k] - step) * returns[k]


TEAM_BASELINES = ("group_time", "batch", "none")


def team_local(t):
    """Local reward of a call under the team scheme: format penalty + call cost (+ validator penalty, 0 unless
    configured), kept out of every return."""
    return t["format_reward"] + t["validator_reward"] + t.get("call_cost", 0.0)


def team_advantages(transitions, curves, gamma=0.99, baseline="group_time", stats=None):
    """A = team_return - b + local, b by `baseline`:
      group_time: leave-one-out mean of team_return_at(t) over the other episodes of the call's group (same order,
          same update; `t["group"]`), t the call's timestep: the episodes are compared at the same time, after the
          same number of environment steps (an episode that ended counts with 0 left to gain);
      batch: mean team_return of the role's calls in the batch (REINFORCE with a baseline);
      none: no baseline.
    A call with a format penalty (invalid output) gets min(A, 0) + local < 0: never reinforced. No division by a
    std (no amplification of small differences)."""
    if baseline not in TEAM_BASELINES:
        raise ValueError(f"unknown team_baseline {baseline}")
    members = defaultdict(set)
    for t in transitions:
        members[t["group"]].add(t["episode"])
    role_mean = {}
    for role in (0, 1):
        values = [t["team_return"] for t in transitions if t["agent_index"] == role]
        role_mean[role] = sum(values) / len(values) if values else 0.0
    result, alone = [], 0
    for t in transitions:
        g = t["team_return"]
        if baseline == "group_time":
            others = [e for e in members[t["group"]] if e != t["episode"]]
            if others:
                b = sum(team_return_at(curves[e], t["timestep"], gamma) for e in others) / len(others)
            else:
                b, alone = 0.0, alone + 1
        elif baseline == "batch":
            b = role_mean[t["agent_index"]]
        else:
            b = 0.0
        a = g - b
        if t["format_reward"] < 0:
            a = min(a, 0.0)
        result.append(a + team_local(t))
    if stats is not None:
        stats["alone_frac"] = alone / max(1, len(transitions))
    return result


def team_metrics(transitions, advantages):
    """Team-scheme monitoring: final team progress and return per episode, and the mean advantage per call kind
    (the offline gate of docs/reward_redesign.md: progress > message ~ wait ~ other valid > rejected > invalid)."""
    final = defaultdict(float)
    for t in transitions:
        final[t["episode"]] = max(final[t["episode"]], t.get("team_progress", 0.0))
    kinds = defaultdict(list)
    for t, a in zip(transitions, advantages):
        role = "chef" if t["agent_index"] == 0 else "asst"
        if t["format_reward"] < 0:
            kind = "invalid"
        elif t.get("action_mode") == "communication":
            kind = "message"
        elif t.get("action_mode") == "wait":
            kind = "wait"
        elif t["validator_reward"] < 0 or t.get("validator_error"):
            kind = "rejected"
        elif t.get("validated_action"):
            kind = "action"
        else:
            kind = "other"
        kinds[f"{role}_{kind}"].append(a)
        kinds[f"share_{role}_{kind}"].append(1)
    n = {r: sum(1 for t in transitions if t["agent_index"] == i) for i, r in enumerate(("chef", "asst"))}
    result = {"team/progress_final": sum(final.values()) / max(1, len(final)),
              "team/return_mean": sum(t["team_return"] for t in transitions) / max(1, len(transitions))}
    for k, v in kinds.items():
        if k.startswith("share_"):
            result[f"team/{k}"] = len(v) / max(1, n[k.split("_")[1]])
        else:
            result[f"team/adv_{k}"] = sum(v) / len(v)
    return result
