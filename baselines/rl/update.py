"""One policy update from a batch of transitions, with the AgentGym or the MARSHAL actor objective.

Both objectives are PPO on response tokens, with old log-probs recomputed by the current adapters before the
update and a KL penalty to the frozen initial adapters (estimator exp(ref - new) - (ref - new) - 1).

agentgym  (AgentGym-RL / verl DataParallelPPOActor): each role is updated separately; its calls are taken in
          order in minibatches of 16; per-call token-mean PPO-clip loss (0.2) plus kl_coef * per-call token-mean
          KL, summed / 16; gradient clipping at 1.0.
marshal   (our GRPO actor): calls of both roles are shuffled together into pairs; per-call token-mean
          dual-clip PPO loss (0.15, dual clip 1.3) averaged over the pair plus kl_coef * pair token-mean KL
          (clamped to [0, 10] per token); 8 pairs per optimizer step; gradient clipping at 0.5.

A fresh Adam optimizer is used for every update. The math is the same as scoring one call at a time; calls
are batched up to `tokens_per_batch` tokens per forward pass for speed. With `world` > 1 (torch.distributed),
the forward passes of every optimizer step are split across ranks and the gradients are summed before the step.
"""

import random
import time

import torch

OBJECTIVES = {
    "agentgym": dict(clip=0.2, dual_clip=None, kl_coef=0.02, grad_clip=1.0, minibatch=16, shuffle=False),
    "marshal": dict(clip=0.15, dual_clip=1.3, kl_coef=0.02, grad_clip=0.5, minibatch=2, steps_per_update=8,
                    shuffle=True),
}


def next_lr_scale(adaptive, scale, update_kl):
    """Learning-rate scale of the next update. `adaptive`: None (constant learning rate) or
    {kl_threshold, decay, min_scale}: the scale is multiplied by `decay` when the mean update KL
    (k3 estimator, per pair for marshal, per call for agentgym) exceeds `kl_threshold`."""
    if not adaptive:
        return 1.0
    if update_kl > adaptive["kl_threshold"]:
        scale = max(adaptive["min_scale"], scale * adaptive["decay"])
    return scale


def _common_prefix(a, b, limit):
    mismatch = (a[:limit] != b[:limit]).nonzero()
    return int(mismatch[0]) if len(mismatch) else limit


def forward_chunks(samples, indices, budget):
    """(role, chunk) pairs covering `indices`: one role per chunk, at most `budget` tokens per forward pass.

    Calls are sorted by prompt so that calls sharing a long prompt prefix are batched together (the prefix is
    encoded once, see LoraPolicy.token_logprobs). Old log-probs are computed with the same chunks as the
    training forward passes, so the first PPO ratio of every call is exactly 1 (bf16 results depend on the
    batch composition)."""
    for role in (0, 1):
        order = sorted((i for i in indices if samples[i]["agent_index"] == role),
                       key=lambda i: samples[i]["prompt_ids"].tolist())
        chunk, shared, longest = [], 0, 0
        for i in order:
            prompt = samples[i]["prompt_ids"]
            length = len(prompt) + len(samples[i]["response_ids"])
            if chunk:
                first = samples[chunk[0]]["prompt_ids"]
                new_shared = min(shared, _common_prefix(first, prompt, min(len(first), len(prompt)) - 1))
                new_longest = max(longest, length)
                if new_shared + (len(chunk) + 1) * (new_longest - new_shared) > budget:
                    yield role, chunk
                    chunk = []
            if not chunk:
                chunk, shared, longest = [i], len(prompt) - 1, length
            else:
                chunk.append(i)
                shared, longest = new_shared, new_longest
        if chunk:
            yield role, chunk


def score(policy, samples, chunks, adapters, temperature, grad=False, entropy=None):
    """Token log-probs of the samples in `chunks`, with the adapter of each chunk's role; a dict `entropy` receives
    the token entropies."""
    result = {}
    for role, chunk in chunks:
        with torch.set_grad_enabled(grad):
            values = policy.token_logprobs(adapters[role], [samples[i]["prompt_ids"] for i in chunk],
                                           [samples[i]["response_ids"] for i in chunk], temperature,
                                           with_entropy=entropy is not None)
        if entropy is not None:
            values, entropies = values
            entropy.update(zip(chunk, entropies))
        result.update(zip(chunk, values))
    return result


def _kl(new, ref):
    x = ref - new
    return torch.exp(x) - x - 1


def _optimizer_steps(samples, objective, rng, update_steps=None):
    """List of optimizer steps; each is a list of (sample index, pg weight, kl weight, kl normalizer group).
    update_steps (agentgym only): instead of minibatches of 16 calls in call order (so the number of optimizer
    steps grows with the number of calls, and a minibatch holds consecutive calls of one episode), each role's
    calls are shuffled and split into `update_steps` minibatches of (nearly) equal size, loss = their mean."""
    cfg = OBJECTIVES[objective]
    steps = []
    if objective == "agentgym" and update_steps:
        for role in (0, 1):
            role_indices = [i for i, s in enumerate(samples) if s["agent_index"] == role]
            rng.shuffle(role_indices)
            k = min(update_steps, len(role_indices))
            for j in range(k):
                chunk = role_indices[j::k]
                steps.append([(i, 1.0 / len(chunk), 1.0 / len(chunk), None) for i in chunk])
        return steps
    if objective == "agentgym":
        for role in (0, 1):
            role_indices = [i for i, s in enumerate(samples) if s["agent_index"] == role]
            for start in range(0, len(role_indices), cfg["minibatch"]):
                chunk = role_indices[start:start + cfg["minibatch"]]
                weight = 1.0 / cfg["minibatch"]
                steps.append([(i, weight, weight, None) for i in chunk])
        return steps
    order = list(range(len(samples)))
    if cfg["shuffle"]:
        rng.shuffle(order)
    pairs = [order[k:k + cfg["minibatch"]] for k in range(0, len(order), cfg["minibatch"])]
    for start in range(0, len(pairs), cfg["steps_per_update"]):
        step = []
        for p, pair in enumerate(pairs[start:start + cfg["steps_per_update"]]):
            weight = 1.0 / cfg["steps_per_update"]
            step += [(i, weight / len(pair), weight, (start + p, pair)) for i in pair]
        steps.append(step)
    return steps


def ppo_update(policy, samples, objective="agentgym", temperature=0.7, lr=1e-5, tokens_per_batch=8192,
               kl_coef=None, entropy_coef=0.0, seed=0, rank=0, world=1, min_token_logprob=None, update_steps=None):
    """`samples`: dicts with agent_index, prompt_ids, response_ids (LongTensor) and advantage; identical on all
    ranks. `entropy_coef`: verl actor entropy bonus, - entropy_coef * token-mean entropy per call (weighted as the
    pg loss). `min_token_logprob`: tokens whose recomputed old log-prob is below it (sampled by vLLM but nearly
    impossible under the trained policy: the source of the KL spikes) are left out of the pg and KL terms.
    Returns averaged metrics."""
    cfg = dict(OBJECTIVES[objective])
    if kl_coef is not None:
        cfg["kl_coef"] = kl_coef
    samples = [s for s in samples if len(s["response_ids"]) > 0]
    steps = _optimizer_steps(samples, objective, random.Random(seed), update_steps)
    step_chunks = [list(forward_chunks(samples, [i for i, *_ in step], tokens_per_batch))[rank::world]
                   for step in steps]
    all_chunks = [c for chunks in step_chunks for c in chunks]
    timing = {"started": time.time()}
    old = score(policy, samples, all_chunks, policy.trainable, temperature)
    timing["old"] = time.time()
    ref = score(policy, samples, all_chunks, policy.references, temperature) if policy.references else None
    timing["ref"] = time.time()
    params = policy.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    metrics = {"pg_loss": [], "kl": [], "kl_clamped": [], "masked_tokens": [], "ppo_kl": [], "clipfrac": [],
               "grad_norm": [], "entropy": []}
    update_kl = {}  # group (pair or call) -> [sum, tokens] of (ratio - 1) - log ratio

    for step, chunks in zip(steps, step_chunks):
        optimizer.zero_grad(set_to_none=True)
        weights = {i: (pg_w, kl_w, group) for i, pg_w, kl_w, group in step}
        pair_tokens = {group[0]: sum(len(samples[j]["response_ids"]) for j in group[1])
                       for _, _, group in weights.values() if group is not None}
        for role, chunk in chunks:
            entropy = {} if entropy_coef else None
            new = score(policy, samples, [(role, chunk)], policy.trainable, temperature, grad=True, entropy=entropy)
            loss = 0.0
            for i in chunk:
                pg_w, kl_w, group = weights[i]
                advantage = float(samples[i]["advantage"])
                token_advantage = samples[i].get("token_advantage")
                keep = None
                if min_token_logprob is not None:
                    keep = (old[i] >= min_token_logprob).to(new[i].dtype)
                    metrics["masked_tokens"].append(float(1 - keep.mean()))
                log_ratio = new[i] - old[i]
                if keep is not None:
                    log_ratio = log_ratio * keep  # masked tokens: ratio 1, no pg gradient
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"])
                if token_advantage is not None:  # per-token advantages (think reward on the reasoning tokens)
                    advantage = token_advantage.to(ratio.device)
                    pg = torch.maximum(-advantage * ratio, -advantage * clipped)
                    if cfg["dual_clip"] is not None:
                        pg = torch.where(advantage < 0, torch.minimum(pg, -cfg["dual_clip"] * advantage), pg)
                else:
                    pg = torch.maximum(-advantage * ratio, -advantage * clipped)
                    if cfg["dual_clip"] is not None and advantage < 0:
                        pg = torch.minimum(pg, torch.full_like(pg, -cfg["dual_clip"] * advantage))
                if keep is not None:
                    pg = pg * keep
                loss = loss + pg_w * (pg.sum() / keep.sum().clamp(min=1) if keep is not None else pg.mean())
                if entropy is not None:
                    loss = loss - entropy_coef * pg_w * entropy[i].mean()
                    metrics["entropy"].append(float(entropy[i].detach().mean()))
                if ref is not None:
                    kl = _kl(new[i], ref[i])
                    if keep is not None:
                        kl = kl * keep
                    if group is None:
                        loss = loss + cfg["kl_coef"] * kl_w * torch.clamp(kl, -10, 10).mean()
                    else:
                        loss = loss + cfg["kl_coef"] * kl_w * torch.clamp(kl, 0, 10).sum() / pair_tokens[group[0]]
                    metrics["kl"].append(float(kl.detach().mean()))
                    metrics["kl_clamped"].append(float(torch.clamp(kl.detach(), 0, 10).mean()))
                metrics["pg_loss"].append(float(pg.detach().mean()))
                metrics["ppo_kl"].append(float(-log_ratio.detach().mean()))
                metrics["clipfrac"].append(float(((ratio.detach() - 1).abs() > cfg["clip"]).float().mean()))
                part = update_kl.setdefault(group[0] if group is not None else ("call", i), [0.0, 0])
                part[0] += float((ratio.detach() - 1 - log_ratio.detach()).sum())
                part[1] += len(log_ratio)
            loss.backward()
        if world > 1:  # sum gradients; parameters without a gradient on any rank keep None (Adam skips them)
            has_grad = torch.tensor([p.grad is not None for p in params], dtype=torch.uint8, device=params[0].device)
            torch.distributed.all_reduce(has_grad, op=torch.distributed.ReduceOp.MAX)
            for p, flag in zip(params, has_grad.tolist()):
                if flag:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    torch.distributed.all_reduce(p.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        optimizer.step()
        metrics["grad_norm"].append(float(grad_norm))
    if world > 1:
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, (metrics, update_kl))
        metrics = {k: [x for m, _ in gathered for x in m[k]] for k in metrics}
        metrics["grad_norm"] = gathered[0][0]["grad_norm"]
        merged = {}
        for _, parts in gathered:
            for key, (total, n) in parts.items():
                merged.setdefault(key, [0.0, 0])
                merged[key][0] += total
                merged[key][1] += n
        update_kl = merged
    result = {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}
    result["update_kl"] = (sum(max(0.0, total / n) for total, n in update_kl.values()) / len(update_kl)
                           if update_kl else 0.0)
    result["optimizer_steps"] = len(steps)
    result["forward_passes"] = len(all_chunks)
    result["prompt_tokens"] = sum(len(s["prompt_ids"]) for s in samples)
    result["seconds_old_logprobs"] = timing["old"] - timing["started"]
    result["seconds_ref_logprobs"] = timing["ref"] - timing["old"]
    result["seconds_train"] = time.time() - timing["ref"]
    result["calls"] = len(samples)
    return result


# ---- critic (PPO with GAE, the AgentGym-RL / verl DataParallelPPOCritic settings) ----------------------
CRITIC = dict(lr=1e-5, cliprange_value=0.5, minibatch=16, grad_clip=1.0)


def critic_values(policy, samples, tokens_per_batch=8192, token_level=False):
    """Value of every sample's prompt under its role's critic (no gradient); with `token_level` a list of the values
    before each response token."""
    result = [0.0] * len(samples)
    with torch.no_grad():
        for role, chunk in forward_chunks(samples, list(range(len(samples))), tokens_per_batch):
            prompts = [samples[i]["prompt_ids"] for i in chunk]
            if token_level:
                values = [v.tolist() for v in policy.token_values(role, prompts, [samples[i]["response_ids"] for i in chunk])]
            else:
                values = policy.values(role, prompts).tolist()
            for i, v in zip(chunk, values):
                result[i] = v
    return result


def critic_update(policy, samples, returns, old_values, tokens_per_batch=8192, lr=None, cliprange_value=None,
                  head_lr=None, epochs=1, persistent=False, seed=0):
    """`epochs` passes over the calls of each role, minibatches of 16, clipped value loss as verl
    compute_value_loss: 0.5 * mean(max((V - R)^2, (clip(V, V_old -+ c) - R)^2)) (cliprange_value None: no clip).
    A fresh Adam per update and role, or with `persistent` one Adam per role kept across updates (policy attribute);
    `head_lr`: learning rate of the value head (default `lr`; the head starts at zero, the LoRA from the actor).
    With per-token `returns` / `old_values` (lists) the mean is over the response tokens of the minibatch."""
    cfg = dict(CRITIC, **{k: v for k, v in (("lr", lr), ("cliprange_value", cliprange_value)) if v is not None})
    no_clip = cliprange_value is None
    token_level = bool(samples) and isinstance(returns[0], list)
    metrics = {"value_loss": [], "value_clipfrac": [], "grad_norm": []}
    optimizers = getattr(policy, "critic_optimizers", None) if persistent else None
    if persistent and optimizers is None:
        optimizers = policy.critic_optimizers = {}
    rng = random.Random(seed)
    for role in sorted(policy.critics):
        params = policy.critic_parameters(role)
        head = list(policy.value_heads[role].parameters())
        groups = [{"params": [p for p in params if all(p is not h for h in head)], "lr": cfg["lr"]},
                  {"params": head, "lr": head_lr or cfg["lr"]}]
        optimizer = optimizers.get(role) if optimizers is not None else None
        if optimizer is None:
            optimizer = torch.optim.Adam(groups)
            if optimizers is not None:
                optimizers[role] = optimizer
        index = [i for i, s in enumerate(samples) if s["agent_index"] == role
                 and (not token_level or len(s["response_ids"]))]
        order = []
        for epoch in range(epochs):  # the first pass in order (as before), later passes shuffled
            batch_order = list(index)
            if epoch:
                rng.shuffle(batch_order)
            order += [batch_order[k:k + cfg["minibatch"]] for k in range(0, len(batch_order), cfg["minibatch"])]
        for batch in order:
            denominator = sum(len(returns[i]) for i in batch) if token_level else len(batch)
            optimizer.zero_grad(set_to_none=True)
            for _, chunk in forward_chunks(samples, batch, tokens_per_batch):
                prompts = [samples[i]["prompt_ids"] for i in chunk]
                if token_level:
                    values = torch.cat(policy.token_values(role, prompts, [samples[i]["response_ids"] for i in chunk]))
                    target = torch.tensor([r for i in chunk for r in returns[i]], device=values.device)
                    old = torch.tensor([v for i in chunk for v in old_values[i]], device=values.device)
                else:
                    values = policy.values(role, prompts)
                    target = torch.tensor([returns[i] for i in chunk], device=values.device)
                    old = torch.tensor([old_values[i] for i in chunk], device=values.device)
                clipped = values if no_clip else old + torch.clamp(values - old, -cfg["cliprange_value"],
                                                                    cfg["cliprange_value"])
                loss = 0.5 * torch.maximum((values - target) ** 2, (clipped - target) ** 2)
                (loss.sum() / denominator).backward()
                metrics["value_loss"] += loss.detach().tolist()
                metrics["value_clipfrac"] += ((values - old).abs() > cfg["cliprange_value"]).float().tolist()
            metrics["grad_norm"].append(float(torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])))
            optimizer.step()
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in metrics.items()}
