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


def score(policy, samples, chunks, adapters, temperature, grad=False):
    """Token log-probs of the samples in `chunks`, with the adapter of each chunk's role."""
    result = {}
    for role, chunk in chunks:
        with torch.set_grad_enabled(grad):
            values = policy.token_logprobs(adapters[role], [samples[i]["prompt_ids"] for i in chunk],
                                           [samples[i]["response_ids"] for i in chunk], temperature)
        result.update(zip(chunk, values))
    return result


def _kl(new, ref):
    x = ref - new
    return torch.exp(x) - x - 1


def _optimizer_steps(samples, objective, rng):
    """List of optimizer steps; each is a list of (sample index, pg weight, kl weight, kl normalizer group)."""
    cfg = OBJECTIVES[objective]
    steps = []
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
               kl_coef=None, seed=0, rank=0, world=1):
    """`samples`: dicts with agent_index, prompt_ids, response_ids (LongTensor) and advantage; identical on all
    ranks. Returns averaged metrics."""
    cfg = dict(OBJECTIVES[objective])
    if kl_coef is not None:
        cfg["kl_coef"] = kl_coef
    samples = [s for s in samples if len(s["response_ids"]) > 0]
    steps = _optimizer_steps(samples, objective, random.Random(seed))
    step_chunks = [list(forward_chunks(samples, [i for i, *_ in step], tokens_per_batch))[rank::world]
                   for step in steps]
    all_chunks = [c for chunks in step_chunks for c in chunks]
    old = score(policy, samples, all_chunks, policy.trainable, temperature)
    ref = score(policy, samples, all_chunks, policy.references, temperature) if policy.references else None
    params = policy.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    metrics = {"pg_loss": [], "kl": [], "ppo_kl": [], "clipfrac": [], "grad_norm": []}
    update_kl = {}  # group (pair or call) -> [sum, tokens] of (ratio - 1) - log ratio

    for step, chunks in zip(steps, step_chunks):
        optimizer.zero_grad(set_to_none=True)
        weights = {i: (pg_w, kl_w, group) for i, pg_w, kl_w, group in step}
        pair_tokens = {group[0]: sum(len(samples[j]["response_ids"]) for j in group[1])
                       for _, _, group in weights.values() if group is not None}
        for role, chunk in chunks:
            new = score(policy, samples, [(role, chunk)], policy.trainable, temperature, grad=True)
            loss = 0.0
            for i in chunk:
                pg_w, kl_w, group = weights[i]
                advantage = float(samples[i]["advantage"])
                log_ratio = new[i] - old[i]
                ratio = torch.exp(log_ratio)
                clipped = torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"])
                pg = torch.maximum(-advantage * ratio, -advantage * clipped)
                if cfg["dual_clip"] is not None and advantage < 0:
                    pg = torch.minimum(pg, torch.full_like(pg, -cfg["dual_clip"] * advantage))
                loss = loss + pg_w * pg.mean()
                if ref is not None:
                    kl = _kl(new[i], ref[i])
                    if group is None:
                        loss = loss + cfg["kl_coef"] * kl_w * torch.clamp(kl, -10, 10).mean()
                    else:
                        loss = loss + cfg["kl_coef"] * kl_w * torch.clamp(kl, 0, 10).sum() / pair_tokens[group[0]]
                    metrics["kl"].append(float(kl.detach().mean()))
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
    result["calls"] = len(samples)
    return result
