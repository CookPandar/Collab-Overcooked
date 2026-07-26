"""Agent-specific MARSHAL-style GRPO advantage estimation.

MARSHAL does not group different trajectories by timestep. It places credit at
the turn/transition level, computes reward-to-go, then normalizes distinct
advantage values separately per player/agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class GrpoAdvantageConfig:
    """Configuration for MARSHAL-style GRPO advantage estimation."""

    norm_scope: str = "agent"
    normalize: str = "mean_std"
    gamma: float = 1.0
    eps: float = 1e-6
    clip: Optional[float] = None
    whiten_advantages: bool = False
    positive_advantage_requires_positive_return: bool = False
    positive_return_threshold: float = 0.0
    positive_only_advantages: bool = False


def _as_float_tensor(values: Sequence[float]) -> torch.Tensor:
    return torch.tensor([float(v) for v in values], dtype=torch.float32)


def discounted_returns(
    rewards: Sequence[float],
    *,
    agent_indices: Sequence[int],
    trajectory_ids: Optional[Sequence[Any]] = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Compute per-agent reward-to-go inside each rollout trajectory."""

    rewards_t = _as_float_tensor(rewards)
    returns = torch.zeros_like(rewards_t)
    if trajectory_ids is not None and len(trajectory_ids) != len(rewards):
        raise ValueError("trajectory_ids must have the same length as rewards.")
    running: Dict[Tuple[Any, int], float] = {}
    for idx in reversed(range(len(rewards))):
        agent = int(agent_indices[idx])
        trajectory = trajectory_ids[idx] if trajectory_ids is not None else 0
        key = (trajectory, agent)
        returns[idx] = float(rewards_t[idx].item()) + float(gamma) * running.get(key, 0.0)
        running[key] = float(returns[idx].item())
    return returns


def normalize_unique_values(
    values: torch.Tensor,
    *,
    normalize: str = "mean_std",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize distinct values and map the normalized values back."""

    mode = (normalize or "none").strip().lower()
    if values.numel() == 0 or mode in {"none", "identity"}:
        return values.clone()
    unique_values = torch.unique(values)
    if unique_values.numel() <= 1:
        return torch.zeros_like(values)
    mean = unique_values.mean()
    if mode in {"mean", "center", "centered"}:
        normalized_unique = unique_values - mean
    elif mode in {"mean_std", "std", "zscore"}:
        std = unique_values.std(unbiased=False).clamp(min=float(eps))
        normalized_unique = (unique_values - mean) / std
    else:
        raise ValueError(
            f"Unsupported GRPO normalize={normalize!r}; expected none|mean|mean_std."
        )

    output = torch.zeros_like(values)
    for original, normalized in zip(unique_values, normalized_unique):
        output[values == original] = normalized
    return output


def normalize_unique_values_by_agent(
    values: torch.Tensor,
    *,
    agent_indices: Sequence[int],
    norm_scope: str = "agent",
    normalize: str = "mean_std",
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize distinct values globally or separately for each agent."""

    scope = (norm_scope or "agent").strip().lower()
    if scope in {"batch", "global", "all"}:
        return normalize_unique_values(values, normalize=normalize, eps=eps)
    if scope not in {"agent", "player", "agent_specific"}:
        raise ValueError(f"Unsupported GRPO norm_scope={norm_scope!r}; expected agent|batch.")

    output = torch.zeros_like(values)
    buckets: Dict[int, List[int]] = {}
    for idx, agent in enumerate(agent_indices):
        buckets.setdefault(int(agent), []).append(idx)
    for indices in buckets.values():
        idx_t = torch.tensor(indices, dtype=torch.long)
        output[idx_t] = normalize_unique_values(
            values[idx_t],
            normalize=normalize,
            eps=eps,
        )
    return output


def whiten_values(values: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    if values.numel() == 0:
        return values.clone()
    if values.numel() == 1:
        return torch.zeros_like(values)
    std = values.std(unbiased=False).clamp(min=float(eps))
    return (values - values.mean()) / std


def _unique_return_counts_by_agent(
    returns: torch.Tensor,
    agent_indices: Sequence[int],
) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for agent in sorted({int(agent) for agent in agent_indices}):
        indices = [idx for idx, item in enumerate(agent_indices) if int(item) == agent]
        counts[agent] = int(torch.unique(returns[indices]).numel()) if indices else 0
    return counts


def compute_grpo_advantages(
    *,
    rewards: Sequence[float],
    agent_indices: Sequence[int],
    timesteps: Sequence[Optional[int]],
    trajectory_ids: Optional[Sequence[Any]] = None,
    positive_return_rewards: Optional[Sequence[float]] = None,
    config: GrpoAdvantageConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Return `(advantages, returns, metrics)` for Collab-Overcooked transitions.

    `timesteps` are accepted to keep the trainer interface stable, but are not
    used as group keys. This is intentional: same timestep across different
    trajectories can represent very different histories and progress states.
    """

    if not (len(rewards) == len(agent_indices) == len(timesteps)):
        raise ValueError("rewards, agent_indices and timesteps must have the same length.")
    if trajectory_ids is not None and len(trajectory_ids) != len(rewards):
        raise ValueError("trajectory_ids must have the same length as rewards.")
    if positive_return_rewards is not None and len(positive_return_rewards) != len(rewards):
        raise ValueError("positive_return_rewards must have the same length as rewards.")
    if not rewards:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, {"grpo/norm_bucket_count": 0.0}

    returns = discounted_returns(
        rewards,
        agent_indices=agent_indices,
        trajectory_ids=trajectory_ids,
        gamma=float(config.gamma),
    )
    advantages = normalize_unique_values_by_agent(
        returns,
        agent_indices=agent_indices,
        norm_scope=config.norm_scope,
        normalize=config.normalize,
        eps=float(config.eps),
    )
    if config.whiten_advantages:
        advantages = whiten_values(advantages, eps=float(config.eps))
    zeroed_positive_advantages = 0
    zeroed_negative_advantages = 0
    positive_gate_returns = returns
    if positive_return_rewards is not None:
        positive_gate_returns = discounted_returns(
            positive_return_rewards,
            agent_indices=agent_indices,
            trajectory_ids=trajectory_ids,
            gamma=float(config.gamma),
        )
    if config.positive_advantage_requires_positive_return:
        zero_mask = (
            positive_gate_returns <= float(config.positive_return_threshold)
        ) & (advantages > 0.0)
        zeroed_positive_advantages = int(zero_mask.sum().item())
        advantages = torch.where(zero_mask, torch.zeros_like(advantages), advantages)
    if config.positive_only_advantages:
        negative_mask = advantages < 0.0
        zeroed_negative_advantages = int(negative_mask.sum().item())
        advantages = torch.where(negative_mask, torch.zeros_like(advantages), advantages)
    if config.clip is not None:
        clip = abs(float(config.clip))
        advantages = torch.clamp(advantages, min=-clip, max=clip)

    unique_counts = _unique_return_counts_by_agent(returns, agent_indices)
    trajectory_count = (
        float(len({str(item) for item in trajectory_ids}))
        if trajectory_ids is not None
        else 1.0
    )
    agent_scoped = (config.norm_scope or "agent").strip().lower() in {
        "agent",
        "player",
        "agent_specific",
    }
    metrics = {
        "grpo/global_transition_count": float(len(rewards)),
        "grpo/trajectory_count": trajectory_count,
        "grpo/norm_bucket_count": (
            float(sum(unique_counts.values()))
            if agent_scoped
            else float(torch.unique(returns).numel())
        ),
        "grpo/return_mean": float(returns.mean().item()),
        "grpo/return_std": float(returns.std(unbiased=False).item()) if returns.numel() > 1 else 0.0,
        "grpo/positive_gate_return_mean": float(positive_gate_returns.mean().item()),
        "grpo/adv_mean": float(advantages.mean().item()),
        "grpo/adv_std": float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        "grpo/nonpositive_return_positive_adv_zeroed": float(zeroed_positive_advantages),
        "grpo/negative_adv_zeroed": float(zeroed_negative_advantages),
        "grpo/positive_adv_count": float((advantages > 0.0).sum().item()),
        "grpo/negative_adv_count": float((advantages < 0.0).sum().item()),
        "grpo/zero_adv_count": float((advantages == 0.0).sum().item()),
    }
    for agent, count in unique_counts.items():
        metrics[f"grpo/agent{agent}_unique_return_count"] = float(count)
    return advantages, returns, metrics
