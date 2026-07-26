import torch

from Grpo.advantages import GrpoAdvantageConfig, compute_grpo_advantages


def test_marshal_unique_value_norm_is_agent_specific():
    adv, returns, metrics = compute_grpo_advantages(
        rewards=[1.0, 3.0, 10.0, 20.0],
        agent_indices=[0, 0, 1, 1],
        timesteps=[0, 0, 0, 1],
        config=GrpoAdvantageConfig(norm_scope="agent", normalize="mean", gamma=0.0),
    )

    assert torch.allclose(returns, torch.tensor([1.0, 3.0, 10.0, 20.0]))
    assert torch.allclose(adv[:2], torch.tensor([-1.0, 1.0]))
    assert torch.allclose(adv[2:], torch.tensor([-5.0, 5.0]))
    assert metrics["grpo/norm_bucket_count"] == 4.0


def test_agent_specific_discounted_returns_do_not_mix_agents():
    adv, returns, _ = compute_grpo_advantages(
        rewards=[1.0, 10.0, 2.0, 20.0],
        agent_indices=[0, 1, 0, 1],
        timesteps=[0, 0, 1, 1],
        config=GrpoAdvantageConfig(norm_scope="agent", normalize="mean", gamma=1.0),
    )

    assert torch.allclose(returns, torch.tensor([3.0, 30.0, 2.0, 20.0]))
    assert torch.allclose(adv, torch.tensor([0.5, 5.0, -0.5, -5.0]))


def test_discounted_returns_reset_at_rollout_boundary():
    _, returns, _ = compute_grpo_advantages(
        rewards=[1.0, 2.0, 10.0, 20.0],
        agent_indices=[0, 0, 0, 0],
        timesteps=[0, 1, 0, 1],
        trajectory_ids=["rank0", "rank0", "rank1", "rank1"],
        config=GrpoAdvantageConfig(norm_scope="agent", normalize="none", gamma=1.0),
    )

    assert torch.allclose(returns, torch.tensor([3.0, 2.0, 30.0, 20.0]))


def test_positive_advantage_can_require_positive_return():
    adv, returns, metrics = compute_grpo_advantages(
        rewards=[0.0, -2.0, -4.0],
        agent_indices=[0, 0, 0],
        timesteps=[0, 1, 2],
        trajectory_ids=["r0", "r1", "r2"],
        config=GrpoAdvantageConfig(
            norm_scope="agent",
            normalize="mean",
            gamma=1.0,
            positive_advantage_requires_positive_return=True,
        ),
    )

    assert torch.allclose(returns, torch.tensor([0.0, -2.0, -4.0]))
    assert torch.allclose(adv, torch.tensor([0.0, 0.0, -2.0]))
    assert metrics["grpo/nonpositive_return_positive_adv_zeroed"] == 1.0


def test_positive_advantage_gate_can_use_task_success_signal():
    adv, returns, metrics = compute_grpo_advantages(
        rewards=[1.0, 35.0, 30.0, 40.0],
        agent_indices=[0, 0, 0, 0],
        timesteps=[0, 0, 0, 0],
        trajectory_ids=["fail_low", "fail_dense", "success_a", "success_b"],
        positive_return_rewards=[0.0, 0.0, 20.0, 20.0],
        config=GrpoAdvantageConfig(
            norm_scope="agent",
            normalize="mean",
            gamma=0.0,
            positive_advantage_requires_positive_return=True,
        ),
    )

    assert torch.allclose(returns, torch.tensor([1.0, 35.0, 30.0, 40.0]))
    assert torch.allclose(adv, torch.tensor([-25.5, 0.0, 3.5, 13.5]))
    assert metrics["grpo/nonpositive_return_positive_adv_zeroed"] == 1.0
    assert metrics["grpo/positive_gate_return_mean"] == 10.0


def test_positive_only_advantages_zeroes_negative_updates():
    adv, returns, metrics = compute_grpo_advantages(
        rewards=[1.0, 35.0, 30.0, 40.0],
        agent_indices=[0, 0, 0, 0],
        timesteps=[0, 0, 0, 0],
        trajectory_ids=["fail_low", "fail_dense", "success_a", "success_b"],
        positive_return_rewards=[0.0, 0.0, 20.0, 20.0],
        config=GrpoAdvantageConfig(
            norm_scope="agent",
            normalize="mean",
            gamma=0.0,
            positive_advantage_requires_positive_return=True,
            positive_only_advantages=True,
        ),
    )

    assert torch.allclose(returns, torch.tensor([1.0, 35.0, 30.0, 40.0]))
    assert torch.allclose(adv, torch.tensor([0.0, 0.0, 3.5, 13.5]))
    assert metrics["grpo/nonpositive_return_positive_adv_zeroed"] == 1.0
    assert metrics["grpo/negative_adv_zeroed"] == 1.0
    assert metrics["grpo/positive_adv_count"] == 2.0
    assert metrics["grpo/negative_adv_count"] == 0.0
    assert metrics["grpo/zero_adv_count"] == 2.0


def test_whiten_advantages_centers_normalized_values():
    adv, returns, _ = compute_grpo_advantages(
        rewards=[20.0, 0.0, 0.0, 0.0],
        agent_indices=[0, 0, 0, 0],
        timesteps=[0, 1, 2, 3],
        trajectory_ids=["good", "bad1", "bad2", "bad3"],
        config=GrpoAdvantageConfig(
            norm_scope="agent",
            normalize="mean_std",
            gamma=1.0,
            whiten_advantages=True,
        ),
    )

    assert torch.allclose(returns, torch.tensor([20.0, 0.0, 0.0, 0.0]))
    assert abs(float(adv.mean().item())) < 1e-6
    assert torch.isclose(adv.std(unbiased=False), torch.tensor(1.0), atol=1e-6)
