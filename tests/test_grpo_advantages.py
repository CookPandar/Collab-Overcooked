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
