"""CPU checks of the RL baseline: advantages, update batching and the learning-rate control."""
import pytest

torch = pytest.importorskip("torch")

from baselines.rl.advantages import compute_advantages
from baselines.rl.update import _optimizer_steps, forward_chunks, next_lr_scale


def call(timestep, agent=0, reward=0.0, terminal=0.0, fmt=0.0, val=0.0, episode=0):
    return dict(episode=episode, agent_index=agent, timestep=timestep, reward=reward + terminal + fmt + val,
                terminal_reward=terminal, format_reward=fmt, validator_reward=val)


def test_advantages_discount_terminal_reward_over_timesteps():
    calls = [call(0, reward=0.5), call(0), call(2, val=-0.1), call(3, terminal=2.0), call(1, agent=1),
             call(0, fmt=-0.2, reward=0.0), call(0, episode=1)]
    a = compute_advantages(calls, gamma=0.9)
    assert a[0] == pytest.approx(0.5 + 2 * 0.9 ** 3)
    assert a[1] == pytest.approx(2 * 0.9 ** 3)  # a second call of the same timestep gets the same credit
    assert a[2] == pytest.approx(-0.1)          # validator penalty: no terminal credit
    assert a[3] == pytest.approx(2.0)
    assert a[4] == 0.0 and a[6] == 0.0          # other agent / other episode
    assert a[5] == pytest.approx(-0.2)


def test_advantages_clip_and_no_positive_advantage_with_format_penalty():
    a = compute_advantages([call(0, reward=50.0), call(1, reward=1.0, fmt=-0.2)])
    assert a == [10.0, 0.0]


def sample(role, prompt, response=3):
    return dict(agent_index=role, prompt_ids=torch.tensor(prompt), response_ids=torch.ones(response, dtype=torch.long),
                advantage=1.0)


def test_forward_chunks_cover_each_role_once_within_budget():
    shared = list(range(100))
    samples = [sample(i % 2, shared + [1000 + i] * (i % 7)) for i in range(30)]
    chunks = list(forward_chunks(samples, range(30), budget=300))
    seen = sorted(i for _, chunk in chunks for i in chunk)
    assert seen == list(range(30))
    for role, chunk in chunks:
        assert all(samples[i]["agent_index"] == role for i in chunk)


def test_optimizer_steps():
    import random
    samples = [sample(i % 2, [1, 2, 3]) for i in range(40)]
    agentgym = _optimizer_steps(samples, "agentgym", random.Random(0))
    assert [len(s) for s in agentgym] == [16, 4, 16, 4]
    assert all(len({samples[i]["agent_index"] for i, *_ in s}) == 1 for s in agentgym)
    marshal = _optimizer_steps(samples, "marshal", random.Random(0))
    assert [len(s) for s in marshal] == [16, 16, 8]
    assert sorted(i for s in marshal for i, *_ in s) == list(range(40))


def test_adaptive_learning_rate():
    rule = dict(kl_threshold=0.05, decay=0.5, min_scale=0.1)
    assert next_lr_scale(None, 1.0, 1.0) == 1.0
    assert next_lr_scale(rule, 1.0, 0.01) == 1.0
    assert next_lr_scale(rule, 1.0, 0.06) == 0.5
    assert next_lr_scale(rule, 0.15, 0.06) == 0.1
