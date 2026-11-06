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


# ---- corrected GRPO / PPO options (defaults must reproduce the old estimators) -------------------------
from baselines.rl.advantages import (advantage_metrics, gae_advantages, grpo_advantages,  # noqa: E402
                                     step_advantages)


def gcall(episode, timestep, group=0, agent=0, ites=0.0, terminal=0.0, fmt=0.0, val=0.0, obs="", success=False,
          mode="planner_main"):
    return dict(episode=episode, agent_index=agent, timestep=timestep, call_index=timestep, group=group,
                order="o", reward=ites + terminal + fmt + val, sequence_reward=ites, terminal_reward=terminal,
                format_reward=fmt, validator_reward=val, success=success, action_mode=mode,
                observation=f"Kitchen states: {obs}\nAgent State: x")


def test_grpo_defaults_are_verl_outcome_grpo():
    calls = [gcall(0, 0, ites=1.0), gcall(0, 1, val=-0.1), gcall(1, 0), gcall(2, 0, ites=0.5)]
    scores = [0.9, 0.0, 0.5]
    mean = sum(scores) / 3
    std = (sum((s - mean) ** 2 for s in scores) / 2) ** 0.5
    a = grpo_advantages(calls)
    assert a[0] == a[1] == pytest.approx((0.9 - mean) / (std + 1e-6))
    assert a[2] == pytest.approx((0.0 - mean) / (std + 1e-6))
    assert a == grpo_advantages(calls, norm="std", score="all", min_spread=None, step_weight=0.0)


def test_grpo_dr_norm_and_local_penalties():
    calls = [gcall(0, 0, ites=1.0), gcall(0, 1, val=-0.1), gcall(1, 0), gcall(1, 1, fmt=-0.2)]
    a = grpo_advantages(calls, norm="none")
    assert a[0] == a[1] == pytest.approx(0.9 - 0.35)  # scores 0.9 / -0.2 incl. penalties, no std division
    a = grpo_advantages(calls, norm="none", score="outcome")
    assert a[0] == pytest.approx(0.5)                 # outcome scores 1.0 / 0.0: mean 0.5
    assert a[1] == pytest.approx(0.5 - 0.1)           # the penalty stays on its own call
    assert a[2] == pytest.approx(-0.5) and a[3] == pytest.approx(-0.5 - 0.2)


def test_grpo_drops_zero_spread_groups():
    calls = [gcall(0, 0, val=-0.1), gcall(1, 0), gcall(2, 0, group=1, ites=1.0), gcall(3, 0, group=1)]
    stats = {}
    a = grpo_advantages(calls, norm="none", score="outcome", min_spread="1e-6", stats=stats)
    assert a[0] == pytest.approx(-0.1) and a[1] == 0.0  # flat outcome group: only the local penalty
    assert a[2] == pytest.approx(0.5) and a[3] == pytest.approx(-0.5)
    assert stats["dropped_groups"] == 1 and stats["groups"] == 2 and stats["dropped_frac"] == 0.5
    # the old score (with penalties) is not flat, so nothing is dropped
    stats = {}
    grpo_advantages(calls, min_spread=1e-6, stats=stats)
    assert stats["dropped_groups"] == 0


def test_grpo_step_level_groups_same_anchor_state():
    calls = [gcall(0, 0, obs="s"), gcall(0, 1, obs="t", ites=1.0), gcall(1, 0, obs="s"), gcall(1, 1, obs="u")]
    step = step_advantages(calls, norm="none", gamma=0.5)
    assert step[0] == pytest.approx(0.25) and step[2] == pytest.approx(-0.25)  # returns 0.5 vs 0 at state s
    assert step[1] == 0.0 and step[3] == 0.0                                   # alone in their step groups
    a = grpo_advantages(calls, norm="none", score="outcome", step_weight=1.0, step_gamma=0.5)
    assert a[0] == pytest.approx(0.5 + 0.25) and a[2] == pytest.approx(-0.5 - 0.25)


def test_gae_local_penalties_and_std_floor():
    calls = [gcall(0, t) for t in range(3)] + [gcall(1, t) for t in range(3)]
    calls[2]["validator_reward"] = calls[2]["reward"] = -0.1
    calls[5]["sequence_reward"] = calls[5]["reward"] = 1.0
    values = [0.0] * 6
    old, old_returns = gae_advantages(calls, values, 1.0, 1.0)
    assert (old, old_returns) == gae_advantages(calls, values, 1.0, 1.0, local_penalties=False, std_floor=0.0)
    assert old_returns[0] == pytest.approx(-0.1)          # the penalty reaches the earlier calls
    _, returns = gae_advantages(calls, values, 0.9, 0.5, local_penalties=True)
    assert returns[0] == 0.0 and returns[2] == 0.0          # ... not with local penalties
    assert returns[4] == pytest.approx(0.9 * 0.5 * 1.0)   # A_4 = (gamma lam) delta_5 with V = 0
    floor, _ = gae_advantages(calls, values, 0.9, 0.5, local_penalties=True, std_floor=100.0)
    raw = [0.0, 0.0, -0.1, 0.45 * 0.45, 0.45, 1.0]
    mean = sum(raw) / 6
    assert floor == pytest.approx([(r - mean) / 100.0 for r in raw])


def test_advantage_metrics():
    calls = [gcall(0, 0, ites=1.0), gcall(0, 1, mode="communication"), gcall(0, 2, val=-0.1),
             gcall(1, 0, ites=1.0, success=True)]
    m = advantage_metrics(calls, [1.0, 0.0, -1.0, 5.0])
    assert m["adv/fail_progress"] == 1.0 and m["adv/fail_msg_wait"] == 0.0 and m["adv/fail_rejected"] == -1.0
    assert m["adv/fail_corr_timestep"] == pytest.approx(-1.0)


# ---- team-grounded return (docs/reward_redesign.md) ----
from baselines.rl.advantages import team_advantages, team_progress, team_return_at, team_returns  # noqa: E402

REFS = {0: [["a1", "a2"]], 1: [["b1", "b2"]]}


def tcall(episode, timestep, agent, action=None, fmt=0.0, val=0.0, success=False, group=0, call_index=0):
    return dict(episode=episode, order="o", agent_index=agent, timestep=timestep, call_index=call_index,
                validated_action=action, format_reward=fmt, validator_reward=val, call_cost=-0.01,
                success=success, group=group)


def test_team_progress_counts_both_agents_in_reference_order():
    calls = [tcall(0, 0, 1, "b1"), tcall(0, 1, 1, "x"), tcall(0, 2, 0, "a2"), tcall(0, 3, 0, "a1"), tcall(0, 4, 1, "b2")]
    assert team_progress(calls, lambda order: REFS) == [0.25, 0.25, 0.25, 0.5, 0.75]


def test_team_returns_share_progress_across_roles_and_discount_by_timestep():
    # episode 0: the chef asks at t0 (no action), the assistant makes progress at t1 and t3, success at t3
    calls = [tcall(0, 0, 0, success=True), tcall(0, 1, 1, "b1", success=True), tcall(0, 3, 1, "b2", success=True)]
    progress = team_progress(calls, lambda order: REFS)
    curves = team_returns(calls, progress, gamma=0.5, success_reward=1.0)
    r1, r3 = 0.25, 0.25 + 1.0
    assert calls[2]["team_return"] == pytest.approx(r3)
    assert calls[1]["team_return"] == pytest.approx(r1 + 0.25 * r3)  # t3 is 2 timesteps later
    assert calls[0]["team_return"] == pytest.approx(0.5 * calls[1]["team_return"])  # the chef's request gets credit
    assert calls[0]["team_reward"] == pytest.approx(r1 + r3)  # the chef's only call carries all team rewards (critic)
    assert team_return_at(curves[0], 2, 0.5) == pytest.approx(0.5 * r3) and team_return_at(curves[0], 9, 0.5) == 0.0


def test_team_advantages_group_time_baseline_and_invalid_outputs():
    good = [tcall(0, 0, 0), tcall(0, 1, 1, "b1")]
    bad = [tcall(1, 0, 0), tcall(1, 1, 1, fmt=-0.5)]
    calls = good + bad
    progress = team_progress(calls, lambda order: REFS)
    curves = team_returns(calls, progress, gamma=0.5)
    a = team_advantages(calls, curves, gamma=0.5, baseline="group_time")
    assert a[0] == pytest.approx(0.125 - 0.0 - 0.01)      # same timestep in the other episode: return 0
    assert a[2] == pytest.approx(0.0 - 0.125 - 0.01)      # the episode whose request led nowhere is below
    assert a[3] == pytest.approx(min(0.0 - 0.25, 0.0) - 0.5 - 0.01)
    assert all(x < 0 for x, t in zip(a, calls) if t["format_reward"] < 0)
    lone = team_advantages([dict(c, group=k) for k, c in enumerate(calls)], curves, 0.5, "group_time")
    assert lone[3] < 0  # an invalid output never gets a positive advantage, even without a baseline


def test_update_steps_fixed_and_shuffled():
    import random
    samples = [{"agent_index": i % 2} for i in range(40)]
    steps = _optimizer_steps(samples, "agentgym", random.Random(0), update_steps=4)
    assert len(steps) == 8
    assert sorted(i for s in steps for i, *_ in s) == list(range(40))
    assert all(abs(sum(w for _, w, *_ in s) - 1.0) < 1e-9 for s in steps)


def test_empty_reply_is_invalid_only_with_empty_invalid():
    pytest.importorskip("overcooked_ai_py")
    from baselines.rl.credit import CallCredit
    record = {"response": "Think: I will coordinate with the Chef and", "format_error": False, "validator_error": False,
              "ites": 0.0}
    assert CallCredit("baked_pumpkin_slices").rewards(record)[1] == 0.0
    assert CallCredit("baked_pumpkin_slices", format_penalty=0.5, empty_invalid=True).rewards(record)[:2] == ("empty", -0.5)
