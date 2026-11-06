"""CPU checks of the reward-shaping options (docs/reward_shaping.md): repeat tracking, the uptake collaboration
reward, the per-update communication metrics and the think-reward redesign (rules_combine, scale ratio, center)."""
import pytest

pytest.importorskip("torch")

from baselines.rl.credit import CallCredit

ORDER = "baked_pumpkin_slices"  # Assistant reference: pickup(pumpkin,ingredient_dispenser), put_obj_in_utensil(...)
FETCH = "pickup(pumpkin,ingredient_dispenser)"
FETCH_MSG = "Collab(request(Assistant,pickup(pumpkin, ingredient_dispenser)))"


def respond(credit, agent, timestep, call_index, action):
    call = {"agent_index": agent, "timestep": timestep, "call_index": call_index, "call_type": "planner_main"}
    credit.on_llm_call(None, call, "prompt", f"Think: t\nRecent Goal: g\nAction: {action}", {})
    return call


def execute(credit, call, action):
    credit.on_validation(None, action, call, True, "", False)
    credit.on_action_executed(None, action, call, None, None)


def record(credit, call):
    return credit.calls[credit.key(call)]


def test_repeat_is_tracked_without_penalty_and_resets_on_execution():
    credit = CallCredit(ORDER)
    a = respond(credit, 0, 0, 0, FETCH_MSG)
    b = respond(credit, 0, 0, 1, "Collab(request(Assistant,wait(1)))")
    c = respond(credit, 0, 0, 2, FETCH_MSG)
    assert [record(credit, x)["repeat"] for x in (a, b, c)] == [False, False, True]
    assert record(credit, c)["uncooperative"] is None  # tracked, not penalized (uncoop_penalty 0)
    execute(credit, respond(credit, 1, 0, 0, FETCH), FETCH)
    assert record(credit, respond(credit, 0, 1, 0, FETCH_MSG))["repeat"] is False


def test_repeat_penalty_unchanged():
    credit = CallCredit(ORDER, uncoop_penalty=0.1, uncoop_kinds=["repeat"])
    respond(credit, 0, 0, 0, FETCH_MSG)
    c = respond(credit, 0, 0, 1, FETCH_MSG)
    assert record(credit, c)["uncooperative"] == "repeat"
    assert credit.collab_reward(record(credit, c)) == pytest.approx(-0.1)


def test_promise_mode_default_credits_at_send_time():
    credit = CallCredit(ORDER, collab_weight=0.5)
    m = respond(credit, 0, 0, 0, FETCH_MSG)
    assert record(credit, m)["collab"] > 0


def test_uptake_credits_the_request_when_the_teammate_executes_it():
    credit = CallCredit(ORDER, collab_weight=0.5, collab_mode="uptake", collab_window=5)
    m = respond(credit, 0, 0, 0, FETCH_MSG)
    again = respond(credit, 0, 0, 1, FETCH_MSG)  # repeated request: nothing extra
    assert record(credit, m)["collab"] == 0.0 and record(credit, m)["requests"] == 1
    assert record(credit, again)["requests"] == 0
    act = respond(credit, 1, 2, 0, FETCH)
    execute(credit, act, FETCH)
    gain = record(credit, act)["ites"]
    assert gain > 0
    assert record(credit, m)["collab"] == pytest.approx(0.5 * gain) and record(credit, m)["uptaken"] == 1
    assert record(credit, again)["collab"] == 0.0
    assert record(credit, act)["collab"] == 0.0  # the teammate gets only its ITES
    assert credit.collab_reward(record(credit, m)) == pytest.approx(0.5 * gain)


def test_uptake_expires_after_the_window():
    credit = CallCredit(ORDER, collab_weight=0.5, collab_mode="uptake", collab_window=5)
    m = respond(credit, 0, 0, 0, FETCH_MSG)
    act = respond(credit, 1, 6, 0, FETCH)
    execute(credit, act, FETCH)
    assert record(credit, act)["ites"] > 0 and record(credit, m)["collab"] == 0.0


def test_uptake_needs_positive_ites():
    credit = CallCredit(ORDER, collab_weight=0.5, collab_mode="uptake")
    m = respond(credit, 0, 0, 0, "Collab(request(Assistant,cut(chopping_board0)))")
    act = respond(credit, 1, 1, 0, "cut(chopping_board0)")
    execute(credit, act, "cut(chopping_board0)")  # not the next reference step of the Assistant: no TES gain
    assert record(credit, act)["ites"] <= 0 and record(credit, m)["collab"] == 0.0


def test_unknown_collab_mode():
    with pytest.raises(ValueError):
        CallCredit(ORDER, collab_mode="other")


def test_comm_metrics():
    from baselines.rl.train import comm_metrics
    calls = [dict(episode=0, agent_index=0, timestep=0, action_mode="communication", repeat=False, collab_requests=1),
             dict(episode=0, agent_index=0, timestep=0, action_mode="communication", repeat=True, uptaken=1),
             dict(episode=0, agent_index=1, timestep=0, action_mode="planner_main"),
             dict(episode=1, agent_index=1, timestep=0, action_mode="wait")]
    m = comm_metrics(calls)
    assert m["comm/repeat_share"] == 0.5
    assert m["comm/calls_per_agent_step"] == pytest.approx(4 / 3)
    assert m["comm/msg_share_chef"] == 1.0 and m["comm/msg_share_assistant"] == 0.0
    assert m["comm/wait_share_assistant"] == 0.5 and m["comm/action_share_assistant"] == 0.5
    assert m["comm/uptake_rate"] == 1.0


def test_rules_combine_mean():
    from baselines.rl.rule_rewards import rule_signal
    result = {"ground": 1.0, "mate_v2": -0.5, "plan": 0.5, "mate": 0.0}
    weights = {"ground": 1, "mate_v2": 1, "plan": 1}
    assert rule_signal(result, weights) == pytest.approx(1.0)
    assert rule_signal(result, weights, "mean") == pytest.approx(1.0 / 3)
    assert rule_signal(None, weights, "mean") is None
    with pytest.raises(ValueError):
        rule_signal(result, weights, "max")


def think_call(agent, mode):
    return {"agent_index": agent, "action_mode": mode}


def test_center_role_type_is_neutral_per_action_kind():
    from baselines.rl.advantages import centered_judge_scores
    calls = [think_call(0, "communication"), think_call(0, "communication"), think_call(0, "planner_main"),
             think_call(0, "wait"), think_call(1, "planner_main"), think_call(0, "planner_main")]
    scores = [1.0, 0.0, -1.0, 0.3, 2.0, None]
    role = centered_judge_scores(calls, scores)
    assert role[0] == pytest.approx(1.0 - 0.075)  # default: per role (unchanged)
    typed = centered_judge_scores(calls, scores, "role_type")
    assert typed == pytest.approx([0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        centered_judge_scores(calls, scores, "type")


def test_ratio_scale_sets_the_think_std_per_role():
    from baselines.rl.advantages import ratio_scale
    calls = [think_call(r, "planner_main") for r in (0, 0, 0, 0, 1, 1)]
    advantages = [1.0, -1.0, 1.0, -1.0, 2.0, -2.0]   # std 1 (role 0), 2 (role 1)
    think = [0.5, -0.5, 0.0, 0.0, 0.1, -0.1]
    raw = [1.0, 0.0, 0.5, None, 0.3, 0.1]            # role 0: the unscored call does not enter the std
    scaled, factors = ratio_scale(calls, advantages, think, raw, 0.25)
    spread0 = (sum(v * v for v in think[:3]) / 3) ** 0.5
    assert factors[0] == pytest.approx(0.25 * 1.0 / spread0)
    assert factors[1] == pytest.approx(0.25 * 2.0 / 0.1)
    assert scaled[4] == pytest.approx(0.1 * factors[1])
    flat, f = ratio_scale(calls, advantages, [0.0] * 6, raw, 0.25)
    assert f == {0: 0.0, 1: 0.0} and flat == [0.0] * 6


def test_chatter_counts_messages_since_the_last_executed_action():
    credit = CallCredit(ORDER, uncoop_penalty=0.1, uncoop_kinds=["repeat", "chatter"], chatter_limit=2)
    a = respond(credit, 0, 0, 0, FETCH_MSG)
    b = respond(credit, 0, 0, 1, 'Collab(ack(Assistant, "go"))')
    c = respond(credit, 0, 0, 2, 'Collab(ack(Assistant, "go now"))')   # paraphrase: not a repeat, but chatter
    d = respond(credit, 0, 0, 3, FETCH_MSG)                            # repeat wins over chatter
    assert [record(credit, x)["msg_index"] for x in (a, b, c, d)] == [0, 1, 2, 3]
    assert [record(credit, x)["uncooperative"] for x in (a, b, c, d)] == [None, None, "chatter", "repeat"]
    execute(credit, respond(credit, 1, 0, 0, FETCH), FETCH)
    e = respond(credit, 0, 1, 0, 'Collab(ack(Assistant, "go now"))')
    assert record(credit, e)["msg_index"] == 0 and record(credit, e)["uncooperative"] is None


def test_chatter_off_by_default():
    credit = CallCredit(ORDER, uncoop_penalty=0.1, uncoop_kinds=["repeat"])
    calls = [respond(credit, 0, 0, k, f'Collab(ack(Assistant, "m{k}"))') for k in range(4)]
    assert all(record(credit, x)["uncooperative"] is None for x in calls)


def test_normalize_std_floor():
    from baselines.rl.advantages import normalize_advantages
    calls = [dict(agent_index=0, format_reward=0.0)] * 4
    small = [0.1, -0.1, 0.1, -0.1]
    assert normalize_advantages(calls, small) == pytest.approx([1, -1, 1, -1], abs=1e-4)
    assert normalize_advantages(calls, small, std_floor=1.0) == pytest.approx(small, abs=1e-5)
    large = [2.0, -2.0, 2.0, -2.0]
    assert normalize_advantages(calls, large, std_floor=1.0) == pytest.approx([1, -1, 1, -1], abs=1e-4)


def test_call_categories_rule_critic():
    from baselines.rl.advantages import call_categories, category_scores

    def call(ts, ci, agent, mode, response, val=0.0, ites=0.0, fmt=0.0):
        return {"episode": 0, "timestep": ts, "call_index": ci, "agent_index": agent, "action_mode": mode,
                "response": response, "validator_reward": val, "sequence_reward": ites, "format_reward": fmt}

    msg_a = 'Think: x\nAction: Collab("Assistant: pickup(pumpkin)")'
    msg_b = 'Think: x\nAction: Collab("Assistant: cut(pumpkin)")'
    calls = [call(0, 0, 0, "communication", msg_a), call(0, 1, 0, "communication", msg_b),
             call(0, 2, 0, "communication", msg_a), call(0, 3, 0, "communication", 'Think: x\nAction: Collab("c")'),
             call(0, 4, 1, "planner_main", "Action: pickup(pumpkin)", ites=0.2),
             call(1, 5, 0, "communication", msg_a), call(1, 6, 1, "planner_main", "Action: cut(x)", val=-0.1),
             call(1, 7, 1, "wait", "Action: wait(1)"), call(1, 8, 1, "planner_main", "Action: x", fmt=-0.2)]
    assert call_categories(calls) == ["msg_pos1", "msg_pos2", "msg_repeat", "msg_pos3", "act_progress", "msg_pos1",
                                      "act_rejected", "wait", None]
    scores = category_scores(calls, {"act_progress": 0.8, "msg_pos1": 0.0})
    assert scores[4] == (0.8, "act_progress") and scores[0] == (0.0, "msg_pos1") and scores[1] == (None, "msg_pos2")
