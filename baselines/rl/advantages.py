"""Advantages: immediate local reward + discounted terminal reward (no baseline, no normalization).

For a call of agent i at timestep t of an episode,
    A = r_local + keep * sum_{t' >= t} gamma^(t' - t) * r_terminal(i, t')
where r_local = ITES + format + validator rewards of the call and keep = 0 for calls with a format or
validator penalty (they never share the delivery reward). A call with a format penalty never gets a
positive advantage. Advantages are clipped to [-clip, clip].
"""

from collections import defaultdict


def compute_advantages(transitions, gamma=0.99, clip=10.0):
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
    advantages = []
    for t in transitions:
        local = t["reward"] - t["terminal_reward"]
        invalid = t["format_reward"] < 0 or t["validator_reward"] < 0
        value = local + (0.0 if invalid else terminal_return[(t["episode"], t["agent_index"], t["timestep"])])
        if t["format_reward"] < 0 and value > 0:
            value = 0.0
        advantages.append(max(-clip, min(clip, value)))
    return advantages
