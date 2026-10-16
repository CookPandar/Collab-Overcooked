"""Collect training episodes: two benchmark agents whose LLM calls go to the policy being trained."""

import json
import urllib.request

from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from collab_overcooked.agent_factory import make_agent

from .credit import CallCredit

ROLES = ("chef", "assistant")
SEED_MODULUS = 2**31 - 1


class VLLMPolicy:
    """Generates replies with a vLLM OpenAI-compatible server that serves one LoRA adapter per role.

    The prompt is rendered and tokenized here, so the token ids used for training are exactly the ones
    the reply was sampled from."""

    def __init__(self, base_url, tokenizer, adapters, temperature=0.7, max_tokens=128, timeout=600):
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer
        self.adapters = adapters  # {agent_index: served adapter name}
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def prompt_ids(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                 enable_thinking=False)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return [int(i) for i in ids]

    def generate(self, agent_index, messages, seed):
        prompt_ids = self.prompt_ids(messages)
        request = {
            "model": self.adapters[agent_index],
            "prompt": prompt_ids,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": seed,
            "return_token_ids": True,
            "skip_special_tokens": True,
        }
        http = urllib.request.Request(self.base_url + "/v1/completions", data=json.dumps(request).encode(),
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=self.timeout) as response:
            choice = json.load(response)["choices"][0]
        response_ids = [int(i) for i in choice["token_ids"]]
        return choice["text"], {"prompt_ids": prompt_ids, "response_ids": response_ids,
                                "token_count": len(response_ids)}


def run_episode(policy, order, horizon, *, seed, response_protocol="action_only", history_window=0,
                reflection=False, reward=None, terminal_reward=2.0, layout="new_env"):
    """Play one episode; return its transitions (one per LLM call) and summary.

    `seed` makes the sampling of the episode reproducible: call k of the episode uses seed + k."""
    reward = reward or {}
    mdp = OvercookedGridworld.from_layout_name(layout)
    mdp.start_order_list = [order]
    mdp.one_task_mode = True
    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()
    credit = CallCredit(order, **reward)
    calls = [0]

    def backend_for(agent_index):
        def backend(messages, call_context):
            text, metadata = policy.generate(agent_index, messages, (seed + calls[0]) % SEED_MODULUS)
            calls[0] += 1
            return text, metadata
        return backend

    agents = [make_agent(mdp, layout, observer=credit, actor=role, model="policy", backend=backend_for(i),
                         response_protocol=response_protocol, history_window=history_window,
                         reflection=reflection)
              for i, role in enumerate(ROLES)]
    team = AgentGroup(*agents)
    team.reset()

    success, steps = False, 0
    for steps in range(1, horizon + 1):
        joint_action, pickup = team.joint_action(env.state)
        team.reset_dialogue()
        _, step_reward, done, _ = env.step(joint_action, pickup)
        if step_reward > 0:
            success = True
            break
        if done:
            break

    transitions = []
    for record in sorted(credit.calls.values(), key=lambda r: (r["timestep"], r["agent_index"], r["call_index"])):
        mode, fmt, val, ites = credit.rewards(record)
        transitions.append({
            "agent_index": record["agent_index"],
            "timestep": record["timestep"],
            "call_index": record["call_index"],
            "call_type": record["call_type"],
            "action_mode": mode,
            "prompt_ids": record["metadata"]["prompt_ids"],
            "response_ids": record["metadata"]["response_ids"],
            "messages": record["metadata"].get("messages"),  # only kept by policies that return them (SFT data)
            "response": record["response"],
            "format_reward": fmt,
            "validator_reward": val,
            "sequence_reward": ites,
            "terminal_reward": 0.0,
        })
    if success:
        for agent_index in (0, 1):
            last = [t for t in transitions if t["agent_index"] == agent_index]
            if last:
                last[-1]["terminal_reward"] = float(terminal_reward)
    for t in transitions:
        t["reward"] = t["format_reward"] + t["validator_reward"] + t["sequence_reward"] + t["terminal_reward"]
    return {"order": order, "success": success, "steps": steps, "transitions": transitions}
