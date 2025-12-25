"""Main-style RL session that reuses Collab-Overcooked prompts and agents.

This module exposes a thin environment wrapper which instantiates the same
`LLMAgents` used by :mod:`collab_overcooked.main`, but swaps their planner
module so that all LLM queries are routed through a user-provided policy
function.  The policy function can in turn call a HuggingFace model (e.g.,
Qwen2.5) to generate Think/Action text, while PPO-style trainers capture logits
and token statistics for optimization.

The session orchestrates an Overcooked environment, process reward tracker, and
agent group.  Each environment step closely mirrors the behavior of the CLI
entrypoint: prompts are built via the real prompt templates, parsed responses
are converted to primitive actions, and process rewards are aggregated.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

from ..reward import ProcessRewardTracker
from ..agents.collab import LLMAgents
from ..agents.utils import convert_messages_to_prompt
from ..main import (
    check_recipe_parse,
    convert_yaml_to_variant,
    load_config_from_yaml,
    make_agent_from_config,
)

# ---------------------------------------------------------------------------
# Interfaces & dataclasses
# ---------------------------------------------------------------------------

RLPolicyFn = Callable[
    [int, List[Dict[str, str]], Dict[str, Any]],
    Tuple[str, Dict[str, Any]],
]
"""Callable signature expected by :class:`RLPlannerProxy`.

Args:
    agent_index: index of the querying agent (0 for Chef, 1 for Assistant).
    messages: full list of role/user messages that will be fed into the LLM.
    context: auxiliary dict containing runtime info (layout, timestep, etc.).

Returns:
    response_text: textual completion produced by the policy model.
    metadata: arbitrary dict (e.g., token counts, logits) recorded by the
        trainer for PPO updates.  Common keys:
            - ``token_count``: total tokens consumed by the completion.
            - ``logits`` / ``values``: tensors needed for policy/value losses.
"""


@dataclass
class PolicyCallRecord:
    """Structured log of a single planner query routed through RL policy."""

    agent_index: int
    messages: List[Dict[str, str]]
    prompt: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    timestep: Optional[int] = None


@dataclass
class SessionStep:
    """Return value of :meth:`CollabMainSession.step`."""

    timestep: int
    reward: float
    done: bool
    observation: Dict[str, Any]
    env_info: Dict[str, Any]
    process_reward: Optional[Dict[str, Any]]
    policy_records: List[PolicyCallRecord]


# ---------------------------------------------------------------------------
# Planner proxy that redirects prompt generation to RL policy
# ---------------------------------------------------------------------------


class RLPlannerProxy:
    """Proxy that wraps a ``Module`` instance and overrides ``query``.

    All other attribute accesses are delegated to the original module so that
    :class:`LLMAgents` can continue to interact with it transparently.
    """

    def __init__(self, planner, agent_index: int, policy_fn: RLPolicyFn):
        object.__setattr__(self, "_planner", planner)
        object.__setattr__(self, "agent_index", agent_index)
        object.__setattr__(self, "policy_fn", policy_fn)
        object.__setattr__(self, "_records", [])  # type: ignore[var-annotated]

    def __getattr__(self, item):
        return getattr(self._planner, item)

    def __setattr__(self, key, value):
        if key in {"_planner", "agent_index", "policy_fn", "_records"}:
            object.__setattr__(self, key, value)
        else:
            setattr(self._planner, key, value)

    # pylint: disable=too-many-arguments
    def query(
        self,
        key=None,
        proxy=None,
        stop=None,
        temperature: float = 0.7,
        debug_mode: str = "Y",
        trace: bool = True,
        rethink: bool = False,
        map: str = "",
    ):
        """Intercept planner queries and forward them to ``policy_fn``."""
        messages = self._planner.query_messages(rethink)
        self._planner.cache_list = self._planner.get_cache()

        if not trace and not rethink:
            messages[-1]["content"] += (
                " Based on the failure explanation and scene description, analyze and plan again."
            )

        print(
            "[RLPlannerProxy] agent="
            f"{self.agent_index} timestep={getattr(self._planner, 'current_timestep', None)} "
            f"trace={trace} rethink={rethink} prompt_messages={len(messages)}"
        )

        context = {
            "temperature": temperature,
            "debug_mode": debug_mode,
            "trace": trace,
            "rethink": rethink,
            "map": map,
            "agent_name": getattr(self._planner, "name", None),
            "timestep": getattr(self._planner, "current_timestep", None),
        }
        call_context = getattr(self._planner, "_rl_call_context", None)
        if isinstance(call_context, dict):
            context.update(call_context)
            setattr(self._planner, "_rl_call_context", None)

        response_text, metadata = self.policy_fn(self.agent_index, messages, context)
        token_count = metadata.get("token_count")
        if token_count is None:
            # Fallback heuristic if trainer did not report token usage.
            token_count = len(response_text.split())
            metadata = dict(metadata)
            metadata["token_count"] = token_count

        record = PolicyCallRecord(
            agent_index=self.agent_index,
            messages=copy.deepcopy(messages),
            prompt=convert_messages_to_prompt(messages),
            response=response_text,
            metadata=metadata,
            context=context,
            timestep=context.get("timestep"),
        )
        self._records.append(record)
        print(
            "[RLPlannerProxy] agent="
            f"{self.agent_index} response_tokens={token_count} meta_keys={sorted(metadata.keys())}"
        )
        return response_text, token_count

    def disable_remote_calls(self):
        setattr(self._planner, "_rl_block_network", True)

    def consume_records(self) -> List[PolicyCallRecord]:
        """Return and clear queued policy call records."""
        records = list(self._records)
        self._records.clear()
        return records


# ---------------------------------------------------------------------------
# Main-style RL Session
# ---------------------------------------------------------------------------


class CollabMainSession:
    """Emulates ``collab_overcooked.main`` for RL training.

    The session constructs Overcooked environments, reward trackers, and LLM
    agents directly from a YAML config.  Every planner query goes through the
    supplied ``policy_fn``, enabling PPO trainers to inject their own models
    while preserving prompt formats and intermediate rewards.
    """

    def __init__(
        self,
        variant: Dict[str, Any],
        policy_fn: RLPolicyFn,
        *,
        history_window: Optional[int] = None,
    ) -> None:
        if make_agent_from_config is None:
            raise ImportError(
                "collab_overcooked.main.make_agent_from_config is unavailable. "
                "Ensure optional dependencies for the new agent system are installed."
            )

        self.policy_fn = policy_fn
        self.variant = copy.deepcopy(variant)
        self.history_window = (
            int(history_window)
            if history_window is not None
            else int(self.variant.get("history_window", 3) or 0)
        )

        layout = self.variant.get("layout", "cramped_room")
        order = self.variant.get("order", "")
        horizon = self.variant.get("horizon", 120)

        self.mdp = OvercookedGridworld.from_layout_name(layout)
        if order and check_recipe_parse(self.variant):
            self.mdp.start_order_list = [order]
            self.mdp.one_task_mode = True

        self.env = OvercookedEnv(self.mdp, horizon=horizon)

        reward_settings = self.variant.get("reward", {})
        self.reward_tracker: Optional[ProcessRewardTracker]
        try:
            self.reward_tracker = ProcessRewardTracker(
                order=order,
                mdp=self.mdp,
                reference_dir=self._prompt_reference_dir(),
                settings=reward_settings,
            )
        except Exception:
            self.reward_tracker = None

        self.agents = self._build_agents()
        self.team = AgentGroup(*self.agents)
        self.rl_modules: List[RLPlannerProxy] = []
        for idx, agent in enumerate(self.team.agents):
            if isinstance(agent, LLMAgents):
                proxy = RLPlannerProxy(agent.planner, idx, self.policy_fn)
                agent.planner = proxy
                proxy.disable_remote_calls()
                self.rl_modules.append(proxy)
                print(
                    "[CollabMainSession] Attached RLPlannerProxy to agent "
                    f"{idx} ({getattr(agent, 'name', 'unknown')})"
                )
        self.reset()

    @classmethod
    def from_yaml(cls, config_path: str, policy_fn: RLPolicyFn) -> "CollabMainSession":
        config = load_config_from_yaml(config_path)
        variant = convert_yaml_to_variant(config)
        variant["yaml_config"] = config
        return cls(variant, policy_fn)

    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.env.reset()
        if self.reward_tracker:
            self.reward_tracker.reset()
        self.team.reset()
        for proxy in self.rl_modules:
            proxy.consume_records()
        return self._build_observation(self.env.state)

    def step(self) -> SessionStep:
        state = self.env.state
        print(f"[CollabMainSession] Beginning step at timestep {state.timestep}")
        joint_action, pickup_parm = self.team.joint_action(state)
        obs, reward, done, env_info = self.env.step(joint_action, pickup_parm)

        process_reward = None
        if self.reward_tracker:
            process_reward = self.reward_tracker.after_step(
                state.timestep, obs.ml_actions, self.env.state
            )

        records: List[PolicyCallRecord] = []
        for proxy in self.rl_modules:
            agent_records = proxy.consume_records()
            if not agent_records:
                continue
            for record in agent_records:
                if record.timestep is None:
                    record.timestep = state.timestep
            records.extend(agent_records)
        self._assign_call_rewards(records, process_reward, done)
        print(
            "[CollabMainSession] "
            f"timestep={state.timestep} reward={reward:.3f} done={done} policy_calls={len(records)}"
        )

        return SessionStep(
            timestep=state.timestep,
            reward=reward,
            done=done,
            observation=self._build_observation(self.env.state),
            env_info={"ml_actions": obs.ml_actions, "raw_info": env_info},
            process_reward=process_reward,
            policy_records=records,
        )

    # ------------------------------------------------------------------
    def _build_agents(self) -> List[LLMAgents]:
        agents = []
        agent_cfgs = self.variant.get("agent_configs") or {}
        if not agent_cfgs:
            raise ValueError(
                "RL session requires YAML configs with `agents` section."
            )
        for agent_id, cfg in agent_cfgs.items():
            if not agent_id.startswith("agent_"):
                continue
            agent = make_agent_from_config(
                cfg,
                self.mdp,
                self.variant.get("layout", "cramped_room"),
                history_window=self.history_window,
                reward_tracker=self.reward_tracker,
            )
            if not isinstance(agent, LLMAgents):
                raise TypeError(
                    f"Agent '{agent_id}' is not LLMAgents; RL session expects LLM drivers."
                )
            agents.append(agent)
        if len(agents) != 2:
            raise ValueError(
                f"Expected exactly two agents in config, got {len(agents)}."
            )
        return agents

    def _build_observation(self, state: OvercookedState) -> Dict[str, Any]:
        counters = self.mdp.get_counter_objects_dict(
            state, list(self.mdp.terrain_pos_dict.get("X", []))
        )
        return {
            "timestep": state.timestep,
            "grid": self.mdp.state_string(state).replace("ø", "o"),
            "players": {
                0: self._format_player(state.players[0]),
                1: self._format_player(state.players[1]),
            },
            "orders": list(state.current_k_order),
            "counters": counters,
            "pot_states": self.mdp.get_pot_states(state),
        }

    def _assign_call_rewards(
        self,
        records: List[PolicyCallRecord],
        process_reward: Optional[Dict[str, Any]],
        done_flag: bool,
    ):
        if not records:
            return
        reward_queues: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
        if process_reward and isinstance(process_reward, dict):
            per_agent = process_reward.get("per_agent") or []
            for agent_idx in range(min(len(per_agent), 2)):
                agent_calls = per_agent[agent_idx].get("calls") if isinstance(per_agent[agent_idx], dict) else None
                if agent_calls:
                    reward_queues[agent_idx].extend(agent_calls)

        action_call_types = {"planner_main"}
        for idx, record in enumerate(records):
            call_type = (record.context or {}).get("call_type")
            metadata = dict(record.metadata)
            if call_type:
                metadata.setdefault("call_type", call_type)
            if call_type in action_call_types:
                queue = reward_queues.get(record.agent_index, [])
                reward_entry = queue.pop(0) if queue else None
                if reward_entry:
                    seq_reward = float(reward_entry.get("sequence_reward", 0.0))
                    fmt_reward = float(reward_entry.get("format_reward", 0.0))
                    record.reward = seq_reward + fmt_reward
                    breakdown = {
                        "sequence_reward": seq_reward,
                        "format_reward": fmt_reward,
                        "call_type": reward_entry.get("call_type"),
                        "raw": reward_entry,
                    }
                    metadata["reward_breakdown"] = breakdown
                else:
                    record.reward = 0.0
                    metadata.setdefault("reward_breakdown", {}).setdefault("missing_reward_entry", True)
            else:
                record.reward = 0.0
                metadata.setdefault("reward_breakdown", {}).setdefault("communication_reward", 0.0)
            record.metadata = metadata
            record.done = bool(done_flag) if idx == len(records) - 1 else False

    def _format_player(self, player_state) -> Dict[str, Any]:
        held = player_state.held_object
        return {
            "position": player_state.position,
            "orientation": player_state.orientation,
            "has_object": held is not None,
            "object": held.name if held is not None else None,
        }

    def _prompt_reference_dir(self):
        from pathlib import Path
        from .. import __file__ as pkg_file

        base = Path(pkg_file).resolve().parent
        return base / "prompts" / "reference"


__all__ = [
    "RLPolicyFn",
    "PolicyCallRecord",
    "SessionStep",
    "CollabMainSession",
]
