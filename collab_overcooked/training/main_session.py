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
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

from ..reward import ProcessRewardTracker
from ..agents.collab import LLMAgents
from .snapshots import build_session_snapshot as _build_session_snapshot
from .snapshots import load_session_snapshot as _load_session_snapshot
from ..agents.utils import convert_messages_to_prompt
from ..main import (
    check_recipe_parse,
    convert_yaml_to_variant,
    load_config_from_yaml,
    make_agent_from_config,
)


def _strip_action_code_fence(action: str) -> str:
    stripped = (action or "").strip()
    if not stripped.startswith("```"):
        return stripped.strip("`").strip()
    content = stripped[3:]
    closing = content.rfind("```")
    if closing != -1:
        content = content[:closing]
    content = content.strip()
    if "\n" in content:
        first_line, remainder = content.split("\n", 1)
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", first_line.strip()):
            content = remainder
    return content.strip().strip("`").strip()


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
    micro_step: Optional[int] = None


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
    executed_action_sources: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planner proxy that redirects prompt generation to RL policy
# ---------------------------------------------------------------------------


class RLPlannerProxy:
    """Proxy that wraps a ``Module`` instance and overrides ``query``.

    All other attribute accesses are delegated to the original module so that
    :class:`LLMAgents` can continue to interact with it transparently.
    """

    def __init__(
        self,
        planner,
        agent_index: int,
        policy_fn: RLPolicyFn,
        shared_trace_store: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        object.__setattr__(self, "_planner", planner)
        object.__setattr__(self, "agent_index", agent_index)
        object.__setattr__(self, "policy_fn", policy_fn)
        object.__setattr__(self, "_records", [])  # type: ignore[var-annotated]
        object.__setattr__(self, "_shared_trace_store", shared_trace_store if shared_trace_store is not None else {})

    def __getattr__(self, item):
        return getattr(self._planner, item)

    def __setattr__(self, key, value):
        if key in {"_planner", "agent_index", "policy_fn", "_records", "_shared_trace_store"}:
            object.__setattr__(self, key, value)
        else:
            setattr(self._planner, key, value)

    @staticmethod
    def _compact_text(text: Optional[str], limit: int = 600) -> str:
        content = (text or "").strip()
        if len(content) <= limit:
            return content
        return content[: limit - 3].rstrip() + "..."

    @classmethod
    def _extract_observation_excerpt(cls, text: Optional[str]) -> str:
        content = text or ""
        anchors = [
            "Current Observation:",
            "Current observation:",
            "Observation:",
            "Scene:",
        ]
        for anchor in anchors:
            start = content.find(anchor)
            if start == -1:
                continue
            end_candidates = [
                pos for pos in (
                    content.find("\n\n", start + len(anchor)),
                    content.find("Think:", start + len(anchor)),
                    content.find("Recent Goal:", start + len(anchor)),
                    content.find("Action:", start + len(anchor)),
                )
                if pos != -1
            ]
            end = min(end_candidates) if end_candidates else len(content)
            return cls._compact_text(content[start:end].strip(), limit=900)
        return cls._compact_text(content, limit=900)

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
        if os.environ.get("RL_STAGE_PHASE", "").strip().lower() == "eval":
            temperature = 0.0
        messages = self._planner.query_messages(rethink)
        self._planner.cache_list = self._planner.get_cache()

        if not trace and not rethink:
            messages[-1]["content"] += (
                " Based on the failure explanation and scene description, analyze and plan again."
            )

        print(
            "[RLPlannerProxy] agent="
            f"{self.agent_index} timestep={getattr(self._planner, 'current_timestep', None)} "
            f"trace={trace} rethink={rethink} temperature={temperature} "
            f"prompt_messages={len(messages)}"
        )

        context = {
            "temperature": temperature,
            "debug_mode": debug_mode,
            "stop": stop,
            "trace": trace,
            "rethink": rethink,
            "map": map,
            "agent_name": getattr(self._planner, "name", None),
            "timestep": getattr(self._planner, "current_timestep", None),
        }
        latest_user_message = next(
            (msg for msg in reversed(messages) if msg.get("role") == "user"),
            {},
        )
        latest_user_content = latest_user_message.get("content", "")
        context["prompt_excerpt"] = self._compact_text(latest_user_content, limit=1200)
        context["observation_excerpt"] = self._extract_observation_excerpt(
            latest_user_content
        )
        context["global_observation"] = copy.deepcopy(
            getattr(self._planner, "_rl_global_observation", None)
        )
        context["shared_agent_traces"] = copy.deepcopy(self._shared_trace_store)
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
            micro_step=context.get("micro_step"),
        )
        self._records.append(record)
        self._shared_trace_store[self.agent_index] = {
            "agent_index": self.agent_index,
            "agent_name": context.get("agent_name"),
            "timestep": context.get("timestep"),
            "call_type": context.get("call_type"),
            "observation_excerpt": context.get("observation_excerpt", ""),
            "prompt_excerpt": context.get("prompt_excerpt", ""),
            "response_excerpt": self._compact_text(response_text, limit=800),
        }
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

    def relabel_last_record(self, call_type: str, extra_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Override the recorded call_type/metadata for the most recent query."""
        if not self._records:
            return False
        record = self._records[-1]
        context = dict(record.context or {})
        context["call_type"] = call_type
        record.context = context
        if extra_metadata:
            metadata = dict(record.metadata or {})
            metadata.update(extra_metadata)
            record.metadata = metadata
        return True

    def annotate_last_record(self, extra_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Attach metadata to the most recent query record without changing call_type."""
        if not self._records or not extra_metadata:
            return False
        record = self._records[-1]
        metadata = dict(record.metadata or {})
        metadata.update(extra_metadata)
        record.metadata = metadata
        return True


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
            else int(self.variant.get("history_window", 0) or 0)
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

        print("[CollabMainSession] before _build_agents")
        self.agents = self._build_agents()
        print("[CollabMainSession] after _build_agents")
        self.team = AgentGroup(*self.agents)
        print("[CollabMainSession] after AgentGroup")
        self._runtime_log_context: Dict[str, Any] = {}
        self._init_policy_record_logging(len(self.agents))
        print("[CollabMainSession] after _init_policy_record_logging")
        self._shared_agent_traces: Dict[int, Dict[str, Any]] = {}
        self.rl_modules: List[RLPlannerProxy] = []
        for idx, agent in enumerate(self.team.agents):
            if isinstance(agent, LLMAgents):
                proxy = RLPlannerProxy(
                    agent.planner,
                    idx,
                    self.policy_fn,
                    shared_trace_store=self._shared_agent_traces,
                )
                agent.planner = proxy
                proxy.disable_remote_calls()
                self.rl_modules.append(proxy)
                print(
                    "[CollabMainSession] Attached RLPlannerProxy to agent "
                    f"{idx} ({getattr(agent, 'name', 'unknown')})"
                )
        print("[CollabMainSession] before reset")
        self.reset()
        print("[CollabMainSession] after reset")

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
        self._shared_agent_traces.clear()
        for proxy in self.rl_modules:
            proxy.consume_records()
        return self._build_observation(self.env.state)

    def set_logging_context(
        self,
        *,
        update_idx: Optional[int] = None,
        phase: Optional[str] = None,
        run_label: Optional[str] = None,
        loop_round_idx: Optional[int] = None,
        rotate_session: bool = False,
    ) -> None:
        context = dict(getattr(self, "_runtime_log_context", {}))
        if update_idx is not None:
            context["update_idx"] = int(update_idx)
        if phase:
            context["phase"] = str(phase)
        if run_label:
            context["run_label"] = str(run_label)
        if loop_round_idx is not None:
            context["loop_round_idx"] = int(loop_round_idx)
        self._runtime_log_context = context
        if rotate_session:
            self._init_policy_record_logging(len(self.agents))

    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture a serializable snapshot of the current session state."""
        return _build_session_snapshot(self)

    def load_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Restore session state from :meth:`capture_snapshot` output."""
        _load_session_snapshot(self, snapshot)
        return self._build_observation(self.env.state)

    def step(self) -> SessionStep:
        state = self.env.state
        current_observation = self._build_observation(state)
        for proxy in self.rl_modules:
            setattr(proxy, "_rl_global_observation", current_observation)
        joint_action, pickup_parm = self.team.joint_action(state)
        print(
            f"[CollabMainSession] joint_action={joint_action} pickup_parm={pickup_parm}"
        )
        executed_action_sources = self._collect_executed_action_sources()
        obs, reward, done, env_info = self.env.step(joint_action, pickup_parm)

        process_reward = None
        if self.reward_tracker:
            process_reward = self.reward_tracker.after_step(
                state.timestep,
                obs.ml_actions,
                self.env.state,
                executed_action_sources=executed_action_sources,
            )

        records: List[PolicyCallRecord] = []
        for proxy in self.rl_modules:
            agent_records = proxy.consume_records()
            if not agent_records:
                continue
            for record in agent_records:
                if record.timestep is None:
                    record.timestep = state.timestep
                if record.micro_step is None:
                    record.micro_step = self._record_micro_step(record)
            records.extend(agent_records)
        self._assign_call_rewards(records, process_reward, done)
        self._persist_policy_records(records)
        self._shrink_policy_records(records)
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
            executed_action_sources=executed_action_sources,
        )

    # ------------------------------------------------------------------
    def _collect_executed_action_sources(self) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        for agent in getattr(self, "agents", []):
            source = getattr(agent, "last_executed_action_source", None)
            if isinstance(source, dict):
                sources.append(copy.deepcopy(source))
        return sources

    # ------------------------------------------------------------------
    @staticmethod
    def _record_micro_step(record: PolicyCallRecord) -> Optional[int]:
        raw = (record.metadata or {}).get("call_index")
        if raw is None:
            raw = (record.context or {}).get("micro_step")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

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

    def _init_policy_record_logging(self, agent_count: int) -> None:
        raw_dir = (
            self.variant.get("record_log_dir")
            or self.variant.get("log_dir")
            or self.variant.get("trainer", {}).get("record_log_dir")
        )
        record_mode, run_label, update_tag, session_meta = self._policy_record_log_context()
        base_dir = Path(raw_dir) if raw_dir else Path("runs") / "rl_policy_records"
        session_dir = (
            base_dir
            / record_mode
            / run_label
            / update_tag
            / f"session_{int(time.time())}_{abs(id(self))}"
        )
        session_dir.mkdir(parents=True, exist_ok=True)
        self._policy_record_session_dir = session_dir
        self._policy_record_session_meta = dict(session_meta)
        self._policy_record_session_meta["session_dir"] = str(session_dir)
        self._policy_record_paths: List[Path] = []
        for idx in range(agent_count):
            path = session_dir / f"agent_{idx}.jsonl"
            path.touch(exist_ok=True)
            self._policy_record_paths.append(path)
        try:
            (session_dir / "session_meta.json").write_text(
                json.dumps(self._policy_record_session_meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass

    @staticmethod
    def _sanitize_log_component(value: Optional[str], fallback: str) -> str:
        text = (value or "").strip()
        if not text:
            text = fallback
        text = text.replace("\\", "/")
        text = text.split("/")[-1]
        text = re.sub(r"[^0-9A-Za-z._-]+", "_", text).strip("._-")
        return text or fallback

    def _policy_record_log_context(self) -> Tuple[str, str, str, Dict[str, Any]]:
        yaml_config = self.variant.get("yaml_config") or {}
        trainer_cfg = yaml_config.get("trainer") if isinstance(yaml_config, dict) else {}
        if not isinstance(trainer_cfg, dict):
            trainer_cfg = {}
        runtime_ctx = getattr(self, "_runtime_log_context", {}) or {}

        train_only = bool(trainer_cfg.get("train_only", False))
        collect_only = bool(trainer_cfg.get("collect_only", False))
        output_dir = str(trainer_cfg.get("output_dir", "") or "")
        rollout_dir = str(trainer_cfg.get("rollout_dir", "") or "")
        explicit_mode = str(
            self.variant.get("record_log_mode")
            or trainer_cfg.get("record_log_mode")
            or ""
        ).strip().lower()

        if runtime_ctx.get("phase"):
            record_mode = self._sanitize_log_component(str(runtime_ctx["phase"]), "run")
        elif explicit_mode:
            record_mode = self._sanitize_log_component(explicit_mode, "run")
        elif train_only:
            record_mode = "train"
        elif collect_only:
            output_hint = output_dir.lower()
            rollout_hint = rollout_dir.lower()
            if "eval" in output_hint or "eval" in rollout_hint:
                record_mode = "eval"
            else:
                record_mode = "collect"
        else:
            record_mode = "run"

        explicit_label = str(
            self.variant.get("record_log_name")
            or trainer_cfg.get("record_log_name")
            or ""
        ).strip()
        default_label = (
            Path(output_dir).name
            if output_dir
            else (Path(rollout_dir).name if rollout_dir else self.variant.get("order", "default"))
        )
        runtime_run_label = runtime_ctx.get("run_label")
        run_label = self._sanitize_log_component(
            str(runtime_run_label) if runtime_run_label else (explicit_label or default_label),
            "default",
        )
        update_idx = runtime_ctx.get("update_idx")
        update_tag = (
            f"update_u{int(update_idx):05d}"
            if update_idx is not None
            else "update_unknown"
        )

        session_meta = {
            "log_mode": record_mode,
            "log_run_label": run_label,
            "update_idx": int(update_idx) if update_idx is not None else None,
            "update_tag": update_tag,
            "loop_round_idx": (
                int(runtime_ctx["loop_round_idx"])
                if runtime_ctx.get("loop_round_idx") is not None
                else None
            ),
            "order": self.variant.get("order"),
            "layout": self.variant.get("layout"),
            "horizon": self.variant.get("horizon"),
            "trainer_collect_only": collect_only,
            "trainer_train_only": train_only,
            "trainer_output_dir": output_dir,
            "trainer_rollout_dir": rollout_dir,
        }
        return record_mode, run_label, update_tag, session_meta

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
        has_reward_entries = False
        if process_reward and isinstance(process_reward, dict):
            for agent_reward in process_reward.get("per_agent") or []:
                if isinstance(agent_reward, dict) and agent_reward.get("calls"):
                    has_reward_entries = True
                    break
        if not records and not has_reward_entries:
            return
        reward_queues: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
        reward_by_source_key: Dict[int, Dict[Tuple[int, int], List[Dict[str, Any]]]] = {0: {}, 1: {}}
        reward_default_ts = None
        indexed_entry_ids = set()

        def _index_reward_entry(
            entry: Dict[str, Any],
            agent_idx: int,
            *,
            add_to_queue: bool,
        ) -> None:
            if not isinstance(entry, dict):
                return
            entry_id = id(entry)
            if entry_id in indexed_entry_ids:
                return
            indexed_entry_ids.add(entry_id)
            if add_to_queue:
                reward_queues[agent_idx].append(entry)
            call_index = entry.get("call_index")
            if call_index is None:
                return
            source_ts = entry.get(
                "source_timestamp",
                entry.get("timestamp", reward_default_ts),
            )
            if source_ts is None:
                source_ts = -1
            try:
                reward_by_source_key[agent_idx].setdefault(
                    (int(source_ts), int(call_index)), []
                ).append(entry)
            except (TypeError, ValueError):
                pass

        if process_reward and isinstance(process_reward, dict):
            reward_default_ts = process_reward.get("timestamp")
            per_agent = process_reward.get("per_agent") or []
            for agent_idx in range(min(len(per_agent), 2)):
                agent_reward = per_agent[agent_idx]
                agent_calls = agent_reward.get("calls") if isinstance(agent_reward, dict) else None
                if agent_calls:
                    for entry in agent_calls:
                        _index_reward_entry(entry, agent_idx, add_to_queue=True)
            for entry in process_reward.get("source_entries") or []:
                if not isinstance(entry, dict):
                    continue
                try:
                    source_agent_idx = int(entry.get("agent_index"))
                except (TypeError, ValueError):
                    continue
                if source_agent_idx not in reward_by_source_key:
                    continue
                _index_reward_entry(entry, source_agent_idx, add_to_queue=False)

        def _record_action(record: PolicyCallRecord, metadata: Dict[str, Any]) -> str:
            return self._normalize_reward_action(str(
                metadata.get("action") or self._extract_action_text(record.response)
            ))

        def _entry_action(entry: Dict[str, Any]) -> str:
            return self._normalize_reward_action(str(entry.get("action") or ""))

        def _entry_reward_magnitude(entry: Dict[str, Any]) -> float:
            reward_keys = (
                "sequence_reward",
                "format_reward",
                "validator_reward",
                "communication_reward",
                "collab_reward",
                "paired_comm_reward",
            )
            return sum(abs(float(entry.get(key, 0.0) or 0.0)) for key in reward_keys)

        def _record_action_is_embodied(action: str) -> bool:
            lowered = action.lower()
            return bool(
                action
                and not lowered.startswith(("collab(", "request(", "seek(", "ack(", "deny("))
                and not lowered.startswith("wait")
            )

        def _pop_entry_from_queue(agent_idx: int, target_entry: Dict[str, Any]) -> None:
            queue = reward_queues.get(agent_idx, [])
            reward_queues[agent_idx] = [
                entry for entry in queue if entry is not target_entry
            ]

        def _pop_source_entry(
            agent_idx: int,
            key: Tuple[int, int],
            record: PolicyCallRecord,
            metadata: Dict[str, Any],
            context: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]:
            candidates = reward_by_source_key.get(agent_idx, {}).get(key)
            if not candidates:
                return None
            target_action = _record_action(record, metadata)
            target_call_type = (
                metadata.get("semantic_call_type")
                or metadata.get("call_type")
                or context.get("call_type")
            )
            selected_idx: Optional[int] = None
            selected_score: Optional[Tuple[int, int, int, float]] = None
            target_is_embodied = _record_action_is_embodied(target_action)
            for candidate_idx, candidate in enumerate(candidates):
                candidate_action = _entry_action(candidate)
                candidate_call_type = candidate.get("call_type")
                actions_match = bool(
                    target_action and candidate_action and candidate_action == target_action
                )
                if target_action and candidate_action and candidate_action != target_action:
                    continue
                if (
                    not actions_match
                    and len(candidates) > 1
                    and
                    target_call_type
                    and candidate_call_type
                    and target_call_type != candidate_call_type
                ):
                        continue
                score = (
                    1 if actions_match else 0,
                    1 if target_call_type and candidate_call_type == target_call_type else 0,
                    1
                    if target_is_embodied
                    and candidate_call_type in {"planner_main", "validator_correction"}
                    else 0,
                    _entry_reward_magnitude(candidate),
                )
                if selected_score is None or score > selected_score:
                    selected_idx = candidate_idx
                    selected_score = score
            if selected_idx is None:
                return None
            entry = candidates.pop(selected_idx)
            if not candidates:
                reward_by_source_key.get(agent_idx, {}).pop(key, None)
            _pop_entry_from_queue(agent_idx, entry)
            return entry

        def _remove_source_entry(agent_idx: int, target_entry: Dict[str, Any]) -> None:
            by_key = reward_by_source_key.get(agent_idx, {})
            empty_keys = []
            for key, entries in by_key.items():
                remaining = [entry for entry in entries if entry is not target_entry]
                if remaining:
                    by_key[key] = remaining
                else:
                    empty_keys.append(key)
            for key in empty_keys:
                by_key.pop(key, None)

        for idx, record in enumerate(records):
            metadata = dict(record.metadata)
            context = dict(record.context or {})
            context_call_type = context.get("call_type")
            action_mode = metadata.get("action_mode")
            semantic_call_type = (
                action_mode
                if action_mode in {"planner_main", "communication"}
                else context_call_type
            )
            if context_call_type:
                metadata.setdefault("call_type", context_call_type)
            if semantic_call_type:
                metadata.setdefault("semantic_call_type", semantic_call_type)
            if metadata.get("call_index") is None and "call_index" in context:
                metadata["call_index"] = context.get("call_index")
            if metadata.get("replay_action") and not metadata.get("action"):
                metadata["action"] = metadata.get("replay_action")
            if record.micro_step is not None:
                metadata.setdefault("micro_step", int(record.micro_step))
            reward_entry = None
            call_index = metadata.get("call_index")
            if call_index is not None:
                try:
                    call_index_int = int(call_index)
                except (TypeError, ValueError):
                    call_index_int = None
                if call_index_int is not None:
                    record_ts = None
                    try:
                        record_ts = int(record.timestep)
                    except (TypeError, ValueError):
                        record_ts = None
                    if record_ts is not None:
                        reward_entry = _pop_source_entry(
                            record.agent_index,
                            (record_ts, call_index_int),
                            record,
                            metadata,
                            context,
                        )
                    if reward_entry is None:
                        reward_entry = _pop_source_entry(
                            record.agent_index,
                            (-1, call_index_int),
                            record,
                            metadata,
                            context,
                        )
            if reward_entry is None:
                queue = reward_queues.get(record.agent_index, [])
                for queue_idx, candidate in enumerate(queue):
                    if candidate.get("call_index") is not None:
                        continue
                    reward_entry = queue.pop(queue_idx)
                    _remove_source_entry(record.agent_index, reward_entry)
                    break
            if reward_entry is None and metadata.get("action"):
                target_action = self._normalize_reward_action(str(metadata.get("action") or ""))
                target_call_type = metadata.get("call_type") or context.get("call_type")
                queue = reward_queues.get(record.agent_index, [])
                for queue_idx, candidate in enumerate(queue):
                    candidate_action = self._normalize_reward_action(
                        str(candidate.get("action") or "")
                    )
                    candidate_call_type = candidate.get("call_type")
                    if candidate_action != target_action:
                        continue
                    if target_call_type and candidate_call_type and target_call_type != candidate_call_type:
                        continue
                    reward_entry = queue.pop(queue_idx)
                    _remove_source_entry(record.agent_index, reward_entry)
                    break
            if reward_entry:
                normalizer = getattr(
                    getattr(self, "reward_tracker", None),
                    "normalize_reward_entry",
                    None,
                )
                if normalizer is not None:
                    reward_entry = normalizer(reward_entry)
                seq_reward = float(reward_entry.get("sequence_reward", 0.0))
                fmt_reward = float(reward_entry.get("format_reward", 0.0))
                validator_reward = float(reward_entry.get("validator_reward", 0.0))
                validator_penalty_floor = getattr(
                    getattr(self, "reward_tracker", None),
                    "validator_penalty_value",
                    None,
                )
                if validator_penalty_floor is not None:
                    validator_penalty_floor = float(validator_penalty_floor)
                    if validator_reward < validator_penalty_floor:
                        validator_reward = validator_penalty_floor
                entry_penalties = list(reward_entry.get("penalties") or [])
                if any(item.get("type") == "format" for item in entry_penalties):
                    validator_reward = 0.0
                    reward_entry = dict(reward_entry)
                    reward_entry["validator_reward"] = 0.0
                    reward_entry["penalties"] = [
                        item
                        for item in entry_penalties
                        if item.get("type") != "validator"
                    ]
                    reward_entry["total"] = (
                        float(reward_entry.get("sequence_reward", 0.0) or 0.0)
                        + float(reward_entry.get("format_reward", 0.0) or 0.0)
                        + float(reward_entry.get("communication_reward", 0.0) or 0.0)
                        + float(reward_entry.get("collab_reward", 0.0) or 0.0)
                        + float(reward_entry.get("paired_comm_reward", 0.0) or 0.0)
                    )
                can_receive_validated_reward = self._record_can_receive_validated_reward_from_entry(
                    record, reward_entry
                )
                if not can_receive_validated_reward:
                    seq_reward = 0.0
                    validator_reward = 0.0
                    fmt_reward = 0.0
                communication_reward = float(
                    reward_entry.get("communication_reward", 0.0)
                )
                repeat_communication_reward = float(
                    reward_entry.get("repeat_communication_reward", 0.0)
                )
                forced_communication_reward = float(
                    reward_entry.get("forced_communication_reward", 0.0)
                )
                paired_comm_reward = float(
                    reward_entry.get("paired_comm_reward", 0.0)
                )
                if (
                    paired_comm_reward < 0.0
                    and (fmt_reward < 0.0 or validator_reward < 0.0)
                ):
                    paired_comm_reward = 0.0
                collab_reward = float(reward_entry.get("collab_reward", 0.0))
                record_action = self._normalize_reward_action(str(
                    metadata.get("action") or self._extract_action_text(record.response)
                ))
                if record_action.lower().startswith("wait"):
                    seq_reward = 0.0
                    validator_reward = 0.0
                    communication_reward = 0.0
                    repeat_communication_reward = 0.0
                    forced_communication_reward = 0.0
                    paired_comm_reward = 0.0
                    collab_reward = 0.0
                record.reward = (
                    seq_reward
                    + fmt_reward
                    + validator_reward
                    + communication_reward
                    + collab_reward
                    + paired_comm_reward
                )
                breakdown = {
                    "sequence_reward": seq_reward,
                    "format_reward": fmt_reward,
                    "validator_reward": validator_reward,
                    "communication_reward": communication_reward,
                    "repeat_communication_reward": repeat_communication_reward,
                    "forced_communication_reward": forced_communication_reward,
                    "collab_reward": collab_reward,
                    "paired_comm_reward": paired_comm_reward,
                    "call_type": reward_entry.get("call_type"),
                    "raw": reward_entry,
                }
                metadata["reward_breakdown"] = breakdown
            else:
                fallback = self._fallback_communication_reward(record, metadata)
                if fallback is not None:
                    record.reward = float(fallback.get("total", 0.0) or 0.0)
                    metadata["reward_breakdown"] = fallback
                else:
                    record.reward = 0.0
                    metadata.setdefault("reward_breakdown", {}).setdefault(
                        "missing_reward_entry", True
                    )
            record.metadata = metadata

        for idx, record in enumerate(records):
            record.done = bool(done_flag) if idx == len(records) - 1 else False

    @staticmethod
    def _record_can_receive_validated_action_reward(record: PolicyCallRecord) -> bool:
        metadata = record.metadata or {}
        action_mode = metadata.get("action_mode")
        semantic = metadata.get("semantic_call_type") or metadata.get("call_type")
        context_call_type = (record.context or {}).get("call_type")
        if action_mode in {"communication", "mixed", "malformed", "multi_embodied"}:
            return False
        if action_mode in {"planner_main", "validator_correction"}:
            return True
        if semantic == "communication" or context_call_type == "communication":
            return False
        rewardable_types = {"planner_main", "validator_correction"}
        return semantic in rewardable_types or context_call_type in rewardable_types

    @staticmethod
    def _reward_entry_is_validated_action(entry: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(entry, dict):
            return False
        action = str(entry.get("action") or "").strip()
        return (
            entry.get("call_type") in {"planner_main", "validator_correction"}
            and bool(action)
            and not action.lower().startswith(("collab(", "request(", "seek(", "ack(", "deny("))
            and not action.lower().startswith("wait")
        )

    @staticmethod
    def _record_has_embodied_action_text(record: PolicyCallRecord) -> bool:
        text = str((record.metadata or {}).get("action") or "")
        if not text:
            text = CollabMainSession._extract_action_text(record.response)
        text = text.strip()
        if not text:
            return False
        lowered = text.lower()
        return not lowered.startswith(("collab(", "request(", "seek(", "ack(", "deny(", "wait"))

    @staticmethod
    def _record_can_receive_validated_reward_from_entry(
        record: PolicyCallRecord, entry: Optional[Dict[str, Any]]
    ) -> bool:
        if CollabMainSession._record_can_receive_validated_action_reward(record):
            return True
        if not CollabMainSession._reward_entry_is_validated_action(entry):
            return False
        metadata = record.metadata or {}
        action_mode = metadata.get("action_mode")
        if action_mode in {"mixed", "multi_embodied"}:
            return False
        if action_mode in {"communication", "malformed"} and not CollabMainSession._record_has_embodied_action_text(record):
            return False
        return True

    def _fallback_communication_reward(
        self,
        record: PolicyCallRecord,
        metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Score communication records that were not present in after_step calls.

        The process/action rewards remain strictly bound through reward tracker
        entries. This fallback only covers Collab(...) communication process
        rewards, which are based on the communication text itself.
        """
        debug = bool(os.environ.get("RL_DEBUG_COMM_FALLBACK"))
        if not getattr(self, "reward_tracker", None):
            if debug:
                print("[comm-fallback] no reward_tracker", flush=True)
            return None
        action = str(metadata.get("action") or metadata.get("replay_action") or "")
        if not action:
            action = self._extract_action_text(record.response)
        normalized = self._normalize_reward_action(action)
        if not normalized or not normalized.lower().startswith(
            ("collab(", "request(", "seek(", "ack(", "deny(")
        ):
            if debug:
                print(
                    f"[comm-fallback] skip non-collab agent={record.agent_index} "
                    f"t={record.timestep} action={action!r} normalized={normalized!r} "
                    f"meta_keys={sorted(metadata.keys())}",
                    flush=True,
                )
            return None

        paired_comm_reward = 0.0
        paired_meta: Dict[str, Any] = {}
        if getattr(self.reward_tracker, "enable_paired_comm_reward", False):
            paired_comm_reward, paired_meta = self.reward_tracker._process_paired_comm_reward(  # noqa: SLF001
                int(record.agent_index),
                int(record.timestep) if record.timestep is not None else -1,
                normalized,
            )
            paired_comm_reward = float(paired_comm_reward or 0.0)
        collab_reward = 0.0
        if getattr(self.reward_tracker, "enable_collab_reward", False):
            collab_reward = float(
                self.reward_tracker._process_collab_reward(  # noqa: SLF001
                    int(record.agent_index),
                    normalized,
                )
                or 0.0
            )

        total = collab_reward + paired_comm_reward
        if total == 0.0:
            if debug:
                print(
                    f"[comm-fallback] zero total agent={record.agent_index} "
                    f"t={record.timestep} normalized={normalized!r} "
                    f"collab={collab_reward} paired={paired_comm_reward} "
                    f"paired_meta={paired_meta}",
                    flush=True,
                )
            return None
        if debug:
            print(
                f"[comm-fallback] reward agent={record.agent_index} t={record.timestep} "
                f"normalized={normalized!r} collab={collab_reward} "
                f"paired={paired_comm_reward} total={total}",
                flush=True,
            )
        raw = {
            "timestamp": int(record.timestep) if record.timestep is not None else -1,
            "source_timestamp": int(record.timestep) if record.timestep is not None else -1,
            "agent_index": int(record.agent_index),
            "call_index": metadata.get("call_index"),
            "call_type": "communication",
            "action": normalized,
            "sequence_reward": 0.0,
            "progress_reward": 0.0,
            "communication_reward": 0.0,
            "repeat_communication_reward": 0.0,
            "forced_communication_reward": 0.0,
            "collab_reward": collab_reward,
            "paired_comm_reward": paired_comm_reward,
            "paired_comm_role": paired_meta.get("role"),
            "paired_comm_result": paired_meta.get("result"),
            "paired_comm_target": paired_meta.get("target_agent"),
            "paired_comm_request_action": paired_meta.get("request_action"),
            "paired_comm_request_helpful": paired_meta.get("request_helpful"),
            "format_reward": 0.0,
            "validator_reward": 0.0,
            "is_collab": True,
            "total": total,
        }
        return {
            "sequence_reward": 0.0,
            "format_reward": 0.0,
            "validator_reward": 0.0,
            "communication_reward": 0.0,
            "repeat_communication_reward": 0.0,
            "forced_communication_reward": 0.0,
            "collab_reward": collab_reward,
            "paired_comm_reward": paired_comm_reward,
            "call_type": "communication",
            "raw": raw,
        }

    @staticmethod
    def _extract_action_text(response: str) -> str:
        match = re.search(
            r"Action\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
            response or "",
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        return (match.group(1) if match else "").strip()

    @staticmethod
    def _normalize_reward_action(action: str) -> str:
        text = (action or "").strip()
        if not text:
            return ""
        text = text.replace("<|im_end|>", "").strip()
        match = re.search(
            r"Action\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        if match:
            text = match.group(1).strip()
        text = _strip_action_code_fence(text)
        return text.replace(" ", "")

    def _metadata_snapshot(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for key, value in metadata.items():
            if key in {"prompt_ids", "response_ids", "critic_input_ids"}:
                length = 0
                if hasattr(value, "numel"):
                    try:
                        length = int(value.numel())
                    except Exception:
                        length = 0
                elif hasattr(value, "__len__"):
                    try:
                        length = len(value)  # type: ignore[arg-type]
                    except Exception:
                        length = 0
                snapshot[f"{key}_length"] = length
                continue
            if isinstance(value, (int, float, str, bool)):
                snapshot[key] = value
            elif isinstance(value, dict):
                child = {
                    c_key: c_val
                    for c_key, c_val in value.items()
                    if isinstance(c_val, (int, float, str, bool))
                }
                if child:
                    snapshot[key] = child
        return snapshot

    def _persist_policy_records(self, records: List[PolicyCallRecord]) -> None:
        if not records or not getattr(self, "_policy_record_paths", None):
            return
        for record in records:
            agent_idx = getattr(record, "agent_index", None)
            if agent_idx is None:
                continue
            try:
                log_path = self._policy_record_paths[agent_idx]
            except (IndexError, TypeError):
                continue
            payload = {
                "timestep": record.timestep,
                "agent_index": agent_idx,
                "call_type": (record.context or {}).get("call_type"),
                "log_mode": getattr(self, "_policy_record_session_meta", {}).get("log_mode"),
                "log_run_label": getattr(self, "_policy_record_session_meta", {}).get("log_run_label"),
                "update_idx": getattr(self, "_policy_record_session_meta", {}).get("update_idx"),
                "update_tag": getattr(self, "_policy_record_session_meta", {}).get("update_tag"),
                "loop_round_idx": getattr(self, "_policy_record_session_meta", {}).get("loop_round_idx"),
                "reward": record.reward,
                "done": record.done,
                "prompt": record.prompt,
                "response": record.response,
                "messages": record.messages,
                "metadata": self._metadata_snapshot(record.metadata),
                "session_meta": getattr(self, "_policy_record_session_meta", {}),
            }
            try:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except OSError:
                continue

    @staticmethod
    def _shrink_policy_records(records: List[PolicyCallRecord]) -> None:
        for record in records:
            if record.response:
                metadata = record.metadata if isinstance(record.metadata, dict) else {}
                metadata.setdefault("_response_text_for_reward", record.response)
                record.metadata = metadata
            record.messages = []
            record.prompt = ""
            record.response = ""

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
