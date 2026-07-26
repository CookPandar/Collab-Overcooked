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
        self._pending_reward_records: Dict[Tuple[int, int, int], PolicyCallRecord] = {}
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
        self._pending_reward_records.clear()
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
        print(f"[CollabMainSession] Beginning step at timestep {state.timestep}")
        joint_action, pickup_parm = self.team.joint_action(state)
        print(
            f"[CollabMainSession] joint_action={joint_action} pickup_parm={pickup_parm}"
        )
        obs, reward, done, env_info = self.env.step(joint_action, pickup_parm)
        executed_action_sources = self._consume_executed_action_sources(obs.ml_actions)

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
        records = self._merge_pending_reward_records(records, process_reward)
        self._assign_call_rewards(records, process_reward, done)
        self._update_pending_reward_records(records)
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
        )

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

    def _consume_executed_action_sources(
        self, ml_actions: Optional[List[Optional[str]]] = None
    ) -> List[Optional[Dict[str, Any]]]:
        sources: List[Optional[Dict[str, Any]]] = []
        for idx, agent in enumerate(self.team.agents):
            executed_action = (
                ml_actions[idx]
                if ml_actions is not None and idx < len(ml_actions)
                else None
            )
            source = getattr(agent, "last_executed_action_source", None)
            if executed_action and not isinstance(source, dict):
                resolver = getattr(agent, "_resolve_executed_action_source", None)
                if callable(resolver):
                    try:
                        source = resolver(executed_action)
                    except Exception:
                        source = None
            if executed_action and isinstance(source, dict):
                enriched = dict(source)
                enriched.setdefault("agent_index", idx)
                enriched.setdefault("submitted_action", executed_action)
                sources.append(enriched)
                try:
                    setattr(agent, "last_executed_action_source", None)
                except Exception:
                    pass
            else:
                sources.append(None)
                # Do not consume the source on movement steps or failed interacts.
                # It must stay attached until the environment reports the actual
                # medium-level action in state.ml_actions.
        return sources

    def _process_reward_call_indices(
        self, process_reward: Optional[Dict[str, Any]]
    ) -> Dict[int, set]:
        by_agent: Dict[int, set] = {0: set(), 1: set()}
        if not isinstance(process_reward, dict):
            return by_agent
        default_ts = process_reward.get("timestamp")
        per_agent = process_reward.get("per_agent") or []
        for agent_idx in range(min(len(per_agent), 2)):
            agent_reward = per_agent[agent_idx]
            calls = agent_reward.get("calls") if isinstance(agent_reward, dict) else None
            if not calls:
                continue
            for entry in calls:
                if not isinstance(entry, dict):
                    continue
                call_index = entry.get("call_index")
                if call_index is None:
                    continue
                try:
                    source_ts = entry.get("source_timestamp", entry.get("timestamp", default_ts))
                    by_agent[agent_idx].add((int(source_ts), int(call_index)))
                except (TypeError, ValueError):
                    continue
        return by_agent

    def _record_source_key(
        self, record: PolicyCallRecord
    ) -> Optional[Tuple[int, int, int]]:
        if record.agent_index not in (0, 1):
            return None
        raw = (record.metadata or {}).get("call_index")
        if raw is None:
            raw = record.micro_step
        try:
            return int(record.agent_index), int(record.timestep), int(raw)
        except (TypeError, ValueError):
            return None

    def _merge_pending_reward_records(
        self,
        records: List[PolicyCallRecord],
        process_reward: Optional[Dict[str, Any]],
    ) -> List[PolicyCallRecord]:
        if not hasattr(self, "_pending_reward_records"):
            self._pending_reward_records = {}
        wanted = self._process_reward_call_indices(process_reward)
        merged = list(records)
        present = {
            key for key in (self._record_source_key(record) for record in merged) if key
        }
        for agent_idx, source_keys in wanted.items():
            for source_ts, call_index in source_keys:
                key = (agent_idx, source_ts, call_index)
                if key in present:
                    continue
                pending = self._pending_reward_records.pop(key, None)
                if pending is not None:
                    merged.append(pending)
                    present.add(key)
        return merged

    def _update_pending_reward_records(self, records: List[PolicyCallRecord]) -> None:
        if not hasattr(self, "_pending_reward_records"):
            self._pending_reward_records = {}
        for record in records:
            key = self._record_source_key(record)
            if key is None:
                continue
            breakdown = (record.metadata or {}).get("reward_breakdown") or {}
            seq_reward = float(breakdown.get("sequence_reward", 0.0) or 0.0)
            format_reward = float(breakdown.get("format_reward", 0.0) or 0.0)
            has_penalty = bool(
                format_reward < 0.0
                or float(breakdown.get("validator_reward", 0.0) or 0.0)
                or float(breakdown.get("communication_reward", 0.0) or 0.0)
                or float(breakdown.get("paired_comm_reward", 0.0) or 0.0)
            )
            if seq_reward or has_penalty or not self._record_can_receive_sequence_reward(record):
                self._pending_reward_records.pop(key, None)
            else:
                self._pending_reward_records[key] = record

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
        if not hasattr(self, "_pending_reward_records"):
            self._pending_reward_records = {}
        has_reward_entries = False
        if process_reward and isinstance(process_reward, dict):
            for agent_reward in process_reward.get("per_agent") or []:
                if isinstance(agent_reward, dict) and agent_reward.get("calls"):
                    has_reward_entries = True
                    break
        if not records and not has_reward_entries:
            return
        reward_queues: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
        reward_by_source_key: Dict[int, Dict[Tuple[int, int], Dict[str, Any]]] = {0: {}, 1: {}}
        executed_sequence_reward: Dict[int, float] = {0: 0.0, 1: 0.0}
        reward_default_ts = None
        if process_reward and isinstance(process_reward, dict):
            reward_default_ts = process_reward.get("timestamp")
            per_agent = process_reward.get("per_agent") or []
            for agent_idx in range(min(len(per_agent), 2)):
                agent_reward = per_agent[agent_idx]
                if isinstance(agent_reward, dict):
                    executed_sequence_reward[agent_idx] = float(
                        agent_reward.get("sequence_reward", 0.0) or 0.0
                    )
                agent_calls = agent_reward.get("calls") if isinstance(agent_reward, dict) else None
                if agent_calls:
                    for entry in agent_calls:
                        if not isinstance(entry, dict):
                            continue
                        reward_queues[agent_idx].append(entry)
                        call_index = entry.get("call_index")
                        if call_index is not None:
                            source_ts = entry.get(
                                "source_timestamp",
                                entry.get("timestamp", reward_default_ts),
                            )
                            if source_ts is None:
                                source_ts = -1
                            try:
                                reward_by_source_key[agent_idx][
                                    (int(source_ts), int(call_index))
                                ] = entry
                            except (TypeError, ValueError):
                                pass

        for idx, record in enumerate(records):
            metadata = dict(record.metadata)
            context_call_type = (record.context or {}).get("call_type")
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
            if record.micro_step is not None:
                metadata.setdefault("micro_step", int(record.micro_step))
            reward_entry = None
            call_index = metadata.get("call_index")
            reward_source_key = None
            if call_index is not None:
                try:
                    call_index_int = int(call_index)
                    record_ts_int = int(record.timestep)
                    reward_source_key = (record.agent_index, record_ts_int, call_index_int)
                except (TypeError, ValueError):
                    call_index_int = None
                    reward_source_key = None
                if call_index_int is not None:
                    record_ts = None
                    try:
                        record_ts = int(record.timestep)
                    except (TypeError, ValueError):
                        record_ts = None
                    if record_ts is not None:
                        reward_entry = reward_by_source_key.get(record.agent_index, {}).pop(
                            (record_ts, call_index_int), None
                        )
                    if reward_entry is None:
                        reward_entry = reward_by_source_key.get(record.agent_index, {}).pop(
                            (-1, call_index_int), None
                        )
                    if reward_entry is not None:
                        queue = reward_queues.get(record.agent_index, [])
                        reward_queues[record.agent_index] = [
                            entry for entry in queue if entry is not reward_entry
                        ]
            if reward_entry is None:
                queue = reward_queues.get(record.agent_index, [])
                for queue_idx, candidate in enumerate(queue):
                    if candidate.get("call_index") is not None:
                        continue
                    reward_entry = queue.pop(queue_idx)
                    break
            if reward_entry:
                if reward_source_key is not None:
                    self._pending_reward_records.pop(reward_source_key, None)
                seq_reward = float(reward_entry.get("sequence_reward", 0.0))
                fmt_reward = float(reward_entry.get("format_reward", 0.0))
                validator_reward = float(reward_entry.get("validator_reward", 0.0))
                can_receive_execution_reward = self._record_can_receive_execution_reward(record)
                if not can_receive_execution_reward:
                    action_mode = metadata.get("action_mode")
                    semantic = metadata.get("semantic_call_type") or metadata.get("call_type")
                    context_call_type = (record.context or {}).get("call_type")
                    entry_call_type = reward_entry.get("call_type")
                    explicitly_non_execution = (
                        action_mode in {"communication", "mixed", "malformed", "multi_embodied"}
                        or (
                            action_mode not in {"planner_main", "validator_correction"}
                            and (semantic == "communication" or context_call_type == "communication")
                        )
                    )
                    can_receive_execution_reward = (
                        not explicitly_non_execution
                        and entry_call_type in {"planner_main", "validator_correction"}
                    )
                if not can_receive_execution_reward:
                    seq_reward = 0.0
                    validator_reward = 0.0
                    if fmt_reward > 0.0:
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
                record.reward = (
                    seq_reward
                    + fmt_reward
                    + validator_reward
                    + communication_reward
                    + paired_comm_reward
                )
                breakdown = {
                    "sequence_reward": seq_reward,
                    "format_reward": fmt_reward,
                    "validator_reward": validator_reward,
                    "communication_reward": communication_reward,
                    "repeat_communication_reward": repeat_communication_reward,
                    "forced_communication_reward": forced_communication_reward,
                    "paired_comm_reward": paired_comm_reward,
                    "call_type": reward_entry.get("call_type"),
                    "raw": reward_entry,
                }
                metadata["reward_breakdown"] = breakdown
            else:
                record.reward = 0.0
                metadata.setdefault("reward_breakdown", {}).setdefault(
                    "missing_reward_entry", True
                )
            record.metadata = metadata

        self._normalize_executed_sequence_rewards(records, executed_sequence_reward)
        for idx, record in enumerate(records):
            record.done = bool(done_flag) if idx == len(records) - 1 else False

    @staticmethod
    def _record_can_receive_execution_reward(record: PolicyCallRecord) -> bool:
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
    def _record_can_receive_sequence_reward(record: PolicyCallRecord) -> bool:
        return CollabMainSession._record_can_receive_execution_reward(record)

    @staticmethod
    def _set_record_sequence_reward(record: PolicyCallRecord, sequence_reward: float) -> None:
        metadata = dict(record.metadata or {})
        breakdown = dict(metadata.get("reward_breakdown") or {})
        old_sequence = float(breakdown.get("sequence_reward", 0.0) or 0.0)
        delta = float(sequence_reward) - old_sequence
        record.reward = float(record.reward or 0.0) + delta
        breakdown["sequence_reward"] = float(sequence_reward)
        raw = dict(breakdown.get("raw") or {})
        if raw:
            old_raw_sequence = float(raw.get("sequence_reward", old_sequence) or 0.0)
            raw["sequence_reward"] = float(sequence_reward)
            raw["progress_reward"] = float(sequence_reward)
            raw["total"] = float(raw.get("total", 0.0) or 0.0) + (
                float(sequence_reward) - old_raw_sequence
            )
            breakdown["raw"] = raw
        metadata["reward_breakdown"] = breakdown
        record.metadata = metadata

    def _normalize_executed_sequence_rewards(
        self,
        records: List[PolicyCallRecord],
        executed_sequence_reward: Dict[int, float],
    ) -> None:
        """Ensure one executed env action gives sequence reward to at most one transition."""
        records_by_agent: Dict[int, List[PolicyCallRecord]] = {0: [], 1: []}
        for record in records:
            if record.agent_index in records_by_agent:
                records_by_agent[record.agent_index].append(record)

        for agent_idx, agent_records in records_by_agent.items():
            target_sequence = float(executed_sequence_reward.get(agent_idx, 0.0) or 0.0)
            if target_sequence == 0.0:
                continue
            kept = False
            for record in agent_records:
                metadata = record.metadata or {}
                breakdown = metadata.get("reward_breakdown") or {}
                current_sequence = float(
                    breakdown.get("sequence_reward", 0.0) or 0.0
                )
                if current_sequence == 0.0:
                    continue
                if kept or not self._record_can_receive_sequence_reward(record):
                    self._set_record_sequence_reward(record, 0.0)
                    continue
                self._set_record_sequence_reward(record, min(current_sequence, target_sequence))
                kept = True

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
