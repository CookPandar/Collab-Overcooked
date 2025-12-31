"""Utilities for serializing / restoring RL session snapshots."""

from __future__ import annotations

from typing import Any, Dict, Optional

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState


def serialize_env_state(state: OvercookedState) -> Dict[str, Any]:
    """Serialize :class:`OvercookedState` (including auxiliary fields)."""
    payload = state.to_dict()
    payload["ml_actions"] = list(getattr(state, "ml_actions", [None, None]))
    payload["communicate_history"] = list(
        getattr(state, "communicate_history", []) or []
    )
    payload["error_message"] = list(getattr(state, "error_message", []) or [])
    return payload


def deserialize_env_state(data: Dict[str, Any]) -> OvercookedState:
    """Inverse of :func:`serialize_env_state`."""
    base = dict(data)
    ml_actions = base.pop("ml_actions", None)
    comm_history = base.pop("communicate_history", None)
    error_message = base.pop("error_message", None)
    state = OvercookedState.from_dict(base)
    if ml_actions is not None:
        state.ml_actions = list(ml_actions)
    if comm_history is not None:
        state.communicate_history = list(comm_history)
    if error_message is not None:
        state.error_message = list(error_message)
    return state


def build_agent_snapshot(agent) -> Optional[Dict[str, Any]]:
    """Capture runtime metadata from an :class:`LLMAgents` instance if supported."""
    export_fn = getattr(agent, "export_runtime_state", None)
    if callable(export_fn):
        return export_fn()
    return None


def apply_agent_snapshot(agent, snapshot: Optional[Dict[str, Any]]):
    """Restore runtime metadata to agent if possible."""
    if snapshot is None:
        return
    import_fn = getattr(agent, "import_runtime_state", None)
    if callable(import_fn):
        import_fn(snapshot)


def build_session_snapshot(session) -> Dict[str, Any]:
    """Build a serializable snapshot from :class:`CollabMainSession`."""
    env_state = serialize_env_state(session.env.state)
    tracker_state = (
        session.reward_tracker.export_state() if session.reward_tracker else None
    )
    agent_data: Dict[str, Any] = {}
    for idx, agent in enumerate(getattr(session.team, "agents", [])):
        payload = build_agent_snapshot(agent)
        if payload is not None:
            agent_data[str(idx)] = payload
    return {
        "timestep": session.env.state.timestep,
        "env_state": env_state,
        "reward_tracker": tracker_state,
        "agents": agent_data,
        "variant": {
            "layout": session.variant.get("layout"),
            "order": session.variant.get("order"),
            "horizon": session.variant.get("horizon"),
        },
    }


def load_session_snapshot(session, snapshot: Dict[str, Any]):
    """Restore :class:`CollabMainSession` state from :func:`build_session_snapshot`."""
    env_state_data = snapshot.get("env_state")
    if not env_state_data:
        raise ValueError("Snapshot missing env_state payload.")
    new_state = deserialize_env_state(env_state_data)
    session.env.state = new_state
    session.env.t = int(snapshot.get("env_time", new_state.timestep) or new_state.timestep)
    session.env.cumulative_sparse_rewards = 0
    session.env.cumulative_shaped_rewards = 0
    tracker_state = snapshot.get("reward_tracker")
    if tracker_state and session.reward_tracker:
        session.reward_tracker.import_state(tracker_state)
    agents_payload = snapshot.get("agents", {})
    if isinstance(agents_payload, dict):
        for idx, agent in enumerate(getattr(session.team, "agents", [])):
            payload = agents_payload.get(str(idx))
            apply_agent_snapshot(agent, payload)
    # Clear any pending planner records since we just teleported the session.
    for proxy in getattr(session, "rl_modules", []):
        proxy.consume_records()
