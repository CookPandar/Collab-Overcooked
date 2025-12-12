"""
Utility wrappers to expose Collab-Overcooked as a lightweight RL environment.

The goal of this module is to decouple reinforcement-learning style rollouts
from the heavy logging / CLI workflow implemented in ``collab_overcooked.main``.

Typical usage::

    from collab_overcooked.training.env_wrapper import CollabOvercookedEnv

    env = CollabOvercookedEnv(
        layout="cramped_room",
        order="boiled_egg",
        horizon=120,
    )

    obs = env.reset()
    done = False
    while not done:
        joint_action = (env.ACTION_ID["stay"], env.ACTION_ID["stay"])
        step_result = env.step(joint_action)
        done = step_result.done
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState

from ..reward import ProcessRewardTracker


@dataclass
class StepResult:
    """Container returned by :meth:`CollabOvercookedEnv.step`."""

    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    process_reward: Optional[Dict[str, Any]] = None


class CollabOvercookedEnv:
    """Minimal RL-style wrapper around ``OvercookedEnv`` with process rewards."""

    ACTION_ID = {
        "stay": Action.STAY,
        "north": Action.NORTH,
        "south": Action.SOUTH,
        "east": Action.EAST,
        "west": Action.WEST,
        "interact": Action.INTERACT,
    }

    ID_TO_ACTION = {idx: name for name, idx in ACTION_ID.items()}

    def __init__(
        self,
        layout: str,
        order: str,
        horizon: int = 120,
        reward_settings: Optional[Dict[str, float]] = None,
    ) -> None:
        self.layout_name = layout
        self.order_name = order
        self.horizon = horizon

        self.mdp = OvercookedGridworld.from_layout_name(layout)
        if self.order_name:
            self._configure_single_order(self.order_name)

        self.env = OvercookedEnv(self.mdp, horizon=horizon)
        self.reward_tracker = ProcessRewardTracker(
            order=self.order_name,
            mdp=self.mdp,
            reference_dir=self._reference_dir,
            settings=reward_settings or {},
        )

        self._last_state: Optional[OvercookedState] = None

    @property
    def _reference_dir(self):
        from pathlib import Path
        from .. import __file__ as pkg_file

        base = Path(pkg_file).resolve().parent
        return base / "prompts" / "reference"

    def _configure_single_order(self, order: str):
        self.mdp.start_order_list = [order]
        self.mdp.one_task_mode = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        """Reset env + reward tracker and return initial observation."""
        self.env.reset()
        self.reward_tracker.reset()
        self._last_state = self.env.state
        return self._build_observation(self._last_state)

    def step(
        self,
        joint_action: Sequence[int],
        *,
        ml_actions: Optional[Sequence[Optional[str]]] = None,
        parm: Optional[Tuple[Any, Any]] = None,
    ) -> StepResult:
        """Perform one environment step.

        Args:
            joint_action: length-2 iterable of integer action ids (using ``ACTION_ID``).
            ml_actions: optional list of medium-level action strings for process rewards.
            parm: optional auxiliary argument passed to ``OvercookedEnv.step``. Set to
                ``None`` when issuing primitive actions directly.
        """
        if len(joint_action) != 2:
            raise ValueError("joint_action must contain exactly two actions (Chef, Assistant)")
        if ml_actions is None:
            ml_actions = [None, None]

        action_tuple = tuple(int(a) for a in joint_action)
        state, sparse_reward, done, info = self.env.step(action_tuple, parm)
        self._last_state = state

        process_reward = self.reward_tracker.after_step(
            timestep=state.timestep - 1,
            ml_actions=list(ml_actions),
            state=state,
        )
        observation = self._build_observation(state)
        step_info = {
            "sparse_reward": sparse_reward,
            "horizon_done": done,
            "env_info": info,
        }
        return StepResult(
            observation=observation,
            reward=process_reward["team_total"],
            done=done,
            info=step_info,
            process_reward=process_reward,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_observation(self, state: OvercookedState) -> Dict[str, Any]:
        """Build a lightweight dictionary observation for both agents."""
        counters = self.mdp.get_counter_objects_dict(state, list(self.mdp.terrain_pos_dict["X"]))
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

    def _format_player(self, player_state) -> Dict[str, Any]:
        held = player_state.held_object
        return {
            "position": player_state.position,
            "orientation": player_state.orientation,
            "has_object": held is not None,
            "object": held.name if held is not None else None,
        }


__all__ = ["CollabOvercookedEnv", "StepResult"]
