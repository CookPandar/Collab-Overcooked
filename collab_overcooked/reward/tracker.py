from __future__ import annotations

import json
import copy
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ProcessRewardTracker:
    """
    Track dense process rewards for benchmarking:
    - sequence reward: compares executed actions with reference demonstrations
    - product reward: grants credit whenever intermediate recipe products appear in the environment
    - penalty hooks: format / validator errors
    """

    def __init__(self, order: str, mdp, reference_dir: Path, settings: Optional[dict] = None):
        if not order:
            raise ValueError("Order name must be specified for reward tracking.")
        self.order = order
        self.mdp = mdp
        self.reference_dir = Path(reference_dir)
        self.settings = settings or {}

        self.sequence_metric = str(self.settings.get("sequence_metric", "tes")).lower()
        self.sequence_weight = float(self.settings.get("sequence_weight", 1.0))
        self.sequence_progress_reward = float(
            self.settings.get("process_progress_reward", 1.0)
        )
        self.product_reward_value = float(self.settings.get("product_reward", 0.5))
        self.format_penalty_value = -abs(self.settings.get("format_penalty", 20.0))
        self.validator_penalty_value = -abs(self.settings.get("validator_penalty", 10.0))
        self.format_success_reward_value = float(
            self.settings.get("format_success_reward", 0.05)
        )
        self.validator_success_reward_value = float(
            self.settings.get("validator_success_reward", 0.05)
        )
        self.communication_penalty_value = -abs(
            self.settings.get("communication_penalty", 0.1)
        )
        self.forced_communication_penalty_value = -abs(
            self.settings.get(
                "forced_communication_penalty",
                abs(self.communication_penalty_value),
            )
        )
        self.enable_collab_reward = bool(self.settings.get("collab_reward_enabled", False))
        self.enable_paired_comm_reward = bool(
            self.settings.get("paired_comm_reward_enabled", False)
        )
        self.paired_comm_request_positive_reward = float(
            self.settings.get("paired_comm_request_positive_reward", 0.5)
        )
        self.paired_comm_request_negative_reward = -abs(
            self.settings.get("paired_comm_request_negative_reward", 0.1)
        )
        self.paired_comm_response_positive_reward = float(
            self.settings.get("paired_comm_response_positive_reward", 0.5)
        )
        self.paired_comm_response_negative_reward = -abs(
            self.settings.get("paired_comm_response_negative_reward", 0.1)
        )
        self.paired_comm_deny_reward = float(
            self.settings.get("paired_comm_deny_reward", 1.0)
        )

        self.references = self._load_references()
        self.sequence_histories: List[List[str]] = [[], []]
        self.sequence_scores: List[float] = [0.0, 0.0]
        self.collab_sequence_scores: List[float] = [0.0, 0.0]
        self.last_action_signatures: List[Optional[str]] = [None, None]

        self.recipe_lookup = self._build_recipe_lookup()
        self.intermediate_targets = self._resolve_recipe_targets(self.order)
        self.observed_targets = set()

        self.penalty_queue: List[List[Dict[str, str]]] = [[], []]
        self.call_events: List[Dict] = []
        self.step_call_records: Dict[int, List[List[Dict]]] = {}
        self.call_records_by_source: Dict[Tuple[int, int, int], Dict] = {}
        self.reward_source_mismatch_events: List[Dict[str, Any]] = []
        self.pending_paired_comm_requests: List[List[Dict[str, Any]]] = [[], []]
        self.paired_comm_request_counter = 0
        self.active_paired_comm_request_ids = set()
        self.pending_collab_execution_requests: List[List[Dict[str, Any]]] = [[], []]
        self.rewardable_action_call_types = {"planner_main", "validator_correction"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register_format_error(self, agent_index: int, detail: str):
        if agent_index is None:
            return
        self.penalty_queue[agent_index].append({"type": "format", "detail": detail})

    def register_validator_error(self, agent_index: int, detail: str):
        if agent_index is None:
            return
        self.penalty_queue[agent_index].append({"type": "validator", "detail": detail})

    def reset(self):
        """Reset histories when the environment starts a new episode."""
        self.sequence_histories = [[], []]
        self.sequence_scores = [0.0, 0.0]
        self.collab_sequence_scores = [0.0, 0.0]
        self.last_action_signatures = [None, None]
        self.observed_targets.clear()
        self.call_events.clear()
        self.step_call_records.clear()
        self.call_records_by_source.clear()
        self.reward_source_mismatch_events.clear()
        self.pending_paired_comm_requests = [[], []]
        self.paired_comm_request_counter = 0
        self.active_paired_comm_request_ids.clear()
        self.pending_collab_execution_requests = [[], []]
        for queue in self.penalty_queue:
            queue.clear()

    def export_state(self) -> Dict[str, Any]:
        """Serialize tracker progress for snapshot replay."""
        return {
            "sequence_histories": copy.deepcopy(self.sequence_histories),
            "sequence_scores": list(self.sequence_scores),
            "collab_sequence_scores": list(self.collab_sequence_scores),
            "last_action_signatures": list(self.last_action_signatures),
            "observed_targets": list(self.observed_targets),
            "penalty_queue": copy.deepcopy(self.penalty_queue),
            "pending_paired_comm_requests": copy.deepcopy(
                self.pending_paired_comm_requests
            ),
            "paired_comm_request_counter": int(self.paired_comm_request_counter),
            "active_paired_comm_request_ids": list(self.active_paired_comm_request_ids),
            "pending_collab_execution_requests": copy.deepcopy(
                self.pending_collab_execution_requests
            ),
        }

    def import_state(self, data: Optional[Dict[str, Any]]):
        """Restore tracker progress from :meth:`export_state` output."""
        if not data:
            self.reset()
            return
        self.sequence_histories = copy.deepcopy(
            data.get("sequence_histories", [[], []])
        )
        if len(self.sequence_histories) < 2:
            self.sequence_histories = [[], []]
        self.sequence_scores = list(data.get("sequence_scores", [0.0, 0.0]))
        if len(self.sequence_scores) < 2:
            self.sequence_scores = [0.0, 0.0]
        self.collab_sequence_scores = list(
            data.get("collab_sequence_scores", [0.0, 0.0])
        )
        if len(self.collab_sequence_scores) < 2:
            self.collab_sequence_scores = [0.0, 0.0]
        restored_signatures = data.get("last_action_signatures")
        if restored_signatures is None:
            restored_signatures = data.get("last_comm_actions", [None, None])
        self.last_action_signatures = list(restored_signatures)
        if len(self.last_action_signatures) < 2:
            self.last_action_signatures = [None, None]
        observed = data.get("observed_targets", [])
        self.observed_targets = set(observed) if observed else set()
        penalty_state = data.get("penalty_queue")
        if isinstance(penalty_state, list) and len(penalty_state) == len(self.penalty_queue):
            self.penalty_queue = [
                list(queue) if isinstance(queue, list) else []
                for queue in penalty_state
            ]
        else:
            for queue in self.penalty_queue:
                queue.clear()
        pending_pairs = data.get("pending_paired_comm_requests")
        if isinstance(pending_pairs, list) and len(pending_pairs) == 2:
            self.pending_paired_comm_requests = [
                list(queue) if isinstance(queue, list) else []
                for queue in pending_pairs
            ]
        else:
            self.pending_paired_comm_requests = [[], []]
        self.paired_comm_request_counter = int(
            data.get("paired_comm_request_counter", 0) or 0
        )
        active_ids = data.get("active_paired_comm_request_ids")
        if active_ids is None:
            active_ids = [
                request.get("request_id")
                for queue in self.pending_paired_comm_requests
                for request in queue
                if isinstance(request, dict) and request.get("request_id") is not None
            ]
        self.active_paired_comm_request_ids = set(active_ids or [])
        pending_collab_exec = data.get("pending_collab_execution_requests")
        if isinstance(pending_collab_exec, list) and len(pending_collab_exec) == 2:
            self.pending_collab_execution_requests = [
                list(queue) if isinstance(queue, list) else []
                for queue in pending_collab_exec
            ]
        else:
            self.pending_collab_execution_requests = [[], []]
        self.call_events.clear()
        self.step_call_records.clear()
        self.call_records_by_source.clear()

    def bootstrap_histories_from_snapshot(
        self, agents_payload: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Seed executed embodied action histories from snapshot agent payloads.

        Off-policy snapshot files may restore the reward tracker with empty
        ``sequence_histories`` even though each agent snapshot still stores the
        teammate's completed medium-level actions in ``teammate_ml_actions``.
        That causes TES/ITES style rewards to be computed from an empty prefix on
        tail snapshots. When both histories are empty, recover them from the
        cross-observed teammate action lists before continuing rollout.
        """
        if not isinstance(agents_payload, dict):
            return False
        if any(self.sequence_histories[idx] for idx in range(min(len(self.sequence_histories), 2))):
            return False

        restored = False
        for agent_idx in range(2):
            observer_payload = agents_payload.get(str(1 - agent_idx)) or {}
            teammate_actions = observer_payload.get("teammate_ml_actions") or []
            if not isinstance(teammate_actions, list):
                continue
            action_rows: List[Tuple[int, int, str]] = []
            for order_idx, item in enumerate(teammate_actions):
                if not isinstance(item, dict):
                    continue
                normalized = self._normalize_action(str(item.get("action") or ""))
                if not normalized:
                    continue
                if normalized.lower().startswith("wait"):
                    continue
                if self._is_collab_action(normalized):
                    continue
                ts_raw = item.get("timestamp")
                try:
                    ts = int(ts_raw)
                except (TypeError, ValueError):
                    ts = order_idx
                action_rows.append((ts, order_idx, normalized))
            action_rows.sort(key=lambda row: (row[0], row[1]))
            history = [row[2] for row in action_rows]
            if not history:
                continue
            self.sequence_histories[agent_idx] = list(history)
            self.sequence_scores[agent_idx] = self._best_sequence_score_from_history(
                agent_idx, history
            )
            self.collab_sequence_scores[agent_idx] = max(
                self.collab_sequence_scores[agent_idx],
                self.sequence_scores[agent_idx],
            )
            restored = True
        return restored

    def bootstrap_histories_from_ml_actions(
        self, ml_actions: Optional[List[Optional[str]]]
    ) -> bool:
        """
        Seed executed action histories from the snapshot state's current ml_actions.

        Tail snapshots can start in the middle of a medium-level action. In that
        case the environment state already reflects progress, but older snapshot
        files may still contain an empty reward-tracker history. Treat the
        snapshot state's active ml_actions as already-established progress so
        ITES/TES rewards do not restart from zero after teleporting.
        """
        if not ml_actions:
            return False
        restored = False
        for agent_idx, action in enumerate(list(ml_actions)[:2]):
            if self.sequence_histories[agent_idx]:
                continue
            normalized = self._normalize_action(str(action or ""))
            if not normalized:
                continue
            if normalized.lower().startswith("wait"):
                continue
            if self._is_collab_action(normalized):
                continue
            self.sequence_histories[agent_idx] = [normalized]
            self.sequence_scores[agent_idx] = self._best_sequence_score_from_history(
                agent_idx, [normalized]
            )
            self.collab_sequence_scores[agent_idx] = max(
                self.collab_sequence_scores[agent_idx],
                self.sequence_scores[agent_idx],
            )
            restored = True
        return restored

    def clear_pending_penalties(self) -> None:
        """Drop unconsumed historical penalties after snapshot teleport."""
        for queue in self.penalty_queue:
            queue.clear()

    def register_llm_action(
        self,
        agent_index: int,
        timestamp: Optional[int],
        action_text: Optional[str],
        *,
        agent_name: Optional[str] = None,
        call_index: Optional[int] = None,
        call_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Record a single LLM call regardless of success / failure."""
        if agent_index is None:
            return {}

        normalized_action = self._clean_action_text(action_text)
        is_collab = self._is_collab_action(normalized_action)
        similarity_before = self.sequence_scores[agent_index]
        similarity_after = similarity_before
        similarity_delta = 0.0
        meta = metadata or {}
        suppress_repeat_penalty = bool(meta.get("suppress_repeat_penalty", False))
        force_communication_penalty = bool(
            meta.get("force_communication_penalty", False)
        )
        repeat_communication_reward = self._process_communication_reward(
            agent_index,
            normalized_action,
            suppress_penalty=suppress_repeat_penalty,
        )
        forced_communication_reward = 0.0
        _penalty_total, penalty_details = self._consume_penalties(agent_index)
        format_reward = sum(entry["value"] for entry in penalty_details if entry["type"] == "format")
        validator_reward = sum(entry["value"] for entry in penalty_details if entry["type"] == "validator")
        has_action_penalty = format_reward < 0.0 or validator_reward < 0.0
        paired_comm_reward, paired_comm_meta = self._process_paired_comm_reward(
            agent_index,
            ts=-1 if timestamp is None else int(timestamp),
            action=normalized_action,
            action_penalized=has_action_penalty,
        )
        collab_reward = (
            self._process_collab_reward(agent_index, normalized_action)
            if self.enable_collab_reward
            else 0.0
        )
        if force_communication_penalty:
            forced_communication_reward += self.forced_communication_penalty_value
        communication_reward = (
            repeat_communication_reward + forced_communication_reward
        )
        # Sequence/ITES progress is assigned only after the action passes the
        # validator and becomes the current medium-level action.
        seq_reward = 0.0
        if (
            format_reward == 0.0
            and normalized_action
            and normalized_action != "[EMPTY]"
            and call_type in self.rewardable_action_call_types
            and not is_collab
            and not self._is_wait_action(normalized_action)
        ):
            format_reward += self.format_success_reward_value
        total = (
            seq_reward
            + collab_reward
            + communication_reward
            + paired_comm_reward
            + format_reward
            + validator_reward
        )

        ts = -1 if timestamp is None else int(timestamp)
        entry = {
            "timestamp": ts,
            "source_timestamp": ts,
            "agent_index": agent_index,
            "agent": agent_name or f"agent_{agent_index}",
            "call_index": call_index,
            "call_type": call_type,
            "action": normalized_action or "[EMPTY]",
            "sequence_reward": seq_reward,
            "progress_reward": seq_reward,
            "communication_reward": communication_reward,
            "repeat_communication_reward": repeat_communication_reward,
            "forced_communication_reward": forced_communication_reward,
            "collab_reward": collab_reward,
            "paired_comm_reward": paired_comm_reward,
            "paired_comm_role": paired_comm_meta.get("role"),
            "paired_comm_result": paired_comm_meta.get("result"),
            "paired_comm_target": paired_comm_meta.get("target_agent"),
            "paired_comm_request_action": paired_comm_meta.get("request_action"),
            "paired_comm_request_helpful": paired_comm_meta.get("request_helpful"),
            "paired_comm_consumed_requests": paired_comm_meta.get("consumed_requests"),
            "paired_comm_registered_requests": paired_comm_meta.get("registered_requests"),
            "format_reward": format_reward,
            "validator_reward": validator_reward,
            "similarity_before": similarity_before,
            "similarity_after": similarity_after,
            "similarity_delta": similarity_delta,
            "penalties": penalty_details,
            "is_collab": is_collab,
            "total": total,
        }
        self._suppress_validator_when_format_failed(entry)
        self._suppress_negative_paired_comm_when_action_penalized(entry)
        self.call_events.append(entry)
        bucket = self.step_call_records.setdefault(ts, [[], []])
        bucket[agent_index].append(entry)
        if call_index is not None:
            self.call_records_by_source[(agent_index, ts, int(call_index))] = entry
        return entry

    def mark_llm_action_validated(
        self,
        agent_index: int,
        timestamp: Optional[int],
        action_text: Optional[str],
        *,
        agent_name: Optional[str] = None,
        call_index: Optional[int] = None,
        call_type: Optional[str] = None,
    ) -> Dict:
        """Grant validator/process rewards when an LLM action is accepted.

        The trigger is validator success, not later environment execution.
        """
        if agent_index is None:
            return {}
        entry = self.ensure_llm_action_entry(
            agent_index=agent_index,
            timestamp=timestamp,
            action_text=action_text,
            agent_name=agent_name,
            call_index=call_index,
            call_type=call_type,
        )
        return self.mark_llm_action_entry_validated(entry, action_text=action_text)

    def mark_llm_action_entry_validated(
        self,
        entry: Optional[Dict[str, Any]],
        *,
        action_text: Optional[str] = None,
    ) -> Dict:
        if not isinstance(entry, dict):
            return {}

        if action_text is not None:
            self._update_llm_action_entry(entry, action_text)
        self._suppress_validator_when_format_failed(entry)
        self._suppress_negative_paired_comm_when_action_penalized(entry)
        if not self._entry_can_receive_validated_action_reward(entry):
            return entry

        if not entry.get("validator_success_assigned"):
            entry["validator_reward"] = (
                float(entry.get("validator_reward", 0.0) or 0.0)
                + self.validator_success_reward_value
            )
            entry["total"] = (
                float(entry.get("total", 0.0) or 0.0)
                + self.validator_success_reward_value
            )
            entry["validator_success_assigned"] = True

        if not entry.get("sequence_reward_assigned"):
            seq_reward, before, after, delta = self._process_sequence_reward(
                int(entry.get("agent_index", 0)),
                entry.get("action"),
            )
            entry["sequence_reward"] = (
                float(entry.get("sequence_reward", 0.0) or 0.0) + seq_reward
            )
            entry["progress_reward"] = (
                float(entry.get("progress_reward", 0.0) or 0.0) + seq_reward
            )
            entry["total"] = float(entry.get("total", 0.0) or 0.0) + seq_reward
            entry["similarity_before"] = before
            entry["similarity_after"] = after
            entry["similarity_delta"] = delta
            entry["sequence_reward_assigned"] = True
            if seq_reward:
                exec_collab_reward, exec_collab_meta = (
                    self._process_collab_execution_reward(
                        int(entry.get("agent_index", 0)),
                        entry.get("action"),
                        sequence_reward=seq_reward,
                    )
                )
                if exec_collab_reward:
                    entry["collab_reward"] = (
                        float(entry.get("collab_reward", 0.0) or 0.0)
                        + exec_collab_reward
                    )
                    entry["collab_execution_reward"] = (
                        float(entry.get("collab_execution_reward", 0.0) or 0.0)
                        + exec_collab_reward
                    )
                    entry["collab_execution_source"] = exec_collab_meta
                    entry["total"] = (
                        float(entry.get("total", 0.0) or 0.0)
                        + exec_collab_reward
                    )

        return entry

    def _entry_can_receive_validated_action_reward(
        self, entry: Optional[Dict[str, Any]]
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        action = self._normalize_action(str(entry.get("action") or ""))
        if not action or self._is_wait_action(action) or self._is_collab_action(action):
            return False
        if entry.get("call_type") not in self.rewardable_action_call_types:
            return False
        penalties = entry.get("penalties") or []
        if any(item.get("type") in {"format", "validator"} for item in penalties):
            return False
        if float(entry.get("format_reward", 0.0) or 0.0) < 0.0:
            return False
        if float(entry.get("validator_reward", 0.0) or 0.0) < 0.0:
            return False
        return True

    @staticmethod
    def _suppress_validator_when_format_failed(entry: Dict[str, Any]) -> None:
        penalties = entry.get("penalties") or []
        has_format_penalty = any(item.get("type") == "format" for item in penalties)
        if not has_format_penalty:
            return
        validator_reward = float(entry.get("validator_reward", 0.0) or 0.0)
        if validator_reward:
            entry["validator_reward"] = 0.0
            entry["total"] = float(entry.get("total", 0.0) or 0.0) - validator_reward
        entry["penalties"] = [
            item for item in penalties if item.get("type") != "validator"
        ]

    def _suppress_negative_paired_comm_when_action_penalized(self, entry: Dict[str, Any]) -> None:
        """Do not assign paired response credit/penalty to a failed action call."""
        if not self._entry_has_action_penalty(entry):
            return
        paired_reward = float(entry.get("paired_comm_reward", 0.0) or 0.0)
        self._restore_paired_comm_pending_if_needed(entry)
        self._remove_registered_paired_comm_pending(entry)
        if not self._entry_has_paired_comm_metadata(entry):
            return
        entry["paired_comm_reward"] = 0.0
        if abs(paired_reward) > 1e-12:
            entry["paired_comm_reward_suppressed"] = paired_reward
        entry["paired_comm_suppressed_reason"] = "action_format_or_validator_penalty"
        entry["total"] = float(entry.get("total", 0.0) or 0.0) - paired_reward
        for key in (
            "paired_comm_role",
            "paired_comm_result",
            "paired_comm_target",
            "paired_comm_request_action",
            "paired_comm_request_helpful",
            "paired_comm_consumed_requests",
            "paired_comm_registered_requests",
            "paired_comm_pending_restored",
        ):
            entry.pop(key, None)

    @staticmethod
    def _entry_has_action_penalty(entry: Dict[str, Any]) -> bool:
        penalties = entry.get("penalties") or []
        has_action_penalty = any(
            item.get("type") in {"format", "validator"} for item in penalties
        )
        has_action_penalty = has_action_penalty or (
            float(entry.get("format_reward", 0.0) or 0.0) < 0.0
        )
        has_action_penalty = has_action_penalty or (
            float(entry.get("validator_reward", 0.0) or 0.0) < 0.0
        )
        return has_action_penalty

    def _restore_paired_comm_pending_if_needed(self, entry: Dict[str, Any]) -> None:
        if entry.get("paired_comm_pending_restored"):
            return
        consumed = entry.get("paired_comm_consumed_requests")
        if not consumed:
            return
        agent_index = entry.get("agent_index")
        if agent_index not in (0, 1):
            return
        restored = copy.deepcopy(consumed)
        for request in restored:
            request_id = request.get("request_id") if isinstance(request, dict) else None
            if request_id is not None:
                self.active_paired_comm_request_ids.add(request_id)
        self.pending_paired_comm_requests[int(agent_index)] = (
            restored + self.pending_paired_comm_requests[int(agent_index)]
        )
        entry["paired_comm_pending_restored"] = True

    def _remove_registered_paired_comm_pending(self, entry: Dict[str, Any]) -> None:
        registered = list(entry.get("paired_comm_registered_requests") or [])
        if not registered and entry.get("paired_comm_role") == "initiator":
            registered = [
                {
                    "timestamp": entry.get("source_timestamp", entry.get("timestamp")),
                    "initiator_agent": entry.get("agent_index"),
                    "responder_agent": entry.get("paired_comm_target"),
                    "request_action": entry.get("paired_comm_request_action"),
                    "request_helpful": entry.get("paired_comm_request_helpful"),
                }
            ]
        if not any(
            request.get("responder_agent") in (0, 1)
            and request.get("request_action")
            for request in registered
        ):
            registered = self._paired_comm_requests_from_entry_action(entry)
        for request in registered:
            responder = request.get("responder_agent")
            if responder not in (0, 1):
                continue
            queue = self.pending_paired_comm_requests[int(responder)]
            for idx, candidate in enumerate(queue):
                if self._paired_comm_pending_matches(candidate, request):
                    request_id = candidate.get("request_id")
                    if request_id is not None:
                        self.active_paired_comm_request_ids.discard(request_id)
                    del queue[idx]
                    break

    def _paired_comm_requests_from_entry_action(
        self, entry: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Recover pending-request keys from an old action when metadata is stale."""
        try:
            initiator = int(entry.get("agent_index"))
        except (TypeError, ValueError):
            return []
        timestamp = entry.get("source_timestamp", entry.get("timestamp"))
        requests = []
        for target_idx, actions in self._extract_collab_requests(
            str(entry.get("action") or "")
        ):
            if target_idx not in (0, 1) or target_idx == initiator or not actions:
                continue
            requests.append(
                {
                    "timestamp": timestamp,
                    "initiator_agent": initiator,
                    "responder_agent": target_idx,
                    "request_action": self._normalize_action(actions[0]),
                    # None deliberately matches either helpfulness value.
                    "request_helpful": None,
                }
            )
        return requests

    @staticmethod
    def _paired_comm_pending_matches(
        candidate: Dict[str, Any], request: Dict[str, Any]
    ) -> bool:
        expected_request_id = request.get("request_id")
        if expected_request_id is not None:
            return candidate.get("request_id") == expected_request_id
        for key in ("timestamp", "initiator_agent", "responder_agent"):
            expected = request.get(key)
            if expected is not None and candidate.get(key) != expected:
                return False
        expected_action = request.get("request_action")
        if expected_action and candidate.get("request_action") != expected_action:
            return False
        expected_helpful = request.get("request_helpful")
        if expected_helpful is not None and candidate.get("request_helpful") != expected_helpful:
            return False
        return True

    def _next_paired_comm_request_id(self) -> int:
        self.paired_comm_request_counter += 1
        return self.paired_comm_request_counter

    def _pending_paired_comm_request_is_active(self, request: Dict[str, Any]) -> bool:
        request_id = request.get("request_id")
        if request_id is None:
            return True
        return request_id in self.active_paired_comm_request_ids

    def _source_entry_can_be_updated(
        self,
        entry: Dict[str, Any],
        action_text: Optional[str],
        call_type: Optional[str] = None,
    ) -> bool:
        existing_action = self._normalize_action(str(entry.get("action") or ""))
        incoming_action = self._normalize_action(str(action_text or ""))
        if not existing_action or existing_action == "[EMPTY]":
            return True
        if self._is_wait_action(incoming_action):
            return True
        if not incoming_action or incoming_action == "[EMPTY]":
            return False
        if existing_action == incoming_action:
            return True
        if self._entry_has_paired_comm_metadata(entry):
            return False
        return (
            entry.get("call_type") == "communication"
            and call_type in self.rewardable_action_call_types
        )

    @staticmethod
    def _entry_has_paired_comm_metadata(entry: Dict[str, Any]) -> bool:
        if abs(float(entry.get("paired_comm_reward", 0.0) or 0.0)) > 1e-12:
            return True
        for key in (
            "paired_comm_role",
            "paired_comm_result",
            "paired_comm_target",
            "paired_comm_request_action",
            "paired_comm_request_helpful",
            "paired_comm_consumed_requests",
            "paired_comm_registered_requests",
            "paired_comm_reward_suppressed",
        ):
            if entry.get(key):
                return True
        return False

    def _close_paired_comm_requests(self, requests: List[Dict[str, Any]]) -> None:
        for request in requests:
            request_id = request.get("request_id") if isinstance(request, dict) else None
            if request_id is not None:
                self.active_paired_comm_request_ids.discard(request_id)

    def _clear_paired_comm_from_entry(self, entry: Dict[str, Any]) -> None:
        paired_reward = float(entry.get("paired_comm_reward", 0.0) or 0.0)
        if abs(paired_reward) > 1e-12:
            entry["total"] = float(entry.get("total", 0.0) or 0.0) - paired_reward
        if entry.get("paired_comm_role") == "responder":
            self._restore_paired_comm_pending_if_needed(entry)
        self._remove_registered_paired_comm_pending(entry)
        for key in (
            "paired_comm_reward",
            "paired_comm_role",
            "paired_comm_result",
            "paired_comm_target",
            "paired_comm_request_action",
            "paired_comm_request_helpful",
            "paired_comm_consumed_requests",
            "paired_comm_registered_requests",
            "paired_comm_reward_suppressed",
            "paired_comm_suppressed_reason",
            "paired_comm_pending_restored",
        ):
            if key == "paired_comm_reward":
                entry[key] = 0.0
            else:
                entry.pop(key, None)

    def _suppress_paired_comm_for_wait_action(self, entry: Dict[str, Any]) -> None:
        action = self._normalize_action(str(entry.get("action") or ""))
        if not self._is_wait_action(action):
            return
        self._clear_paired_comm_from_entry(entry)

    def _apply_paired_comm_meta_to_entry(
        self, entry: Dict[str, Any], reward: float, meta: Dict[str, Any]
    ) -> None:
        if not meta:
            return
        entry["paired_comm_reward"] = float(reward)
        entry["paired_comm_role"] = meta.get("role")
        entry["paired_comm_result"] = meta.get("result")
        entry["paired_comm_target"] = meta.get("target_agent")
        entry["paired_comm_request_action"] = meta.get("request_action")
        entry["paired_comm_request_helpful"] = meta.get("request_helpful")
        entry["paired_comm_consumed_requests"] = meta.get("consumed_requests")
        entry["paired_comm_registered_requests"] = meta.get("registered_requests")
        entry["total"] = float(entry.get("total", 0.0) or 0.0) + float(reward)

    def normalize_reward_entry(self, entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(entry, dict):
            return {}
        self._suppress_validator_when_format_failed(entry)
        self._suppress_negative_paired_comm_when_action_penalized(entry)
        self._suppress_paired_comm_for_wait_action(entry)
        return entry

    def ensure_llm_action_entry(
        self,
        agent_index: int,
        timestamp: Optional[int],
        action_text: Optional[str],
        *,
        agent_name: Optional[str] = None,
        call_index: Optional[int] = None,
        call_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Ensure a queued/executed action source has a reward entry.

        Action sources may be created before the normal reward-event flush path,
        especially when a communication turn is reclassified as an embodied
        planner action and queued for a later environment step. The executed
        source must still point to an exact reward entry for strict assignment.
        """
        if agent_index is None or call_index is None:
            return {}
        ts = -1 if timestamp is None else int(timestamp)
        key = (int(agent_index), ts, int(call_index))
        existing = self.call_records_by_source.get(key)
        if existing is None or not self._source_entry_can_be_updated(
            existing, action_text, call_type
        ):
            return self.register_llm_action(
                agent_index=agent_index,
                timestamp=timestamp,
                action_text=action_text,
                agent_name=agent_name,
                call_index=call_index,
                call_type=call_type,
                metadata=metadata,
            )
        return self.normalize_reward_entry(
            self._update_llm_action_entry(
                existing,
                action_text,
                agent_name=agent_name,
                call_type=call_type,
            )
        )

    def register_or_update_llm_action(
        self,
        agent_index: int,
        timestamp: Optional[int],
        action_text: Optional[str],
        *,
        agent_name: Optional[str] = None,
        call_index: Optional[int] = None,
        call_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Register a call, or merge into the existing source entry once.

        This prevents duplicate call records when an action source was already
        materialized for future execution before the reward event is flushed.
        Pending penalties are attached only when they belong to this concrete
        action. Parser failures from an earlier retry are consumed here but not
        copied onto a later valid retry with the same source key.
        """
        if agent_index is None or call_index is None:
            return self.register_llm_action(
                agent_index=agent_index,
                timestamp=timestamp,
                action_text=action_text,
                agent_name=agent_name,
                call_index=call_index,
                call_type=call_type,
                metadata=metadata,
            )
        ts = -1 if timestamp is None else int(timestamp)
        key = (int(agent_index), ts, int(call_index))
        existing = self.call_records_by_source.get(key)
        if existing is None or not self._source_entry_can_be_updated(
            existing, action_text, call_type
        ):
            return self.register_llm_action(
                agent_index=agent_index,
                timestamp=timestamp,
                action_text=action_text,
                agent_name=agent_name,
                call_index=call_index,
                call_type=call_type,
                metadata=metadata,
            )

        previous_action_norm = self._normalize_action(str(existing.get("action") or ""))
        incoming_action_norm = self._normalize_action(str(action_text or ""))
        updated = self._update_llm_action_entry(
            existing,
            action_text,
            agent_name=agent_name,
            call_type=call_type,
        )
        _penalty_total, penalty_details = self._consume_penalties(agent_index)
        if penalty_details:
            penalties = updated.setdefault("penalties", [])
            had_format_penalty = any(
                entry.get("type") == "format" for entry in penalties
            ) or float(updated.get("format_reward", 0.0) or 0.0) < 0.0
            had_validator_penalty = any(
                entry.get("type") == "validator" for entry in penalties
            ) or float(updated.get("validator_reward", 0.0) or 0.0) < 0.0
            has_pending_format_penalty = any(
                entry.get("type") == "format" for entry in penalty_details
            )
            action_changed = previous_action_norm != incoming_action_norm
            valid_retry_replaces_format_failure = (
                has_pending_format_penalty
                and action_changed
                and not had_format_penalty
                and incoming_action_norm
                and incoming_action_norm != "[EMPTY]"
                and call_type in self.rewardable_action_call_types
                and not self._is_wait_action(incoming_action_norm)
                and not self._is_collab_action(incoming_action_norm)
            )
            format_delta = (
                0.0
                if valid_retry_replaces_format_failure
                else sum(
                    entry["value"]
                    for entry in penalty_details
                    if entry["type"] == "format"
                )
            )
            has_format_penalty = format_delta != 0.0 or had_format_penalty
            if had_format_penalty:
                format_delta = 0.0
            validator_delta = (
                0.0
                if has_format_penalty or had_validator_penalty
                else sum(
                    entry["value"]
                    for entry in penalty_details
                    if entry["type"] == "validator"
                )
            )
            attach_penalties = []
            for penalty in penalty_details:
                penalty_type = penalty.get("type")
                if valid_retry_replaces_format_failure and penalty_type == "format":
                    continue
                if penalty_type == "format" and had_format_penalty:
                    continue
                if penalty_type == "validator" and (
                    has_format_penalty or had_validator_penalty
                ):
                    continue
                attach_penalties.append(penalty)
            penalties.extend(attach_penalties)
            if has_format_penalty:
                existing_validator = float(updated.get("validator_reward", 0.0) or 0.0)
                if existing_validator > 0.0:
                    validator_delta -= existing_validator
            updated["format_reward"] = (
                float(updated.get("format_reward", 0.0) or 0.0) + format_delta
            )
            updated["validator_reward"] = (
                float(updated.get("validator_reward", 0.0) or 0.0)
                + validator_delta
            )
            updated["total"] = (
                float(updated.get("total", 0.0) or 0.0)
                + format_delta
                + validator_delta
            )
        self._suppress_validator_when_format_failed(updated)
        self._suppress_negative_paired_comm_when_action_penalized(updated)
        return updated

    def _update_llm_action_entry(
        self,
        entry: Dict[str, Any],
        action_text: Optional[str],
        *,
        agent_name: Optional[str] = None,
        call_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_action = (action_text or "").strip() or "[EMPTY]"
        previous_action = str(entry.get("action") or "")
        action_changed = (
            self._normalize_action(previous_action)
            != self._normalize_action(normalized_action)
        )
        if action_changed:
            self._clear_paired_comm_from_entry(entry)
        previous_rewardable = (
            entry.get("call_type") in self.rewardable_action_call_types
            and not entry.get("is_collab")
        )
        entry["action"] = normalized_action
        if agent_name:
            entry["agent"] = agent_name
        if call_type is not None:
            entry["call_type"] = call_type
        entry["is_collab"] = self._is_collab_action(normalized_action)
        now_rewardable = (
            entry.get("call_type") in self.rewardable_action_call_types
            and not entry.get("is_collab")
            and normalized_action != "[EMPTY]"
            and not self._is_wait_action(normalized_action)
        )
        has_format_penalty = any(
            item.get("type") == "format" for item in (entry.get("penalties") or [])
        )
        has_format_credit = float(entry.get("format_reward", 0.0) or 0.0) > 0.0
        if (
            now_rewardable
            and not previous_rewardable
            and not has_format_penalty
            and not has_format_credit
        ):
            entry["format_reward"] = (
                float(entry.get("format_reward", 0.0) or 0.0)
                + self.format_success_reward_value
            )
            entry["total"] = (
                float(entry.get("total", 0.0) or 0.0)
                + self.format_success_reward_value
            )
        if action_changed and not self._entry_has_action_penalty(entry):
            paired_reward, paired_meta = self._process_paired_comm_reward(
                int(entry.get("agent_index", 0)),
                int(entry.get("source_timestamp", entry.get("timestamp", -1))),
                normalized_action,
                action_penalized=False,
            )
            self._apply_paired_comm_meta_to_entry(entry, paired_reward, paired_meta)
        self._suppress_paired_comm_for_wait_action(entry)
        return entry

    def after_step(
        self,
        timestep: int,
        ml_actions: Optional[List[str]],
        state,
        executed_action_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict:
        """
        Aggregate rewards recorded during a control step.

        Process/ITES reward is granted when the validator accepts the LLM action,
        with executed action sources used as a delayed-execution fallback.
        """
        if ml_actions is None:
            ml_actions = [None, None]
        self._mark_executed_action_sources_validated(
            executed_action_sources,
            ml_actions,
        )

        per_agent = []
        team_total = 0.0

        bucket = self.step_call_records.pop(timestep, [[], []])
        source_entries: List[Dict[str, Any]] = []
        source_entry_ids = set()
        for agent_idx in range(2):
            bucket_entries = bucket[agent_idx]
            call_entries = [self.normalize_reward_entry(entry) for entry in bucket_entries]
            for entry in call_entries:
                entry_id = id(entry)
                if entry_id in source_entry_ids:
                    continue
                source_entry_ids.add(entry_id)
                source_entries.append(entry)
            seq_reward = sum(
                entry.get("sequence_reward", 0.0) for entry in call_entries
            )
            communication_reward = sum(
                entry.get("communication_reward", 0.0) for entry in call_entries
            )
            repeat_communication_reward = sum(
                entry.get("repeat_communication_reward", 0.0)
                for entry in call_entries
            )
            forced_communication_reward = sum(
                entry.get("forced_communication_reward", 0.0)
                for entry in call_entries
            )
            paired_comm_reward = sum(
                entry.get("paired_comm_reward", 0.0) for entry in call_entries
            )
            collab_reward = sum(
                entry.get("collab_reward", 0.0) for entry in call_entries
            )
            format_reward = sum(
                entry.get("format_reward", 0.0) for entry in call_entries
            )
            validator_reward = sum(
                entry.get("validator_reward", 0.0) for entry in call_entries
            )
            agent_total = sum(entry["total"] for entry in call_entries)
            penalty_total = format_reward + validator_reward
            penalty_details = []
            for entry in call_entries:
                penalty_details.extend(entry["penalties"])
            per_agent.append(
                {
                    "sequence_reward": seq_reward,
                    "communication_reward": communication_reward,
                    "repeat_communication_reward": repeat_communication_reward,
                    "forced_communication_reward": forced_communication_reward,
                    "collab_reward": collab_reward,
                    "paired_comm_reward": paired_comm_reward,
                    "format_reward": format_reward,
                    "validator_reward": validator_reward,
                    "penalty_total": penalty_total,
                    "penalties": penalty_details,
                    "similarity": self.sequence_scores[agent_idx],
                    "total": agent_total,
                    "call_count": len(call_entries),
                    "calls": call_entries,
                }
            )
            team_total += agent_total

        intermediate_reward, new_items = self._process_intermediate_reward(state)
        team_total += intermediate_reward

        for agent_data in per_agent:
            agent_data["total"] += intermediate_reward / 2.0

        for entry in self.call_records_by_source.values():
            entry_id = id(entry)
            if entry_id in source_entry_ids:
                continue
            source_entry_ids.add(entry_id)
            source_entries.append(self.normalize_reward_entry(entry))

        reward_info = {
            "timestamp": timestep,
            "per_agent": per_agent,
            "source_entries": source_entries,
            "intermediate": {
                "reward": intermediate_reward,
                "items": new_items,
            },
            "team_total": team_total,
        }
        return reward_info

    def _mark_executed_action_sources_validated(
        self,
        executed_action_sources: Optional[List[Dict[str, Any]]],
        ml_actions: Optional[List[str]],
    ) -> None:
        for source in executed_action_sources or []:
            if not isinstance(source, dict):
                continue
            try:
                agent_idx = int(source.get("agent_index"))
                source_ts = int(source.get("source_timestamp", source.get("timestamp")))
                call_index = int(source.get("call_index"))
            except (TypeError, ValueError):
                continue
            if agent_idx not in (0, 1):
                continue
            action_text = (
                source.get("submitted_action")
                or source.get("action")
                or (
                    ml_actions[agent_idx]
                    if ml_actions is not None and agent_idx < len(ml_actions)
                    else None
                )
            )
            normalized_action = self._normalize_action(str(action_text or ""))
            if (
                not normalized_action
                or self._is_wait_action(normalized_action)
                or self._is_collab_action(normalized_action)
            ):
                continue
            if ml_actions is not None and agent_idx < len(ml_actions):
                env_action = self._normalize_action(str(ml_actions[agent_idx] or ""))
                if env_action and env_action != normalized_action:
                    continue
            source_call_type = source.get("call_type") or "planner_main"
            entry = self._executed_source_target_entry(
                agent_idx,
                source_ts,
                call_index,
                normalized_action,
            )
            if not isinstance(entry, dict):
                entry = self.ensure_llm_action_entry(
                    agent_index=agent_idx,
                    timestamp=source_ts,
                    action_text=action_text,
                    agent_name=source.get("agent"),
                    call_index=call_index,
                    call_type=source_call_type,
                )
            elif source_call_type in self.rewardable_action_call_types:
                self._update_llm_action_entry(
                    entry,
                    action_text,
                    agent_name=source.get("agent"),
                    call_type=source_call_type,
                )
            self.mark_llm_action_entry_validated(entry, action_text=action_text)

    def _executed_source_target_entry(
        self,
        agent_idx: int,
        source_ts: int,
        call_index: int,
        normalized_action: str,
    ) -> Optional[Dict[str, Any]]:
        exact = self.call_records_by_source.get((agent_idx, source_ts, call_index))
        if self._entry_matches_executed_action(exact, normalized_action):
            if self._entry_can_receive_validated_action_reward(exact):
                return exact
        candidates: List[Tuple[int, Dict[str, Any]]] = []
        for (entry_agent, entry_ts, entry_call_index), entry in self.call_records_by_source.items():
            if entry_agent != agent_idx or entry_ts != source_ts:
                continue
            if not self._entry_matches_executed_action(entry, normalized_action):
                continue
            if not self._entry_can_receive_validated_action_reward(entry):
                continue
            candidates.append((entry_call_index, entry))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[-1][1]
        if self._entry_matches_executed_action(exact, normalized_action):
            return exact
        return None

    def _entry_matches_executed_action(
        self,
        entry: Optional[Dict[str, Any]],
        normalized_action: str,
    ) -> bool:
        if not isinstance(entry, dict):
            return False
        entry_action = self._normalize_action(str(entry.get("action") or ""))
        return bool(entry_action and entry_action == normalized_action)

    def _validator_success_target_entry(
        self,
        call_entries: List[Dict],
        executed_action: Optional[str],
        executed_source: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """Attach validator success only to the planner call that actually executed."""
        normalized_executed = self._normalize_action(str(executed_action or ""))
        if not normalized_executed or normalized_executed.lower().startswith("wait"):
            return None
        if not executed_source or executed_source.get("call_index") is None:
            return None
        source_call_index = int(executed_source["call_index"])
        for entry in call_entries:
            if entry.get("call_index") is None:
                continue
            if int(entry.get("call_index")) != source_call_index:
                continue
            if self._normalize_action(str(entry.get("action") or "")) != normalized_executed:
                return None
            if (
                entry.get("call_type") in self.rewardable_action_call_types
                and not entry.get("is_collab")
            ):
                return entry
            return None
        return None

    # ------------------------------------------------------------------
    # Sequence reward helpers
    # ------------------------------------------------------------------
    def _process_sequence_reward(
        self, agent_index: int, action: Optional[str]
    ) -> Tuple[float, float, float, float]:
        before_score = self.sequence_scores[agent_index]
        if not action:
            return 0.0, before_score, before_score, 0.0
        normalized = action.strip()
        if not normalized or normalized.lower().startswith("wait"):
            return 0.0, before_score, before_score, 0.0

        self.sequence_histories[agent_index].append(normalized)
        candidate_score = self._best_sequence_score(agent_index)
        delta = max(0.0, candidate_score - before_score)
        if delta > 0:
            self.sequence_scores[agent_index] = candidate_score
            self.collab_sequence_scores[agent_index] = max(
                self.collab_sequence_scores[agent_index], candidate_score
            )
            return (
                self.sequence_progress_reward,
                before_score,
                self.sequence_scores[agent_index],
                delta,
            )
        return 0.0, before_score, self.sequence_scores[agent_index], 0.0

    def _best_sequence_score(self, agent_index: int) -> float:
        history = self.sequence_histories[agent_index]
        refs = self.references.get(agent_index, [])
        if not refs:
            return 0.0
        if self.sequence_metric == "lcs":
            scores = [self._lcs_ratio(history, ref) for ref in refs]
        else:
            scores = [self._tes_score(history, ref) for ref in refs]
        return max(scores)

    def _best_sequence_score_from_history(self, agent_index: int, history: List[str]) -> float:
        refs = self.references.get(agent_index, [])
        if not refs or not history:
            return 0.0
        if self.sequence_metric == "lcs":
            scores = [self._lcs_ratio(history, ref) for ref in refs]
        else:
            scores = [self._tes_score(history, ref) for ref in refs]
        return max(scores) if scores else 0.0

    def _process_collab_reward(self, agent_index: int, action: str) -> float:
        requests = self._extract_collab_requests(action)
        if not requests:
            return 0.0
        reward = 0.0
        for target_idx, actions in requests:
            if target_idx == agent_index or target_idx not in (0, 1):
                continue
            history = list(self.sequence_histories[target_idx])
            history.extend(actions)
            new_score = self._best_sequence_score_from_history(target_idx, history)
            baseline = self.collab_sequence_scores[target_idx]
            delta = max(0.0, new_score - baseline)
            if delta > 0:
                self.collab_sequence_scores[target_idx] = new_score
                reward += delta * self.sequence_weight
                self.pending_collab_execution_requests[target_idx].append(
                    {
                        "initiator_agent": agent_index,
                        "target_agent": target_idx,
                        "action": self._normalize_action(actions[0]),
                        "score_delta": delta,
                        "request_reward": delta * self.sequence_weight,
                    }
                )
        return reward

    def _process_collab_execution_reward(
        self,
        agent_index: int,
        executed_action: Optional[str],
        *,
        sequence_reward: float,
    ) -> Tuple[float, Dict[str, Any]]:
        if not self.enable_collab_reward or sequence_reward <= 0.0:
            return 0.0, {}
        normalized = self._normalize_action(str(executed_action or ""))
        if not normalized or self._is_wait_action(normalized):
            return 0.0, {}
        queue = self.pending_collab_execution_requests[agent_index]
        for idx, request in enumerate(list(queue)):
            if self._normalize_action(str(request.get("action") or "")) != normalized:
                continue
            matched = queue.pop(idx)
            reward = float(matched.get("request_reward", 0.0) or 0.0)
            return reward, {
                "role": "executor",
                "initiator_agent": matched.get("initiator_agent"),
                "target_agent": matched.get("target_agent"),
                "action": normalized,
                "score_delta": matched.get("score_delta"),
            }
        return 0.0, {}

    def _process_communication_reward(
        self, agent_index: int, action: str, *, suppress_penalty: bool = False
    ) -> float:
        signature = self._build_action_signature(action)
        if not signature:
            return 0.0
        previous = self.last_action_signatures[agent_index]
        self.last_action_signatures[agent_index] = signature
        if suppress_penalty:
            return 0.0
        if previous and previous == signature:
            return self.communication_penalty_value
        return 0.0

    def _process_paired_comm_reward(
        self,
        agent_index: int,
        ts: int,
        action: str,
        *,
        action_penalized: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        if not self.enable_paired_comm_reward or not action:
            return 0.0, {}
        if self._is_wait_action(self._normalize_action(action)):
            return 0.0, {}

        response_reward, response_meta = self._resolve_paired_comm_response(
            agent_index,
            action,
            action_penalized=action_penalized,
        )
        if response_meta:
            return response_reward, response_meta

        if action_penalized:
            return 0.0, {}

        request_reward, request_meta = self._register_paired_comm_requests(
            agent_index, ts, action
        )
        if request_meta:
            return request_reward, request_meta

        return 0.0, {}

    def _resolve_paired_comm_response(
        self,
        agent_index: int,
        action: str,
        *,
        action_penalized: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        pending_queue = self.pending_paired_comm_requests[agent_index]
        if pending_queue:
            pending_queue[:] = [
                request
                for request in pending_queue
                if self._pending_paired_comm_request_is_active(request)
            ]
        if not pending_queue:
            return 0.0, {}

        response_kind, response_payload = self._classify_paired_comm_response(action)
        if response_kind is None:
            return 0.0, {}

        pair_idx = self._select_paired_comm_response_pair(
            pending_queue,
            response_kind=response_kind,
            response_payload=response_payload,
        )
        if pair_idx is None:
            return 0.0, {}
        pair = pending_queue[pair_idx]
        request_helpful = bool(pair.get("request_helpful", False))
        request_action = str(pair.get("request_action") or "")
        target_agent = pair.get("initiator_agent")
        consumed_pairs = copy.deepcopy(pending_queue[: pair_idx + 1])
        consumed_live_pairs = list(pending_queue[: pair_idx + 1])

        if action_penalized:
            reward = self.paired_comm_response_negative_reward
            result = (
                "helpful_request_rejected_or_missed"
                if request_helpful
                else "bad_request_followed"
            )
        elif request_helpful:
            self._close_paired_comm_requests(consumed_live_pairs)
            del pending_queue[: pair_idx + 1]
            accepted = response_kind == "ack" or (
                response_kind == "embodied"
                and response_payload == self._normalize_action(request_action)
            )
            reward = (
                self.paired_comm_response_positive_reward
                if accepted
                else self.paired_comm_response_negative_reward
            )
            result = (
                "helpful_request_accepted"
                if accepted
                else "helpful_request_rejected_or_missed"
            )
        else:
            self._close_paired_comm_requests(consumed_live_pairs)
            del pending_queue[: pair_idx + 1]
            denied = response_kind == "deny"
            followed = response_kind in {"ack", "embodied"}
            reward = self.paired_comm_deny_reward if denied else 0.0
            result = (
                "bad_request_denied"
                if denied
                else "bad_request_followed_ignored"
                if followed
                else "bad_request_ignored"
            )

        return reward, {
            "role": "responder",
            "result": result,
            "target_agent": target_agent,
            "request_action": request_action,
            "request_helpful": request_helpful,
            "consumed_requests": (
                consumed_pairs
                if not action_penalized
                else []
            ),
        }

    def _select_paired_comm_response_pair(
        self,
        pending_queue: List[Dict[str, Any]],
        *,
        response_kind: str,
        response_payload: Optional[str],
    ) -> Optional[int]:
        """Prefer the pending request that the current response actually satisfies."""
        if not pending_queue:
            return 0

        if response_kind == "ack":
            target_idx = self._paired_comm_response_target_agent(response_payload)
            for idx, pair in enumerate(pending_queue):
                if target_idx is not None and pair.get("initiator_agent") != target_idx:
                    continue
                if bool(pair.get("request_helpful", False)):
                    return idx
            return self._fallback_paired_comm_response_pair(
                pending_queue, target_idx=target_idx
            )

        if response_kind == "deny":
            target_idx = self._paired_comm_response_target_agent(response_payload)
            for idx, pair in enumerate(pending_queue):
                if target_idx is not None and pair.get("initiator_agent") != target_idx:
                    continue
                if not bool(pair.get("request_helpful", False)):
                    return idx
            return self._fallback_paired_comm_response_pair(
                pending_queue, target_idx=target_idx
            )

        if response_kind == "embodied":
            normalized_payload = self._normalize_action(response_payload or "")
            for idx, pair in enumerate(pending_queue):
                request_action = self._normalize_action(
                    str(pair.get("request_action") or "")
                )
                if request_action == normalized_payload:
                    return idx
            return None

        return 0

    def _fallback_paired_comm_response_pair(
        self,
        pending_queue: List[Dict[str, Any]],
        *,
        target_idx: Optional[int],
    ) -> Optional[int]:
        if target_idx is None:
            return 0
        for idx, pair in enumerate(pending_queue):
            if pair.get("initiator_agent") == target_idx:
                return idx
        return None

    def _paired_comm_response_target_agent(
        self, response_payload: Optional[str]
    ) -> Optional[int]:
        target_raw, _payload_raw = self._split_first_argument(
            (response_payload or "").replace("|", ",", 1)
        )
        return self._agent_index_from_label(target_raw)

    def _register_paired_comm_requests(
        self, agent_index: int, ts: int, action: str
    ) -> Tuple[float, Dict[str, Any]]:
        requests = self._extract_collab_requests(action)
        if not requests:
            return 0.0, {}

        total_reward = 0.0
        request_results: List[str] = []
        registered_requests: List[Dict[str, Any]] = []
        target_agent: Optional[int] = None
        request_action: Optional[str] = None
        request_helpful: Optional[bool] = None

        for target_idx, actions in requests:
            if target_idx == agent_index or target_idx not in (0, 1) or not actions:
                continue
            action_text = self._normalize_action(actions[0])
            helpful, baseline_score, new_score = self._evaluate_request_helpfulness(
                target_idx, actions
            )
            reward = (
                self.paired_comm_request_positive_reward
                if helpful
                else self.paired_comm_request_negative_reward
            )
            total_reward += reward
            request_results.append(
                "request_helpful" if helpful else "request_not_helpful"
            )
            request_id = self._next_paired_comm_request_id()
            request_entry = {
                "request_id": request_id,
                "timestamp": ts,
                "initiator_agent": agent_index,
                "responder_agent": target_idx,
                "request_action": action_text,
                "request_helpful": helpful,
                "baseline_score": baseline_score,
                "new_score": new_score,
            }
            self.active_paired_comm_request_ids.add(request_id)
            self.pending_paired_comm_requests[target_idx].append(request_entry)
            registered_requests.append(
                copy.deepcopy(self.pending_paired_comm_requests[target_idx][-1])
            )
            target_agent = target_idx
            request_action = action_text
            request_helpful = helpful

        if not request_results:
            return 0.0, {}

        result = request_results[0] if len(request_results) == 1 else "multi_request"
        return total_reward, {
            "role": "initiator",
            "result": result,
            "target_agent": target_agent,
            "request_action": request_action,
            "request_helpful": request_helpful,
            "registered_requests": registered_requests,
        }

    def _evaluate_request_helpfulness(
        self, target_idx: int, actions: List[str]
    ) -> Tuple[bool, float, float]:
        history = list(self.sequence_histories[target_idx])
        history.extend(self._normalize_action(action) for action in actions if action)
        new_score = self._best_sequence_score_from_history(target_idx, history)
        baseline = max(
            float(self.sequence_scores[target_idx]),
            float(self.collab_sequence_scores[target_idx]),
        )
        helpful = new_score > baseline
        if helpful:
            self.collab_sequence_scores[target_idx] = max(
                self.collab_sequence_scores[target_idx], new_score
            )
        return helpful, baseline, new_score

    def _classify_paired_comm_response(
        self, action: str
    ) -> Tuple[Optional[str], Optional[str]]:
        normalized = self._normalize_action(action or "")
        if not normalized or normalized.lower().startswith("wait"):
            return None, None
        if not self._is_collab_action(normalized):
            return "embodied", normalized

        body = self._unwrap_function_body(normalized)
        collab_body = (
            body if normalized.lower().startswith("collab(") and body is not None else normalized
        )
        segments = self._split_top_level_segments(collab_body)
        if not segments:
            return None, None
        for segment in segments:
            primitive, payload = self._parse_collab_signature_segment(segment)
            if primitive in {"ack", "deny"}:
                return primitive, payload
        return None, None

    def _build_action_signature(self, action: str) -> str:
        normalized = self._normalize_action(action)
        if not normalized:
            return ""
        if not self._is_collab_action(normalized):
            return f"embodied:{normalized}"

        body = self._unwrap_function_body(normalized)
        collab_body = body if normalized.lower().startswith("collab(") and body is not None else normalized
        segments = self._split_top_level_segments(collab_body)
        if not segments:
            segments = [collab_body]

        parts: List[str] = []
        for segment in segments:
            primitive, payload = self._parse_collab_signature_segment(segment)
            if primitive:
                parts.append(f"{primitive}:{payload}")
        if not parts:
            return f"collab:{normalized}"
        return "collab:" + ";".join(parts)

    def _parse_collab_signature_segment(self, text: str) -> Tuple[str, str]:
        stripped = (text or "").strip()
        if not stripped:
            return "", ""
        lowered = stripped.lower()
        for primitive in ("request", "seek", "ack", "deny"):
            prefix = primitive + "("
            if lowered.startswith(prefix):
                body = self._unwrap_function_body(stripped)
                if body is None:
                    return primitive, stripped
                target_raw, payload_raw = self._split_first_argument(body)
                target_norm = self._normalize_action(target_raw)
                payload_norm = self._normalize_action(payload_raw)
                return primitive, f"{target_norm}|{payload_norm}"
        return "raw", self._normalize_action(stripped)

    def _lcs_ratio(self, seq_a: List[str], seq_b: List[str]) -> float:
        if not seq_a or not seq_b:
            return 0.0
        len_a, len_b = len(seq_a), len(seq_b)
        dp = [0] * (len_b + 1)
        for i in range(1, len_a + 1):
            prev = 0
            for j in range(1, len_b + 1):
                temp = dp[j]
                if self._normalize_action(seq_a[i - 1]) == self._normalize_action(seq_b[j - 1]):
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        lcs_len = dp[-1]
        return lcs_len / len_b if len_b else 0.0

    def _tes_score(self, history: List[str], reference: List[str]) -> float:
        """Similarity using the TES metric from evaluation utils (F1-style)."""
        if not history or not reference:
            return 0.0

        normalized_history = [self._normalize_action(a) for a in history if a]
        normalized_reference = [self._normalize_action(a) for a in reference if a]
        if not normalized_history or not normalized_reference:
            return 0.0

        element_positions: Dict[str, List[int]] = defaultdict(list)
        for idx, action in enumerate(normalized_history):
            element_positions[action].append(idx)

        start_positions = element_positions.get(normalized_reference[0], [])
        if not start_positions:
            return 0.0

        max_depth = 0
        for start in start_positions:
            pos_action = start
            depth = 1
            for ref_action in normalized_reference[1:]:
                positions = element_positions.get(ref_action)
                if not positions:
                    break
                next_idx = bisect_right(positions, pos_action)
                if next_idx >= len(positions):
                    break
                pos_action = positions[next_idx]
                depth += 1
            if depth > max_depth:
                max_depth = depth

        beta = float(self.settings.get("tes_beta", 0.95))
        beta_sq = beta * beta
        denominator = len(normalized_reference) + beta_sq * len(normalized_history)
        if denominator <= 0:
            return 0.0
        numerator = (1 + beta_sq) * max_depth
        return numerator / denominator

    # ------------------------------------------------------------------
    # Intermediate product reward
    # ------------------------------------------------------------------
    def _process_intermediate_reward(self, state) -> Tuple[float, List[str]]:
        if not self.intermediate_targets:
            return 0.0, []

        present_items = self._current_recipe_items(state)
        new_items = sorted(
            item for item in present_items if item in self.intermediate_targets and item not in self.observed_targets
        )
        if not new_items:
            return 0.0, []

        for item in new_items:
            self.observed_targets.add(item)
        reward = len(new_items) * self.product_reward_value
        return reward, new_items

    def _current_recipe_items(self, state) -> List[str]:
        items = []
        for obj in state.objects.values():
            items.extend(self._extract_object_products(obj))
        for player in state.players:
            if player.has_object():
                items.extend(self._extract_object_products(player.get_object()))
        return items

    def _extract_object_products(self, obj) -> List[str]:
        products = []
        if obj.name == "soup" and obj.state:
            product = obj.state[0]
            if isinstance(product, str):
                products.append(product)
            elif isinstance(product, (list, tuple)):
                products.extend(product)
        else:
            products.append(obj.name)
        return products

    # ------------------------------------------------------------------
    # Penalty helpers
    # ------------------------------------------------------------------
    def _consume_penalties(self, agent_index: int) -> Tuple[float, List[Dict[str, str]]]:
        events = self.penalty_queue[agent_index]
        total = 0.0
        details = []
        seen = set()
        seen_types = set()
        has_format_event = any(entry.get("type") == "format" for entry in events)
        for entry in events:
            entry_type = entry.get("type")
            if has_format_event and entry_type == "validator":
                continue
            if entry_type in seen_types:
                continue
            key = (entry_type, entry.get("detail"))
            if key in seen:
                continue
            seen.add(key)
            seen_types.add(entry_type)
            if entry_type == "format":
                value = self.format_penalty_value
            else:
                value = self.validator_penalty_value
            total += value
            details.append({"type": entry["type"], "detail": entry["detail"], "value": value})
        events.clear()
        return total, details

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------
    def _load_references(self) -> Dict[int, List[List[str]]]:
        override = self.settings.get("reference_overrides")
        if isinstance(override, dict):
            references: Dict[int, List[List[str]]] = {0: [], 1: []}
            for agent_key in ("agent_0", "agent_1", 0, 1, "0", "1"):
                sequence = override.get(agent_key)
                if not sequence:
                    continue
                agent_idx = 0 if str(agent_key).endswith("0") else 1
                if (
                    isinstance(sequence, list)
                    and sequence
                    and all(isinstance(item, str) for item in sequence)
                ):
                    sequence = [sequence]
                for ref_seq in sequence:
                    if not isinstance(ref_seq, list):
                        continue
                    cleaned = [
                        self._normalize_action(str(action or ""))
                        for action in ref_seq
                        if self._normalize_action(str(action or ""))
                    ]
                    if cleaned:
                        references[agent_idx].append(cleaned)
            if any(references.values()):
                return references

        pattern = f"*_{self.order}_ref.*"
        candidates = sorted(self.reference_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No reference file found for order '{self.order}' under {self.reference_dir}")

        with candidates[0].open("r") as f:
            content = json.load(f)

        references: Dict[int, List[List[str]]] = {0: [], 1: []}
        for ref_entry in content.values():
            for agent_key in ("agent_0", "agent_1"):
                sequence = ref_entry.get(agent_key)
                if sequence:
                    agent_idx = 0 if agent_key.endswith("0") else 1
                    cleaned = [action.strip() for action in sequence if action.strip()]
                    references[agent_idx].append(cleaned)
        return references

    def _build_recipe_lookup(self) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        recipes = self.mdp.recipe_config.get("recipes", {})
        for section in recipes.values():
            for product, detail in section.items():
                inputs = detail.get("recipe", [])
                mapping[product] = inputs
        return mapping

    def _resolve_recipe_targets(self, target: str) -> set:
        mapping = self.recipe_lookup
        default_ingredients = set(self.mdp.default_ingredients)

        visited = set()
        intermediates = set()
        stack = [target]

        while stack:
            item = stack.pop()
            if item in visited:
                continue
            visited.add(item)
            inputs = mapping.get(item, [])
            for ing in inputs:
                if ing in visited:
                    continue
                if ing not in default_ingredients:
                    intermediates.add(ing)
                    stack.append(ing)
        if target not in default_ingredients:
            intermediates.add(target)
        return intermediates

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _extract_collab_requests(self, action: str) -> List[Tuple[int, List[str]]]:
        if not self._is_collab_action(action or ""):
            return []
        body = self._unwrap_function_body(action)
        if body is None:
            return []
        segments = self._split_top_level_segments(body)
        results: List[Tuple[int, List[str]]] = []
        for segment in segments:
            parsed = self._parse_request_segment(segment)
            if not parsed:
                continue
            target_idx, command = parsed
            normalized = self._normalize_action(command)
            if not normalized:
                continue
            results.append((target_idx, [normalized]))
        return results

    def _parse_request_segment(self, text: str) -> Optional[Tuple[int, str]]:
        stripped = (text or "").strip()
        if not stripped.lower().startswith("request("):
            return None
        body = self._unwrap_function_body(stripped)
        if body is None:
            return None
        target_raw, action_raw = self._split_first_argument(body)
        target_idx = self._agent_index_from_label(target_raw)
        if target_idx is None:
            return None
        command = (action_raw or "").strip()
        if not command:
            return None
        return target_idx, command

    def _split_top_level_segments(self, text: str, delimiter: str = ";") -> List[str]:
        segments: List[str] = []
        depth = 0
        start = 0
        for idx, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == delimiter and depth == 0:
                segment = text[start:idx].strip()
                if segment:
                    segments.append(segment)
                start = idx + 1
        tail = text[start:].strip()
        if tail:
            segments.append(tail)
        return segments

    def _split_first_argument(self, text: str) -> Tuple[str, str]:
        depth = 0
        for idx, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                return text[:idx], text[idx + 1 :]
        return text, ""

    def _unwrap_function_body(self, text: str) -> Optional[str]:
        stripped = (text or "").strip()
        start = stripped.find("(")
        if start == -1:
            return None
        end = len(stripped) - 1
        if stripped[end] != ")":
            return None
        return stripped[start + 1 : end]

    def _agent_index_from_label(self, label: Optional[str]) -> Optional[int]:
        if not label:
            return None
        normalized = label.strip().lower()
        if "assistant" in normalized or "player1" in normalized:
            return 1
        if "chef" in normalized or "player0" in normalized:
            return 0
        return None

    def _normalize_action(self, action: str) -> str:
        return self._clean_action_text(action).replace(" ", "")

    def _clean_action_text(self, action: Optional[str]) -> str:
        text = (action or "").strip()
        if not text:
            return ""
        text = text.replace("<|im_end|>", "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        match = re.search(
            r"Action\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        if match:
            text = match.group(1).strip()
        if text.endswith("```"):
            text = text[:-3].strip()
        return text

    def _is_collab_action(self, action: str) -> bool:
        if not action:
            return False
        lowered = action.strip().lower()
        if lowered.startswith("collab("):
            return True
        return lowered.startswith(("request(", "seek(", "ack(", "deny("))

    def _is_wait_action(self, action: Optional[str]) -> bool:
        if not action:
            return False
        lowered = action.strip().lower()
        return lowered == "wait" or lowered.startswith("wait(")
