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
        self.reward_source_mismatch_events.clear()
        self.pending_paired_comm_requests = [[], []]
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
        paired_comm_reward, paired_comm_meta = self._process_paired_comm_reward(
            agent_index,
            ts=-1 if timestamp is None else int(timestamp),
            action=normalized_action,
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
        # Sequence/ITES progress must be based on actions that actually executed
        # in the environment. LLM calls can include plans, corrections, or queued
        # actions that never execute, so progress is assigned in after_step()
        # from ml_actions instead of here.
        seq_reward = 0.0
        _penalty_total, penalty_details = self._consume_penalties(agent_index)
        format_reward = sum(entry["value"] for entry in penalty_details if entry["type"] == "format")
        validator_reward = sum(entry["value"] for entry in penalty_details if entry["type"] == "validator")
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
        self.call_events.append(entry)
        bucket = self.step_call_records.setdefault(ts, [[], []])
        bucket[agent_index].append(entry)
        if call_index is not None:
            self.call_records_by_source[(agent_index, ts, int(call_index))] = entry
        return entry

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
        if existing is None:
            return self.register_llm_action(
                agent_index=agent_index,
                timestamp=timestamp,
                action_text=action_text,
                agent_name=agent_name,
                call_index=call_index,
                call_type=call_type,
                metadata=metadata,
            )
        return self._update_llm_action_entry(
            existing,
            action_text,
            agent_name=agent_name,
            call_type=call_type,
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
        Pending penalties are still consumed and attached to the existing entry.
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
        if existing is None:
            return self.register_llm_action(
                agent_index=agent_index,
                timestamp=timestamp,
                action_text=action_text,
                agent_name=agent_name,
                call_index=call_index,
                call_type=call_type,
                metadata=metadata,
            )

        updated = self._update_llm_action_entry(
            existing,
            action_text,
            agent_name=agent_name,
            call_type=call_type,
        )
        _penalty_total, penalty_details = self._consume_penalties(agent_index)
        if penalty_details:
            penalties = updated.setdefault("penalties", [])
            penalties.extend(penalty_details)
            format_delta = sum(
                entry["value"] for entry in penalty_details if entry["type"] == "format"
            )
            has_format_penalty = format_delta != 0.0 or any(
                entry.get("type") == "format" for entry in penalties
            )
            validator_delta = 0.0 if has_format_penalty else sum(
                entry["value"]
                for entry in penalty_details
                if entry["type"] == "validator"
            )
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
        return entry

    def after_step(
        self,
        timestep: int,
        ml_actions: Optional[List[str]],
        state,
        executed_action_sources: Optional[List[Optional[Dict[str, Any]]]] = None,
    ) -> Dict:
        """
        Update rewards after a control step. `ml_actions` should contain executed medium-level actions.
        """
        if ml_actions is None:
            ml_actions = [None, None]
        if executed_action_sources is None:
            executed_action_sources = [None, None]

        per_agent = []
        team_total = 0.0

        bucket = self.step_call_records.pop(timestep, [[], []])
        for agent_idx in range(2):
            bucket_entries = bucket[agent_idx]
            call_entries = list(bucket_entries)
            source = (
                executed_action_sources[agent_idx]
                if agent_idx < len(executed_action_sources)
                else None
            )
            executed_action = ml_actions[agent_idx] if agent_idx < len(ml_actions) else None
            source_entry = None
            if source and source.get("call_index") is not None:
                source_entry = self._source_entry_for_executed_action(
                    agent_idx,
                    timestep,
                    executed_action,
                    source,
                    call_entries,
                )
                if source_entry is not None and source_entry not in call_entries:
                    call_entries.append(source_entry)
                source_entry_valid = self._validate_executed_source_entry(
                    agent_idx,
                    timestep,
                    executed_action,
                    source,
                    source_entry,
                    call_entries,
                )
                if not source_entry_valid:
                    source_entry = None
            (
                executed_seq_reward,
                executed_similarity_before,
                executed_similarity_after,
                executed_similarity_delta,
            ) = self._process_sequence_reward(
                agent_idx,
                ml_actions[agent_idx] if agent_idx < len(ml_actions) else None,
            )
            seq_reward = executed_seq_reward
            validator_success_target = (
                source_entry
                if self._entry_can_receive_execution_reward(
                    source_entry, executed_action, source
                )
                else None
            )
            target_has_format_penalty = any(
                item.get("type") == "format"
                for item in (validator_success_target.get("penalties") or [])
            ) if validator_success_target is not None else False
            if validator_success_target is not None and not target_has_format_penalty:
                validator_success_target["validator_reward"] = (
                    validator_success_target.get("validator_reward", 0.0)
                    + self.validator_success_reward_value
                )
                validator_success_target["total"] = (
                    validator_success_target.get("total", 0.0)
                    + self.validator_success_reward_value
                )
            elif validator_success_target is not None:
                self._suppress_validator_when_format_failed(validator_success_target)
            if seq_reward:
                target_entry = None
                if source and source.get("call_index") is not None:
                    target_entry = (
                        source_entry
                        if self._entry_can_receive_execution_reward(
                            source_entry, executed_action, source
                        )
                        else None
                    )
                else:
                    target_entry = self._sequence_reward_target_entry(
                        call_entries,
                        executed_action,
                        None,
                    )
                if target_entry is not None:
                    target_entry["sequence_reward"] = (
                        target_entry.get("sequence_reward", 0.0) + seq_reward
                    )
                    target_entry["progress_reward"] = (
                        target_entry.get("progress_reward", 0.0) + seq_reward
                    )
                    target_entry["total"] = target_entry.get("total", 0.0) + seq_reward
                    target_entry["similarity_before"] = executed_similarity_before
                    target_entry["similarity_after"] = executed_similarity_after
                    target_entry["similarity_delta"] = executed_similarity_delta
                    exec_collab_reward, exec_collab_meta = (
                        self._process_collab_execution_reward(
                            agent_idx,
                            executed_action,
                            sequence_reward=seq_reward,
                        )
                    )
                    if exec_collab_reward:
                        target_entry["collab_reward"] = (
                            target_entry.get("collab_reward", 0.0)
                            + exec_collab_reward
                        )
                        target_entry["total"] = (
                            target_entry.get("total", 0.0) + exec_collab_reward
                        )
                        target_entry["collab_execution_reward"] = (
                            target_entry.get("collab_execution_reward", 0.0)
                            + exec_collab_reward
                        )
                        target_entry["collab_execution_source"] = exec_collab_meta
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
            if seq_reward and not call_entries:
                agent_total += seq_reward
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

        reward_info = {
            "timestamp": timestep,
            "per_agent": per_agent,
            "intermediate": {
                "reward": intermediate_reward,
                "items": new_items,
            },
            "team_total": team_total,
        }
        return reward_info

    def _source_entry_for_executed_action(
        self,
        agent_idx: int,
        timestep: int,
        executed_action: Optional[str],
        source: Dict[str, Any],
        call_entries: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Resolve the reward entry for the exact executed action source.

        Multiple embodied candidates can originate from the same LLM call. In
        that case `(agent, timestep, call_index)` alone is not unique enough:
        later correction/queued actions may overwrite `call_records_by_source`.
        The executed environment action must bind to an entry with the same
        normalized action; otherwise process reward can be assigned to the wrong
        LLM response.
        """
        call_index = source.get("call_index")
        if call_index is None:
            return None
        source_ts = source.get("source_timestamp", source.get("timestamp", timestep))
        try:
            source_ts_int = int(source_ts)
            call_index_int = int(call_index)
        except (TypeError, ValueError):
            return None
        normalized_executed = self._normalize_action(str(executed_action or ""))
        normalized_source = self._normalize_action(str(source.get("action") or ""))
        target_action = normalized_executed or normalized_source
        source_key = (int(agent_idx), source_ts_int, call_index_int)

        def _matches(entry: Optional[Dict[str, Any]]) -> bool:
            if not isinstance(entry, dict):
                return False
            if entry.get("call_index") is not None:
                try:
                    if int(entry.get("call_index")) != call_index_int:
                        return False
                except (TypeError, ValueError):
                    return False
            entry_ts = entry.get("source_timestamp", entry.get("timestamp", source_ts_int))
            try:
                if int(entry_ts) != source_ts_int:
                    return False
            except (TypeError, ValueError):
                return False
            return self._normalize_action(str(entry.get("action") or "")) == target_action

        existing = self.call_records_by_source.get(source_key)
        if _matches(existing):
            return existing
        for entry in reversed(call_entries):
            if _matches(entry):
                self.call_records_by_source[source_key] = entry
                return entry
        for entry in reversed(self.call_events):
            if (
                isinstance(entry, dict)
                and int(entry.get("agent_index", -1)) == int(agent_idx)
                and _matches(entry)
            ):
                self.call_records_by_source[source_key] = entry
                return entry

        return existing

    def _validate_executed_source_entry(
        self,
        agent_idx: int,
        timestep: int,
        executed_action: Optional[str],
        source: Dict[str, Any],
        source_entry: Optional[Dict[str, Any]],
        call_entries: List[Dict[str, Any]],
    ) -> bool:
        normalized_executed = self._normalize_action(str(executed_action or ""))
        if not normalized_executed or normalized_executed.lower().startswith("wait"):
            return True
        call_index = source.get("call_index")
        source_action = self._normalize_action(str(source.get("action") or ""))
        if source_entry is None:
            available = [
                {
                    "call_index": entry.get("call_index"),
                    "call_type": entry.get("call_type"),
                    "action": entry.get("action"),
                }
                for entry in call_entries
            ]
            self._record_reward_source_mismatch(
                "missing_reward_entry",
                agent_idx=agent_idx,
                timestep=timestep,
                call_index=call_index,
                executed_action=normalized_executed,
                source_action=source_action,
                source=source,
                available_calls=available,
            )
            return False
        entry_action = self._normalize_action(str(source_entry.get("action") or ""))
        if entry_action != normalized_executed:
            self._record_reward_source_mismatch(
                "action_mismatch",
                agent_idx=agent_idx,
                timestep=timestep,
                call_index=call_index,
                executed_action=normalized_executed,
                source_action=source_action,
                entry_action=entry_action,
                source=source,
            )
            return False
        if (
            not self._is_rewardable_execution_entry(source_entry)
        ):
            self._record_reward_source_mismatch(
                "not_rewardable_entry",
                agent_idx=agent_idx,
                timestep=timestep,
                call_index=call_index,
                executed_action=normalized_executed,
                source_action=source_action,
                entry_action=entry_action,
                call_type=source_entry.get("call_type"),
                is_collab=source_entry.get("is_collab"),
                source=source,
            )
            return False
        return True

    def _record_reward_source_mismatch(self, reason: str, **payload: Any) -> None:
        event = {"reason": reason, **payload}
        self.reward_source_mismatch_events.append(event)
        print(f"[RewardTracker] source mismatch skipped: {event}", flush=True)

    def _is_rewardable_execution_entry(self, entry: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(entry, dict):
            return False
        return (
            entry.get("call_type") in self.rewardable_action_call_types
            and not entry.get("is_collab")
            and not self._is_collab_action(str(entry.get("action") or ""))
        )

    def _entry_can_receive_execution_reward(
        self,
        entry: Optional[Dict[str, Any]],
        executed_action: Optional[str],
        executed_source: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """True only for the exact LLM call whose embodied action executed."""
        if not self._is_rewardable_execution_entry(entry):
            return False
        normalized_executed = self._normalize_action(str(executed_action or ""))
        if not normalized_executed or normalized_executed.lower().startswith("wait"):
            return False
        if self._normalize_action(str(entry.get("action") or "")) != normalized_executed:
            return False
        if not executed_source or executed_source.get("call_index") is None:
            return False
        try:
            if int(entry.get("call_index")) != int(executed_source.get("call_index")):
                return False
            source_ts = executed_source.get(
                "source_timestamp",
                executed_source.get("timestamp"),
            )
            entry_ts = entry.get("source_timestamp", entry.get("timestamp"))
            if source_ts is not None and entry_ts is not None and int(entry_ts) != int(source_ts):
                return False
        except (TypeError, ValueError):
            return False
        return True

    def _sequence_reward_target_entry(
        self,
        call_entries: List[Dict],
        executed_action: Optional[str],
        executed_source: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict]:
        """Attach executed-action progress only to the call that produced it."""
        normalized_executed = self._normalize_action(str(executed_action or ""))
        if not normalized_executed:
            return None
        if executed_source and executed_source.get("call_index") is not None:
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
        matching_entries = [
            entry
            for entry in call_entries
            if self._normalize_action(str(entry.get("action") or ""))
            == normalized_executed
        ]
        if not matching_entries:
            return None
        for entry in call_entries:
            if (
                entry in matching_entries
                and entry.get("call_type") in self.rewardable_action_call_types
                and not entry.get("is_collab")
            ):
                return entry
        for entry in matching_entries:
            if not entry.get("is_collab"):
                return entry
        return None

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
        self, agent_index: int, ts: int, action: str
    ) -> Tuple[float, Dict[str, Any]]:
        if not self.enable_paired_comm_reward or not action:
            return 0.0, {}

        total_reward = 0.0
        meta: Dict[str, Any] = {}

        response_reward, response_meta = self._resolve_paired_comm_response(
            agent_index, action
        )
        if response_meta:
            total_reward += response_reward
            meta = response_meta

        request_reward, request_meta = self._register_paired_comm_requests(
            agent_index, ts, action
        )
        if request_meta:
            total_reward += request_reward
            meta = request_meta

        return total_reward, meta

    def _resolve_paired_comm_response(
        self, agent_index: int, action: str
    ) -> Tuple[float, Dict[str, Any]]:
        pending_queue = self.pending_paired_comm_requests[agent_index]
        if not pending_queue:
            return 0.0, {}

        response_kind, response_payload = self._classify_paired_comm_response(action)
        if response_kind is None:
            return 0.0, {}

        pair = pending_queue.pop(0)
        request_helpful = bool(pair.get("request_helpful", False))
        request_action = str(pair.get("request_action") or "")
        target_agent = pair.get("initiator_agent")

        if request_helpful:
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
            denied = response_kind == "deny"
            reward = (
                self.paired_comm_deny_reward
                if denied
                else self.paired_comm_response_negative_reward
            )
            result = "bad_request_denied" if denied else "bad_request_followed"

        return reward, {
            "role": "responder",
            "result": result,
            "target_agent": target_agent,
            "request_action": request_action,
            "request_helpful": request_helpful,
        }

    def _register_paired_comm_requests(
        self, agent_index: int, ts: int, action: str
    ) -> Tuple[float, Dict[str, Any]]:
        requests = self._extract_collab_requests(action)
        if not requests:
            return 0.0, {}

        total_reward = 0.0
        request_results: List[str] = []
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
            self.pending_paired_comm_requests[target_idx].append(
                {
                    "timestamp": ts,
                    "initiator_agent": agent_index,
                    "responder_agent": target_idx,
                    "request_action": action_text,
                    "request_helpful": helpful,
                    "baseline_score": baseline_score,
                    "new_score": new_score,
                }
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
        primitive, payload = self._parse_collab_signature_segment(segments[0])
        if primitive in {"ack", "deny"}:
            return primitive, payload
        if primitive in {"request", "seek", "raw"}:
            return "other_collab", payload
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
        has_format_event = any(entry.get("type") == "format" for entry in events)
        for entry in events:
            if has_format_event and entry.get("type") == "validator":
                continue
            key = (entry.get("type"), entry.get("detail"))
            if key in seen:
                continue
            if entry.get("type") == "format" and any(
                item_type == "format" for item_type, _ in seen
            ):
                continue
            seen.add(key)
            if entry["type"] == "format":
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
