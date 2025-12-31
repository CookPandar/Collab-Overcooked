from __future__ import annotations

import json
import copy
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
        self.product_reward_value = float(self.settings.get("product_reward", 0.5))
        self.format_penalty_value = -abs(self.settings.get("format_penalty", 1.0))
        self.validator_penalty_value = -abs(self.settings.get("validator_penalty", 0.5))
        self.enable_collab_reward = bool(self.settings.get("collab_reward_enabled", False))

        self.references = self._load_references()
        self.sequence_histories: List[List[str]] = [[], []]
        self.sequence_scores: List[float] = [0.0, 0.0]
        self.collab_sequence_scores: List[float] = [0.0, 0.0]

        self.recipe_lookup = self._build_recipe_lookup()
        self.intermediate_targets = self._resolve_recipe_targets(self.order)
        self.observed_targets = set()

        self.penalty_queue: List[List[Dict[str, str]]] = [[], []]
        self.call_events: List[Dict] = []
        self.step_call_records: Dict[int, List[List[Dict]]] = {}

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
        self.observed_targets.clear()
        self.call_events.clear()
        self.step_call_records.clear()
        for queue in self.penalty_queue:
            queue.clear()

    def export_state(self) -> Dict[str, Any]:
        """Serialize tracker progress for snapshot replay."""
        return {
            "sequence_histories": copy.deepcopy(self.sequence_histories),
            "sequence_scores": list(self.sequence_scores),
            "collab_sequence_scores": list(self.collab_sequence_scores),
            "observed_targets": list(self.observed_targets),
            "penalty_queue": copy.deepcopy(self.penalty_queue),
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
        self.call_events.clear()
        self.step_call_records.clear()

    def register_llm_action(
        self,
        agent_index: int,
        timestamp: Optional[int],
        action_text: Optional[str],
        *,
        agent_name: Optional[str] = None,
        call_index: Optional[int] = None,
        call_type: Optional[str] = None,
    ) -> Dict:
        """Record a single LLM call regardless of success / failure."""
        if agent_index is None:
            return {}

        normalized_action = (action_text or "").strip()
        is_collab = self._is_collab_action(normalized_action)
        if is_collab:
            if self.enable_collab_reward:
                seq_reward = self._process_collab_reward(agent_index, normalized_action)
            else:
                seq_reward = 0.0
        else:
            seq_reward = self._process_sequence_reward(agent_index, normalized_action)
        penalty_total, penalty_details = self._consume_penalties(agent_index)
        format_reward = sum(entry["value"] for entry in penalty_details if entry["type"] == "format")
        validator_reward = sum(entry["value"] for entry in penalty_details if entry["type"] == "validator")
        total = seq_reward + penalty_total

        ts = -1 if timestamp is None else int(timestamp)
        entry = {
            "timestamp": ts,
            "agent_index": agent_index,
            "agent": agent_name or f"agent_{agent_index}",
            "call_index": call_index,
            "call_type": call_type,
            "action": normalized_action or "[EMPTY]",
            "sequence_reward": seq_reward,
            "progress_reward": seq_reward,
            "format_reward": format_reward,
            "validator_reward": validator_reward,
            "penalties": penalty_details,
            "is_collab": is_collab,
            "total": total,
        }
        self.call_events.append(entry)
        bucket = self.step_call_records.setdefault(ts, [[], []])
        bucket[agent_index].append(entry)
        return entry

    def after_step(self, timestep: int, ml_actions: Optional[List[str]], state) -> Dict:
        """
        Update rewards after a control step. `ml_actions` should contain executed medium-level actions.
        """
        if ml_actions is None:
            ml_actions = [None, None]

        per_agent = []
        team_total = 0.0

        bucket = self.step_call_records.pop(timestep, [[], []])
        for agent_idx in range(2):
            call_entries = bucket[agent_idx]
            seq_reward = sum(entry["sequence_reward"] for entry in call_entries)
            agent_total = sum(entry["total"] for entry in call_entries)
            penalty_total = agent_total - seq_reward
            penalty_details = []
            for entry in call_entries:
                penalty_details.extend(entry["penalties"])
            per_agent.append(
                {
                    "sequence_reward": seq_reward,
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

    # ------------------------------------------------------------------
    # Sequence reward helpers
    # ------------------------------------------------------------------
    def _process_sequence_reward(self, agent_index: int, action: Optional[str]) -> float:
        if not action:
            return 0.0
        normalized = action.strip()
        if not normalized or normalized.lower().startswith("wait"):
            return 0.0

        self.sequence_histories[agent_index].append(normalized)
        new_score = self._best_sequence_score(agent_index)
        delta = max(0.0, new_score - self.sequence_scores[agent_index])
        if delta > 0:
            self.sequence_scores[agent_index] = new_score
            self.collab_sequence_scores[agent_index] = max(
                self.collab_sequence_scores[agent_index], new_score
            )
            return delta * self.sequence_weight
        return 0.0

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
        return reward

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
        for entry in events:
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
        return action.replace(" ", "")

    def _is_collab_action(self, action: str) -> bool:
        return action.strip().lower().startswith("collab(") if action else False
