import json
import queue
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from collab_overcooked.agents.collab import LLMAgents
from collab_overcooked.reward import ProcessRewardTracker
from collab_overcooked.training.main_session import CollabMainSession, PolicyCallRecord
from collab_overcooked.training.mappo_qwen import AdapterSpec, MAPPOTrainer, TextRolloutBuffer

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_LOG_BASE = REPO_ROOT / "assets/data/batch_results/level12_parallel_eval/base_shared_suite/base_shared/json/baked_bell_pepper/worker0_0_7_baked_bell_pepper_0-2026-03-23_14-52-21_345498-6660f9_baked_bell_pepper.json"
REAL_LOG_SFT = REPO_ROOT / "assets/data/batch_results/level12_parallel_eval/sft_shared_lora_suite/sft_shared_lora/json/baked_bell_pepper/worker0_0_7_baked_bell_pepper_0-2026-03-23_14-52-32_135111-3ae403_baked_bell_pepper.json"
REAL_LOG_GPT4O = REPO_ROOT / "assets/data/batch_results/azure-gpt-4o/json/baked_bell_pepper/worker0_0_7_baked_bell_pepper_0-2025-12-17_17-18-46_630114-458910_baked_bell_pepper.json"
REAL_LOG_ROOTS = [
    REPO_ROOT / "assets/data/batch_results/level12_parallel_eval/base_shared_suite/base_shared/json",
    REPO_ROOT / "assets/data/batch_results/level12_parallel_eval/sft_shared_lora_suite/sft_shared_lora/json",
    REPO_ROOT / "assets/data/batch_results/azure-gpt-4o/json",
]
REAL_REFERENCE_DIR = REPO_ROOT / "collab_overcooked/prompts/reference"


class FakeMDP:
    def __init__(self):
        self.recipe_config = {
            "recipes": {
                "main": {
                    "soup": {"recipe": ["onion"]},
                    "dish": {"recipe": ["soup"]},
                }
            }
        }
        self.default_ingredients = {"onion"}


class FakeObject:
    def __init__(self, name, state=None):
        self.name = name
        self.state = state


class FakePlayer:
    def __init__(self, obj=None):
        self._obj = obj

    def has_object(self):
        return self._obj is not None

    def get_object(self):
        return self._obj


class FakeState:
    def __init__(self, objects=None, players=None):
        self.objects = objects or {}
        self.players = players or [FakePlayer(), FakePlayer()]


class RewardAssignmentTest(unittest.TestCase):
    _real_log_cache = {}

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.reference_dir = Path(self.tempdir.name)
        self.mdp = FakeMDP()

    def tearDown(self):
        self.tempdir.cleanup()

    def load_real_log(self, path: Path):
        if not path.exists():
            self.skipTest(f"Missing real log fixture: {path}")
        cache_key = str(path.resolve())
        if cache_key not in self._real_log_cache:
            self._real_log_cache[cache_key] = json.loads(path.read_text(encoding="utf-8"))
        return self._real_log_cache[cache_key]

    @staticmethod
    def classify_action_mode_from_output(response_text: str) -> str:
        text = (response_text or "").strip()
        section_pattern = re.compile(
            r"^\s*(Think|Recent Goal|Action)\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        sections = {
            match.group(1).strip().lower(): (match.group(2) or "").strip()
            for match in section_pattern.finditer(text)
        }
        action_block = sections.get("action", "")
        action_body = action_block.replace("\r\n", "\n").replace("\r", "\n").replace("\n", ";").strip()
        if not action_body:
            return "empty"
        tokens = [token.strip() for token in action_body.split(";") if token.strip()]
        collab_tokens = []
        embodied_tokens = []
        for token in tokens:
            lowered = token.lower()
            if lowered.startswith("collab(") or lowered.startswith(("request(", "seek(", "ack(", "deny(")):
                collab_tokens.append(token)
            else:
                embodied_tokens.append(token)
        if collab_tokens and embodied_tokens:
            return "mixed"
        if collab_tokens:
            return "communication"
        return "planner_main"

    @staticmethod
    def build_records_from_log_calls(raw_calls, inject_action_mode: bool = False):
        records = []
        for call in raw_calls:
            metadata = {"call_index": call.get("call_index")}
            if inject_action_mode:
                metadata["action_mode"] = RewardAssignmentTest.classify_action_mode_from_output(
                    call.get("output") or ""
                )
            records.append(
                PolicyCallRecord(
                    agent_index=0 if call.get("agent") == "Chef" else 1,
                    messages=[],
                    prompt=call.get("input") or "",
                    response=call.get("output") or "",
                    metadata=metadata,
                    context={"call_type": call.get("call_type")},
                    timestep=call.get("timestamp"),
                )
            )
        return records

    def collect_real_reward_cases(self):
        planner_cases = []
        communication_cases = []
        for root in REAL_LOG_ROOTS:
            if not root.exists():
                self.skipTest(f"Missing real log root: {root}")
            for path in sorted(root.glob("*/*.json")):
                data = self.load_real_log(path)
                timeline = data.get("content") or []
                rewards = data.get("process_rewards") or []
                for step_idx, reward_step in enumerate(rewards):
                    if step_idx >= len(timeline):
                        continue
                    original_log = (((timeline[step_idx] or {}).get("content") or {}).get("original_log")) or []
                    for agent_idx, agent_calls in enumerate(original_log):
                        for raw_call in agent_calls or []:
                            semantic_mode = self.classify_action_mode_from_output(
                                raw_call.get("output") or ""
                            )
                            if semantic_mode == "communication":
                                communication_cases.append(
                                    {
                                        "category": "communication",
                                        "path": path,
                                        "step_idx": step_idx,
                                        "agent_idx": agent_idx,
                                        "call_index": raw_call.get("call_index"),
                                    }
                                )
                    for agent_idx, agent_reward in enumerate(reward_step.get("per_agent") or []):
                        call_entries = agent_reward.get("calls") or []
                        raw_lookup = {
                            raw_call.get("call_index"): raw_call
                            for raw_call in (original_log[agent_idx] if agent_idx < len(original_log) else [])
                        }
                        for reward_entry in call_entries:
                            call_index = reward_entry.get("call_index")
                            if call_index not in raw_lookup:
                                continue
                            semantic_mode = self.classify_action_mode_from_output(
                                raw_lookup[call_index].get("output") or ""
                            )
                            if semantic_mode != "planner_main":
                                continue
                            planner_cases.append(
                                {
                                    "category": "planner",
                                    "path": path,
                                    "step_idx": step_idx,
                                    "agent_idx": agent_idx,
                                    "call_index": call_index,
                                    "sequence_reward": float(reward_entry.get("sequence_reward", 0.0) or 0.0),
                                    "format_reward": float(reward_entry.get("format_reward", 0.0) or 0.0),
                                    "validator_reward": float(reward_entry.get("validator_reward", 0.0) or 0.0),
                                    "total": float(reward_entry.get("total", 0.0) or 0.0),
                                }
                            )

        selected = []
        used = set()

        def add_cases(candidates, predicate, limit):
            added = 0
            for case in candidates:
                case_key = (
                    str(case["path"]),
                    int(case["step_idx"]),
                    int(case["agent_idx"]),
                    int(case["call_index"]),
                    case["category"],
                )
                if case_key in used:
                    continue
                if not predicate(case):
                    continue
                used.add(case_key)
                selected.append(case)
                added += 1
                if added >= limit:
                    break
            return added

        add_cases(planner_cases, lambda case: case["format_reward"] != 0.0, 10)
        add_cases(planner_cases, lambda case: case["validator_reward"] != 0.0, 10)
        add_cases(
            planner_cases,
            lambda case: (
                case["sequence_reward"] > 0.0
                and case["format_reward"] == 0.0
                and case["validator_reward"] == 0.0
            ),
            5,
        )
        add_cases(communication_cases, lambda case: True, 5)

        self.assertEqual(len(selected), 30, "Expected exactly 30 sampled real reward cases.")
        return selected

    def collect_real_positive_sequence_cases(self, limit: int = 10):
        cases = []
        for root in REAL_LOG_ROOTS:
            if not root.exists():
                self.skipTest(f"Missing real log root: {root}")
            for path in sorted(root.glob("*/*.json")):
                data = self.load_real_log(path)
                rewards = data.get("process_rewards") or []
                for step_idx, reward_step in enumerate(rewards):
                    timeline = data.get("content") or []
                    original_log = (
                        (((timeline[step_idx] or {}).get("content") or {}).get("original_log")) or []
                        if step_idx < len(timeline)
                        else []
                    )
                    raw_lookup = {
                        agent_idx: {
                            raw_call.get("call_index"): raw_call
                            for raw_call in (agent_calls or [])
                        }
                        for agent_idx, agent_calls in enumerate(original_log)
                    }
                    for agent_idx, agent_reward in enumerate(reward_step.get("per_agent") or []):
                        for reward_entry in agent_reward.get("calls") or []:
                            call_index = reward_entry.get("call_index")
                            raw_call = raw_lookup.get(agent_idx, {}).get(call_index)
                            if raw_call is None:
                                continue
                            semantic_mode = self.classify_action_mode_from_output(
                                raw_call.get("output") or ""
                            )
                            if semantic_mode != "planner_main":
                                continue
                            sequence_reward = float(reward_entry.get("sequence_reward", 0.0) or 0.0)
                            if sequence_reward <= 0.0:
                                continue
                            prefix_safe = True
                            for prefix_step_idx, prefix_reward_step in enumerate(rewards[: step_idx + 1]):
                                prefix_timeline = data.get("content") or []
                                prefix_original_log = (
                                    (((prefix_timeline[prefix_step_idx] or {}).get("content") or {}).get("original_log")) or []
                                    if prefix_step_idx < len(prefix_timeline)
                                    else []
                                )
                                prefix_raw_lookup = {
                                    idx: {
                                        raw_call.get("call_index"): raw_call
                                        for raw_call in (agent_calls or [])
                                    }
                                    for idx, agent_calls in enumerate(prefix_original_log)
                                }
                                for prefix_entry in (
                                    (prefix_reward_step.get("per_agent") or [])[agent_idx].get("calls") or []
                                ):
                                    if prefix_step_idx == step_idx and prefix_entry.get("call_index") == reward_entry.get("call_index"):
                                        break
                                    prefix_raw_call = prefix_raw_lookup.get(agent_idx, {}).get(
                                        prefix_entry.get("call_index")
                                    )
                                    if prefix_raw_call is None:
                                        continue
                                    prefix_mode = self.classify_action_mode_from_output(
                                        prefix_raw_call.get("output") or ""
                                    )
                                    if prefix_mode != "communication":
                                        continue
                                    action_text = str(prefix_entry.get("action") or "").strip().lower()
                                    if action_text.startswith(("request(", "seek(", "ack(", "deny(")):
                                        prefix_safe = False
                                        break
                                if not prefix_safe:
                                    break
                            if not prefix_safe:
                                continue
                            cases.append(
                                {
                                    "path": path,
                                    "order": path.parent.name,
                                    "step_idx": step_idx,
                                    "agent_idx": agent_idx,
                                    "call_index": reward_entry.get("call_index"),
                                    "sequence_reward": sequence_reward,
                                }
                            )
                            if len(cases) >= limit:
                                return cases
        self.assertEqual(len(cases), limit, f"Expected {limit} positive sequence cases.")
        return cases

    def build_tracker(self, order="soup", refs_agent0=None, refs_agent1=None, **settings):
        refs_agent0 = refs_agent0 or []
        refs_agent1 = refs_agent1 or []
        ref_path = self.reference_dir / f"unit_{order}_ref.json"
        payload = {
            "demo": {
                "agent_0": refs_agent0,
                "agent_1": refs_agent1,
            }
        }
        ref_path.write_text(json.dumps(payload), encoding="utf-8")
        return ProcessRewardTracker(
            order=order,
            mdp=self.mdp,
            reference_dir=self.reference_dir,
            settings=settings,
        )

    def build_real_tracker(self, order: str, **settings):
        self.assertTrue(REAL_REFERENCE_DIR.exists(), f"Missing reference dir: {REAL_REFERENCE_DIR}")
        return ProcessRewardTracker(
            order=order,
            mdp=self.mdp,
            reference_dir=REAL_REFERENCE_DIR,
            settings=settings,
        )

    def test_bootstrap_histories_from_snapshot_recovers_prefix_actions(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)", "serve(soup)"],
            refs_agent1=["place_obj_on_counter()", "wash_dish()"],
        )
        tracker.import_state(
            {
                "sequence_histories": [[], []],
                "sequence_scores": [0.0, 0.0],
                "collab_sequence_scores": [0.0, 0.0],
            }
        )
        restored = tracker.bootstrap_histories_from_snapshot(
            {
                "0": {
                    "teammate_ml_actions": [
                        {"timestamp": 1, "action": "place_obj_on_counter()"},
                        {"timestamp": 2, "action": "wash_dish()"},
                    ]
                },
                "1": {
                    "teammate_ml_actions": [
                        {"timestamp": 3, "action": "pickup(onion)"},
                        {"timestamp": 5, "action": "cook(onion)"},
                    ]
                },
            }
        )
        self.assertTrue(restored)
        self.assertEqual(
            tracker.sequence_histories[0],
            ["pickup(onion)", "cook(onion)"],
        )
        self.assertEqual(
            tracker.sequence_histories[1],
            ["place_obj_on_counter()", "wash_dish()"],
        )
        self.assertGreater(tracker.sequence_scores[0], 0.0)
        self.assertGreater(tracker.sequence_scores[1], 0.0)

    def test_bootstrap_histories_from_ml_actions_recovers_tail_snapshot_action(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)", "serve(soup)"],
            refs_agent1=["pickup(onion)", "place_obj_on_counter()"],
        )
        tracker.import_state(
            {
                "sequence_histories": [[], []],
                "sequence_scores": [0.0, 0.0],
                "collab_sequence_scores": [0.0, 0.0],
            }
        )

        restored = tracker.bootstrap_histories_from_ml_actions(
            [None, "pickup(onion)"]
        )

        self.assertTrue(restored)
        self.assertEqual(tracker.sequence_histories[0], [])
        self.assertEqual(tracker.sequence_histories[1], ["pickup(onion)"])
        self.assertEqual(tracker.sequence_scores[0], 0.0)
        self.assertGreater(tracker.sequence_scores[1], 0.0)

        tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="pickup(onion)",
            agent_name="Assistant",
            call_index=0,
            call_type="planner_main",
        )
        reward = tracker.after_step(3, [None, "pickup(onion)"], FakeState())
        agent1_calls = reward["per_agent"][1]["calls"]
        self.assertEqual(reward["per_agent"][1]["sequence_reward"], 0.0)
        self.assertEqual(agent1_calls[0]["sequence_reward"], 0.0)

    def test_clear_pending_penalties_drops_snapshot_validator_debt(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)"],
            refs_agent1=["place_obj_on_counter()"],
        )
        tracker.import_state(
            {
                "sequence_histories": [[], []],
                "sequence_scores": [0.0, 0.0],
                "collab_sequence_scores": [0.0, 0.0],
                "penalty_queue": [
                    [{"type": "validator", "detail": "old chef error"}],
                    [{"type": "validator", "detail": "old assistant error"}],
                ],
            }
        )
        tracker.clear_pending_penalties()
        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        self.assertEqual(tracker.penalty_queue, [[], []])
        self.assertEqual(entry["validator_reward"], 0.0)

    def test_format_failed_action_never_gets_sequence_progress(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion,counter)"],
            format_penalty=0.5,
        )

        tracker.register_format_error(0, "missing_think")
        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=3,
            call_type="planner_main",
        )

        self.assertEqual(entry["sequence_reward"], 0.0)
        self.assertEqual(entry["format_reward"], -0.5)
        self.assertEqual(entry["validator_reward"], 0.0)
        self.assertEqual(entry["total"], -0.5)

        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=3,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=0,
            ml_actions=["pickup(onion)", None],
            state=FakeState(),
        )
        call_entry = reward_info["per_agent"][0]["calls"][0]
        self.assertEqual(call_entry["sequence_reward"], 0.0)
        self.assertEqual(call_entry["format_reward"], -0.5)
        self.assertEqual(call_entry["validator_reward"], 0.0)
        self.assertEqual(call_entry["total"], -0.5)

        # Penalties should be consumed by exactly one call.
        next_entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=4,
            call_type="planner_main",
        )
        self.assertEqual(next_entry["format_reward"], 0.05)
        self.assertEqual(next_entry["sequence_reward"], 0.0)

    def test_default_format_and_validator_penalties_are_large_negative(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
        )
        tracker.register_format_error(0, "bad format")
        tracker.register_validator_error(0, "invalid action")

        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )

        self.assertEqual(entry["format_reward"], -20.0)
        self.assertEqual(entry["validator_reward"], 0.0)
        self.assertEqual(entry["total"], -20.0)

    def test_valid_planner_action_gets_format_success_reward(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
        )

        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        comm_entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="request(Assistant,pickup(onion))",
            agent_name="Chef",
            call_index=2,
            call_type="communication",
        )

        self.assertEqual(entry["format_reward"], 0.05)
        self.assertEqual(entry["validator_reward"], 0.0)
        self.assertEqual(entry["total"], 0.05)
        self.assertEqual(comm_entry["format_reward"], 0.0)

    def test_valid_wait_action_gets_no_positive_reward(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )

        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="wait(3)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=3,
            ml_actions=["wait(3)", None],
            state=FakeState(),
        )

        self.assertEqual(entry["format_reward"], 0.0)
        self.assertEqual(entry["validator_reward"], 0.0)
        self.assertEqual(entry["sequence_reward"], 0.0)
        self.assertEqual(entry["total"], 0.0)
        self.assertEqual(reward_info["per_agent"][0]["calls"][0], entry)

    def test_validated_planner_action_gets_validator_and_sequence_reward(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
        )
        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=7,
            call_type="planner_main",
        )

        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=7,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=0,
            ml_actions=["pickup(onion)", None],
            state=FakeState(),
        )

        call = reward_info["per_agent"][0]["calls"][0]
        self.assertIs(call, entry)
        self.assertEqual(call["format_reward"], 0.05)
        self.assertEqual(call["validator_reward"], 0.05)
        self.assertEqual(call["sequence_reward"], 1.0)
        self.assertEqual(call["total"], 1.1)

    def test_validator_correction_action_gets_format_and_validated_rewards(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
        )
        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=8,
            call_type="validator_correction",
        )

        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=8,
            call_type="validator_correction",
        )
        reward_info = tracker.after_step(
            timestep=0,
            ml_actions=["pickup(onion)", None],
            state=FakeState(),
        )

        call = reward_info["per_agent"][0]["calls"][0]
        self.assertIs(call, entry)
        self.assertEqual(call["format_reward"], 0.05)
        self.assertEqual(call["validator_reward"], 0.05)
        self.assertEqual(call["sequence_reward"], 1.0)
        self.assertEqual(call["total"], 1.1)

    def test_generated_but_unvalidated_action_does_not_get_validator_success_reward(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)"],
            refs_agent1=[],
        )
        early = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        actual = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="cook(onion)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )

        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=0,
            action_text="cook(onion)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )
        tracker.after_step(
            timestep=0,
            ml_actions=["cook(onion)", None],
            state=FakeState(),
        )

        self.assertEqual(early["validator_reward"], 0.0)
        self.assertEqual(actual["validator_reward"], 0.05)

    def test_after_step_does_not_infer_reward_from_executed_source(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
        )
        wrong_entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=0,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=2,
            call_type="planner_main",
        )

        reward_info = tracker.after_step(
            timestep=0,
            ml_actions=[None, "place_obj_on_counter()"],
            state=FakeState(),
        )

        self.assertEqual(wrong_entry["sequence_reward"], 0.0)
        self.assertEqual(wrong_entry["validator_reward"], 0.0)
        self.assertEqual(reward_info["per_agent"][1]["sequence_reward"], 0.0)

    def test_communication_plan_queues_primary_action_with_source(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 3

        agent._replace_pending_action_plan(
            ["place_obj_on_counter()", "pickup(bell_pepper, counter)"],
            call_index=4,
            call_type="planner_main",
            include_primary=True,
        )

        self.assertEqual(list(agent.action_wait_parse.queue), [
            "place_obj_on_counter()",
            "pickup(bell_pepper, counter)",
        ])
        first_source = agent._pending_action_sources.get()
        second_source = agent._pending_action_sources.get()
        self.assertEqual(first_source["call_index"], 4)
        self.assertEqual(first_source["action"], "place_obj_on_counter()")
        self.assertEqual(second_source["action"], "pickup(bell_pepper, counter)")

    def test_reclassified_communication_action_source_survives_outer_parse(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 3
        agent.reward_tracker = None
        agent._current_action_source = None
        agent._active_action_source = None
        agent.last_executed_action_source = None
        agent.time_to_wait = 0
        agent.mdp = type("FakeParserMDP", (), {"interact_actions": []})()
        agent._clear_pending_action_sources = LLMAgents._clear_pending_action_sources.__get__(agent, LLMAgents)
        agent._make_action_source = LLMAgents._make_action_source.__get__(agent, LLMAgents)
        agent._sync_reward_entry_for_action_source = LLMAgents._sync_reward_entry_for_action_source.__get__(agent, LLMAgents)
        agent._queue_action_source = LLMAgents._queue_action_source.__get__(agent, LLMAgents)
        agent._queue_pending_action_source = LLMAgents._queue_pending_action_source.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._sanitize_action_text = LLMAgents._sanitize_action_text.__get__(agent, LLMAgents)
        agent._split_action_tokens = LLMAgents._split_action_tokens.__get__(agent, LLMAgents)
        agent.parse_response = LLMAgents.parse_response.__get__(agent, LLMAgents)
        agent.parse_params_in_action = LLMAgents.parse_params_in_action.__get__(agent, LLMAgents)
        agent.parse_ml_action = LLMAgents.parse_ml_action.__get__(agent, LLMAgents)
        agent.parse_wait_string = LLMAgents.parse_wait_string.__get__(agent, LLMAgents)
        agent.planner = type(
            "FakePlanner",
            (),
            {"add_msg_to_dialog_history": lambda self, msg: None},
        )()
        agent._report_action_format_error = lambda *args, **kwargs: None

        ml_action = LLMAgents.parse_ml_action_top(
            agent,
            "Action: place_obj_on_counter()",
            True,
            source_call_index=5,
            source_call_type="planner_main",
        )

        self.assertEqual(ml_action, "place_obj_on_counter()")
        self.assertEqual(agent._current_action_source["call_index"], 5)
        self.assertEqual(agent._current_action_source["call_type"], "planner_main")
        self.assertEqual(agent._current_action_source["action"], "place_obj_on_counter()")

        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
        )
        entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=5,
            call_type="planner_main",
        )
        tracker.mark_llm_action_validated(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=5,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=3,
            ml_actions=[None, "place_obj_on_counter()"],
            state=FakeState(),
        )

        self.assertIs(reward_info["per_agent"][1]["calls"][0], entry)
        self.assertEqual(entry["sequence_reward"], 1.0)
        self.assertEqual(entry["validator_reward"], 0.05)

    def test_llm_call_index_does_not_reuse_within_timestep_after_log_flush(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.current_timestep = 3
        agent.name = "Chef"
        agent.pending_llm_logs = []
        agent.planner = SimpleNamespace()
        agent._allocate_llm_call_index = LLMAgents._allocate_llm_call_index.__get__(agent, LLMAgents)
        agent._log_llm_call = LLMAgents._log_llm_call.__get__(agent, LLMAgents)

        agent._log_llm_call("communication", "prompt", "Action: Collab(seek(Assistant))", 1)
        self.assertEqual(agent.pending_llm_logs[-1]["call_index"], 0)
        agent.pending_llm_logs = []
        agent._log_llm_call("planner_main", "prompt", "Action: pickup(onion,counter)", 1)
        self.assertEqual(agent.pending_llm_logs[-1]["call_index"], 1)

        agent.current_timestep = 4
        agent.pending_llm_logs = []
        agent._log_llm_call("planner_main", "prompt", "Action: wait(1)", 1)
        self.assertEqual(agent.pending_llm_logs[-1]["call_index"], 0)

    def test_active_action_source_survives_pending_source_during_long_action(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 0
        agent.name = "Chef"
        agent.current_timestep = 8
        agent.current_ml_action = "pickup(bell_pepper,counter)"
        agent.current_ml_action_steps = 1
        agent.last_executed_action_source = None
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = {
            "agent_index": 0,
            "agent": "Chef",
            "timestamp": 8,
            "source_timestamp": 8,
            "call_index": 0,
            "call_type": "planner_main",
            "action": "pickup(bell_pepper,counter)",
        }
        agent._active_action_source = dict(agent._current_action_source)
        agent._locked_action_source = dict(agent._current_action_source)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._normalize_source_action = LLMAgents._normalize_source_action.__get__(agent, LLMAgents)
        agent._action_source_matches = LLMAgents._action_source_matches.__get__(agent, LLMAgents)
        agent._take_matching_pending_action_source = LLMAgents._take_matching_pending_action_source.__get__(agent, LLMAgents)
        agent._resolve_executed_action_source = LLMAgents._resolve_executed_action_source.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)

        agent._pending_action_sources.put(
            {
                "agent_index": 0,
                "agent": "Chef",
                "timestamp": 12,
                "source_timestamp": 12,
                "call_index": 3,
                "call_type": "validator_correction",
                "action": "cut(chopping_board0)",
            }
        )

        source = agent._resolve_executed_action_source("pickup(bell_pepper, counter)")

        self.assertIsNotNone(source)
        self.assertEqual(source["source_timestamp"], 8)
        self.assertEqual(source["call_index"], 0)
        self.assertEqual(source["action"], "pickup(bell_pepper,counter)")

        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(bell_pepper, counter)"],
            refs_agent1=[],
        )
        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=8,
            action_text="pickup(bell_pepper, counter)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=8,
            action_text="pickup(bell_pepper, counter)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=8,
            ml_actions=["pickup(bell_pepper, counter)", None],
            state=FakeState(),
        )
        self.assertIs(reward_info["per_agent"][0]["calls"][0], entry)
        self.assertEqual(entry["sequence_reward"], 1.0)
        self.assertEqual(entry["validator_reward"], 0.05)

    def test_success_episode_requires_full_episode_return(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer._last_rollout_stats = {
            "env_steps": 0,
            "episodes_completed": 0,
            "success_episodes": 0,
            "env_reward_sum": 0.0,
            "positive_reward_steps": 0,
            "policy_calls": 0,
            "episode_return_sum": 0.0,
            "episode_lengths_sum": 0,
            "agent0_custom_return_sum": 0.0,
            "agent1_custom_return_sum": 0.0,
            "team_custom_return_sum": 0.0,
        }
        trainer._current_episode_return = 0.0
        trainer._current_episode_length = 0
        trainer._current_episode_has_positive = False
        trainer._current_episode_custom_stats = {
            "agent0_total": 0.0,
            "agent0_sequence": 0.0,
            "agent0_format": 0.0,
            "agent0_validator": 0.0,
            "agent0_comm": 0.0,
            "agent0_paired_comm": 0.0,
            "agent1_total": 0.0,
            "agent1_sequence": 0.0,
            "agent1_format": 0.0,
            "agent1_validator": 0.0,
            "agent1_comm": 0.0,
            "agent1_paired_comm": 0.0,
            "team_total": 0.0,
        }
        trainer._episode_log_path = None
        trainer._runtime_worker_id = lambda: 0
        trainer._current_update_idx = 1
        trainer._episode_counter = 0
        trainer._extract_step_custom_reward_stats = lambda process_reward=None: {
            "agent0_total": 0.0,
            "agent0_sequence": 0.0,
            "agent0_format": 0.0,
            "agent0_validator": 0.0,
            "agent0_comm": 0.0,
            "agent0_paired_comm": 0.0,
            "agent1_total": 0.0,
            "agent1_sequence": 0.0,
            "agent1_format": 0.0,
            "agent1_validator": 0.0,
            "agent1_comm": 0.0,
            "agent1_paired_comm": 0.0,
            "team_total": 0.0,
        }
        trainer._log_episode_return = lambda **kwargs: None

        MAPPOTrainer._update_rollout_stats(
            trainer,
            step_reward=5.0,
            done=False,
            policy_calls=1,
            process_reward=None,
        )
        MAPPOTrainer._update_rollout_stats(
            trainer,
            step_reward=10.0,
            done=False,
            policy_calls=1,
            process_reward=None,
        )
        MAPPOTrainer._update_rollout_stats(
            trainer,
            step_reward=5.0,
            done=True,
            policy_calls=1,
            process_reward=None,
        )
        self.assertEqual(trainer._last_rollout_stats["episodes_completed"], 1)
        self.assertEqual(trainer._last_rollout_stats["success_episodes"], 1)

        trainer._last_rollout_stats = {
            "env_steps": 0,
            "episodes_completed": 0,
            "success_episodes": 0,
            "env_reward_sum": 0.0,
            "positive_reward_steps": 0,
            "policy_calls": 0,
            "episode_return_sum": 0.0,
            "episode_lengths_sum": 0,
            "agent0_custom_return_sum": 0.0,
            "agent1_custom_return_sum": 0.0,
            "team_custom_return_sum": 0.0,
        }
        trainer._current_episode_return = 0.0
        trainer._current_episode_length = 0
        trainer._current_episode_has_positive = False
        MAPPOTrainer._update_rollout_stats(
            trainer,
            step_reward=1.0,
            done=False,
            policy_calls=1,
            process_reward=None,
        )
        MAPPOTrainer._update_rollout_stats(
            trainer,
            step_reward=1.0,
            done=True,
            policy_calls=1,
            process_reward=None,
        )
        self.assertEqual(trainer._last_rollout_stats["episodes_completed"], 1)
        self.assertEqual(trainer._last_rollout_stats["success_episodes"], 0)

    def test_partial_success_reward_targets_configured_agent_record(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.partial_success_agent_index = 0
        trainer.partial_success_reward = 20.0
        trainer.partial_success_type = "agent_holds"
        trainer.partial_success_shared_terminal_reward = False

        records = [
            SimpleNamespace(agent_index=0, reward=2.1, done=False, metadata={}),
            SimpleNamespace(agent_index=1, reward=0.0, done=False, metadata={}),
        ]

        MAPPOTrainer._apply_partial_success_reward(trainer, records)

        self.assertAlmostEqual(records[0].reward, 22.1)
        self.assertTrue(records[0].done)
        self.assertAlmostEqual(records[1].reward, 0.0)
        self.assertFalse(records[1].done)
        self.assertEqual(records[0].metadata["partial_success_agent_index"], 0)
        self.assertAlmostEqual(
            records[0].metadata["reward_breakdown"]["partial_success_reward"],
            20.0,
        )

    def test_shared_partial_success_rewards_both_agent_records(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.partial_success_agent_index = 0
        trainer.partial_success_reward = 20.0
        trainer.partial_success_type = "agent_holds"
        trainer.partial_success_shared_terminal_reward = True
        trainer.num_agents = 2

        records = [
            SimpleNamespace(agent_index=0, reward=2.1, done=False, metadata={}),
            SimpleNamespace(agent_index=1, reward=0.1, done=False, metadata={}),
        ]

        MAPPOTrainer._apply_partial_success_reward(trainer, records)

        self.assertAlmostEqual(records[0].reward, 22.1)
        self.assertAlmostEqual(records[1].reward, 20.1)
        self.assertTrue(records[0].done)
        self.assertTrue(records[1].done)
        self.assertAlmostEqual(
            records[0].metadata["reward_breakdown"]["partial_success_reward"],
            20.0,
        )
        self.assertAlmostEqual(
            records[1].metadata["reward_breakdown"]["partial_success_reward"],
            20.0,
        )

    def test_partial_success_done_marks_each_agent_without_moving_reward(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer._active_rollout_id = "rollout0"
        trainer._active_rollout_episode_idx = 0
        trainer._runtime_worker_id = lambda: 0
        trainer._runtime_stage_round_idx = lambda: 1

        def record(agent_index, reward, done=False):
            return PolicyCallRecord(
                agent_index=agent_index,
                messages=[],
                prompt="p",
                response="Action: wait(1)",
                metadata={
                    "prompt_ids": torch.tensor([1]),
                    "response_ids": torch.tensor([2]),
                    "log_prob": 0.0,
                    "value": 0.0,
                    "entropy": 0.0,
                    "action_mode": "planner_main",
                    "reward_breakdown": {
                        "sequence_reward": 0.0,
                        "format_reward": 0.0,
                        "validator_reward": 0.0,
                        "paired_comm_reward": 0.0,
                        "partial_success_reward": max(0.0, reward),
                        "raw": {"partial_success_reward": max(0.0, reward)},
                    },
                },
                reward=reward,
                done=done,
                timestep=7,
            )

        records = [record(0, 20.0, True), record(1, 0.0, False)]
        MAPPOTrainer._append_policy_records(trainer, records, 20.0, True)

        self.assertEqual(len(trainer.buffer.storage), 2)
        agent0, agent1 = trainer.buffer.storage
        self.assertEqual(agent0.agent_index, 0)
        self.assertEqual(agent1.agent_index, 1)
        self.assertAlmostEqual(agent0.reward, 20.0)
        self.assertAlmostEqual(agent1.reward, 0.0)
        self.assertEqual(agent0.done, 1.0)
        self.assertEqual(agent1.done, 1.0)

    def test_partial_success_without_target_record_backfills_target_agent_buffer(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer.partial_success_agent_index = 0
        trainer.partial_success_reward = 20.0
        trainer._active_rollout_id = "rollout0"
        trainer._active_rollout_episode_idx = 0
        trainer._runtime_worker_id = lambda: 0
        trainer._runtime_stage_round_idx = lambda: 1

        trainer.buffer.add(
            prompt_ids=torch.tensor([1]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=2.1,
            done=0.0,
            agent_index=0,
            entropy=0.0,
            timestep=5,
            sequence_reward=2.0,
            process_reward=2.0,
            breakdown_total_reward=2.1,
            rollout_id="rollout0",
        )

        current_record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="p",
            response="",
            metadata={
                "prompt_ids": torch.tensor([3]),
                "response_ids": torch.tensor([4]),
                "log_prob": 0.0,
                "value": 0.0,
                "entropy": 0.0,
                "action_mode": "planner_main",
                "reward_breakdown": {"raw": {}},
            },
            reward=0.0,
            done=False,
            timestep=7,
        )

        self.assertIsNone(
            MAPPOTrainer._select_partial_success_record(
                trainer,
                [current_record],
                allow_fallback=False,
            )
        )
        self.assertTrue(MAPPOTrainer._apply_partial_success_reward_to_buffer(trainer))
        MAPPOTrainer._append_policy_records(trainer, [current_record], 0.0, True)

        agent0_transition = trainer.buffer.storage[0]
        agent1_transition = trainer.buffer.storage[1]
        self.assertEqual(agent0_transition.agent_index, 0)
        self.assertEqual(agent1_transition.agent_index, 1)
        self.assertAlmostEqual(agent0_transition.reward, 22.1)
        self.assertAlmostEqual(agent0_transition.breakdown_total_reward, 22.1)
        self.assertAlmostEqual(agent1_transition.reward, 0.0)
        self.assertEqual(agent0_transition.done, 1.0)
        self.assertEqual(agent1_transition.done, 1.0)

    def test_partial_success_without_target_record_or_buffer_does_not_reward_other_agent(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer.partial_success_agent_index = 0
        trainer.partial_success_reward = 20.0
        trainer._active_rollout_id = "rollout0"
        trainer._active_rollout_episode_idx = 0
        trainer._runtime_worker_id = lambda: 0
        trainer._runtime_stage_round_idx = lambda: 1

        current_record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="p",
            response="Action: wait(1)",
            metadata={
                "prompt_ids": torch.tensor([3]),
                "response_ids": torch.tensor([4]),
                "log_prob": 0.0,
                "value": 0.0,
                "entropy": 0.0,
                "reward_breakdown": {"raw": {}},
            },
            reward=0.0,
            done=False,
            timestep=7,
        )

        self.assertIsNone(
            MAPPOTrainer._select_partial_success_record(
                trainer,
                [current_record],
                allow_fallback=False,
            )
        )
        self.assertFalse(MAPPOTrainer._apply_partial_success_reward_to_buffer(trainer))
        MAPPOTrainer._append_policy_records(trainer, [current_record], 0.0, True)

        self.assertEqual(len(trainer.buffer.storage), 1)
        transition = trainer.buffer.storage[0]
        self.assertEqual(transition.agent_index, 1)
        self.assertAlmostEqual(transition.reward, 0.0)
        self.assertAlmostEqual(transition.breakdown_total_reward, 0.0)
        self.assertEqual(transition.done, 1.0)

    def test_shared_partial_success_backfills_missing_teammate_buffer(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer.partial_success_agent_index = 0
        trainer.partial_success_reward = 20.0
        trainer.partial_success_type = "agent_holds"
        trainer.partial_success_shared_terminal_reward = True
        trainer._active_rollout_id = "rollout0"
        trainer._active_rollout_episode_idx = 0
        trainer._runtime_worker_id = lambda: 0
        trainer._runtime_stage_round_idx = lambda: 1

        trainer.buffer.add(
            prompt_ids=torch.tensor([1]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=0.1,
            done=0.0,
            agent_index=1,
            entropy=0.0,
            timestep=4,
            format_reward=0.05,
            validator_reward=0.05,
            breakdown_total_reward=0.1,
            rollout_id="rollout0",
        )
        records = [
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="p",
                response="Action: pickup(bell_pepper,counter)",
                metadata={
                    "prompt_ids": torch.tensor([3]),
                    "response_ids": torch.tensor([4]),
                    "log_prob": 0.0,
                    "value": 0.0,
                    "entropy": 0.0,
                    "action_mode": "planner_main",
                    "reward_breakdown": {
                        "sequence_reward": 2.0,
                        "format_reward": 0.05,
                        "validator_reward": 0.05,
                        "raw": {},
                    },
                },
                reward=2.1,
                done=False,
                timestep=5,
            )
        ]

        self.assertTrue(
            MAPPOTrainer._apply_shared_partial_success_reward(trainer, records, None)
        )
        MAPPOTrainer._append_policy_records(trainer, records, 0.0, True)

        teammate_transition = trainer.buffer.storage[0]
        chef_transition = trainer.buffer.storage[1]
        self.assertEqual(teammate_transition.agent_index, 1)
        self.assertEqual(chef_transition.agent_index, 0)
        self.assertAlmostEqual(teammate_transition.reward, 20.1)
        self.assertAlmostEqual(teammate_transition.partial_success_reward, 20.0)
        self.assertAlmostEqual(chef_transition.reward, 22.1)
        self.assertAlmostEqual(chef_transition.partial_success_reward, 20.0)
        self.assertEqual(teammate_transition.done, 1.0)
        self.assertEqual(chef_transition.done, 1.0)

    def test_shared_partial_success_prefers_executed_source_over_current_record(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer.partial_success_agent_index = 0
        trainer.partial_success_reward = 20.0
        trainer.partial_success_type = "agent_holds"
        trainer.partial_success_shared_terminal_reward = True
        trainer._active_rollout_id = "rollout0"
        trainer._active_rollout_episode_idx = 0
        trainer._runtime_worker_id = lambda: 0
        trainer._runtime_stage_round_idx = lambda: 1

        trainer.buffer.add(
            prompt_ids=torch.tensor([1]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=2.1,
            done=0.0,
            agent_index=0,
            entropy=0.0,
            timestep=5,
            sequence_reward=2.0,
            format_reward=0.05,
            validator_reward=0.05,
            process_reward=2.0,
            breakdown_total_reward=2.1,
            reward_source_key=(0, 5, 0),
            action_mode="planner_main",
            rollout_id="rollout0",
        )
        trainer.buffer.add(
            prompt_ids=torch.tensor([7]),
            response_ids=torch.tensor([8]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=0.1,
            done=0.0,
            agent_index=1,
            entropy=0.0,
            timestep=5,
            sequence_reward=0.0,
            format_reward=0.05,
            validator_reward=0.05,
            process_reward=0.0,
            breakdown_total_reward=0.1,
            action_mode="planner_main",
            rollout_id="rollout0",
        )

        records = [
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="p",
                response="Action: Collab(request(Assistant,place_obj_on_counter()))",
                metadata={
                    "prompt_ids": torch.tensor([3]),
                    "response_ids": torch.tensor([4]),
                    "log_prob": 0.0,
                    "value": 0.0,
                    "entropy": 0.0,
                    "action_mode": "communication",
                    "reward_breakdown": {"raw": {}},
                },
                reward=0.0,
                done=False,
                timestep=6,
            ),
            PolicyCallRecord(
                agent_index=1,
                messages=[],
                prompt="p",
                response="Action: wait(1)",
                metadata={
                    "prompt_ids": torch.tensor([5]),
                    "response_ids": torch.tensor([6]),
                    "log_prob": 0.0,
                    "value": 0.0,
                    "entropy": 0.0,
                    "action_mode": "planner_main",
                    "reward_breakdown": {"raw": {}},
                },
                reward=0.0,
                done=False,
                timestep=6,
            ),
        ]

        applied = MAPPOTrainer._apply_shared_partial_success_reward(
            trainer,
            records,
            [
                {
                    "agent_index": 0,
                    "source_timestamp": 5,
                    "call_index": 0,
                    "action": "pickup(bell_pepper,counter)",
                }
            ],
        )
        MAPPOTrainer._append_policy_records(trainer, records, 0.0, True)

        self.assertTrue(applied)
        chef_action, teammate_action, chef_comm, teammate_wait = trainer.buffer.storage
        self.assertAlmostEqual(chef_action.reward, 22.1)
        self.assertAlmostEqual(chef_action.partial_success_reward, 20.0)
        self.assertAlmostEqual(teammate_action.reward, 20.1)
        self.assertAlmostEqual(teammate_action.partial_success_reward, 20.0)
        self.assertAlmostEqual(chef_comm.reward, 0.0)
        self.assertAlmostEqual(chef_comm.partial_success_reward, 0.0)
        self.assertAlmostEqual(teammate_wait.reward, 0.0)
        self.assertAlmostEqual(teammate_wait.partial_success_reward, 0.0)

    def test_shared_partial_success_prefers_teammate_causal_buffer_over_later_record(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer.partial_success_agent_index = 0
        trainer.partial_success_reward = 20.0
        trainer.partial_success_type = "agent_holds"
        trainer.partial_success_shared_terminal_reward = True
        trainer._active_rollout_id = "rollout0"
        trainer._active_rollout_episode_idx = 0
        trainer._runtime_worker_id = lambda: 0
        trainer._runtime_stage_round_idx = lambda: 1

        trainer.buffer.add(
            prompt_ids=torch.tensor([1]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=2.6,
            done=0.0,
            agent_index=1,
            entropy=0.0,
            timestep=3,
            sequence_reward=2.0,
            format_reward=0.05,
            validator_reward=0.05,
            paired_comm_reward=0.5,
            process_reward=2.0,
            breakdown_total_reward=2.6,
            action_mode="planner_main",
            rollout_id="rollout0",
        )
        trainer.buffer.add(
            prompt_ids=torch.tensor([3]),
            response_ids=torch.tensor([4]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=0.1,
            done=0.0,
            agent_index=1,
            entropy=0.0,
            timestep=5,
            sequence_reward=0.0,
            format_reward=0.05,
            validator_reward=0.05,
            process_reward=0.0,
            breakdown_total_reward=0.1,
            action_mode="planner_main",
            rollout_id="rollout0",
        )

        records = [
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="p",
                response="Action: pickup(bell_pepper,counter)",
                metadata={
                    "prompt_ids": torch.tensor([5]),
                    "response_ids": torch.tensor([6]),
                    "log_prob": 0.0,
                    "value": 0.0,
                    "entropy": 0.0,
                    "action_mode": "planner_main",
                    "reward_breakdown": {
                        "sequence_reward": 2.0,
                        "format_reward": 0.05,
                        "validator_reward": 0.05,
                        "raw": {},
                    },
                },
                reward=2.1,
                done=False,
                timestep=5,
            ),
            PolicyCallRecord(
                agent_index=1,
                messages=[],
                prompt="p",
                response="Action: pickup(bell_pepper,counter)",
                metadata={
                    "prompt_ids": torch.tensor([7]),
                    "response_ids": torch.tensor([8]),
                    "log_prob": 0.0,
                    "value": 0.0,
                    "entropy": 0.0,
                    "action_mode": "planner_main",
                    "reward_breakdown": {
                        "sequence_reward": 0.0,
                        "format_reward": 0.05,
                        "validator_reward": 0.05,
                        "raw": {},
                    },
                },
                reward=0.1,
                done=False,
                timestep=6,
            ),
        ]

        self.assertTrue(
            MAPPOTrainer._apply_shared_partial_success_reward(trainer, records, None)
        )
        MAPPOTrainer._append_policy_records(trainer, records, 0.0, True)

        early_teammate, later_teammate, chef_record, current_teammate = (
            trainer.buffer.storage
        )
        self.assertAlmostEqual(early_teammate.reward, 22.6)
        self.assertAlmostEqual(early_teammate.partial_success_reward, 20.0)
        self.assertEqual(early_teammate.done, 1.0)
        self.assertAlmostEqual(later_teammate.reward, 0.1)
        self.assertAlmostEqual(later_teammate.partial_success_reward, 0.0)
        self.assertAlmostEqual(chef_record.reward, 22.1)
        self.assertAlmostEqual(chef_record.partial_success_reward, 20.0)
        self.assertAlmostEqual(current_teammate.reward, 0.1)
        self.assertAlmostEqual(current_teammate.partial_success_reward, 0.0)

    def test_late_source_update_backfills_existing_delayed_action_transition(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.partial_success_agent_index = 0
        trainer.partial_success_reward = 20.0
        trainer._active_rollout_id = "rollout0"

        trainer.buffer.add(
            prompt_ids=torch.tensor([1]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=0.05,
            done=0.0,
            agent_index=0,
            entropy=0.0,
            timestep=5,
            format_reward=0.05,
            validator_reward=0.0,
            process_reward=0.0,
            sequence_reward=0.0,
            breakdown_total_reward=0.05,
            reward_source_key=(0, 5, 0),
            rollout_id="rollout0",
        )
        trainer.buffer.add(
            prompt_ids=torch.tensor([3]),
            response_ids=torch.tensor([4]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=0.0,
            done=0.0,
            agent_index=0,
            entropy=0.0,
            timestep=5,
            communication_reward=0.0,
            breakdown_total_reward=0.0,
            rollout_id="rollout0",
        )

        process_reward = {
            "source_entries": [
                {
                    "agent_index": 0,
                    "source_timestamp": 5,
                    "timestamp": 7,
                    "call_index": 0,
                    "action": "pickup(bell_pepper,counter)",
                    "sequence_reward": 0.0,
                    "format_reward": 0.05,
                    "validator_reward": 0.05,
                    "communication_reward": 0.0,
                    "repeat_communication_reward": 0.0,
                    "forced_communication_reward": 0.0,
                    "collab_reward": 0.0,
                    "paired_comm_reward": 0.0,
                    "total": 0.1,
                }
            ]
        }

        updated = MAPPOTrainer._apply_process_reward_source_updates(
            trainer,
            process_reward,
        )
        applied = MAPPOTrainer._apply_partial_success_reward_to_sources(
            trainer,
            [
                {
                    "agent_index": 0,
                    "source_timestamp": 5,
                    "call_index": 0,
                    "action": "pickup(bell_pepper,counter)",
                }
            ],
        )

        delayed_action = trainer.buffer.storage[0]
        later_comm = trainer.buffer.storage[1]
        self.assertEqual(updated, 1)
        self.assertTrue(applied)
        self.assertAlmostEqual(delayed_action.reward, 20.1)
        self.assertAlmostEqual(delayed_action.format_reward, 0.05)
        self.assertAlmostEqual(delayed_action.validator_reward, 0.05)
        self.assertAlmostEqual(delayed_action.breakdown_total_reward, 20.1)
        self.assertEqual(delayed_action.done, 1.0)
        self.assertAlmostEqual(later_comm.reward, 0.0)
        self.assertEqual(later_comm.done, 0.0)

    def test_late_source_update_matches_inline_fenced_action_transition(self):
        class FakeTokenizer:
            def decode(self, values, skip_special_tokens=True):
                return (
                    "Think: ok\n"
                    "Recent Goal: act\n"
                    "Action:\n"
                    "```pickup(bell_pepper, counter)```"
                )

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.policy_tokenizer = FakeTokenizer()
        trainer.validator_penalty_value = -0.1
        trainer._active_rollout_id = "rollout0"

        trainer.buffer.add(
            prompt_ids=torch.tensor([1]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=0.0,
            done=0.0,
            agent_index=1,
            entropy=0.0,
            timestep=4,
            reward_source_key=(1, 4, 2),
            action_mode="planner_main",
            rollout_id="rollout0",
        )

        updated = MAPPOTrainer._apply_process_reward_source_updates(
            trainer,
            {
                "source_entries": [
                    {
                        "agent_index": 1,
                        "source_timestamp": 4,
                        "timestamp": 5,
                        "call_index": 2,
                        "call_type": "planner_main",
                        "action": "pickup(bell_pepper,counter)",
                        "sequence_reward": 0.0,
                        "format_reward": 0.05,
                        "validator_reward": 0.05,
                        "communication_reward": 0.0,
                        "repeat_communication_reward": 0.0,
                        "forced_communication_reward": 0.0,
                        "collab_reward": 0.0,
                        "paired_comm_reward": 0.0,
                        "total": 0.1,
                    }
                ]
            },
        )

        transition = trainer.buffer.storage[0]
        self.assertEqual(updated, 1)
        self.assertAlmostEqual(transition.reward, 0.1)
        self.assertAlmostEqual(transition.format_reward, 0.05)
        self.assertAlmostEqual(transition.validator_reward, 0.05)

    def test_source_updates_match_duplicate_source_key_by_decoded_action(self):
        class FakeTokenizer:
            def decode(self, values, skip_special_tokens=True):
                action = "place_obj_on_counter()" if values[0] == 1 else "bake(oven0)"
                return f"Think: ok\nRecent Goal: act\nAction: Collab(request(Assistant,{action}))"

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.policy_tokenizer = FakeTokenizer()
        trainer.validator_penalty_value = -0.1
        trainer._active_rollout_id = "rollout0"
        shared_key = (0, 3, 0)
        for token in (1, 2):
            trainer.buffer.add(
                prompt_ids=torch.tensor([0]),
                response_ids=torch.tensor([token]),
                log_prob=0.0,
                policy_temperature=None,
                value=0.0,
                reward=0.0,
                done=0.0,
                agent_index=0,
                entropy=0.0,
                timestep=3,
                reward_source_key=shared_key,
                action_mode="communication",
                rollout_id="rollout0",
            )

        updated = MAPPOTrainer._apply_process_reward_source_updates(
            trainer,
            {
                "source_entries": [
                    {
                        "timestamp": 3,
                        "source_timestamp": 3,
                        "agent_index": 0,
                        "call_index": 0,
                        "call_type": "communication",
                        "action": "Collab(request(Assistant,place_obj_on_counter()))",
                        "sequence_reward": 0.0,
                        "format_reward": 0.0,
                        "validator_reward": 0.0,
                        "communication_reward": 0.0,
                        "repeat_communication_reward": 0.0,
                        "forced_communication_reward": 0.0,
                        "collab_reward": 0.0,
                        "paired_comm_reward": 0.5,
                        "paired_comm_role": "initiator",
                        "paired_comm_result": "request_helpful",
                        "paired_comm_target": 1,
                        "paired_comm_request_action": "place_obj_on_counter()",
                        "paired_comm_request_helpful": True,
                        "paired_comm_registered_requests": [
                            {"target_agent": 1, "request_action": "place_obj_on_counter()"}
                        ],
                        "penalties": [],
                        "total": 0.5,
                    }
                ]
            },
        )

        self.assertEqual(updated, 1)
        self.assertAlmostEqual(trainer.buffer.storage[0].paired_comm_reward, 0.5)
        self.assertAlmostEqual(trainer.buffer.storage[1].paired_comm_reward, 0.0)
        self.assertEqual(
            trainer.buffer.storage[0].paired_comm_request_action,
            "place_obj_on_counter()",
        )

    def test_source_update_does_not_apply_entry_to_same_key_wrong_action(self):
        class FakeTokenizer:
            def decode(self, values, skip_special_tokens=True):
                return "Think: ok\nRecent Goal: act\nAction: Collab(request(Assistant,bake(oven0)))"

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.policy_tokenizer = FakeTokenizer()
        trainer.validator_penalty_value = -0.1
        trainer._active_rollout_id = "rollout0"
        shared_key = (0, 3, 0)
        trainer.buffer.add(
            prompt_ids=torch.tensor([0]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=0.0,
            done=0.0,
            agent_index=0,
            entropy=0.0,
            timestep=3,
            reward_source_key=shared_key,
            action_mode="communication",
            rollout_id="rollout0",
        )

        updated = MAPPOTrainer._apply_process_reward_source_updates(
            trainer,
            {
                "source_entries": [
                    {
                        "timestamp": 3,
                        "source_timestamp": 3,
                        "agent_index": 0,
                        "call_index": 0,
                        "call_type": "communication",
                        "action": "Collab(request(Assistant,place_obj_on_counter()))",
                        "sequence_reward": 0.0,
                        "format_reward": 0.0,
                        "validator_reward": 0.0,
                        "communication_reward": 0.0,
                        "repeat_communication_reward": 0.0,
                        "forced_communication_reward": 0.0,
                        "collab_reward": 0.0,
                        "paired_comm_reward": 0.5,
                        "paired_comm_role": "initiator",
                        "paired_comm_result": "request_helpful",
                        "paired_comm_target": 1,
                        "paired_comm_request_action": "place_obj_on_counter()",
                        "paired_comm_request_helpful": True,
                        "paired_comm_registered_requests": [
                            {"target_agent": 1, "request_action": "place_obj_on_counter()"}
                        ],
                        "penalties": [],
                        "total": 0.5,
                    }
                ]
            },
        )

        self.assertEqual(updated, 0)
        self.assertAlmostEqual(trainer.buffer.storage[0].paired_comm_reward, 0.0)
        self.assertIsNone(trainer.buffer.storage[0].paired_comm_request_action)

    def test_source_update_does_not_erase_existing_format_penalty(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer._active_rollout_id = "rollout0"

        trainer.buffer.add(
            prompt_ids=torch.tensor([1]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=-0.2,
            done=0.0,
            agent_index=1,
            entropy=0.0,
            timestep=5,
            format_reward=-0.2,
            validator_reward=0.0,
            process_reward=0.0,
            sequence_reward=0.0,
            breakdown_total_reward=-0.2,
            reward_source_key=(1, 5, 0),
            rollout_id="rollout0",
        )
        process_reward = {
            "source_entries": [
                {
                    "timestamp": 6,
                    "source_timestamp": 5,
                    "agent_index": 1,
                    "call_index": 0,
                    "call_type": "planner_main",
                    "action": "pickup(bell_pepper,counter)",
                    "sequence_reward": 2.0,
                    "format_reward": 0.05,
                    "validator_reward": 0.05,
                    "communication_reward": 0.0,
                    "repeat_communication_reward": 0.0,
                    "forced_communication_reward": 0.0,
                    "collab_reward": 0.0,
                    "paired_comm_reward": 0.0,
                    "total": 2.1,
                }
            ]
        }

        updated = MAPPOTrainer._apply_process_reward_source_updates(
            trainer,
            process_reward,
        )

        transition = trainer.buffer.storage[0]
        self.assertEqual(updated, 0)
        self.assertAlmostEqual(transition.reward, -0.2)
        self.assertAlmostEqual(transition.breakdown_total_reward, -0.2)
        self.assertAlmostEqual(transition.format_reward, -0.2)
        self.assertAlmostEqual(transition.validator_reward, 0.0)
        self.assertAlmostEqual(transition.sequence_reward, 0.0)

    def test_source_update_does_not_add_validator_to_communication_transition(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer._active_rollout_id = "rollout0"

        trainer.buffer.add(
            prompt_ids=torch.tensor([1]),
            response_ids=torch.tensor([2]),
            log_prob=0.0,
            policy_temperature=None,
            value=0.0,
            reward=0.05,
            done=0.0,
            agent_index=0,
            entropy=0.0,
            timestep=4,
            format_reward=0.05,
            validator_reward=0.0,
            process_reward=0.0,
            sequence_reward=0.0,
            breakdown_total_reward=0.05,
            reward_source_key=(0, 4, 1),
            action_mode="communication",
            rollout_id="rollout0",
        )
        process_reward = {
            "source_entries": [
                {
                    "timestamp": 4,
                    "source_timestamp": 4,
                    "agent_index": 0,
                    "call_index": 1,
                    "call_type": "planner_main",
                    "action": "Collab(request(Assistant,pickup(bell_pepper,counter)))",
                    "sequence_reward": 0.0,
                    "format_reward": 0.05,
                    "validator_reward": -0.1,
                    "communication_reward": 0.0,
                    "repeat_communication_reward": 0.0,
                    "forced_communication_reward": 0.0,
                    "collab_reward": 0.0,
                    "paired_comm_reward": 0.0,
                    "total": -0.05,
                }
            ]
        }

        updated = MAPPOTrainer._apply_process_reward_source_updates(
            trainer,
            process_reward,
        )

        transition = trainer.buffer.storage[0]
        self.assertEqual(updated, 0)
        self.assertAlmostEqual(transition.reward, 0.05)
        self.assertAlmostEqual(transition.breakdown_total_reward, 0.05)
        self.assertAlmostEqual(transition.format_reward, 0.05)
        self.assertAlmostEqual(transition.validator_reward, 0.0)
        self.assertAlmostEqual(transition.sequence_reward, 0.0)

    def test_append_policy_records_logs_cleaned_reward_as_breakdown_total(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer._active_rollout_id = "rollout0"
        trainer._active_rollout_episode_idx = 0
        trainer._runtime_worker_id = lambda: 0
        trainer._runtime_stage_round_idx = lambda: 1

        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="p",
            response="",
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "entropy": 0.0,
                "action_mode": "planner_main",
                "reward_breakdown": {
                    "sequence_reward": 0.0,
                    "format_reward": 0.0,
                    "validator_reward": 0.0,
                    "paired_comm_reward": -0.1,
                    "raw": {"total": -0.2},
                },
            },
            reward=-0.1,
            done=False,
            timestep=3,
        )

        MAPPOTrainer._append_policy_records(trainer, [record], 0.0, False)

        self.assertEqual(len(trainer.buffer.storage), 1)
        transition = trainer.buffer.storage[0]
        self.assertAlmostEqual(transition.reward, -0.1)
        self.assertAlmostEqual(transition.breakdown_total_reward, -0.1)

    def test_append_policy_records_drops_paired_comm_for_wait_action(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer._active_rollout_id = "rollout0"
        trainer._active_rollout_episode_idx = 0
        trainer._runtime_worker_id = lambda: 0
        trainer._runtime_stage_round_idx = lambda: 1

        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="p",
            response="Think: ok\nRecent Goal: wait\nAction: wait(3)",
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "entropy": 0.0,
                "action_mode": "planner_main",
                "reward_breakdown": {
                    "sequence_reward": 0.0,
                    "format_reward": 0.0,
                    "validator_reward": 0.0,
                    "paired_comm_reward": -0.1,
                    "raw": {"total": -0.1},
                },
            },
            reward=-0.1,
            done=False,
            timestep=3,
        )

        MAPPOTrainer._append_policy_records(trainer, [record], 0.0, False)

        transition = trainer.buffer.storage[0]
        self.assertEqual(transition.action_mode, "wait")
        self.assertAlmostEqual(transition.reward, 0.0)
        self.assertAlmostEqual(transition.breakdown_total_reward, 0.0)
        self.assertAlmostEqual(transition.paired_comm_reward, 0.0)

    def test_append_policy_records_persists_paired_comm_metadata(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer._active_rollout_id = "rollout0"

        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="p",
            response="Think: ok\nRecent Goal: place\nAction: place_obj_on_counter()",
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "entropy": 0.0,
                "reward_breakdown": {
                    "sequence_reward": 2.0,
                    "format_reward": 0.05,
                    "validator_reward": 0.05,
                    "paired_comm_reward": 0.5,
                    "raw": {
                        "paired_comm_reward": 0.5,
                        "paired_comm_role": "responder",
                        "paired_comm_result": "helpful_request_accepted",
                        "paired_comm_target": 1,
                        "paired_comm_request_action": "place_obj_on_counter()",
                        "paired_comm_request_helpful": True,
                        "paired_comm_consumed_requests": [
                            {"request_action": "place_obj_on_counter()"}
                        ],
                    },
                },
            },
            reward=2.6,
            done=False,
            timestep=3,
        )

        MAPPOTrainer._append_policy_records(trainer, [record], 0.0, False)

        transition = trainer.buffer.storage[0]
        self.assertAlmostEqual(transition.reward, 2.6)
        self.assertAlmostEqual(transition.paired_comm_reward, 0.5)
        self.assertEqual(transition.paired_comm_role, "responder")
        self.assertEqual(transition.paired_comm_result, "helpful_request_accepted")
        self.assertEqual(transition.paired_comm_target, 1)
        self.assertEqual(
            transition.paired_comm_request_action,
            "place_obj_on_counter()",
        )
        self.assertTrue(transition.paired_comm_request_helpful)
        self.assertEqual(
            transition.paired_comm_consumed_requests,
            [{"request_action": "place_obj_on_counter()"}],
        )

    def test_append_policy_records_keeps_negative_paired_comm_for_valid_communication(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = TextRolloutBuffer()
        trainer.validator_penalty_value = -0.1
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer.num_agents = 2
        trainer._active_rollout_id = "rollout0"

        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="p",
            response=(
                "Think: ok\n"
                "Recent Goal: ask\n"
                "Action: Collab(request(Assistant, drop(onion)))"
            ),
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "entropy": 0.0,
                "reward_breakdown": {
                    "sequence_reward": 0.0,
                    "format_reward": 0.0,
                    "validator_reward": 0.0,
                    "paired_comm_reward": -0.1,
                    "raw": {
                        "paired_comm_reward": -0.1,
                        "paired_comm_role": "initiator",
                        "paired_comm_result": "request_not_helpful",
                        "paired_comm_target": 1,
                        "paired_comm_request_action": "drop(onion)",
                        "paired_comm_request_helpful": False,
                    },
                },
            },
            reward=-0.1,
            done=False,
            timestep=3,
        )

        MAPPOTrainer._append_policy_records(trainer, [record], 0.0, False)

        transition = trainer.buffer.storage[0]
        self.assertEqual(transition.action_mode, "communication")
        self.assertAlmostEqual(transition.reward, -0.1)
        self.assertAlmostEqual(transition.breakdown_total_reward, -0.1)
        self.assertAlmostEqual(transition.paired_comm_reward, -0.1)
        self.assertEqual(transition.paired_comm_role, "initiator")
        self.assertEqual(transition.paired_comm_result, "request_not_helpful")
        self.assertEqual(transition.paired_comm_target, 1)
        self.assertEqual(transition.paired_comm_request_action, "drop(onion)")
        self.assertFalse(transition.paired_comm_request_helpful)

    def test_sequence_reward_uses_validated_action_once_per_call(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)"],
            refs_agent1=[],
        )

        first = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )

        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=0,
            ml_actions=["pickup(onion)", None],
            state=FakeState(),
        )

        calls = reward_info["per_agent"][0]["calls"]
        self.assertIs(calls[0], first)
        self.assertIs(calls[1], second)
        self.assertEqual(sum(call["sequence_reward"] for call in calls), 1.0)
        self.assertEqual(calls[0]["sequence_reward"], 1.0)
        self.assertEqual(calls[1]["sequence_reward"], 0.0)
        self.assertEqual(tracker.sequence_histories[0], ["pickup(onion)"])

        tracker.register_llm_action(
            agent_index=0,
            timestamp=1,
            action_text="cook(onion)",
            agent_name="Chef",
            call_index=3,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=1,
            ml_actions=["cook(onion)", None],
            state=FakeState(),
        )
        self.assertEqual(
            sum(call["sequence_reward"] for call in reward_info["per_agent"][0]["calls"]),
            0.0,
        )
        self.assertEqual(tracker.sequence_histories[0], ["pickup(onion)"])

    def test_validated_out_of_order_action_does_not_get_sequence_reward(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)"],
            refs_agent1=[],
        )

        early = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        actual = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="cook(onion)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )

        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=0,
            action_text="cook(onion)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=0,
            ml_actions=["cook(onion)", None],
            state=FakeState(),
        )

        calls = reward_info["per_agent"][0]["calls"]
        self.assertEqual(calls[0]["sequence_reward"], 0.0)
        self.assertEqual(calls[1]["sequence_reward"], 0.0)
        self.assertEqual(reward_info["per_agent"][0]["sequence_reward"], 0.0)
        self.assertEqual(tracker.sequence_histories[0], ["cook(onion)"])
        self.assertEqual(early["validator_reward"], 0.0)
        self.assertEqual(actual["validator_reward"], 0.05)

    def test_sequence_reward_goes_to_validated_action_call_only(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)"],
            refs_agent1=[],
        )

        wrong = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="cook(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        right = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )

        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=0,
            ml_actions=["pickup(onion)", None],
            state=FakeState(),
        )

        calls = reward_info["per_agent"][0]["calls"]
        self.assertIs(calls[0], wrong)
        self.assertIs(calls[1], right)
        self.assertEqual(calls[0]["sequence_reward"], 0.0)
        self.assertEqual(calls[1]["sequence_reward"], 1.0)
        self.assertEqual(sum(call["sequence_reward"] for call in calls), 1.0)
        self.assertEqual(tracker.sequence_histories[0], ["pickup(onion)"])

    def test_after_step_ignores_executed_action_source_call_index(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)"],
            refs_agent1=[],
        )

        first = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )

        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=0,
            ml_actions=["pickup(onion)", None],
            state=FakeState(),
        )

        calls = reward_info["per_agent"][0]["calls"]
        self.assertIs(calls[0], first)
        self.assertIs(calls[1], second)
        self.assertEqual(calls[0]["sequence_reward"], 1.0)
        self.assertEqual(calls[1]["sequence_reward"], 0.0)

    def test_unvalidated_later_env_action_does_not_backfill_source_call(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)", "cook(onion)"],
            refs_agent1=[],
        )
        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=7,
            call_type="planner_main",
        )
        tracker.after_step(
            timestep=0,
            ml_actions=[None, None],
            state=FakeState(),
        )

        reward_info = tracker.after_step(
            timestep=1,
            ml_actions=["pickup(onion)", None],
            state=FakeState(),
        )

        self.assertEqual(reward_info["per_agent"][0]["calls"], [])
        self.assertEqual(entry["sequence_reward"], 0.0)

    def test_reclassified_communication_plan_gets_sequence_reward(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
        )
        entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=11,
            call_type="planner_main",
        )

        tracker.mark_llm_action_validated(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=11,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=3,
            ml_actions=[None, "place_obj_on_counter()"],
            state=FakeState(),
        )

        calls = reward_info["per_agent"][1]["calls"]
        self.assertIs(calls[0], entry)
        self.assertEqual(calls[0]["sequence_reward"], 1.0)

    def test_after_step_splits_intermediate_reward_evenly(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=[],
            product_reward=1.0,
        )
        state = FakeState(
            objects={"pot": FakeObject("soup", state=("soup",))},
            players=[FakePlayer(), FakePlayer()],
        )

        reward_info = tracker.after_step(timestep=0, ml_actions=[None, None], state=state)

        self.assertEqual(reward_info["intermediate"]["reward"], 1.0)
        self.assertEqual(reward_info["team_total"], 1.0)
        self.assertEqual(reward_info["per_agent"][0]["total"], 0.5)
        self.assertEqual(reward_info["per_agent"][1]["total"], 0.5)

    def test_tracker_treats_bare_request_as_communication_not_embodied(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=["pickup(onion)"],
        )

        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="request(Assistant,pickup(onion))",
            agent_name="Chef",
            call_index=0,
            call_type="communication",
        )

        self.assertTrue(entry["is_collab"])
        self.assertEqual(entry["sequence_reward"], 0.0)

    def test_tracker_repeated_communication_gets_negative_penalty(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=["pickup(onion)"],
            communication_penalty=0.1,
        )

        first = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="Collab(request(Assistant,pickup(onion,ingredient_dispenser)))",
            agent_name="Chef",
            call_index=0,
            call_type="communication",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="Collab(request(Assistant,pickup(onion,ingredient_dispenser)))",
            agent_name="Chef",
            call_index=1,
            call_type="communication",
        )

        self.assertEqual(first["communication_reward"], 0.0)
        self.assertEqual(second["communication_reward"], -0.1)
        self.assertEqual(second["total"], -0.1)

    def test_assign_call_rewards_matches_planner_entry_without_communication_shift(self):
        session = CollabMainSession.__new__(CollabMainSession)
        records = [
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="p0",
                response="r0",
                metadata={"call_index": 10},
                context={"call_type": "communication"},
                timestep=0,
            ),
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="p1",
                response="r1",
                metadata={"call_index": 11},
                context={"call_type": "planner_main"},
                timestep=0,
            ),
        ]
        process_reward = {
            "per_agent": [
                {
                    "calls": [
                        {
                            "call_type": "communication",
                            "call_index": 10,
                            "sequence_reward": 0.0,
                            "format_reward": 0.0,
                            "validator_reward": -0.5,
                            "total": -0.5,
                        },
                        {
                            "call_type": "planner_main",
                            "call_index": 11,
                            "sequence_reward": 1.0,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "total": 1.0,
                        },
                    ]
                },
                {"calls": []},
            ]
        }

        session._assign_call_rewards(records, process_reward, done_flag=False)

        self.assertEqual(records[0].reward, 0.0)
        self.assertEqual(records[1].reward, 1.0)
        self.assertEqual(
            records[0].metadata["reward_breakdown"]["call_type"],
            "communication",
        )
        self.assertEqual(
            records[0].metadata["reward_breakdown"]["validator_reward"],
            0.0,
        )
        self.assertEqual(
            records[1].metadata["reward_breakdown"]["call_type"],
            "planner_main",
        )

    def test_assign_call_rewards_uses_late_source_entry_for_reclassified_action(self):
        session = CollabMainSession.__new__(CollabMainSession)
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="",
            response=(
                "Think: ok\n"
                "Recent Goal: place the pepper\n"
                "Action: place_obj_on_counter()"
            ),
            metadata={"call_index": 2, "action_mode": "planner_main"},
            context={"call_type": "planner_main"},
            timestep=3,
        )
        stale_entry = {
            "timestamp": 3,
            "source_timestamp": 3,
            "agent_index": 1,
            "call_index": 2,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
            "sequence_reward": 0.0,
            "format_reward": 0.0,
            "validator_reward": 0.0,
            "communication_reward": 0.0,
            "repeat_communication_reward": 0.0,
            "forced_communication_reward": 0.0,
            "collab_reward": 0.0,
            "paired_comm_reward": 0.0,
            "penalties": [],
            "total": 0.0,
        }
        validated_source_entry = {
            "timestamp": 4,
            "source_timestamp": 3,
            "agent_index": 1,
            "call_index": 2,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
            "sequence_reward": 0.0,
            "format_reward": 0.05,
            "validator_reward": 0.05,
            "communication_reward": 0.0,
            "repeat_communication_reward": 0.0,
            "forced_communication_reward": 0.0,
            "collab_reward": 0.0,
            "paired_comm_reward": 0.0,
            "penalties": [],
            "total": 0.1,
        }
        process_reward = {
            "timestamp": 3,
            "per_agent": [
                {"calls": []},
                {"calls": [stale_entry]},
            ],
            "source_entries": [validated_source_entry],
        }

        session._assign_call_rewards([record], process_reward, done_flag=False)

        breakdown = record.metadata["reward_breakdown"]
        self.assertAlmostEqual(record.reward, 0.1)
        self.assertAlmostEqual(breakdown["format_reward"], 0.05)
        self.assertAlmostEqual(breakdown["validator_reward"], 0.05)
        self.assertIs(breakdown["raw"], validated_source_entry)

    def test_assign_call_rewards_drops_intermediate_reward_from_policy_record(self):
        session = CollabMainSession.__new__(CollabMainSession)
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="p",
            response="r",
            context={"call_type": "planner_main"},
            timestep=0,
        )
        process_reward = {
            "per_agent": [
                {
                    "calls": [
                        {
                            "call_type": "planner_main",
                            "sequence_reward": 1.0,
                            "format_reward": -0.2,
                            "validator_reward": -0.1,
                            "total": 1.2,
                        }
                    ],
                    "total": 1.7,
                },
                {"calls": [], "total": 0.5},
            ],
            "intermediate": {"reward": 1.0, "items": ["soup"]},
            "team_total": 2.2,
        }

        session._assign_call_rewards([record], process_reward, done_flag=True)

        # Current record reward ignores the intermediate/product reward share.
        self.assertAlmostEqual(record.reward, 0.7)
        self.assertNotIn("intermediate_reward", record.metadata.get("reward_breakdown", {}))
        self.assertTrue(record.done)

    def test_assign_call_rewards_uses_action_mode_for_embodied_actions(self):
        session = CollabMainSession.__new__(CollabMainSession)
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="p",
            response="r",
            metadata={"action_mode": "planner_main", "call_index": 3},
            context={"call_type": "communication"},
            timestep=0,
        )
        process_reward = {
            "per_agent": [
                {
                    "calls": [
                        {
                            "call_type": "planner_main",
                            "call_index": 3,
                            "sequence_reward": 1.0,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "total": 1.0,
                        }
                    ]
                },
                {"calls": []},
            ]
        }

        session._assign_call_rewards([record], process_reward, done_flag=False)

        self.assertEqual(record.reward, 1.0)
        self.assertEqual(
            record.metadata["reward_breakdown"]["call_type"],
            "planner_main",
        )

    def test_assign_call_rewards_falls_back_to_queue_when_call_index_missing(self):
        session = CollabMainSession.__new__(CollabMainSession)
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="p",
            response="r",
            context={"call_type": "planner_main"},
            timestep=0,
        )
        process_reward = {
            "per_agent": [
                {
                    "calls": [
                        {
                            "call_type": "planner_main",
                            "sequence_reward": 0.4,
                            "format_reward": -0.2,
                            "validator_reward": -0.1,
                            "total": 0.1,
                        }
                    ]
                },
                {"calls": []},
            ]
        }

        session._assign_call_rewards([record], process_reward, done_flag=False)

        self.assertAlmostEqual(record.reward, 0.1)
        self.assertEqual(record.metadata["reward_breakdown"]["call_type"], "planner_main")

    def test_assign_call_rewards_treats_validator_correction_as_rewardable_action(self):
        session = CollabMainSession.__new__(CollabMainSession)
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="p",
            response="r",
            metadata={"call_index": 5},
            context={"call_type": "validator_correction"},
            timestep=0,
        )
        process_reward = {
            "per_agent": [
                {
                    "sequence_reward": 1.0,
                    "calls": [
                        {
                            "call_index": 5,
                            "call_type": "validator_correction",
                            "sequence_reward": 1.0,
                            "format_reward": 0.05,
                            "validator_reward": 0.05,
                            "communication_reward": 0.0,
                            "paired_comm_reward": 0.0,
                            "total": 1.1,
                        }
                    ],
                },
                {"calls": []},
            ]
        }

        session._assign_call_rewards([record], process_reward, done_flag=False)

        self.assertAlmostEqual(record.reward, 1.1)
        self.assertEqual(
            record.metadata["reward_breakdown"]["call_type"],
            "validator_correction",
        )
        self.assertFalse(
            record.metadata["reward_breakdown"].get("missing_reward_entry", False)
        )

    def test_assign_call_rewards_clamps_duplicate_validator_penalty(self):
        session = CollabMainSession.__new__(CollabMainSession)
        session.reward_tracker = SimpleNamespace(validator_penalty_value=-0.1)
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="p",
            response="Action: pickup(bell_pepper, counter)",
            metadata={"call_index": 2, "action": "pickup(bell_pepper, counter)"},
            context={"call_type": "planner_main"},
            timestep=6,
        )
        process_reward = {
            "timestamp": 6,
            "per_agent": [
                {"calls": []},
                {
                    "calls": [
                        {
                            "timestamp": 6,
                            "source_timestamp": 6,
                            "call_index": 2,
                            "call_type": "planner_main",
                            "action": "pickup(bell_pepper, counter)",
                            "sequence_reward": 0.0,
                            "format_reward": 0.05,
                            "validator_reward": -0.2,
                            "communication_reward": 0.0,
                            "collab_reward": 0.0,
                            "paired_comm_reward": -0.1,
                            "total": -0.25,
                        }
                    ]
                },
            ],
        }

        session._assign_call_rewards([record], process_reward, done_flag=False)

        breakdown = record.metadata["reward_breakdown"]
        self.assertAlmostEqual(breakdown["validator_reward"], -0.1)
        self.assertAlmostEqual(breakdown["paired_comm_reward"], 0.0)
        self.assertAlmostEqual(record.reward, -0.05)

    def test_assign_call_rewards_keeps_agent_queues_isolated(self):
        session = CollabMainSession.__new__(CollabMainSession)
        records = [
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="p0",
                response="r0",
                metadata={"call_index": 1},
                context={"call_type": "planner_main"},
                timestep=0,
            ),
            PolicyCallRecord(
                agent_index=1,
                messages=[],
                prompt="p1",
                response="r1",
                metadata={"call_index": 2},
                context={"call_type": "planner_main"},
                timestep=0,
            ),
        ]
        process_reward = {
            "per_agent": [
                {
                    "calls": [
                        {
                            "call_type": "planner_main",
                            "call_index": 1,
                            "sequence_reward": 0.6,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "total": 0.6,
                        }
                    ]
                },
                {
                    "calls": [
                        {
                            "call_type": "planner_main",
                            "call_index": 2,
                            "sequence_reward": 0.2,
                            "format_reward": -0.3,
                            "validator_reward": 0.0,
                            "total": -0.1,
                        }
                    ]
                },
            ]
        }

        session._assign_call_rewards(records, process_reward, done_flag=False)

        self.assertAlmostEqual(records[0].reward, 0.6)
        self.assertAlmostEqual(records[1].reward, -0.1)
        self.assertEqual(records[0].metadata["reward_breakdown"]["call_type"], "planner_main")
        self.assertEqual(records[1].metadata["reward_breakdown"]["call_type"], "planner_main")

    def test_assign_call_rewards_strips_execution_reward_when_action_mode_is_communication(self):
        session = CollabMainSession.__new__(CollabMainSession)
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="p",
            response="r",
            metadata={"action_mode": "communication", "call_index": 8},
            context={"call_type": "planner_main"},
            timestep=0,
        )
        process_reward = {
            "per_agent": [
                {
                    "calls": [
                        {
                            "call_type": "planner_main",
                            "call_index": 8,
                            "sequence_reward": 1.0,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "total": 1.0,
                        }
                    ]
                },
                {"calls": []},
            ]
        }

        session._assign_call_rewards([record], process_reward, done_flag=False)

        self.assertEqual(record.reward, 0.0)
        self.assertEqual(record.metadata["semantic_call_type"], "communication")
        self.assertEqual(record.metadata["reward_breakdown"]["communication_reward"], 0.0)
        self.assertEqual(record.metadata["reward_breakdown"]["sequence_reward"], 0.0)

    def test_assign_call_rewards_marks_only_last_record_done(self):
        session = CollabMainSession.__new__(CollabMainSession)
        records = [
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="p0",
                response="r0",
                metadata={"call_index": 1},
                context={"call_type": "planner_main"},
                timestep=0,
            ),
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="p1",
                response="r1",
                metadata={"call_index": 2},
                context={"call_type": "planner_main"},
                timestep=0,
            ),
        ]
        process_reward = {
            "per_agent": [
                {
                    "calls": [
                        {
                            "call_type": "planner_main",
                            "call_index": 1,
                            "sequence_reward": 0.3,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "total": 0.3,
                        },
                        {
                            "call_type": "planner_main",
                            "call_index": 2,
                            "sequence_reward": 0.7,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "total": 0.7,
                        },
                    ]
                },
                {"calls": []},
            ]
        }

        session._assign_call_rewards(records, process_reward, done_flag=True)

        self.assertFalse(records[0].done)
        self.assertTrue(records[1].done)

    def test_real_log_base_step0_agent0_preserves_per_call_rewards(self):
        self.skipTest("Legacy real-log reward fixture stores pre-validator-immediate reward values.")
        session = CollabMainSession.__new__(CollabMainSession)
        data = self.load_real_log(REAL_LOG_BASE)
        raw_calls = data["content"][0]["content"]["original_log"][0]
        records = self.build_records_from_log_calls(raw_calls)
        process_reward = data["process_rewards"][0]

        session._assign_call_rewards(records, process_reward, done_flag=False)

        self.assertEqual(len(records), 4)
        self.assertAlmostEqual(records[3].reward, -2.0)
        self.assertEqual(records[3].metadata["reward_breakdown"]["call_type"], "planner_main")

    def test_real_log_sft_step0_agent1_keeps_communication_and_planner_rewards_separate(self):
        self.skipTest("Legacy real-log reward fixture stores pre-validator-immediate reward values.")
        session = CollabMainSession.__new__(CollabMainSession)
        data = self.load_real_log(REAL_LOG_SFT)
        raw_calls = data["content"][0]["content"]["original_log"][1]
        records = self.build_records_from_log_calls(raw_calls)
        process_reward = data["process_rewards"][0]

        session._assign_call_rewards(records, process_reward, done_flag=False)

        self.assertEqual(len(records), 3)
        self.assertAlmostEqual(records[1].reward, -0.1554694229112834)
        self.assertEqual(records[1].metadata["reward_breakdown"]["call_type"], "planner_main")
        self.assertEqual(records[2].metadata["reward_breakdown"]["call_type"], "planner_main")

    def test_real_log_semantic_communication_does_not_zero_reward(self):
        self.skipTest("Legacy real-log reward fixture stores pre-validator-immediate reward values.")
        session = CollabMainSession.__new__(CollabMainSession)
        data = self.load_real_log(REAL_LOG_BASE)
        raw_calls = data["content"][4]["content"]["original_log"][0]
        records = self.build_records_from_log_calls(raw_calls, inject_action_mode=True)
        process_reward = data["process_rewards"][4]

        session._assign_call_rewards(records, process_reward, done_flag=False)

        self.assertTrue(any(r.metadata.get("action_mode") == "communication" for r in records))
        self.assertEqual(records[-1].metadata["action_mode"], "communication")
        self.assertEqual(records[-1].metadata["semantic_call_type"], "communication")
        self.assertNotEqual(records[-1].reward, 0.0)

    def test_real_log_gpt4o_step0_agent1_planner_call_keeps_positive_reward(self):
        self.skipTest("Legacy real-log reward fixture stores pre-validator-immediate reward values.")
        session = CollabMainSession.__new__(CollabMainSession)
        data = self.load_real_log(REAL_LOG_GPT4O)
        raw_calls = data["content"][0]["content"]["original_log"][1]
        records = self.build_records_from_log_calls(raw_calls)
        process_reward = data["process_rewards"][0]

        session._assign_call_rewards(records, process_reward, done_flag=False)

        self.assertEqual(records[0].reward, 0.0)
        self.assertAlmostEqual(records[1].reward, 0.6554694229112834)
        self.assertEqual(records[1].metadata["reward_breakdown"]["call_type"], "planner_main")

    def test_all_azure_gpt4o_baked_bell_pepper_logs_preserve_reward_assignment(self):
        self.skipTest("Legacy real-log reward fixture stores pre-validator-immediate reward values.")
        session = CollabMainSession.__new__(CollabMainSession)
        root = REPO_ROOT / "assets/data/batch_results/azure-gpt-4o/json/baked_bell_pepper"
        self.assertTrue(root.exists(), f"Missing azure-gpt-4o fixture dir: {root}")
        checked_calls = 0

        for path in sorted(root.glob("*.json")):
            data = self.load_real_log(path)
            timeline = data.get("content") or []
            rewards = data.get("process_rewards") or []
            self.assertEqual(
                len(timeline),
                len(rewards),
                f"Timeline/process_rewards length mismatch in {path.name}",
            )

            for step_idx, process_reward in enumerate(rewards):
                original_log = (((timeline[step_idx] or {}).get("content") or {}).get("original_log")) or []
                records = []
                for agent_calls in original_log:
                    records.extend(
                        self.build_records_from_log_calls(
                            agent_calls, inject_action_mode=True
                        )
                    )

                session._assign_call_rewards(records, process_reward, done_flag=False)

                for agent_idx, agent_reward in enumerate(process_reward.get("per_agent") or []):
                    reward_calls = agent_reward.get("calls") or []
                    if not reward_calls:
                        continue
                    record_lookup = {
                        int(record.metadata.get("call_index")): record
                        for record in records
                        if record.agent_index == agent_idx
                        and record.metadata.get("call_index") is not None
                    }

                    for reward_entry in reward_calls:
                        call_index = reward_entry.get("call_index")
                        self.assertIn(
                            int(call_index),
                            record_lookup,
                            f"Missing record for {path.name} step={step_idx} agent={agent_idx} call_index={call_index}",
                        )
                        record = record_lookup[int(call_index)]
                        breakdown = record.metadata.get("reward_breakdown") or {}
                        expected_seq = float(
                            reward_entry.get("sequence_reward", 0.0) or 0.0
                        )
                        expected_fmt = float(
                            reward_entry.get("format_reward", 0.0) or 0.0
                        )
                        expected_validator = float(
                            reward_entry.get("validator_reward", 0.0) or 0.0
                        )
                        expected_comm = float(
                            reward_entry.get("communication_reward", 0.0) or 0.0
                        )
                        expected_paired_comm = float(
                            reward_entry.get("paired_comm_reward", 0.0) or 0.0
                        )
                        expected_total = (
                            expected_seq
                            + expected_fmt
                            + expected_validator
                            + expected_comm
                            + expected_paired_comm
                        )

                        with self.subTest(
                            path=path.name,
                            step=step_idx,
                            agent=agent_idx,
                            call_index=call_index,
                            semantic_call_type=record.metadata.get("semantic_call_type"),
                        ):
                            self.assertAlmostEqual(record.reward, expected_total)
                            self.assertAlmostEqual(
                                float(breakdown.get("sequence_reward", 0.0) or 0.0),
                                expected_seq,
                            )
                            self.assertAlmostEqual(
                                float(breakdown.get("format_reward", 0.0) or 0.0),
                                expected_fmt,
                            )
                            self.assertAlmostEqual(
                                float(
                                    breakdown.get("validator_reward", 0.0) or 0.0
                                ),
                                expected_validator,
                            )
                            self.assertAlmostEqual(
                                float(
                                    breakdown.get("communication_reward", 0.0)
                                    or 0.0
                                ),
                                expected_comm,
                            )
                            self.assertAlmostEqual(
                                float(breakdown.get("paired_comm_reward", 0.0) or 0.0),
                                expected_paired_comm,
                            )
                            action_mode = record.metadata.get("action_mode")
                            semantic_call_type = record.metadata.get(
                                "semantic_call_type"
                            ) or record.metadata.get("call_type")
                            if action_mode in {"planner_main", "communication"}:
                                self.assertEqual(
                                    semantic_call_type,
                                    action_mode,
                                )
                            checked_calls += 1

        self.assertGreater(checked_calls, 0)

    def test_batch_sampled_real_cases_match_expected_reward_values(self):
        self.skipTest("Legacy real-log reward fixture stores pre-validator-immediate reward values.")
        session = CollabMainSession.__new__(CollabMainSession)
        sampled_cases = self.collect_real_reward_cases()
        fmt_hits = 0
        validator_hits = 0
        communication_hits = 0

        for case in sampled_cases:
            data = self.load_real_log(case["path"])
            original_log = data["content"][case["step_idx"]]["content"]["original_log"]
            process_reward = data["process_rewards"][case["step_idx"]]
            records = []
            for agent_calls in original_log:
                records.extend(self.build_records_from_log_calls(agent_calls, inject_action_mode=True))

            session._assign_call_rewards(records, process_reward, done_flag=False)
            target = next(
                record
                for record in records
                if record.agent_index == case["agent_idx"]
                and record.metadata.get("call_index") == case["call_index"]
            )
            semantic_call_type = target.metadata.get("semantic_call_type") or target.metadata.get("call_type")

            with self.subTest(
                path=str(case["path"].name),
                step=case["step_idx"],
                agent=case["agent_idx"],
                call_index=case["call_index"],
                category=case["category"],
            ):
                if case["category"] == "communication":
                    communication_hits += 1
                    self.assertEqual(semantic_call_type, "communication")
                    breakdown = target.metadata["reward_breakdown"]
                    self.assertEqual(
                        target.reward,
                        float(breakdown.get("sequence_reward", 0.0))
                        + float(breakdown.get("format_reward", 0.0))
                        + float(breakdown.get("validator_reward", 0.0))
                        + float(breakdown.get("communication_reward", 0.0))
                        + float(breakdown.get("paired_comm_reward", 0.0)),
                    )
                    continue

                reward_entry = next(
                    entry
                    for entry in (process_reward["per_agent"][case["agent_idx"]].get("calls") or [])
                    if entry.get("call_index") == case["call_index"]
                )

                expected_scalar = (
                    case["sequence_reward"] + case["format_reward"] + case["validator_reward"]
                )
                if case["format_reward"] != 0.0:
                    fmt_hits += 1
                if case["validator_reward"] != 0.0:
                    validator_hits += 1

                penalty_list = reward_entry.get("penalties") or []
                format_penalty_sum = sum(
                    float(item.get("value", 0.0) or 0.0)
                    for item in penalty_list
                    if item.get("type") == "format"
                )
                validator_penalty_sum = sum(
                    float(item.get("value", 0.0) or 0.0)
                    for item in penalty_list
                    if item.get("type") == "validator"
                )
                self.assertAlmostEqual(format_penalty_sum, case["format_reward"])
                self.assertAlmostEqual(validator_penalty_sum, case["validator_reward"])

                if semantic_call_type == "planner_main":
                    breakdown = target.metadata["reward_breakdown"]
                    self.assertAlmostEqual(target.reward, expected_scalar)
                    self.assertAlmostEqual(breakdown["sequence_reward"], case["sequence_reward"])
                    self.assertAlmostEqual(breakdown["format_reward"], case["format_reward"])
                    self.assertAlmostEqual(breakdown["validator_reward"], case["validator_reward"])
                    self.assertAlmostEqual(
                        breakdown["sequence_reward"]
                        + breakdown["format_reward"]
                        + breakdown["validator_reward"],
                        target.reward,
                    )
                else:
                    self.assertEqual(semantic_call_type, "communication")
                    breakdown = target.metadata["reward_breakdown"]
                    self.assertAlmostEqual(
                        target.reward,
                        breakdown["sequence_reward"]
                        + breakdown["format_reward"]
                        + breakdown["validator_reward"]
                        + breakdown["communication_reward"]
                        + float(breakdown.get("paired_comm_reward", 0.0)),
                    )

        self.assertGreaterEqual(fmt_hits, 10)
        self.assertGreaterEqual(validator_hits, 10)
        self.assertGreaterEqual(communication_hits, 5)

    def test_positive_sequence_rewards_become_fixed_progress_reward_on_real_logs(self):
        self.skipTest("Legacy real-log reward fixture stores pre-validator-immediate reward values.")
        positive_cases = self.collect_real_positive_sequence_cases(limit=10)

        for case in positive_cases:
            tracker = self.build_real_tracker(case["order"])
            data = self.load_real_log(case["path"])
            observed_sequence_reward = None
            target_seen = False

            for step_idx, reward_step in enumerate(data.get("process_rewards") or []):
                for reward_entry in (reward_step.get("per_agent") or [])[case["agent_idx"]].get("calls") or []:
                    emitted = tracker.register_llm_action(
                        agent_index=case["agent_idx"],
                        timestamp=step_idx,
                        action_text=reward_entry.get("action"),
                        agent_name="Chef" if case["agent_idx"] == 0 else "Assistant",
                        call_index=reward_entry.get("call_index"),
                        call_type=reward_entry.get("call_type"),
                    )
                    if step_idx == case["step_idx"] and reward_entry.get("call_index") == case["call_index"]:
                        observed_sequence_reward = emitted["sequence_reward"]
                        target_seen = True
                        break
                if target_seen:
                    break

            with self.subTest(
                path=str(case["path"].name),
                step=case["step_idx"],
                agent=case["agent_idx"],
                call_index=case["call_index"],
            ):
                self.assertTrue(target_seen)
                self.assertIsNotNone(observed_sequence_reward)
                self.assertAlmostEqual(observed_sequence_reward, 1.0)

    def test_validated_action_assigns_process_reward_in_source_step(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion,counter)", "cook(onion)"],
            refs_agent1=[],
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )
        session = CollabMainSession.__new__(CollabMainSession)

        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=5,
            action_text="pickup(onion,counter)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="",
            response="Action: pickup(onion,counter)",
            metadata={"call_index": 0},
            context={"call_type": "planner_main"},
            timestep=5,
            micro_step=0,
        )
        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=5,
            action_text="pickup(onion,counter)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        session._assign_call_rewards([record], {"timestamp": 5, "per_agent": [{"calls": [entry]}, {"calls": []}]}, False)

        breakdown = record.metadata["reward_breakdown"]
        self.assertAlmostEqual(breakdown["sequence_reward"], 1.0)
        self.assertAlmostEqual(breakdown["validator_reward"], 0.05)
        self.assertAlmostEqual(record.reward, 1.10)

    def test_later_env_action_does_not_backfill_source_call(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )
        session = CollabMainSession.__new__(CollabMainSession)

        entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=0,
            call_type="planner_main",
        )
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="",
            response="Action: place_obj_on_counter()",
            metadata={"call_index": 0},
            context={"call_type": "planner_main"},
            timestep=3,
            micro_step=0,
        )

        session._assign_call_rewards(
            [record],
            {"timestamp": 3, "per_agent": [{"calls": []}, {"calls": [entry]}]},
            False,
        )
        self.assertAlmostEqual(record.reward, 0.05)

        reward_info = tracker.after_step(
            4,
            [None, "place_obj_on_counter()"],
            FakeState(),
        )

        breakdown = record.metadata["reward_breakdown"]
        self.assertAlmostEqual(breakdown["sequence_reward"], 0.0)
        self.assertAlmostEqual(breakdown["validator_reward"], 0.0)
        self.assertAlmostEqual(record.reward, 0.05)
        self.assertEqual(reward_info["per_agent"][1]["calls"], [])

    def test_executed_source_rewards_matching_final_action_entry(self):
        tracker = self.build_tracker(
            order="baked_bell_pepper",
            refs_agent0=[],
            refs_agent1=["pickup(bell_pepper,ingredient_dispenser)"],
            format_penalty=0.2,
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )

        tracker.register_format_error(1, "missing_action")
        stale = tracker.register_llm_action(
            agent_index=1,
            timestamp=5,
            action_text="pickup(bell_pepper,ingredient_dispenser)",
            agent_name="Assistant",
            call_index=0,
            call_type="planner_main",
        )
        final = tracker.register_llm_action(
            agent_index=1,
            timestamp=5,
            action_text="pickup(bell_pepper,ingredient_dispenser)",
            agent_name="Assistant",
            call_index=2,
            call_type="planner_main",
        )

        reward_info = tracker.after_step(
            6,
            [None, "pickup(bell_pepper,ingredient_dispenser)"],
            FakeState(),
            executed_action_sources=[
                {
                    "agent_index": 1,
                    "agent": "Assistant",
                    "timestamp": 6,
                    "source_timestamp": 5,
                    "call_index": 0,
                    "call_type": "planner_main",
                    "action": "pickup(bell_pepper,ingredient_dispenser)",
                    "submitted_action": "pickup(bell_pepper,ingredient_dispenser)",
                }
            ],
        )

        self.assertAlmostEqual(stale["format_reward"], -0.2)
        self.assertAlmostEqual(stale["sequence_reward"], 0.0)
        self.assertAlmostEqual(stale["validator_reward"], 0.0)
        self.assertAlmostEqual(final["format_reward"], 0.05)
        self.assertAlmostEqual(final["sequence_reward"], 1.0)
        self.assertAlmostEqual(final["validator_reward"], 0.05)
        source_entries = reward_info["source_entries"]
        self.assertIn(final, source_entries)

    def test_same_call_index_on_later_timestep_does_not_receive_source_reward(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion,counter)"],
            refs_agent1=[],
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )
        session = CollabMainSession.__new__(CollabMainSession)
        session.reward_tracker = tracker

        old_entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=5,
            action_text="pickup(onion,counter)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        old_record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="",
            response="Action: pickup(onion,counter)",
            metadata={"call_index": 0},
            context={"call_type": "planner_main"},
            timestep=5,
            micro_step=0,
        )
        session._assign_call_rewards([old_record], {"timestamp": 5, "per_agent": [{"calls": [old_entry]}, {"calls": []}]}, False)

        new_record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="",
            response="Action: wait(1)",
            metadata={"call_index": 0},
            context={"call_type": "planner_main"},
            timestep=6,
            micro_step=0,
        )
        reward_info = tracker.after_step(
            6,
            ["pickup(onion,counter)", None],
            FakeState(),
        )
        session._assign_call_rewards([new_record], reward_info, False)

        self.assertEqual(new_record.metadata["reward_breakdown"]["missing_reward_entry"], True)
        self.assertAlmostEqual(old_record.metadata["reward_breakdown"]["sequence_reward"], 0.0)
        self.assertAlmostEqual(old_record.reward, 0.05)

    def test_reclassified_communication_reward_entry_becomes_rewardable(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 16
        agent.reward_tracker = tracker
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = None
        agent._active_action_source = None
        agent.last_executed_action_source = None

        entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=16,
            action_text="Collab(request(place_obj_on_counter()))",
            agent_name="Assistant",
            call_index=1,
            call_type="communication",
        )
        self.assertEqual(entry["call_type"], "communication")

        agent._queue_action_source("place_obj_on_counter()", 1, "planner_main")
        tracker.mark_llm_action_validated(
            agent_index=1,
            timestamp=16,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=1,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            16,
            [None, "place_obj_on_counter()"],
            FakeState(),
        )

        calls = reward_info["per_agent"][1]["calls"]
        self.assertEqual(calls[0]["call_type"], "planner_main")
        self.assertEqual(calls[0]["action"], "place_obj_on_counter()")
        self.assertAlmostEqual(calls[0]["validator_reward"], 0.05)
        self.assertAlmostEqual(calls[0]["sequence_reward"], 1.0)

    def test_ignored_communication_plan_does_not_override_pending_action_source(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 8
        agent.reward_tracker = None
        agent.trace = True
        agent.action_wait_parse = queue.Queue()
        agent.action_wait_parse.put("place_obj_on_counter()")
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 8,
            "source_timestamp": 8,
            "call_index": 1,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
        }
        agent._active_action_source = dict(agent._current_action_source)
        agent._forced_action_override = None
        agent.pending_llm_logs = [{"call_index": 2, "metadata": {}}]
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._set_action_override_from_plan = LLMAgents._set_action_override_from_plan.__get__(agent, LLMAgents)
        agent._replace_pending_action_plan = LLMAgents._replace_pending_action_plan.__get__(agent, LLMAgents)
        agent._clear_pending_action_sources = LLMAgents._clear_pending_action_sources.__get__(agent, LLMAgents)
        agent._relabel_last_llm_call = lambda *args, **kwargs: 2

        new_plan = ["place_obj_on_counter()"]
        communication_plan_call_index = agent._relabel_last_llm_call(
            "planner_main",
            {"reclassified_from": "communication", "produced_actions": list(new_plan)},
        )
        clean_plan = [a for a in new_plan if not agent._is_collab_action(a)]
        if agent.action_wait_parse.qsize() == 0:
            agent._set_action_override_from_plan(clean_plan, communication_plan_call_index)
            agent._replace_pending_action_plan(
                clean_plan,
                communication_plan_call_index,
                "planner_main",
                include_primary=True,
            )

        self.assertEqual(agent.action_wait_parse.qsize(), 1)
        self.assertIsNone(agent._forced_action_override)
        self.assertEqual(agent._current_action_source["call_index"], 1)
        self.assertEqual(agent._active_action_source["call_index"], 1)

    def test_active_embodied_source_is_not_overwritten_by_later_collab_call(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 3
        agent._current_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 3,
            "source_timestamp": 3,
            "micro_step": 2,
            "call_index": 2,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
        }
        agent._active_action_source = dict(agent._current_action_source)
        agent.last_executed_action_source = None
        agent.current_ml_action = 'Collab(seek(Chef, "What should I do next?"))'
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._set_active_action_source_from_current = (
            LLMAgents._set_active_action_source_from_current.__get__(agent, LLMAgents)
        )

        agent._set_active_action_source_from_current()

        self.assertEqual(agent._active_action_source["call_index"], 2)
        self.assertEqual(agent._active_action_source["action"], "place_obj_on_counter()")

    def test_active_embodied_source_switches_to_matching_later_different_call(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 3
        agent._pending_action_sources = queue.Queue()
        agent._active_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 3,
            "source_timestamp": 3,
            "micro_step": 2,
            "call_index": 2,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
        }
        agent._current_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 3,
            "source_timestamp": 3,
            "micro_step": 4,
            "call_index": 4,
            "call_type": "planner_main",
            "action": "pickup(bell_pepper,counter)",
        }
        agent.last_executed_action_source = None
        agent.current_ml_action = "pickup(bell_pepper,counter)"
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._normalize_source_action = LLMAgents._normalize_source_action.__get__(agent, LLMAgents)
        agent._action_source_matches = LLMAgents._action_source_matches.__get__(agent, LLMAgents)
        agent._take_matching_pending_action_source = LLMAgents._take_matching_pending_action_source.__get__(agent, LLMAgents)
        agent._set_active_action_source_from_current = (
            LLMAgents._set_active_action_source_from_current.__get__(agent, LLMAgents)
        )

        agent._set_active_action_source_from_current()

        self.assertEqual(agent._active_action_source["call_index"], 4)
        self.assertEqual(agent._active_action_source["action"], "pickup(bell_pepper,counter)")

    def test_finalize_discards_stale_executed_action_source_mismatch(self):
        from overcooked_ai_py.mdp.actions import Action

        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 10
        agent.current_ml_action = "pickup(bell_pepper,counter)"
        agent.pending_llm_logs = []
        agent.turn_statistics_dict = {"content": {"original_log": [[], []]}}
        stale_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 10,
            "source_timestamp": 8,
            "micro_step": 3,
            "call_index": 3,
            "call_type": "planner_main",
            "action": "wait(3)",
        }
        agent._pending_action_sources = queue.Queue()
        agent._active_action_source = dict(stale_source)
        agent._current_action_source = dict(stale_source)
        agent.last_executed_action_source = None
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._normalize_source_action = LLMAgents._normalize_source_action.__get__(agent, LLMAgents)
        agent._action_source_matches = LLMAgents._action_source_matches.__get__(agent, LLMAgents)
        agent._take_matching_pending_action_source = LLMAgents._take_matching_pending_action_source.__get__(agent, LLMAgents)
        agent._resolve_executed_action_source = LLMAgents._resolve_executed_action_source.__get__(agent, LLMAgents)
        agent._finalize_action_return = LLMAgents._finalize_action_return.__get__(agent, LLMAgents)

        agent._finalize_action_return(Action.INTERACT, "bell_pepper")

        self.assertIsNone(agent.last_executed_action_source)
        self.assertIsNone(agent._active_action_source)
        self.assertIsNone(agent._current_action_source)

    def test_finalize_uses_matching_pending_source_not_stale_active_source(self):
        from overcooked_ai_py.mdp.actions import Action

        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 10
        agent.current_ml_action = "pickup(bell_pepper,counter)"
        agent.pending_llm_logs = []
        agent.turn_statistics_dict = {"content": {"original_log": [[], []]}}
        agent._active_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 10,
            "source_timestamp": 8,
            "micro_step": 3,
            "call_index": 3,
            "call_type": "planner_main",
            "action": "wait(3)",
        }
        agent._current_action_source = dict(agent._active_action_source)
        agent._pending_action_sources = queue.Queue()
        agent._pending_action_sources.put(
            {
                "agent_index": 1,
                "agent": "Assistant",
                "timestamp": 10,
                "source_timestamp": 10,
                "micro_step": 5,
                "call_index": 5,
                "call_type": "planner_main",
                "action": "pickup(bell_pepper,counter)",
            }
        )
        agent.last_executed_action_source = None
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._normalize_source_action = LLMAgents._normalize_source_action.__get__(agent, LLMAgents)
        agent._action_source_matches = LLMAgents._action_source_matches.__get__(agent, LLMAgents)
        agent._take_matching_pending_action_source = LLMAgents._take_matching_pending_action_source.__get__(agent, LLMAgents)
        agent._resolve_executed_action_source = LLMAgents._resolve_executed_action_source.__get__(agent, LLMAgents)
        agent._finalize_action_return = LLMAgents._finalize_action_return.__get__(agent, LLMAgents)

        agent._finalize_action_return(Action.INTERACT, "bell_pepper")

        self.assertIsNotNone(agent.last_executed_action_source)
        self.assertEqual(agent.last_executed_action_source["call_index"], 5)
        self.assertEqual(agent.last_executed_action_source["action"], "pickup(bell_pepper,counter)")

    def test_future_executed_action_source_creates_missing_reward_entry(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 12
        agent.reward_tracker = tracker
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = None
        agent._active_action_source = None
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._make_action_source = LLMAgents._make_action_source.__get__(agent, LLMAgents)
        agent._sync_reward_entry_for_action_source = (
            LLMAgents._sync_reward_entry_for_action_source.__get__(agent, LLMAgents)
        )
        agent._queue_pending_action_source = (
            LLMAgents._queue_pending_action_source.__get__(agent, LLMAgents)
        )

        agent._queue_pending_action_source(
            "place_obj_on_counter()",
            call_index=0,
            call_type="planner_main",
        )

        source_entry = tracker.call_records_by_source[(1, 12, 0)]
        self.assertEqual(source_entry["action"], "place_obj_on_counter()")
        self.assertEqual(source_entry["call_type"], "planner_main")
        self.assertAlmostEqual(source_entry["format_reward"], 0.05)

        queued_source = agent._pending_action_sources.get()
        queued_source["timestamp"] = 14
        reward_info = tracker.after_step(
            14,
            [None, "place_obj_on_counter()"],
            FakeState(),
        )

        calls = reward_info["per_agent"][1]["calls"]
        self.assertEqual(calls, [])
        self.assertAlmostEqual(source_entry["validator_reward"], 0.0)
        self.assertAlmostEqual(source_entry["sequence_reward"], 0.0)
        self.assertAlmostEqual(source_entry["total"], 0.05)

    def test_parameterized_place_on_counter_fails_format_and_is_not_queued(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
            format_penalty=20.0,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.actor = "assistant"
        agent.current_timestep = 3
        agent.reward_tracker = tracker
        agent.pending_llm_logs = [{"call_index": 0, "call_type": "planner_main", "metadata": {}}]
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = None
        agent._active_action_source = None
        agent.time_to_wait = 0
        agent.planner = type("Planner", (), {"add_msg_to_dialog_history": lambda *args, **kwargs: None})()
        agent.mdp = type(
            "MDP",
            (),
            {
                "utensil_list": [],
                "default_ingredients": {"bell_pepper"},
                "all_ingredients": ["bell_pepper"],
                "interact_actions": {},
            },
        )()
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent._forced_action_override = None
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._sanitize_action_text = LLMAgents._sanitize_action_text.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent.parse_response = LLMAgents.parse_response.__get__(agent, LLMAgents)
        agent.parse_params_in_action = LLMAgents.parse_params_in_action.__get__(agent, LLMAgents)
        agent.parse_wait_string = LLMAgents.parse_wait_string.__get__(agent, LLMAgents)
        agent._queue_reward_event = LLMAgents._queue_reward_event.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)
        agent._report_action_format_error = LLMAgents._report_action_format_error.__get__(agent, LLMAgents)
        agent._register_penalty = LLMAgents._register_penalty.__get__(agent, LLMAgents)
        agent._apply_penalty_to_last_entry = LLMAgents._apply_penalty_to_last_entry.__get__(agent, LLMAgents)
        agent._ensure_penalty_reward_entry = LLMAgents._ensure_penalty_reward_entry.__get__(agent, LLMAgents)
        agent._make_action_source = LLMAgents._make_action_source.__get__(agent, LLMAgents)
        agent._queue_action_source = LLMAgents._queue_action_source.__get__(agent, LLMAgents)
        agent._sync_reward_entry_for_action_source = (
            LLMAgents._sync_reward_entry_for_action_source.__get__(agent, LLMAgents)
        )
        agent._clear_pending_action_sources = LLMAgents._clear_pending_action_sources.__get__(agent, LLMAgents)
        agent.parse_ml_action = LLMAgents.parse_ml_action.__get__(agent, LLMAgents)
        agent.parse_ml_action_top = LLMAgents.parse_ml_action_top.__get__(agent, LLMAgents)

        selected = agent.parse_ml_action_top("Action: place_obj_on_counter(bell_pepper)", True, 0, "planner_main")

        self.assertEqual(selected, "wait(1)")
        self.assertIsNone(agent._current_action_source)
        entry = tracker.call_records_by_source[(1, 3, 0)]
        self.assertEqual(entry["action"], "place_obj_on_counter(bell_pepper)")
        self.assertAlmostEqual(entry["format_reward"], -20.0)
        self.assertAlmostEqual(entry["validator_reward"], 0.0)

    def test_multiple_embodied_actions_fail_format_and_are_not_queued(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=[
                "pickup(bell_pepper, counter)",
                "put_obj_in_utensil(oven0)",
            ],
            format_penalty=20.0,
            validator_penalty=10.0,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.actor = "assistant"
        agent.current_timestep = 8
        agent.reward_tracker = tracker
        agent.pending_llm_logs = [{"call_index": 1, "call_type": "planner_main", "metadata": {}}]
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = None
        agent._active_action_source = None
        agent.time_to_wait = 0
        agent.planner = type("Planner", (), {"add_msg_to_dialog_history": lambda *args, **kwargs: None})()
        agent.mdp = type(
            "MDP",
            (),
            {
                "utensil_list": ["oven0"],
                "default_ingredients": {"bell_pepper"},
                "all_ingredients": ["bell_pepper"],
                "interact_actions": {"bake": ["oven0"]},
            },
        )()
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent._forced_action_override = None
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._sanitize_action_text = LLMAgents._sanitize_action_text.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent.parse_response = LLMAgents.parse_response.__get__(agent, LLMAgents)
        agent.parse_params_in_action = LLMAgents.parse_params_in_action.__get__(agent, LLMAgents)
        agent.parse_wait_string = LLMAgents.parse_wait_string.__get__(agent, LLMAgents)
        agent._queue_reward_event = LLMAgents._queue_reward_event.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)
        agent._report_action_format_error = LLMAgents._report_action_format_error.__get__(agent, LLMAgents)
        agent._register_penalty = LLMAgents._register_penalty.__get__(agent, LLMAgents)
        agent._apply_penalty_to_last_entry = LLMAgents._apply_penalty_to_last_entry.__get__(agent, LLMAgents)
        agent._ensure_penalty_reward_entry = LLMAgents._ensure_penalty_reward_entry.__get__(agent, LLMAgents)
        agent._make_action_source = LLMAgents._make_action_source.__get__(agent, LLMAgents)
        agent._queue_action_source = LLMAgents._queue_action_source.__get__(agent, LLMAgents)
        agent._sync_reward_entry_for_action_source = (
            LLMAgents._sync_reward_entry_for_action_source.__get__(agent, LLMAgents)
        )
        agent._clear_pending_action_sources = LLMAgents._clear_pending_action_sources.__get__(agent, LLMAgents)
        agent.parse_ml_action = LLMAgents.parse_ml_action.__get__(agent, LLMAgents)
        agent.parse_ml_action_top = LLMAgents.parse_ml_action_top.__get__(agent, LLMAgents)

        selected = agent.parse_ml_action_top(
            "Action: pickup(bell_pepper, counter); put_obj_in_utensil(oven0)",
            True,
            1,
            "planner_main",
        )

        self.assertEqual(selected, "wait(1)")
        self.assertEqual(agent.action_wait_parse.qsize(), 0)
        self.assertTrue(agent._pending_action_sources.empty())
        self.assertIsNone(agent._current_action_source)
        self.assertIsNone(agent._active_action_source)
        entry = tracker.call_records_by_source[(1, 8, 1)]
        self.assertEqual(entry["action"], "pickup(bell_pepper, counter); put_obj_in_utensil(oven0)")
        self.assertAlmostEqual(entry["format_reward"], -20.0)
        self.assertAlmostEqual(entry["validator_reward"], 0.0)

    def test_multiple_collab_actions_do_not_trigger_multi_embodied_format_error(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._sanitize_action_text = LLMAgents._sanitize_action_text.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._split_action_tokens = LLMAgents._split_action_tokens.__get__(agent, LLMAgents)
        agent._inspect_action_mode = LLMAgents._inspect_action_mode.__get__(agent, LLMAgents)

        info = agent._inspect_action_mode(
            "Action: Collab(request(Assistant,place_obj_on_counter())); Collab(ack(Assistant))"
        )

        self.assertEqual(info["mode"], "communication")
        self.assertEqual(info["embodied_tokens"], [])

    def test_action_source_is_reported_only_on_interact_execution_step(self):
        from overcooked_ai_py.mdp.actions import Action

        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 4
        agent.current_ml_action = "place_obj_on_counter()"
        agent.pending_llm_logs = []
        agent.turn_statistics_dict = {"content": {"original_log": [[], []]}}
        source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 4,
            "source_timestamp": 4,
            "micro_step": 2,
            "call_index": 2,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
        }
        agent._active_action_source = dict(source)
        agent._current_action_source = dict(source)
        agent.last_executed_action_source = None
        agent._finalize_action_return = LLMAgents._finalize_action_return.__get__(agent, LLMAgents)

        agent._finalize_action_return((1, 0), "")
        self.assertIsNone(agent.last_executed_action_source)
        self.assertEqual(agent._active_action_source["action"], "place_obj_on_counter()")

        agent.current_timestep = 5
        agent._finalize_action_return(Action.INTERACT, "")
        self.assertIsNotNone(agent.last_executed_action_source)
        self.assertEqual(agent.last_executed_action_source["source_timestamp"], 4)
        self.assertEqual(agent.last_executed_action_source["call_index"], 2)
        self.assertEqual(agent.last_executed_action_source["submitted_action"], "place_obj_on_counter()")

    def test_completed_action_without_executed_source_clears_stale_active_source(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent._active_action_source = {"action": "place_obj_on_counter()"}
        agent._current_action_source = {"action": "place_obj_on_counter()"}
        agent._locked_action_source = {"action": "place_obj_on_counter()"}
        agent.last_executed_action_source = None
        agent._clear_active_action_source = LLMAgents._clear_active_action_source.__get__(
            agent, LLMAgents
        )
        agent._clear_active_action_source_after_completion = (
            LLMAgents._clear_active_action_source_after_completion.__get__(
                agent, LLMAgents
            )
        )

        agent._clear_active_action_source_after_completion()

        self.assertIsNone(agent._active_action_source)
        self.assertIsNone(agent._current_action_source)
        self.assertIsNone(agent._locked_action_source)

    def test_completed_action_clears_source_even_if_execution_source_exists(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent._active_action_source = {"action": "place_obj_on_counter()"}
        agent._current_action_source = {"action": "place_obj_on_counter()"}
        agent._locked_action_source = {"action": "place_obj_on_counter()"}
        agent.last_executed_action_source = {
            "action": "place_obj_on_counter()",
            "call_index": 2,
        }
        agent._clear_active_action_source = LLMAgents._clear_active_action_source.__get__(
            agent, LLMAgents
        )
        agent._clear_active_action_source_after_completion = (
            LLMAgents._clear_active_action_source_after_completion.__get__(
                agent, LLMAgents
            )
        )

        agent._clear_active_action_source_after_completion()

        self.assertIsNone(agent._active_action_source)
        self.assertIsNone(agent._current_action_source)
        self.assertIsNone(agent._locked_action_source)

    def test_snapshot_preserves_pending_action_source_queue(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.name = "Assistant"
        agent.agent_index = 1
        agent.order = "baked_bell_pepper"
        agent.conversation_history = []
        agent.current_turn_conversation = []
        agent.current_recent_goal_text = "[EMPTY]"
        agent.history_records = []
        agent.conversation_history_timestamp = None
        agent.teammate_ml_actions = []
        agent.pending_collab_reply = False
        agent._communication_turn_counter = 0
        agent._communication_turn_timestamp = None
        agent._collab_ack_consumed = False
        agent.current_ml_action = "wait(1)"
        agent.current_ml_action_steps = 0
        agent._active_action_source = None
        agent._current_action_source = None
        agent.time_to_wait = 0
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent.failed_history = []
        source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 4,
            "source_timestamp": 4,
            "micro_step": 2,
            "call_index": 2,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
        }
        agent.action_wait_parse.put("place_obj_on_counter()")
        agent._pending_action_sources.put(dict(source))
        agent.export_runtime_state = LLMAgents.export_runtime_state.__get__(agent, LLMAgents)
        agent.import_runtime_state = LLMAgents.import_runtime_state.__get__(agent, LLMAgents)
        agent._clear_pending_action_sources = LLMAgents._clear_pending_action_sources.__get__(agent, LLMAgents)

        state = agent.export_runtime_state()

        restored = LLMAgents.__new__(LLMAgents)
        restored.name = "Assistant"
        restored.agent_index = 1
        restored.current_recent_goal_text = "[EMPTY]"
        restored.conversation_history_timestamp = None
        restored.action_wait_parse = queue.Queue()
        restored._pending_action_sources = queue.Queue()
        restored._clear_pending_action_sources = LLMAgents._clear_pending_action_sources.__get__(restored, LLMAgents)
        restored.import_runtime_state = LLMAgents.import_runtime_state.__get__(restored, LLMAgents)
        restored.import_runtime_state(state)

        self.assertEqual(list(restored.action_wait_parse.queue), ["place_obj_on_counter()"])
        self.assertEqual(restored._pending_action_sources.qsize(), 1)
        restored_source = restored._pending_action_sources.get()
        self.assertEqual(restored_source["source_timestamp"], 4)
        self.assertEqual(restored_source["call_index"], 2)

    def test_empty_place_on_counter_params_are_not_counted(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.parse_params_in_action = LLMAgents.parse_params_in_action.__get__(agent, LLMAgents)

        action, params = agent.parse_params_in_action("place_obj_on_counter()")

        self.assertEqual(action, "place_obj_on_counter")
        self.assertEqual(params, [])

    def test_collab_internal_semicolon_is_not_mixed_action(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._sanitize_action_text = LLMAgents._sanitize_action_text.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._split_action_tokens = LLMAgents._split_action_tokens.__get__(agent, LLMAgents)
        agent._inspect_action_mode = LLMAgents._inspect_action_mode.__get__(agent, LLMAgents)

        info = agent._inspect_action_mode(
            "Collab(request(Assistant,pickup(bell_pepper, counter);place_obj_on_counter()))"
        )

        self.assertEqual(info["mode"], "communication")
        self.assertEqual(len(info["tokens"]), 1)

    def test_top_level_multiple_embodied_actions_are_format_error_mode(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._strip_code_fences = LLMAgents._strip_code_fences.__get__(agent, LLMAgents)
        agent._sanitize_action_text = LLMAgents._sanitize_action_text.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._split_action_tokens = LLMAgents._split_action_tokens.__get__(agent, LLMAgents)
        agent._inspect_action_mode = LLMAgents._inspect_action_mode.__get__(agent, LLMAgents)

        info = agent._inspect_action_mode(
            "pickup(bell_pepper, ingredient_dispenser); place_obj_on_counter()"
        )

        self.assertEqual(info["mode"], "multi_embodied")

    def test_markdown_fenced_single_action_is_one_planner_action(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._strip_code_fences = LLMAgents._strip_code_fences.__get__(agent, LLMAgents)
        agent._sanitize_action_text = LLMAgents._sanitize_action_text.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._split_action_tokens = LLMAgents._split_action_tokens.__get__(agent, LLMAgents)
        agent._inspect_action_mode = LLMAgents._inspect_action_mode.__get__(agent, LLMAgents)

        info = agent._inspect_action_mode("Action:\n```\nput_obj_on_counter()\n```")

        self.assertEqual(info["mode"], "planner_main")
        self.assertEqual(info["tokens"], ["put_obj_on_counter()"])

    def test_all_reference_actions_pass_format_parser(self):
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.actor = "assistant"
        agent.reward_tracker = None
        agent.pending_llm_logs = []
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent.mdp = type(
            "MDP",
            (),
            {
                "utensil_list": ["pot0", "chopping_board0", "oven0", "blender0"],
                "default_ingredients": [
                    "bean",
                    "bell_pepper",
                    "broccoli",
                    "carrot",
                    "cauliflower",
                    "chickpea",
                    "corn",
                    "egg",
                    "eggplant",
                    "green_bean",
                    "green_pea",
                    "lentil",
                    "mushroom",
                    "onion",
                    "pea",
                    "potato",
                    "pumpkin",
                    "romaine_lettuce",
                    "spinach",
                    "sweet_potato",
                    "taro",
                    "tomato",
                    "zucchini",
                ],
                "all_ingredients": [
                    "bean",
                    "bell_pepper",
                    "bell_pepper_slices",
                    "boiled_broccoli_slices",
                    "boiled_carrot_slices",
                    "boiled_cauliflower_slices",
                    "boiled_corn_slices",
                    "boiled_egg",
                    "boiled_green_bean_slices",
                    "boiled_mushroom",
                    "boiled_potato_slices",
                    "boiled_romaine_lettuce_slices",
                    "boiled_sweet_potato",
                    "boiled_sweet_potato_slices",
                    "boiled_taro_slices",
                    "boiled_zucchini_slices",
                    "broccoli",
                    "broccoli_slices",
                    "carrot",
                    "carrot_slices",
                    "cauliflower",
                    "cauliflower_slices",
                    "chickpea",
                    "corn",
                    "corn_slices",
                    "egg",
                    "eggplant",
                    "eggplant_slices",
                    "green_bean",
                    "green_bean_slices",
                    "green_pea",
                    "lentil",
                    "mashed_broccoli",
                    "mashed_carrot",
                    "mashed_cauliflower",
                    "mashed_potato",
                    "mashed_romaine_lettuce",
                    "mashed_sweet_potato",
                    "mashed_taro",
                    "mashed_zucchini",
                    "mushroom",
                    "mushroom_slices",
                    "onion",
                    "onion_slices",
                    "pea",
                    "potato",
                    "potato_slices",
                    "pumpkin",
                    "pumpkin_slices",
                    "romaine_lettuce",
                    "romaine_lettuce_slices",
                    "spinach",
                    "sweet_potato",
                    "sweet_potato_slices",
                    "taro",
                    "taro_slices",
                    "tomato",
                    "tomato_slices",
                    "zucchini",
                    "zucchini_slices",
                    "baked_bell_pepper",
                    "baked_bell_pepper_slices",
                    "baked_carrot_slices",
                    "baked_mushroom_slices",
                    "baked_potato_slices",
                    "baked_pumpkin_slices",
                    "baked_sweet_potato",
                ],
                "interact_actions": {
                    "cook": ["pot0"],
                    "cut": ["chopping_board0"],
                    "bake": ["oven0"],
                    "stir": ["blender0"],
                },
            },
        )()
        agent.parse_params_in_action = LLMAgents.parse_params_in_action.__get__(agent, LLMAgents)
        agent.parse_ml_action = LLMAgents.parse_ml_action.__get__(agent, LLMAgents)

        failures = []
        for path in sorted(REAL_REFERENCE_DIR.glob("*_ref.txt")):
            content = json.loads(path.read_text(encoding="utf-8"))
            for ref_name, payload in content.items():
                for agent_key in ("agent_0", "agent_1"):
                    for action_text in payload.get(agent_key, []) or []:
                        valid, parsed_or_error = agent.parse_ml_action(action_text)
                        if not valid:
                            failures.append(
                                f"{path.name}:{ref_name}:{agent_key}:{action_text} -> {parsed_or_error}"
                            )

        self.assertEqual(failures, [])

    def test_reward_event_flush_updates_existing_action_source_entry(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=[],
            format_success_reward=0.05,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 12
        agent.reward_tracker = tracker
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = None
        agent._active_action_source = None
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent.pending_llm_logs = []
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._make_action_source = LLMAgents._make_action_source.__get__(agent, LLMAgents)
        agent._sync_reward_entry_for_action_source = (
            LLMAgents._sync_reward_entry_for_action_source.__get__(agent, LLMAgents)
        )
        agent._queue_pending_action_source = (
            LLMAgents._queue_pending_action_source.__get__(agent, LLMAgents)
        )
        agent._queue_reward_event = LLMAgents._queue_reward_event.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)

        agent._queue_pending_action_source(
            "place_obj_on_counter()",
            call_index=0,
            call_type="planner_main",
        )
        agent._queue_reward_event(
            "place_obj_on_counter()",
            call_index=0,
            call_type="planner_main",
        )
        agent._flush_reward_event()

        self.assertEqual(len(tracker.call_events), 1)
        self.assertEqual(len(tracker.step_call_records[12][1]), 1)
        entry = tracker.call_records_by_source[(1, 12, 0)]
        self.assertIs(agent._last_reward_entry, entry)
        self.assertAlmostEqual(entry["format_reward"], 0.05)
        self.assertAlmostEqual(entry["total"], 0.05)

    def test_format_penalty_does_not_drift_to_later_planner_action(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=[],
            format_penalty=20.0,
            validator_penalty=10.0,
            format_success_reward=0.05,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 0
        agent.name = "Chef"
        agent.current_timestep = 7
        agent.reward_tracker = tracker
        agent.pending_llm_logs = [{"call_index": 1, "call_type": "communication", "metadata": {}}]
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent._annotate_last_log_metadata = lambda **kwargs: None
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._queue_reward_event = LLMAgents._queue_reward_event.__get__(agent, LLMAgents)
        agent._ensure_penalty_reward_entry = LLMAgents._ensure_penalty_reward_entry.__get__(agent, LLMAgents)
        agent._apply_penalty_to_last_entry = LLMAgents._apply_penalty_to_last_entry.__get__(agent, LLMAgents)
        agent._register_penalty = LLMAgents._register_penalty.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)
        agent._handle_format_issues = LLMAgents._handle_format_issues.__get__(agent, LLMAgents)

        agent._handle_format_issues(
            ["missing_action", "missing_recent_goal"],
            "Action: Collab(request(Assistant,wait(1)))",
            "communication",
        )
        self.assertEqual(len(tracker.penalty_queue[0]), 0)

        planner_entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=7,
            action_text="put_obj_in_utensil(oven0)",
            agent_name="Chef",
            call_index=2,
            call_type="planner_main",
        )
        self.assertAlmostEqual(planner_entry["format_reward"], 0.05)
        self.assertAlmostEqual(planner_entry["total"], 0.05)
        prior = tracker.call_records_by_source[(0, 7, 1)]
        self.assertAlmostEqual(prior["format_reward"], -20.0)

    def test_mixed_action_correction_binds_action_source_to_corrected_call(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
            format_penalty=0.2,
            format_success_reward=0.05,
        )

        class Planner:
            def __init__(self):
                self.current_user_message = {"content": "state prompt"}
                self.annotations = []

            def query(self, **_kwargs):
                return (
                    "Think: fixed\n"
                    "Recent Goal: pick onion\n"
                    "Action: pickup(onion,counter)",
                    3,
                )

            def annotate_last_record(self, fields):
                self.annotations.append(dict(fields))
                return True

            def add_msg_to_dialog_history(self, _message):
                return None

        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 0
        agent.name = "Chef"
        agent.current_timestep = 6
        agent.reward_tracker = tracker
        agent.pending_llm_logs = []
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = None
        agent._active_action_source = None
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent.time_to_wait = 0
        agent.proxy = None
        agent.planner = Planner()
        agent.mdp = type(
            "MDP",
            (),
            {
                "utensil_list": [],
                "default_ingredients": {"onion"},
                "all_ingredients": ["onion"],
                "interact_actions": {},
            },
        )()
        agent.turn_statistics_dict = {
            "statistical_data": {
                "error": [
                    {"format_error": {"error_num": 0, "error_message": []}},
                    {"format_error": {"error_num": 0, "error_message": []}},
                ],
                "error_correction": [
                    {"format_correction": {"correction_num": 0, "correction_tokens": []}},
                    {"format_correction": {"correction_num": 0, "correction_tokens": []}},
                ],
            }
        }

        for name in [
            "_strip_action_prefix",
            "_strip_code_fences",
            "_sanitize_action_text",
            "_split_action_tokens",
            "_is_collab_action",
            "_token_has_trailing_text",
            "_inspect_action_mode",
            "_annotate_last_log_metadata",
            "_annotate_last_action_mode",
            "_last_llm_call_index",
            "_set_planner_call_context",
            "_log_llm_call",
            "_queue_reward_event",
            "_ensure_penalty_reward_entry",
            "_apply_penalty_to_last_entry",
            "_register_penalty",
            "_flush_reward_event",
            "_handle_format_issues",
            "_rewrite_mixed_action_response",
            "_ensure_pure_action_response",
            "_make_action_source",
            "_sync_reward_entry_for_action_source",
            "_queue_action_source",
            "_queue_pending_action_source",
            "_clear_pending_action_sources",
            "parse_response",
            "parse_params_in_action",
            "parse_wait_string",
            "parse_ml_action",
            "parse_ml_action_top",
        ]:
            setattr(agent, name, getattr(LLMAgents, name).__get__(agent, LLMAgents))
        agent._has_nonempty_tail_after_fenced_response = LLMAgents._has_nonempty_tail_after_fenced_response
        agent._report_action_format_error = LLMAgents._report_action_format_error.__get__(agent, LLMAgents)

        mixed_response = (
            "Think: mixed\n"
            "Recent Goal: pick onion\n"
            "Action: Collab(request(Assistant,pickup(onion)));wait(1)"
        )
        agent._log_llm_call("planner_main", "state prompt", mixed_response, 2)
        corrected_response, _tokens, action_info = agent._ensure_pure_action_response(
            mixed_response,
            "planner_main",
        )

        self.assertEqual(action_info["corrected_from_call_index"], 0)
        self.assertEqual(action_info["call_index"], 1)
        self.assertEqual(agent.pending_llm_logs[0]["metadata"]["action_mode"], "mixed")
        self.assertIn("mixed_action_types", agent.pending_llm_logs[0]["metadata"]["format_issues"])
        self.assertEqual(agent.pending_llm_logs[1]["metadata"]["action_mode"], "planner_main")
        self.assertAlmostEqual(
            tracker.call_records_by_source[(0, 6, 0)]["format_reward"],
            -0.2,
        )

        selected = agent.parse_ml_action_top(
            corrected_response,
            True,
            source_call_index=action_info["call_index"],
            source_call_type="planner_main",
        )

        self.assertEqual(selected, "pickup(onion,counter)")
        self.assertEqual(agent._current_action_source["call_index"], 1)
        corrected_entry = tracker.call_records_by_source[(0, 6, 1)]
        self.assertEqual(corrected_entry["action"], "pickup(onion,counter)")
        self.assertAlmostEqual(corrected_entry["format_reward"], 0.05)

    def test_fenced_response_tail_action_gets_format_penalty(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=[],
            format_penalty=0.2,
            format_success_reward=0.05,
        )

        class Planner:
            def __init__(self):
                self.current_user_message = {"content": "state prompt"}
                self.annotations = []

            def query(self, **_kwargs):
                return (
                    "Think: fixed\n"
                    "Recent Goal: wait\n"
                    "Action: wait(1)",
                    3,
                )

            def annotate_last_record(self, fields):
                self.annotations.append(dict(fields))
                return True

            def add_msg_to_dialog_history(self, _message):
                return None

        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 0
        agent.name = "Chef"
        agent.current_timestep = 5
        agent.reward_tracker = tracker
        agent.pending_llm_logs = []
        agent.action_wait_parse = queue.Queue()
        agent._pending_action_sources = queue.Queue()
        agent._current_action_source = None
        agent._active_action_source = None
        agent._locked_action_source = None
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent.time_to_wait = 0
        agent.proxy = None
        agent.planner = Planner()
        agent.mdp = type(
            "MDP",
            (),
            {
                "utensil_list": [],
                "default_ingredients": {"bell_pepper"},
                "all_ingredients": ["bell_pepper"],
                "interact_actions": {},
            },
        )()
        agent.turn_statistics_dict = {
            "statistical_data": {
                "error": [
                    {"format_error": {"error_num": 0, "error_message": []}},
                    {"format_error": {"error_num": 0, "error_message": []}},
                ],
                "error_correction": [
                    {"format_correction": {"correction_num": 0, "correction_tokens": []}},
                    {"format_correction": {"correction_num": 0, "correction_tokens": []}},
                ],
            }
        }

        for name in [
            "_append_with_newline",
            "_strip_action_prefix",
            "_strip_code_fences",
            "_sanitize_action_text",
            "_split_action_tokens",
            "_is_collab_action",
            "_token_has_trailing_text",
            "_inspect_action_mode",
            "_annotate_last_log_metadata",
            "_annotate_last_action_mode",
            "_last_llm_call_index",
            "_set_planner_call_context",
            "_log_llm_call",
            "_queue_reward_event",
            "_ensure_penalty_reward_entry",
            "_apply_penalty_to_last_entry",
            "_register_penalty",
            "_flush_reward_event",
            "_handle_format_issues",
            "_rewrite_mixed_action_response",
            "_ensure_pure_action_response",
            "_clear_pending_action_sources",
            "parse_response",
            "parse_params_in_action",
            "parse_wait_string",
            "parse_ml_action",
            "parse_ml_action_top",
        ]:
            setattr(agent, name, getattr(LLMAgents, name).__get__(agent, LLMAgents))
        agent._has_nonempty_tail_after_fenced_response = LLMAgents._has_nonempty_tail_after_fenced_response
        agent._report_action_format_error = LLMAgents._report_action_format_error.__get__(agent, LLMAgents)

        bad_response = (
            "```\n"
            "Think: choose oven\n"
            "Recent Goal: pick and bake\n"
            "Collab(seek(Assistant, \"use the oven\"))\n"
            "```\n"
            "Action: pickup(bell_pepper, counter0);put_obj_in_utensil(oven0)"
        )
        agent._log_llm_call("planner_main", "state prompt", bad_response, 2)

        parsed_action = agent.parse_response(bad_response, "action")
        self.assertIn("pickup(bell_pepper, counter0)", parsed_action)
        self.assertIn("put_obj_in_utensil(oven0)", parsed_action)

        _corrected_response, _tokens, action_info = agent._ensure_pure_action_response(
            bad_response,
            "planner_main",
        )

        self.assertEqual(action_info["corrected_from_call_index"], 0)
        self.assertEqual(action_info["call_index"], 1)
        self.assertEqual(agent.pending_llm_logs[0]["metadata"]["action_mode"], "malformed")
        self.assertIn(
            "malformed_action_types",
            agent.pending_llm_logs[0]["metadata"]["format_issues"],
        )
        entry = tracker.call_records_by_source[(0, 5, 0)]
        self.assertAlmostEqual(entry["format_reward"], -0.2)
        self.assertAlmostEqual(entry["total"], -0.2)

    def test_format_failed_validation_does_not_add_validator_penalty(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=[],
            format_penalty=20.0,
            validator_penalty=10.0,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 6
        agent.current_ml_action = "put_obj_on_counter()"
        agent.reward_tracker = tracker
        agent.pending_llm_logs = [{"call_index": 2, "call_type": "validator_correction", "metadata": {}}]
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent._last_validation_failed_format = True
        agent._annotate_last_log_metadata = lambda **kwargs: None
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._queue_reward_event = LLMAgents._queue_reward_event.__get__(agent, LLMAgents)
        agent._ensure_penalty_reward_entry = LLMAgents._ensure_penalty_reward_entry.__get__(agent, LLMAgents)
        agent._apply_penalty_to_last_entry = LLMAgents._apply_penalty_to_last_entry.__get__(agent, LLMAgents)
        agent._register_penalty = LLMAgents._register_penalty.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)
        agent._record_validation_reward = LLMAgents._record_validation_reward.__get__(agent, LLMAgents)

        agent._ensure_penalty_reward_entry(agent.current_ml_action, "validator_correction")
        agent._register_penalty(
            "format",
            "Please ensure the Action field is a semicolon-separated list of function calls without narration.",
        )
        agent._record_validation_reward("invalid action name")

        entry = tracker.call_records_by_source[(1, 6, 2)]
        self.assertAlmostEqual(entry["format_reward"], -20.0)
        self.assertAlmostEqual(entry["validator_reward"], 0.0)
        self.assertAlmostEqual(entry["total"], -20.0)

    def test_stale_format_failure_flag_does_not_suppress_correction_success(self):
        tracker = self.build_tracker(
            order="baked_bell_pepper",
            refs_agent0=[],
            refs_agent1=["pickup(bell_pepper,counter)"],
            format_success_reward=0.05,
            validator_success_reward=0.05,
            process_progress_reward=2.0,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 5
        agent.current_ml_action = "pickup(bell_pepper,counter)"
        agent.reward_tracker = tracker
        agent.pending_llm_logs = [
            {"call_index": 4, "call_type": "planner_main", "metadata": {}}
        ]
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent._last_validation_failed_format = True
        agent._current_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 5,
            "source_timestamp": 5,
            "call_index": 4,
            "call_type": "planner_main",
            "action": "pickup(bell_pepper,counter)",
        }
        agent._active_action_source = dict(agent._current_action_source)
        agent._locked_action_source = dict(agent._current_action_source)
        agent._annotate_last_log_metadata = lambda **kwargs: None
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._is_wait_action = LLMAgents._is_wait_action.__get__(agent, LLMAgents)
        agent._normalize_source_action = LLMAgents._normalize_source_action.__get__(agent, LLMAgents)
        agent._action_source_matches = LLMAgents._action_source_matches.__get__(agent, LLMAgents)
        agent._reward_entry_for_current_action_source = LLMAgents._reward_entry_for_current_action_source.__get__(agent, LLMAgents)
        agent._ensure_reward_entry_for_current_action_source = LLMAgents._ensure_reward_entry_for_current_action_source.__get__(agent, LLMAgents)
        agent._mark_current_action_validated = LLMAgents._mark_current_action_validated.__get__(agent, LLMAgents)
        agent._handle_validator_failure = LLMAgents._handle_validator_failure.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)
        agent._record_validation_reward = LLMAgents._record_validation_reward.__get__(agent, LLMAgents)

        entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=5,
            action_text="pickup(bell_pepper,counter)",
            agent_name="Assistant",
            call_index=4,
            call_type="planner_main",
        )

        agent._record_validation_reward("success")

        self.assertAlmostEqual(entry["format_reward"], 0.05)
        self.assertAlmostEqual(entry["validator_reward"], 0.05)
        self.assertAlmostEqual(entry["sequence_reward"], 2.0)
        self.assertAlmostEqual(entry["total"], 2.1)

    def test_format_penalty_is_deduplicated_per_llm_call(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
            format_penalty=0.2,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 0
        agent.name = "Chef"
        agent.current_timestep = 3
        agent.reward_tracker = tracker
        agent.pending_llm_logs = [{"call_index": 0, "call_type": "planner_main", "metadata": {}}]
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent._annotate_last_log_metadata = lambda **kwargs: None
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._queue_reward_event = LLMAgents._queue_reward_event.__get__(agent, LLMAgents)
        agent._ensure_penalty_reward_entry = LLMAgents._ensure_penalty_reward_entry.__get__(agent, LLMAgents)
        agent._apply_penalty_to_last_entry = LLMAgents._apply_penalty_to_last_entry.__get__(agent, LLMAgents)
        agent._register_penalty = LLMAgents._register_penalty.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)
        agent._handle_format_issues = LLMAgents._handle_format_issues.__get__(agent, LLMAgents)

        action = "Action:\n```\nput_obj_on_counter()\n```"
        agent._handle_format_issues(
            ["missing_action", "malformed_action_types", "Wrong action name"],
            action,
            "planner_main",
        )

        entry = tracker.call_records_by_source[(0, 3, 0)]
        self.assertAlmostEqual(entry["format_reward"], -0.2)
        self.assertAlmostEqual(entry["total"], -0.2)
        self.assertEqual(
            sum(1 for item in entry["penalties"] if item["type"] == "format"),
            1,
        )

    def test_penalty_queue_deduplicates_repeated_format_penalties(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
            format_penalty=0.2,
        )
        tracker.penalty_queue[0].extend(
            [
                {"type": "format", "detail": "missing_action"},
                {"type": "format", "detail": "malformed_action_types"},
                {"type": "format", "detail": "missing_action"},
            ]
        )

        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="put_obj_on_counter()",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )

        self.assertAlmostEqual(entry["format_reward"], -0.2)
        self.assertAlmostEqual(entry["total"], -0.2)
        self.assertEqual(
            sum(1 for item in entry["penalties"] if item["type"] == "format"),
            1,
        )

    def test_format_penalty_from_failed_retry_does_not_contaminate_valid_retry_update(self):
        tracker = self.build_tracker(
            order="baked_bell_pepper",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
            format_penalty=0.2,
            format_success_reward=0.05,
        )
        entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=6,
            action_text="[EMPTY]",
            agent_name="Assistant",
            call_index=0,
            call_type="communication",
        )
        tracker.register_format_error(1, "previous retry produced malformed output")

        updated = tracker.register_or_update_llm_action(
            agent_index=1,
            timestamp=6,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=0,
            call_type="planner_main",
        )

        self.assertIs(updated, entry)
        self.assertAlmostEqual(updated["format_reward"], 0.05)
        self.assertAlmostEqual(updated["total"], 0.05)
        self.assertEqual(
            [item for item in updated.get("penalties", []) if item["type"] == "format"],
            [],
        )
        self.assertEqual(tracker.penalty_queue[1], [])

    def test_validator_penalty_is_deduplicated_per_existing_llm_call(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
            validator_penalty=0.1,
        )

        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )

        tracker.register_validator_error(0, "first validator failure")
        tracker.register_or_update_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        tracker.register_validator_error(0, "second validator failure")
        tracker.register_or_update_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )

        self.assertIs(entry, tracker.call_records_by_source[(0, 3, 0)])
        self.assertAlmostEqual(entry["validator_reward"], -0.1)
        self.assertAlmostEqual(entry["total"], -0.05)
        self.assertEqual(
            sum(1 for item in entry["penalties"] if item["type"] == "validator"),
            1,
        )

    def test_stale_action_validator_penalty_does_not_contaminate_correction_action(self):
        tracker = self.build_tracker(
            order="baked_bell_pepper",
            refs_agent0=[],
            refs_agent1=["pickup(bell_pepper, ingredient_dispenser)"],
            format_success_reward=0.05,
            validator_success_reward=0.05,
            validator_penalty=0.1,
            process_progress_reward=2.0,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 5
        agent.current_ml_action = "place_obj_on_counter()"
        agent.reward_tracker = tracker
        agent.pending_llm_logs = [{"call_index": 2, "call_type": "planner_main", "metadata": {}}]
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent._current_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 5,
            "source_timestamp": 5,
            "call_index": 2,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
        }
        agent._annotate_last_log_metadata = lambda **kwargs: None
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._queue_reward_event = LLMAgents._queue_reward_event.__get__(agent, LLMAgents)
        agent._ensure_penalty_reward_entry = LLMAgents._ensure_penalty_reward_entry.__get__(agent, LLMAgents)
        agent._apply_penalty_to_last_entry = LLMAgents._apply_penalty_to_last_entry.__get__(agent, LLMAgents)
        agent._reward_entry_for_current_action_source = LLMAgents._reward_entry_for_current_action_source.__get__(agent, LLMAgents)
        agent._register_validator_penalty_for_current_action = LLMAgents._register_validator_penalty_for_current_action.__get__(agent, LLMAgents)
        agent._register_penalty = LLMAgents._register_penalty.__get__(agent, LLMAgents)
        agent._handle_validator_failure = LLMAgents._handle_validator_failure.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)
        agent._record_validation_reward = LLMAgents._record_validation_reward.__get__(agent, LLMAgents)

        stale = tracker.register_llm_action(
            agent_index=1,
            timestamp=5,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=2,
            call_type="planner_main",
        )
        agent._record_validation_reward(
            "There is no object in assistant's hand, so can not place object on counter."
        )
        self.assertAlmostEqual(stale["validator_reward"], -0.1)

        agent.pending_llm_logs = [{"call_index": 3, "call_type": "validator_correction", "metadata": {}}]
        agent._last_reward_entry = None
        agent.current_ml_action = "pickup(bell_pepper, ingredient_dispenser)"
        correction = tracker.register_llm_action(
            agent_index=1,
            timestamp=5,
            action_text=agent.current_ml_action,
            agent_name="Assistant",
            call_index=3,
            call_type="validator_correction",
        )
        agent._current_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 5,
            "source_timestamp": 5,
            "call_index": 3,
            "call_type": "validator_correction",
            "action": agent.current_ml_action,
        }
        agent._last_reward_entry = correction
        agent._record_validation_reward("success")

        self.assertAlmostEqual(correction["format_reward"], 0.05)
        self.assertAlmostEqual(correction["validator_reward"], 0.05)
        self.assertAlmostEqual(correction["sequence_reward"], 2.0)
        self.assertEqual(
            [item for item in correction.get("penalties", []) if item["type"] == "validator"],
            [],
        )
        self.assertAlmostEqual(correction["total"], 2.1)
        self.assertEqual(tracker.penalty_queue[1], [])

    def test_validator_success_binds_to_current_action_source_not_latest_llm_call(self):
        tracker = self.build_tracker(
            order="baked_bell_pepper",
            refs_agent0=[],
            refs_agent1=[
                "pickup(bell_pepper, ingredient_dispenser)",
                "place_obj_on_counter()",
            ],
            format_success_reward=0.05,
            validator_success_reward=0.05,
            process_progress_reward=2.0,
        )
        agent = LLMAgents.__new__(LLMAgents)
        agent.agent_index = 1
        agent.name = "Assistant"
        agent.current_timestep = 4
        agent.current_ml_action = "place_obj_on_counter()"
        agent.reward_tracker = tracker
        agent.pending_llm_logs = [
            {"call_index": 8, "call_type": "communication", "metadata": {}}
        ]
        agent._pending_reward_event = None
        agent._last_reward_entry = None
        agent._last_validation_failed_format = False
        agent._current_action_source = {
            "agent_index": 1,
            "agent": "Assistant",
            "timestamp": 4,
            "source_timestamp": 3,
            "call_index": 6,
            "call_type": "planner_main",
            "action": "place_obj_on_counter()",
        }
        agent._active_action_source = dict(agent._current_action_source)
        agent._locked_action_source = dict(agent._current_action_source)
        agent._annotate_last_log_metadata = lambda **kwargs: None
        agent._strip_action_prefix = LLMAgents._strip_action_prefix.__get__(agent, LLMAgents)
        agent._is_collab_action = LLMAgents._is_collab_action.__get__(agent, LLMAgents)
        agent._is_wait_action = LLMAgents._is_wait_action.__get__(agent, LLMAgents)
        agent._normalize_source_action = LLMAgents._normalize_source_action.__get__(agent, LLMAgents)
        agent._action_source_matches = LLMAgents._action_source_matches.__get__(agent, LLMAgents)
        agent._queue_reward_event = LLMAgents._queue_reward_event.__get__(agent, LLMAgents)
        agent._ensure_penalty_reward_entry = LLMAgents._ensure_penalty_reward_entry.__get__(agent, LLMAgents)
        agent._apply_penalty_to_last_entry = LLMAgents._apply_penalty_to_last_entry.__get__(agent, LLMAgents)
        agent._reward_entry_for_current_action_source = LLMAgents._reward_entry_for_current_action_source.__get__(agent, LLMAgents)
        agent._ensure_reward_entry_for_current_action_source = LLMAgents._ensure_reward_entry_for_current_action_source.__get__(agent, LLMAgents)
        agent._mark_current_action_validated = LLMAgents._mark_current_action_validated.__get__(agent, LLMAgents)
        agent._register_validator_penalty_for_current_action = LLMAgents._register_validator_penalty_for_current_action.__get__(agent, LLMAgents)
        agent._register_penalty = LLMAgents._register_penalty.__get__(agent, LLMAgents)
        agent._handle_validator_failure = LLMAgents._handle_validator_failure.__get__(agent, LLMAgents)
        agent._flush_reward_event = LLMAgents._flush_reward_event.__get__(agent, LLMAgents)
        agent._record_validation_reward = LLMAgents._record_validation_reward.__get__(agent, LLMAgents)

        source_entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=6,
            call_type="planner_main",
        )
        latest_entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=4,
            action_text="Collab(ack(Chef))",
            agent_name="Assistant",
            call_index=8,
            call_type="communication",
        )
        tracker.bootstrap_histories_from_ml_actions(
            [None, "pickup(bell_pepper, ingredient_dispenser)"]
        )

        agent._record_validation_reward("success")

        self.assertAlmostEqual(source_entry["format_reward"], 0.05)
        self.assertAlmostEqual(source_entry["validator_reward"], 0.05)
        self.assertAlmostEqual(source_entry["sequence_reward"], 2.0)
        self.assertAlmostEqual(source_entry["total"], 2.1)
        self.assertAlmostEqual(latest_entry["validator_reward"], 0.0)
        self.assertAlmostEqual(latest_entry["sequence_reward"], 0.0)

    def test_penalty_queue_deduplicates_repeated_validator_penalties(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
            refs_agent1=[],
            validator_penalty=0.1,
        )
        tracker.penalty_queue[0].extend(
            [
                {"type": "validator", "detail": "first validator failure"},
                {"type": "validator", "detail": "second validator failure"},
            ]
        )

        entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )

        self.assertAlmostEqual(entry["validator_reward"], -0.1)
        self.assertAlmostEqual(entry["total"], -0.05)
        self.assertEqual(
            sum(1 for item in entry["penalties"] if item["type"] == "validator"),
            1,
        )

    def test_snapshot_ml_actions_bootstrap_sequence_prefix_before_next_action(self):
        tracker = self.build_tracker(
            order="baked_bell_pepper",
            refs_agent0=[],
            refs_agent1=[
                "pickup(bell_pepper, ingredient_dispenser)",
                "place_obj_on_counter()",
            ],
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )

        restored = tracker.bootstrap_histories_from_ml_actions(
            [None, "pickup(bell_pepper, ingredient_dispenser)"]
        )
        self.assertTrue(restored)
        self.assertEqual(
            tracker.sequence_histories[1],
            ["pickup(bell_pepper,ingredient_dispenser)"],
        )

        tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=1,
            call_type="planner_main",
        )
        tracker.mark_llm_action_validated(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=1,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            3,
            [None, "place_obj_on_counter()"],
            FakeState(),
        )

        call = reward_info["per_agent"][1]["calls"][0]
        self.assertAlmostEqual(call["sequence_reward"], 1.0)
        self.assertAlmostEqual(call["validator_reward"], 0.05)
        self.assertAlmostEqual(call["format_reward"], 0.05)

    def test_reference_overrides_support_two_action_tail_task(self):
        tracker = self.build_tracker(
            order="baked_bell_pepper",
            refs_agent0=[],
            refs_agent1=[],
            process_progress_reward=2.0,
            sequence_metric="lcs",
            reference_overrides={
                "agent_0": ["pickup(bell_pepper,counter)"],
                "agent_1": [
                    "pickup(bell_pepper,ingredient_dispenser)",
                    "place_obj_on_counter()",
                ],
            },
        )

        tracker.import_state(
            {
                "sequence_histories": [
                    [],
                    ["pickup(bell_pepper,ingredient_dispenser)"],
                ],
                "sequence_scores": [0.0, 0.5],
                "collab_sequence_scores": [0.0, 0.5],
                "observed_targets": ["bell_pepper"],
                "penalty_queue": [[], []],
            }
        )
        self.assertEqual(
            tracker.references[0],
            [["pickup(bell_pepper,counter)"]],
        )
        self.assertEqual(
            tracker.references[1],
            [["pickup(bell_pepper,ingredient_dispenser)", "place_obj_on_counter()"]],
        )

        tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=1,
            call_type="planner_main",
        )
        tracker.mark_llm_action_validated(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=1,
            call_type="planner_main",
        )
        assistant_reward = tracker.after_step(
            3,
            [None, "place_obj_on_counter()"],
            FakeState(),
        )
        self.assertAlmostEqual(
            assistant_reward["per_agent"][1]["sequence_reward"],
            2.0,
        )

        tracker.register_llm_action(
            agent_index=0,
            timestamp=5,
            action_text="pickup(bell_pepper,counter)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        tracker.mark_llm_action_validated(
            agent_index=0,
            timestamp=5,
            action_text="pickup(bell_pepper,counter)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        chef_reward = tracker.after_step(
            5,
            ["pickup(bell_pepper,counter)", None],
            FakeState(),
        )
        self.assertAlmostEqual(chef_reward["per_agent"][0]["sequence_reward"], 2.0)

    def test_agent_holds_partial_success_detects_chef_pickup_tail_task(self):
        trainer = object.__new__(MAPPOTrainer)
        trainer.partial_success_enabled = True
        trainer.partial_success_type = "agent_holds"
        trainer.partial_success_agent_index = 0
        trainer.partial_success_item = "bell_pepper"
        trainer.session = type(
            "FakeSession",
            (),
            {
                "env": type(
                    "FakeEnv",
                    (),
                    {
                        "state": FakeState(
                            players=[
                                FakePlayer(FakeObject("bell_pepper")),
                                FakePlayer(),
                            ]
                        )
                    },
                )()
            },
        )()

        self.assertTrue(MAPPOTrainer._partial_success_reached(trainer))

        trainer.session.env.state = FakeState(
            players=[FakePlayer(), FakePlayer(FakeObject("bell_pepper"))]
        )
        self.assertFalse(MAPPOTrainer._partial_success_reached(trainer))

    def test_validated_action_uses_matching_action_when_call_index_is_reused(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(bell_pepper,counter)", "put_obj_in_utensil(oven0)"],
            refs_agent1=[],
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )
        pickup_entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=7,
            action_text="pickup(bell_pepper,counter)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        tracker.call_records_by_source[(0, 7, 0)] = tracker.register_llm_action(
            agent_index=0,
            timestamp=7,
            action_text="put_obj_in_utensil(oven0)",
            agent_name="Chef",
            call_index=0,
            call_type="planner_main",
        )
        tracker.mark_llm_action_entry_validated(pickup_entry)

        reward_info = tracker.after_step(
            7,
            ["pickup(bell_pepper,counter)", None],
            FakeState(),
        )

        call = reward_info["per_agent"][0]["calls"][0]
        self.assertIs(call, pickup_entry)
        self.assertEqual(call["action"], "pickup(bell_pepper,counter)")
        self.assertAlmostEqual(call["sequence_reward"], 1.0)
        self.assertAlmostEqual(call["validator_reward"], 0.05)

    def test_critic_prompt_is_english_and_contains_rule_sections(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.agent_roles = {0: "Chef", 1: "Assistant"}
        trainer.critic_role_prompt = MAPPOTrainer._load_critic_role_prompt(trainer)

        prompt = MAPPOTrainer._build_critic_prompt(
            trainer,
            agent_index=0,
            messages=[
                {
                    "role": "user",
                    "content": "Current Observation:\nCounter has onion.\n\nAction: pickup(onion,counter)",
                }
            ],
            context={
                "call_type": "planner_main",
                "observation_excerpt": "Current Observation:\nCounter has onion.",
                "global_observation": {
                    "timestep": 3,
                    "orders": ["baked_bell_pepper"],
                    "grid": "mock-grid",
                    "pot_states": {},
                    "counters": {"counter0": "onion"},
                    "players": {
                        0: {"position": (1, 1), "orientation": "N", "object": None},
                        1: {"position": (2, 2), "orientation": "S", "object": "dish"},
                    },
                },
                "shared_agent_traces": {
                    1: {
                        "call_type": "communication",
                        "observation_excerpt": "Assistant sees a free counter.",
                        "response_excerpt": "Think: ask chef\nRecent Goal: support chef\nAction: Collab(seek(Chef, what next))",
                    }
                },
            },
            output_text="Think: pick onion\nRecent Goal: start cooking\nAction: pickup(onion,counter)",
        )

        self.assertIn("[Global Observation]", prompt)
        self.assertIn("[Game Rules]", prompt)
        self.assertIn("[Chef Action Space]", prompt)
        self.assertIn("[Assistant Action Space]", prompt)
        self.assertIn("You are a centralized critic", prompt)
        self.assertIn("Please reply in English.", prompt)
        self.assertIn("def pickup(obj, place):", prompt)
        self.assertIn("def cook(pot_name):", prompt)
        self.assertIn("def stir(blender_name):", prompt)
        self.assertIn("acting_agent: Chef", prompt)
        self.assertIn("teammate_agent: Assistant", prompt)

    def test_execution_reward_does_not_drift_to_later_communication_call(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
        )
        executed = tracker.register_llm_action(
            agent_index=1,
            timestamp=8,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=2,
            call_type="planner_main",
        )
        tracker.mark_llm_action_validated(
            agent_index=1,
            timestamp=8,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=2,
            call_type="planner_main",
        )
        communication = tracker.register_llm_action(
            agent_index=1,
            timestamp=8,
            action_text="Collab(ack(Chef))",
            agent_name="Assistant",
            call_index=3,
            call_type="communication",
        )

        reward_info = tracker.after_step(
            timestep=8,
            ml_actions=[None, "place_obj_on_counter()"],
            state=FakeState(),
        )

        calls = reward_info["per_agent"][1]["calls"]
        self.assertEqual(calls[0], executed)
        self.assertEqual(calls[1], communication)
        self.assertAlmostEqual(executed["validator_reward"], 0.05)
        self.assertAlmostEqual(executed["sequence_reward"], 1.0)
        self.assertAlmostEqual(communication["validator_reward"], 0.0)
        self.assertAlmostEqual(communication["sequence_reward"], 0.0)

    def test_collab_reward_goes_to_requester_and_executor_when_request_is_executed(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
            collab_reward_enabled=True,
            sequence_metric="lcs",
            sequence_weight=0.2,
            format_success_reward=0.05,
            validator_success_reward=0.05,
        )
        request = tracker.register_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="Collab(request(Assistant,place_obj_on_counter()))",
            agent_name="Chef",
            call_index=0,
            call_type="communication",
        )
        self.assertAlmostEqual(request["collab_reward"], 0.2)

        executed = tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=1,
            call_type="planner_main",
        )
        tracker.mark_llm_action_validated(
            agent_index=1,
            timestamp=3,
            action_text="place_obj_on_counter()",
            agent_name="Assistant",
            call_index=1,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=3,
            ml_actions=[None, "place_obj_on_counter()"],
            state=FakeState(),
        )

        assistant_call = reward_info["per_agent"][1]["calls"][0]
        self.assertEqual(assistant_call, executed)
        self.assertAlmostEqual(assistant_call["sequence_reward"], 1.0)
        self.assertAlmostEqual(assistant_call["validator_reward"], 0.05)
        self.assertAlmostEqual(assistant_call["collab_reward"], 0.2)
        self.assertEqual(
            assistant_call["collab_execution_source"]["initiator_agent"],
            0,
        )

    def test_collab_execution_reward_requires_executed_sequence_reward(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
            collab_reward_enabled=True,
            sequence_metric="lcs",
            sequence_weight=0.2,
        )
        tracker.register_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="Collab(request(Assistant,place_obj_on_counter()))",
            agent_name="Chef",
            call_index=0,
            call_type="communication",
        )
        executed = tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="pickup(onion,counter)",
            agent_name="Assistant",
            call_index=1,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            timestep=3,
            ml_actions=[None, "pickup(onion,counter)"],
            state=FakeState(),
        )

        assistant_call = reward_info["per_agent"][1]["calls"][0]
        self.assertEqual(assistant_call, executed)
        self.assertAlmostEqual(assistant_call["sequence_reward"], 0.0)
        self.assertAlmostEqual(assistant_call["collab_reward"], 0.0)

    def test_paired_comm_request_reply_does_not_close_pending_request(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion,counter)"],
            refs_agent1=["place_obj_on_counter()"],
            paired_comm_reward_enabled=True,
            paired_comm_request_positive_reward=0.2,
            paired_comm_request_negative_reward=-0.1,
            paired_comm_response_positive_reward=0.2,
            paired_comm_response_negative_reward=-0.1,
            paired_comm_deny_reward=0.2,
        )
        first = tracker.register_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="Collab(request(Assistant,place_obj_on_counter()))",
            agent_name="Chef",
            call_index=0,
            call_type="communication",
        )
        self.assertAlmostEqual(first["paired_comm_reward"], 0.2)

        second = tracker.register_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="Collab(request(Chef,wrong_action()))",
            agent_name="Assistant",
            call_index=1,
            call_type="communication",
        )

        self.assertEqual(second["paired_comm_role"], "initiator")
        self.assertEqual(second["paired_comm_result"], "request_not_helpful")
        self.assertAlmostEqual(second["paired_comm_reward"], -0.1)
        self.assertEqual(len(tracker.pending_paired_comm_requests[1]), 1)
        self.assertEqual(
            tracker.pending_paired_comm_requests[1][0]["request_action"],
            "place_obj_on_counter()",
        )
        self.assertEqual(len(tracker.pending_paired_comm_requests[0]), 1)

    def test_session_strips_execution_reward_from_semantic_communication_record(self):
        session = CollabMainSession.__new__(CollabMainSession)
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="",
            response="Think: ok\nRecent Goal: help\nAction: Collab(ack(Chef))",
            metadata={"call_index": 3, "action_mode": "communication"},
            context={"call_type": "planner_main"},
            timestep=3,
        )
        process_reward = {
            "timestamp": 3,
            "per_agent": [
                {"sequence_reward": 0.0, "calls": []},
                {
                    "sequence_reward": 0.0,
                    "calls": [
                        {
                            "timestamp": 3,
                            "source_timestamp": 3,
                            "agent_index": 1,
                            "call_index": 3,
                            "call_type": "planner_main",
                            "action": "Collab(ack(Chef))",
                            "sequence_reward": 0.0,
                            "format_reward": 0.05,
                            "validator_reward": 0.05,
                            "communication_reward": 0.0,
                            "repeat_communication_reward": 0.0,
                            "forced_communication_reward": 0.0,
                            "paired_comm_reward": 0.0,
                            "penalties": [],
                            "total": 0.1,
                        }
                    ],
                },
            ],
        }

        CollabMainSession._assign_call_rewards(session, [record], process_reward, False)

        breakdown = record.metadata["reward_breakdown"]
        self.assertEqual(record.reward, 0.0)
        self.assertEqual(breakdown["format_reward"], 0.0)
        self.assertEqual(breakdown["validator_reward"], 0.0)
        self.assertEqual(breakdown["sequence_reward"], 0.0)

    def test_assign_call_rewards_matches_duplicate_source_key_by_action(self):
        session = CollabMainSession.__new__(CollabMainSession)
        request_record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="",
            response="Think: ok\nRecent Goal: ask\nAction: Collab(request(Assistant,place_obj_on_counter()))",
            metadata={"call_index": 0, "action_mode": "communication"},
            context={"call_type": "communication"},
            timestep=3,
        )
        later_record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="",
            response="Think: ok\nRecent Goal: ask\nAction: Collab(request(Assistant,bake(oven0)))",
            metadata={"call_index": 0, "action_mode": "communication"},
            context={"call_type": "communication"},
            timestep=3,
        )
        process_reward = {
            "timestamp": 3,
            "per_agent": [
                {
                    "sequence_reward": 0.0,
                    "calls": [
                        {
                            "timestamp": 3,
                            "source_timestamp": 3,
                            "agent_index": 0,
                            "call_index": 0,
                            "call_type": "communication",
                            "action": "Collab(request(Assistant,place_obj_on_counter()))",
                            "sequence_reward": 0.0,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "communication_reward": 0.0,
                            "repeat_communication_reward": 0.0,
                            "forced_communication_reward": 0.0,
                            "paired_comm_reward": 0.5,
                            "paired_comm_role": "initiator",
                            "paired_comm_result": "request_helpful",
                            "paired_comm_target": 1,
                            "paired_comm_request_action": "place_obj_on_counter()",
                            "paired_comm_request_helpful": True,
                            "paired_comm_registered_requests": [
                                {"target_agent": 1, "request_action": "place_obj_on_counter()"}
                            ],
                            "penalties": [],
                            "total": 0.5,
                        },
                        {
                            "timestamp": 3,
                            "source_timestamp": 3,
                            "agent_index": 0,
                            "call_index": 0,
                            "call_type": "communication",
                            "action": "Collab(request(Assistant,bake(oven0)))",
                            "sequence_reward": 0.0,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "communication_reward": 0.0,
                            "repeat_communication_reward": 0.0,
                            "forced_communication_reward": 0.0,
                            "paired_comm_reward": 0.0,
                            "penalties": [],
                            "total": 0.0,
                        },
                    ],
                },
                {"sequence_reward": 0.0, "calls": []},
            ],
        }

        CollabMainSession._assign_call_rewards(
            session, [request_record, later_record], process_reward, False
        )

        self.assertAlmostEqual(request_record.reward, 0.5)
        self.assertAlmostEqual(later_record.reward, 0.0)
        request_breakdown = request_record.metadata["reward_breakdown"]
        self.assertAlmostEqual(request_breakdown["paired_comm_reward"], 0.5)
        self.assertEqual(
            request_breakdown["raw"]["paired_comm_result"],
            "request_helpful",
        )

    def test_assign_call_rewards_does_not_steal_duplicate_source_key_request_reward(self):
        session = CollabMainSession.__new__(CollabMainSession)
        records = [
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="",
                response=(
                    "Think: ok\nRecent Goal: ask\n"
                    "Action: Collab(request(Assistant,place_obj_on_counter());"
                    "request(Assistant,pickup(bell_pepper, ingredient_dispenser)))"
                ),
                metadata={"call_index": 0, "action_mode": "communication"},
                context={"call_type": "communication"},
                timestep=3,
            ),
            PolicyCallRecord(
                agent_index=0,
                messages=[],
                prompt="",
                response=(
                    "Think: ok\nRecent Goal: ask\n"
                    "Action: Collab(request(Assistant,bake(oven0)))"
                ),
                metadata={"call_index": 0, "action_mode": "communication"},
                context={"call_type": "communication"},
                timestep=3,
            ),
            PolicyCallRecord(
                agent_index=1,
                messages=[],
                prompt="",
                response=(
                    "Think: ok\nRecent Goal: reply\n"
                    "Action: Collab(Ack(Chef, \"I have placed the bell_pepper on the counter.\"))"
                ),
                metadata={"call_index": 1, "action_mode": "communication"},
                context={"call_type": "communication"},
                timestep=3,
            ),
        ]
        process_reward = {
            "timestamp": 3,
            "per_agent": [
                {
                    "sequence_reward": 0.0,
                    "calls": [
                        {
                            "timestamp": 3,
                            "source_timestamp": 3,
                            "agent_index": 0,
                            "call_index": 0,
                            "call_type": "communication",
                            "action": "Collab(request(Assistant,place_obj_on_counter());request(Assistant,pickup(bell_pepper, ingredient_dispenser)))",
                            "sequence_reward": 0.0,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "communication_reward": 0.0,
                            "repeat_communication_reward": 0.0,
                            "forced_communication_reward": 0.0,
                            "paired_comm_reward": 0.4,
                            "paired_comm_role": "initiator",
                            "paired_comm_result": "multi_request",
                            "paired_comm_target": 1,
                            "paired_comm_request_action": "pickup(bell_pepper,ingredient_dispenser)",
                            "paired_comm_request_helpful": False,
                            "paired_comm_registered_requests": [
                                {
                                    "request_id": 1,
                                    "timestamp": 3,
                                    "initiator_agent": 0,
                                    "responder_agent": 1,
                                    "request_action": "place_obj_on_counter()",
                                    "request_helpful": True,
                                },
                                {
                                    "request_id": 2,
                                    "timestamp": 3,
                                    "initiator_agent": 0,
                                    "responder_agent": 1,
                                    "request_action": "pickup(bell_pepper,ingredient_dispenser)",
                                    "request_helpful": False,
                                },
                            ],
                            "penalties": [],
                            "total": 0.4,
                        },
                        {
                            "timestamp": 3,
                            "source_timestamp": 3,
                            "agent_index": 0,
                            "call_index": 0,
                            "call_type": "communication",
                            "action": "Collab(request(Assistant,bake(oven0)))",
                            "sequence_reward": 0.0,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "communication_reward": 0.0,
                            "repeat_communication_reward": 0.0,
                            "forced_communication_reward": 0.0,
                            "paired_comm_reward": 0.0,
                            "penalties": [],
                            "total": 0.0,
                        },
                    ],
                },
                {
                    "sequence_reward": 0.0,
                    "calls": [
                        {
                            "timestamp": 3,
                            "source_timestamp": 3,
                            "agent_index": 1,
                            "call_index": 1,
                            "call_type": "communication",
                            "action": "Collab(Ack(Chef, \"I have placed the bell_pepper on the counter.\"))",
                            "sequence_reward": 0.0,
                            "format_reward": 0.0,
                            "validator_reward": 0.0,
                            "communication_reward": 0.0,
                            "repeat_communication_reward": 0.0,
                            "forced_communication_reward": 0.0,
                            "paired_comm_reward": 0.5,
                            "paired_comm_role": "responder",
                            "paired_comm_result": "helpful_request_accepted",
                            "paired_comm_target": 0,
                            "paired_comm_request_action": "place_obj_on_counter()",
                            "paired_comm_request_helpful": True,
                            "paired_comm_consumed_requests": [
                                {
                                    "request_id": 1,
                                    "timestamp": 3,
                                    "initiator_agent": 0,
                                    "responder_agent": 1,
                                    "request_action": "place_obj_on_counter()",
                                    "request_helpful": True,
                                }
                            ],
                            "penalties": [],
                            "total": 0.5,
                        }
                    ],
                },
            ],
        }

        CollabMainSession._assign_call_rewards(session, records, process_reward, False)

        request_breakdown = records[0].metadata["reward_breakdown"]
        later_breakdown = records[1].metadata["reward_breakdown"]
        ack_breakdown = records[2].metadata["reward_breakdown"]
        self.assertAlmostEqual(records[0].reward, 0.4)
        self.assertAlmostEqual(request_breakdown["paired_comm_reward"], 0.4)
        self.assertEqual(request_breakdown["raw"]["paired_comm_result"], "multi_request")
        self.assertAlmostEqual(records[1].reward, 0.0)
        self.assertAlmostEqual(later_breakdown["paired_comm_reward"], 0.0)
        self.assertAlmostEqual(records[2].reward, 0.5)
        self.assertEqual(
            ack_breakdown["raw"]["paired_comm_result"],
            "helpful_request_accepted",
        )

    def test_duplicate_source_key_update_does_not_clear_consumed_request_reward(self):
        tracker = self.build_tracker(
            order="baked_bell_pepper",
            refs_agent0=[],
            refs_agent1=["place_obj_on_counter()"],
            paired_comm_reward_enabled=True,
            paired_comm_request_positive_reward=0.5,
            paired_comm_request_negative_reward=0.1,
            paired_comm_response_positive_reward=0.5,
            paired_comm_response_negative_reward=0.1,
            paired_comm_deny_reward=1.0,
        )
        request = tracker.register_or_update_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="Collab(request(Assistant,place_obj_on_counter()))",
            agent_name="Chef",
            call_index=1,
            call_type="communication",
        )
        response = tracker.register_or_update_llm_action(
            agent_index=1,
            timestamp=3,
            action_text="Collab(ack(Chef))",
            agent_name="Assistant",
            call_index=0,
            call_type="communication",
        )
        later = tracker.register_or_update_llm_action(
            agent_index=0,
            timestamp=3,
            action_text="pickup(bell_pepper,counter)",
            agent_name="Chef",
            call_index=1,
            call_type="planner_main",
        )
        reward_info = tracker.after_step(
            3,
            ["pickup(bell_pepper,counter)", None],
            FakeState(),
        )
        agent0_calls = reward_info["per_agent"][0]["calls"]

        self.assertIs(agent0_calls[0], request)
        self.assertIs(agent0_calls[1], later)
        self.assertAlmostEqual(request["paired_comm_reward"], 0.5)
        self.assertEqual(request["paired_comm_result"], "request_helpful")
        self.assertAlmostEqual(response["paired_comm_reward"], 0.5)
        self.assertEqual(response["paired_comm_result"], "helpful_request_accepted")
        self.assertAlmostEqual(later["paired_comm_reward"], 0.0)
        self.assertEqual(later["action"], "pickup(bell_pepper,counter)")

    def test_session_strips_validator_penalty_from_semantic_communication_record(self):
        session = CollabMainSession.__new__(CollabMainSession)
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="",
            response="Think: ok\nRecent Goal: ask\nAction: Collab(request(Assistant,place_obj_on_counter()))",
            metadata={"call_index": 4, "action_mode": "communication"},
            context={"call_type": "planner_main"},
            timestep=8,
        )
        process_reward = {
            "timestamp": 8,
            "per_agent": [
                {
                    "sequence_reward": 0.0,
                    "calls": [
                        {
                            "timestamp": 8,
                            "source_timestamp": 8,
                            "agent_index": 0,
                            "call_index": 4,
                            "call_type": "planner_main",
                            "action": "Collab(request(Assistant,place_obj_on_counter()))",
                            "sequence_reward": 0.0,
                            "format_reward": 0.05,
                            "validator_reward": -0.1,
                            "communication_reward": 0.0,
                            "repeat_communication_reward": 0.0,
                            "forced_communication_reward": 0.0,
                            "paired_comm_reward": 0.0,
                            "penalties": [{"type": "validator", "detail": "invalid", "value": -0.1}],
                            "total": -0.05,
                        }
                    ],
                },
                {"sequence_reward": 0.0, "calls": []},
            ],
        }

        CollabMainSession._assign_call_rewards(session, [record], process_reward, False)

        breakdown = record.metadata["reward_breakdown"]
        self.assertEqual(record.reward, 0.0)
        self.assertEqual(breakdown["format_reward"], 0.0)
        self.assertEqual(breakdown["validator_reward"], 0.0)
        self.assertEqual(breakdown["sequence_reward"], 0.0)

    def test_agent_action_mode_flags_malformed_and_multi_embodied_outputs(self):
        agent = LLMAgents.__new__(LLMAgents)

        multi = agent._inspect_action_mode(
            "Action: pickup(bell_pepper, counter); put_obj_in_utensil(oven0)"
        )
        malformed = agent._inspect_action_mode(
            "Action: Collab(ack(Chef))\n```\n```python\nCollab(ack(Chef))"
        )
        nested_collab = agent._inspect_action_mode(
            "Action: Collab(request(Assistant, place_obj_on_counter()))"
        )

        self.assertEqual(multi["mode"], "multi_embodied")
        self.assertEqual(malformed["mode"], "malformed")
        self.assertEqual(nested_collab["mode"], "communication")

    def test_append_policy_records_does_not_backfill_existing_source_key(self):
        class ExistingTransition:
            reward = 0.0
            format_reward = 0.0
            validator_reward = 0.0
            process_reward = 0.0
            sequence_reward = 0.0

        class FakeBuffer:
            def __init__(self):
                self.rows = []
                self.existing = ExistingTransition()

            def add(self, **kwargs):
                self.rows.append(kwargs)

            def get_by_source_key(self, key):
                return self.existing

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = FakeBuffer()
        trainer.format_penalty_value = -0.2
        trainer.validator_penalty_value = -0.1
        trainer._active_rollout_id = "unit"
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="",
            response="Think: ok\nRecent Goal: place\nAction: place_obj_on_counter()",
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "call_index": 3,
                "reward_breakdown": {
                    "sequence_reward": 2.0,
                    "format_reward": 0.05,
                    "validator_reward": 0.05,
                    "raw": {"total": 2.1},
                },
            },
            context={"call_type": "planner_main"},
            timestep=5,
            micro_step=3,
        )
        record.reward = 2.1

        added = trainer._append_policy_records([record], default_reward=0.0, done_flag=False)

        self.assertEqual(added, 1)
        self.assertAlmostEqual(trainer.buffer.rows[0]["reward"], 2.1)
        self.assertAlmostEqual(trainer.buffer.rows[0]["sequence_reward"], 2.0)
        self.assertAlmostEqual(trainer.buffer.existing.reward, 0.0)
        self.assertAlmostEqual(trainer.buffer.existing.sequence_reward, 0.0)

    def test_trainer_fallback_penalizes_multi_embodied_saved_transition(self):
        class FakeBuffer:
            def __init__(self):
                self.rows = []

            def add(self, **kwargs):
                self.rows.append(kwargs)

            def get_by_source_key(self, key):
                return None

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = FakeBuffer()
        trainer.format_penalty_value = -0.2
        trainer._active_rollout_id = "unit"
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="",
            response=(
                "Think: ok\n"
                "Recent Goal: finish\n"
                "Action: pickup(bell_pepper, counter); put_obj_in_utensil(oven0)"
            ),
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
            },
            context={"call_type": "planner_main"},
            timestep=4,
        )

        added = trainer._append_policy_records([record], default_reward=0.0, done_flag=False)

        self.assertEqual(added, 1)
        row = trainer.buffer.rows[0]
        self.assertAlmostEqual(row["reward"], -0.2)
        self.assertAlmostEqual(row["format_reward"], -0.2)
        self.assertAlmostEqual(row["validator_reward"], 0.0)
        self.assertAlmostEqual(row["sequence_reward"], 0.0)
        self.assertAlmostEqual(row["process_reward"], 0.0)
        self.assertAlmostEqual(row["paired_comm_reward"], 0.0)
        self.assertAlmostEqual(row["breakdown_total_reward"], -0.2)
        self.assertEqual(record.metadata["action_mode"], "multi_embodied")
        self.assertTrue(record.metadata["format_fallback_penalty"])

    def test_trainer_fallback_penalizes_extra_fenced_text_after_action(self):
        class FakeBuffer:
            def __init__(self):
                self.rows = []

            def add(self, **kwargs):
                self.rows.append(kwargs)

            def get_by_source_key(self, key):
                return None

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = FakeBuffer()
        trainer.format_penalty_value = -0.2
        trainer._active_rollout_id = "unit"
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="",
            response=(
                "```\n"
                "Think: ok\n"
                "Recent Goal: Confirm if the oven is ready.\n"
                "Action: put_obj_in_utensil(oven0)\n"
                "```\n\n"
                "```recent goal: Confirm if the oven is ready.\n"
            ),
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "reward_breakdown": {
                    "format_reward": 0.05,
                    "validator_reward": -0.1,
                    "sequence_reward": 0.0,
                },
            },
            context={"call_type": "planner_main"},
            timestep=3,
        )

        added = trainer._append_policy_records([record], default_reward=0.0, done_flag=False)

        self.assertEqual(added, 1)
        row = trainer.buffer.rows[0]
        self.assertAlmostEqual(row["reward"], -0.2)
        self.assertAlmostEqual(row["format_reward"], -0.2)
        self.assertAlmostEqual(row["validator_reward"], 0.0)
        self.assertAlmostEqual(row["sequence_reward"], 0.0)
        self.assertAlmostEqual(row["process_reward"], 0.0)
        self.assertAlmostEqual(row["breakdown_total_reward"], -0.2)
        self.assertEqual(record.metadata["action_mode"], "malformed")

    def test_trainer_fallback_checks_full_response_before_metadata_action_mode(self):
        class FakeBuffer:
            def __init__(self):
                self.rows = []

            def add(self, **kwargs):
                self.rows.append(kwargs)

            def get_by_source_key(self, key):
                return None

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = FakeBuffer()
        trainer.format_penalty_value = -0.2
        trainer._active_rollout_id = "unit"
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="",
            response=(
                "```\n"
                "Think: ok\n"
                "Recent Goal: answer the request.\n"
                "Action: Collab(ack(Chef))\n"
                "```\n\n"
                "```\n"
                "Think: extra answer block\n"
                "Recent Goal: place the item.\n"
                "Action: place_obj_on_counter()\n"
                "```\n"
            ),
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "action_mode": "communication",
                "reward_breakdown": {
                    "format_reward": 0.0,
                    "validator_reward": 0.0,
                    "sequence_reward": 0.0,
                },
            },
            context={"call_type": "communication"},
            timestep=3,
        )

        added = trainer._append_policy_records([record], default_reward=0.0, done_flag=False)

        self.assertEqual(added, 1)
        row = trainer.buffer.rows[0]
        self.assertAlmostEqual(row["reward"], -0.2)
        self.assertAlmostEqual(row["format_reward"], -0.2)
        self.assertAlmostEqual(row["validator_reward"], 0.0)
        self.assertAlmostEqual(row["sequence_reward"], 0.0)
        self.assertAlmostEqual(row["breakdown_total_reward"], -0.2)
        self.assertEqual(record.metadata["action_mode"], "malformed")
        self.assertTrue(record.metadata["format_fallback_penalty"])

    def test_trainer_fallback_uses_stored_response_after_session_shrink(self):
        class FakeBuffer:
            def __init__(self):
                self.rows = []

            def add(self, **kwargs):
                self.rows.append(kwargs)

            def get_by_source_key(self, key):
                return None

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = FakeBuffer()
        trainer.format_penalty_value = -0.2
        trainer.policy_tokenizer = None
        trainer._active_rollout_id = "unit"
        malformed_response = (
            "```\n"
            "Think: ok\n"
            "Recent Goal: answer the request.\n"
            "Action: Collab(ack(Chef))\n"
            "```\n\n"
            "```\n"
            "Think: extra answer block\n"
            "Recent Goal: place the item.\n"
            "Action: place_obj_on_counter()\n"
            "```\n"
        )
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="",
            response="",
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "action_mode": "communication",
                "_response_text_for_reward": malformed_response,
                "reward_breakdown": {
                    "format_reward": 0.0,
                    "validator_reward": 0.0,
                    "sequence_reward": 0.0,
                },
            },
            context={"call_type": "communication"},
            timestep=3,
        )

        added = trainer._append_policy_records([record], default_reward=0.0, done_flag=False)

        self.assertEqual(added, 1)
        row = trainer.buffer.rows[0]
        self.assertAlmostEqual(row["reward"], -0.2)
        self.assertAlmostEqual(row["format_reward"], -0.2)
        self.assertAlmostEqual(row["validator_reward"], 0.0)
        self.assertAlmostEqual(row["sequence_reward"], 0.0)
        self.assertEqual(record.metadata["action_mode"], "malformed")
        self.assertTrue(record.metadata["format_fallback_penalty"])

    def test_append_policy_records_clears_validator_for_communication(self):
        class FakeBuffer:
            def __init__(self):
                self.rows = []

            def add(self, **kwargs):
                self.rows.append(kwargs)

            def get_by_source_key(self, key):
                return None

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = FakeBuffer()
        trainer.format_penalty_value = -0.2
        trainer.validator_penalty_value = -0.1
        trainer.policy_tokenizer = None
        trainer._active_rollout_id = "unit"
        record = PolicyCallRecord(
            agent_index=0,
            messages=[],
            prompt="",
            response="Think: ok\nRecent Goal: ask\nAction: Collab(request(Assistant, cut(chopping_board0)))",
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "reward_breakdown": {
                    "format_reward": 0.05,
                    "validator_reward": -0.1,
                    "sequence_reward": 0.0,
                    "paired_comm_reward": 0.0,
                    "raw": {"total": -0.05},
                },
            },
            context={"call_type": "communication"},
            timestep=3,
        )

        trainer._append_policy_records([record], default_reward=0.05, done_flag=False)

        row = trainer.buffer.rows[0]
        self.assertAlmostEqual(row["reward"], 0.05)
        self.assertAlmostEqual(row["format_reward"], 0.05)
        self.assertAlmostEqual(row["validator_reward"], 0.0)
        self.assertAlmostEqual(row["sequence_reward"], 0.0)
        self.assertAlmostEqual(row["breakdown_total_reward"], 0.05)

    def test_append_policy_records_drops_paired_comm_penalty_for_malformed(self):
        class FakeBuffer:
            def __init__(self):
                self.rows = []

            def add(self, **kwargs):
                self.rows.append(kwargs)

            def get_by_source_key(self, key):
                return None

        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.buffer = FakeBuffer()
        trainer.format_penalty_value = -0.2
        trainer.validator_penalty_value = -0.1
        trainer.policy_tokenizer = None
        trainer._active_rollout_id = "unit"
        record = PolicyCallRecord(
            agent_index=1,
            messages=[],
            prompt="",
            response=(
                "```\n"
                "Think: first\n"
                "Recent Goal: place\n"
                "Action: place_obj_on_counter()\n"
                "```\n\n"
                "```\n"
                "Think: extra\n"
                "Recent Goal: ask\n"
                "Action: Collab(request(Chef, put_obj_in_oven0(cut_bell_pepper)))\n"
                "```"
            ),
            metadata={
                "prompt_ids": torch.tensor([1]),
                "response_ids": torch.tensor([2]),
                "log_prob": 0.0,
                "value": 0.0,
                "reward_breakdown": {
                    "format_reward": 0.0,
                    "validator_reward": 0.0,
                    "sequence_reward": 0.0,
                    "paired_comm_reward": -0.1,
                    "raw": {"total": -0.1},
                },
            },
            context={"call_type": "communication"},
            timestep=3,
        )

        trainer._append_policy_records([record], default_reward=-0.1, done_flag=False)

        row = trainer.buffer.rows[0]
        self.assertAlmostEqual(row["reward"], -0.2)
        self.assertAlmostEqual(row["format_reward"], -0.2)
        self.assertAlmostEqual(row["validator_reward"], 0.0)
        self.assertAlmostEqual(row["paired_comm_reward"], 0.0)
        self.assertAlmostEqual(row["breakdown_total_reward"], -0.2)
        self.assertEqual(record.metadata["action_mode"], "malformed")
        self.assertTrue(record.metadata["format_fallback_penalty"])

    def test_single_fenced_answer_block_is_not_marked_malformed(self):
        response = (
            "```\n"
            "Think: ok\n"
            "Recent Goal: place the item.\n"
            "Action: place_obj_on_counter()\n"
            "```"
        )

        self.assertEqual(MAPPOTrainer._classify_response_action_mode(response), "planner_main")

    def test_plain_text_tail_after_fenced_answer_block_is_malformed(self):
        response = (
            "```\n"
            "Think: ok\n"
            "Recent Goal: answer the request.\n"
            "Action: Collab(ack(Chef))\n"
            "```\n\n"
            "The current action is to wait for the chef's response."
        )

        self.assertEqual(MAPPOTrainer._classify_response_action_mode(response), "malformed")
        self.assertTrue(LLMAgents._has_nonempty_tail_after_fenced_response(response))

    def test_vllm_actor_request_uses_agent_specific_adapter_name(self):
        trainer = MAPPOTrainer.__new__(MAPPOTrainer)
        trainer.agents_cfg = {
            "agent_0": {"model": "qwen2.5-7B-instruct", "timeout": 3},
            "agent_1": {"model": "qwen2.5-7B-instruct", "timeout": 3},
        }
        trainer.trainer_cfg = {"max_new_tokens": 32, "vllm_request_timeout": 3}
        trainer.actor_adapters = {
            0: AdapterSpec(name="Chef", lora_path="/tmp/Chef"),
            1: AdapterSpec(name="Assistant", lora_path="/tmp/Assistant"),
        }
        trainer._last_rollout_stats = {}
        trainer._vllm_endpoint = lambda path, service="actor": (0, "http://unit.test/rl/generate")
        captured = []

        def fake_post_json(url, payload, timeout, **kwargs):
            captured.append((url, payload, timeout))
            return {
                "text": "Think: ok\nRecent Goal: test\nAction: wait(1)",
                "response_log_probs": [-0.1],
                "response_tokens": ["x"],
                "log_prob": -0.1,
            }

        trainer._post_json = fake_post_json

        trainer._query_vllm_actor(
            agent_index=0,
            messages=[{"role": "user", "content": "p"}],
            temperature=0.7,
            stop="Scene",
        )
        trainer._query_vllm_actor(
            agent_index=1,
            messages=[{"role": "user", "content": "p"}],
            temperature=0.7,
        )

        self.assertEqual(captured[0][1]["adapter_name"], "Chef")
        self.assertEqual(captured[0][1]["stop"], "Scene")
        self.assertEqual(captured[1][1]["adapter_name"], "Assistant")
        self.assertIsNone(captured[1][1]["stop"])


if __name__ == "__main__":
    unittest.main()
