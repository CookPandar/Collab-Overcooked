import json
import re
import tempfile
import unittest
from pathlib import Path

from collab_overcooked.reward import ProcessRewardTracker
from collab_overcooked.training.main_session import CollabMainSession, PolicyCallRecord
from collab_overcooked.training.mappo_qwen import MAPPOTrainer

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
        self.assertTrue(path.exists(), f"Missing real log fixture: {path}")
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
            self.assertTrue(root.exists(), f"Missing real log root: {root}")
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
                            if raw_call.get("call_type") == "communication":
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
                            if reward_entry.get("call_type") != "planner_main":
                                continue
                            call_index = reward_entry.get("call_index")
                            if call_index not in raw_lookup:
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
            self.assertTrue(root.exists(), f"Missing real log root: {root}")
            for path in sorted(root.glob("*/*.json")):
                data = self.load_real_log(path)
                rewards = data.get("process_rewards") or []
                for step_idx, reward_step in enumerate(rewards):
                    for agent_idx, agent_reward in enumerate(reward_step.get("per_agent") or []):
                        for reward_entry in agent_reward.get("calls") or []:
                            if reward_entry.get("call_type") != "planner_main":
                                continue
                            sequence_reward = float(reward_entry.get("sequence_reward", 0.0) or 0.0)
                            if sequence_reward <= 0.0:
                                continue
                            prefix_safe = True
                            for prefix_step_idx, prefix_reward_step in enumerate(rewards[: step_idx + 1]):
                                for prefix_entry in (
                                    (prefix_reward_step.get("per_agent") or [])[agent_idx].get("calls") or []
                                ):
                                    if prefix_step_idx == step_idx and prefix_entry.get("call_index") == reward_entry.get("call_index"):
                                        break
                                    if prefix_entry.get("call_type") != "communication":
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

    def test_register_llm_action_combines_incremental_sequence_and_penalties(self):
        tracker = self.build_tracker(
            order="soup",
            refs_agent0=["pickup(onion)"],
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

        self.assertEqual(entry["sequence_reward"], 1.0)
        self.assertEqual(entry["format_reward"], -0.5)
        self.assertEqual(entry["validator_reward"], 0.0)
        self.assertEqual(entry["total"], 0.5)

        # Penalties should be consumed by exactly one call.
        next_entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            agent_name="Chef",
            call_index=4,
            call_type="planner_main",
        )
        self.assertEqual(next_entry["format_reward"], 0.0)
        self.assertEqual(next_entry["sequence_reward"], 0.0)

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

        self.assertEqual(records[0].reward, -0.5)
        self.assertEqual(records[1].reward, 1.0)
        self.assertEqual(
            records[0].metadata["reward_breakdown"]["call_type"],
            "communication",
        )
        self.assertEqual(
            records[1].metadata["reward_breakdown"]["call_type"],
            "planner_main",
        )

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

    def test_assign_call_rewards_keeps_reward_even_when_semantic_call_type_is_communication(self):
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

        self.assertEqual(record.reward, 1.0)
        self.assertEqual(record.metadata["semantic_call_type"], "communication")
        self.assertEqual(record.metadata["reward_breakdown"]["communication_reward"], 0.0)

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
        session = CollabMainSession.__new__(CollabMainSession)
        data = self.load_real_log(REAL_LOG_GPT4O)
        raw_calls = data["content"][0]["content"]["original_log"][1]
        records = self.build_records_from_log_calls(raw_calls)
        process_reward = data["process_rewards"][0]

        session._assign_call_rewards(records, process_reward, done_flag=False)

        self.assertGreater(records[0].reward, 0.0)
        self.assertAlmostEqual(records[1].reward, 0.6554694229112834)
        self.assertEqual(records[1].metadata["reward_breakdown"]["call_type"], "planner_main")

    def test_batch_sampled_real_cases_match_expected_reward_values(self):
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
                        + float(breakdown.get("communication_reward", 0.0)),
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
                        + breakdown["communication_reward"],
                    )

        self.assertGreaterEqual(fmt_hits, 10)
        self.assertGreaterEqual(validator_hits, 10)
        self.assertGreaterEqual(communication_hits, 5)

    def test_positive_sequence_rewards_become_fixed_progress_reward_on_real_logs(self):
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


if __name__ == "__main__":
    unittest.main()
