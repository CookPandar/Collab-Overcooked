import json
import tempfile
import unittest
from pathlib import Path

from collab_overcooked.reward import ProcessRewardTracker


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


class CommunicationRewardTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.reference_dir = Path(self.tempdir.name)
        self.mdp = FakeMDP()

    def tearDown(self):
        self.tempdir.cleanup()

    def build_tracker(self, order="soup", **settings):
        ref_path = self.reference_dir / f"unit_{order}_ref.json"
        payload = {
            "demo": {
                "agent_0": ["pickup(onion)"],
                "agent_1": ["place_obj_on_counter()"],
            }
        }
        ref_path.write_text(json.dumps(payload), encoding="utf-8")
        return ProcessRewardTracker(
            order=order,
            mdp=self.mdp,
            reference_dir=self.reference_dir,
            settings=settings,
        )

    def build_paired_tracker(self):
        return self.build_tracker(
            paired_comm_reward_enabled=True,
            paired_comm_request_positive_reward=0.5,
            paired_comm_request_negative_reward=0.1,
            paired_comm_response_positive_reward=0.5,
            paired_comm_response_negative_reward=0.1,
            paired_comm_deny_reward=1.0,
        )

    def _register(
        self,
        tracker,
        agent_index,
        action_text,
        *,
        timestamp,
        agent_name,
        call_index,
        call_type="communication",
    ):
        return tracker.register_llm_action(
            agent_index=agent_index,
            timestamp=timestamp,
            action_text=action_text,
            agent_name=agent_name,
            call_index=call_index,
            call_type=call_type,
        )

    def test_repeated_embodied_action_for_same_agent_gets_penalized(self):
        tracker = self.build_tracker(communication_penalty=0.3)

        first = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            call_type="planner_main",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=1,
            action_text="pickup(onion)",
            call_type="planner_main",
        )

        self.assertEqual(first["communication_reward"], 0.0)
        self.assertEqual(second["communication_reward"], -0.3)

    def test_repeated_action_is_tracked_per_agent(self):
        tracker = self.build_tracker(communication_penalty=0.25)

        tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            call_type="planner_main",
        )
        other_agent = tracker.register_llm_action(
            agent_index=1,
            timestamp=0,
            action_text="pickup(onion)",
            call_type="planner_main",
        )

        self.assertEqual(other_agent["communication_reward"], 0.0)

    def test_repeated_request_with_same_embodied_payload_gets_penalized(self):
        tracker = self.build_tracker(communication_penalty=0.4)
        action = "Collab(request(Assistant,pickup(onion,ingredient_dispenser)))"

        first = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text=action,
            call_type="communication",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=1,
            action_text=action,
            call_type="communication",
        )

        self.assertEqual(first["communication_reward"], 0.0)
        self.assertEqual(second["communication_reward"], -0.4)

    def test_request_with_different_embodied_payload_is_not_penalized(self):
        tracker = self.build_tracker(communication_penalty=0.4)

        tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="Collab(request(Assistant,pickup(onion,ingredient_dispenser)))",
            call_type="communication",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=1,
            action_text="Collab(request(Assistant,place_obj_on_counter()))",
            call_type="communication",
        )

        self.assertEqual(second["communication_reward"], 0.0)

    def test_same_payload_but_different_communication_primitive_is_not_penalized(self):
        tracker = self.build_tracker(communication_penalty=0.2)

        tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text='Collab(seek(Assistant,"pickup(onion,ingredient_dispenser)"))',
            call_type="communication",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=1,
            action_text='Collab(ack(Assistant,"pickup(onion,ingredient_dispenser)"))',
            call_type="communication",
        )

        self.assertEqual(second["communication_reward"], 0.0)

    def test_switching_between_embodied_and_communication_does_not_count_as_repeat(self):
        tracker = self.build_tracker(communication_penalty=0.2)

        tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="pickup(onion)",
            call_type="planner_main",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=1,
            action_text="Collab(request(Assistant,pickup(onion,ingredient_dispenser)))",
            call_type="communication",
        )

        self.assertEqual(second["communication_reward"], 0.0)

    def test_forced_communication_termination_gets_single_communication_penalty(self):
        tracker = self.build_tracker(communication_penalty=0.2)

        tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="wait(1)",
            call_type="planner_main",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=1,
            action_text="wait(1)",
            call_type="planner_main",
            metadata={
                "suppress_repeat_penalty": True,
                "force_communication_penalty": True,
            },
        )

        self.assertEqual(second["communication_reward"], -0.2)

    def test_forced_communication_termination_can_use_stronger_penalty(self):
        tracker = self.build_tracker(
            communication_penalty=0.2,
            forced_communication_penalty=0.6,
        )

        tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="wait(1)",
            call_type="planner_main",
        )
        second = tracker.register_llm_action(
            agent_index=0,
            timestamp=1,
            action_text="wait(1)",
            call_type="planner_main",
            metadata={
                "suppress_repeat_penalty": True,
                "force_communication_penalty": True,
            },
        )

        self.assertEqual(second["communication_reward"], -0.6)

    def test_helpful_request_and_ack_reward_both_agents(self):
        tracker = self.build_tracker(
            paired_comm_reward_enabled=True,
            paired_comm_request_positive_reward=0.5,
            paired_comm_response_positive_reward=0.5,
        )

        request_entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="Collab(request(Assistant,place_obj_on_counter()))",
            call_type="communication",
        )
        response_entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=0,
            action_text="Collab(ack(Chef))",
            call_type="communication",
        )

        self.assertEqual(request_entry["paired_comm_reward"], 0.5)
        self.assertEqual(request_entry["paired_comm_role"], "initiator")
        self.assertTrue(request_entry["paired_comm_request_helpful"])
        self.assertEqual(response_entry["paired_comm_reward"], 0.5)
        self.assertEqual(response_entry["paired_comm_role"], "responder")
        self.assertEqual(response_entry["paired_comm_result"], "helpful_request_accepted")

    def test_unhelpful_request_and_deny_reward_responder(self):
        tracker = self.build_tracker(
            paired_comm_reward_enabled=True,
            paired_comm_request_negative_reward=0.1,
            paired_comm_response_negative_reward=0.1,
            paired_comm_deny_reward=1.0,
        )

        request_entry = tracker.register_llm_action(
            agent_index=0,
            timestamp=0,
            action_text="Collab(request(Assistant,drop(onion)))",
            call_type="communication",
        )
        response_entry = tracker.register_llm_action(
            agent_index=1,
            timestamp=0,
            action_text="Collab(deny(Chef))",
            call_type="communication",
        )

        self.assertEqual(request_entry["paired_comm_reward"], -0.1)
        self.assertEqual(request_entry["paired_comm_result"], "request_not_helpful")
        self.assertFalse(request_entry["paired_comm_request_helpful"])
        self.assertEqual(response_entry["paired_comm_reward"], 1.0)
        self.assertEqual(response_entry["paired_comm_result"], "bad_request_denied")

    def test_detailed_paired_communication_scenarios(self):
        scenarios = [
            {
                "name": "chef_helpful_request_then_assistant_ack",
                "request": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(request(Assistant,place_obj_on_counter()))",
                },
                "response": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "Collab(ack(Chef))",
                },
                "expected": {
                    "request_reward": 0.5,
                    "request_helpful": True,
                    "request_result": "request_helpful",
                    "response_reward": 0.5,
                    "response_result": "helpful_request_accepted",
                },
            },
            {
                "name": "chef_helpful_request_then_assistant_executes_requested_embodied_action",
                "request": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(request(Assistant,place_obj_on_counter()))",
                },
                "response": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "place_obj_on_counter()",
                    "call_type": "planner_main",
                },
                "expected": {
                    "request_reward": 0.5,
                    "request_helpful": True,
                    "request_result": "request_helpful",
                    "response_reward": 0.5,
                    "response_result": "helpful_request_accepted",
                },
            },
            {
                "name": "chef_helpful_request_then_assistant_executes_wrong_embodied_action",
                "request": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(request(Assistant,place_obj_on_counter()))",
                },
                "response": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "pickup(onion)",
                    "call_type": "planner_main",
                },
                "expected": {
                    "request_reward": 0.5,
                    "request_helpful": True,
                    "request_result": "request_helpful",
                    "response_reward": -0.1,
                    "response_result": "helpful_request_rejected_or_missed",
                },
            },
            {
                "name": "chef_helpful_request_then_assistant_replies_with_deny",
                "request": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(request(Assistant,place_obj_on_counter()))",
                },
                "response": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "Collab(deny(Chef))",
                },
                "expected": {
                    "request_reward": 0.5,
                    "request_helpful": True,
                    "request_result": "request_helpful",
                    "response_reward": -0.1,
                    "response_result": "helpful_request_rejected_or_missed",
                },
            },
            {
                "name": "chef_bad_request_then_assistant_ack",
                "request": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(request(Assistant,drop(onion)))",
                },
                "response": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "Collab(ack(Chef))",
                },
                "expected": {
                    "request_reward": -0.1,
                    "request_helpful": False,
                    "request_result": "request_not_helpful",
                    "response_reward": -0.1,
                    "response_result": "bad_request_followed",
                },
            },
            {
                "name": "chef_bad_request_then_assistant_executes_wrong_embodied_action",
                "request": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(request(Assistant,drop(onion)))",
                },
                "response": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "drop(onion)",
                    "call_type": "planner_main",
                },
                "expected": {
                    "request_reward": -0.1,
                    "request_helpful": False,
                    "request_result": "request_not_helpful",
                    "response_reward": -0.1,
                    "response_result": "bad_request_followed",
                },
            },
            {
                "name": "chef_bad_request_then_assistant_deny",
                "request": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(request(Assistant,drop(onion)))",
                },
                "response": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "Collab(deny(Chef))",
                },
                "expected": {
                    "request_reward": -0.1,
                    "request_helpful": False,
                    "request_result": "request_not_helpful",
                    "response_reward": 1.0,
                    "response_result": "bad_request_denied",
                },
            },
            {
                "name": "assistant_helpful_request_then_chef_ack",
                "request": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "Collab(request(Chef,pickup(onion)))",
                },
                "response": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(ack(Assistant))",
                },
                "expected": {
                    "request_reward": 0.5,
                    "request_helpful": True,
                    "request_result": "request_helpful",
                    "response_reward": 0.5,
                    "response_result": "helpful_request_accepted",
                },
            },
            {
                "name": "assistant_bad_request_then_chef_deny",
                "request": {
                    "agent_index": 1,
                    "agent_name": "Assistant",
                    "action": "Collab(request(Chef,drop(onion)))",
                },
                "response": {
                    "agent_index": 0,
                    "agent_name": "Chef",
                    "action": "Collab(deny(Assistant))",
                },
                "expected": {
                    "request_reward": -0.1,
                    "request_helpful": False,
                    "request_result": "request_not_helpful",
                    "response_reward": 1.0,
                    "response_result": "bad_request_denied",
                },
            },
        ]

        for index, scenario in enumerate(scenarios):
            tracker = self.build_paired_tracker()
            request = self._register(
                tracker,
                scenario["request"]["agent_index"],
                scenario["request"]["action"],
                timestamp=index,
                agent_name=scenario["request"]["agent_name"],
                call_index=index * 2,
                call_type=scenario["request"].get("call_type", "communication"),
            )
            response = self._register(
                tracker,
                scenario["response"]["agent_index"],
                scenario["response"]["action"],
                timestamp=index,
                agent_name=scenario["response"]["agent_name"],
                call_index=index * 2 + 1,
                call_type=scenario["response"].get("call_type", "communication"),
            )
            expected = scenario["expected"]

            with self.subTest(scenario=scenario["name"]):
                self.assertEqual(request["paired_comm_role"], "initiator")
                self.assertEqual(
                    request["paired_comm_reward"], expected["request_reward"]
                )
                self.assertEqual(
                    request["paired_comm_request_helpful"],
                    expected["request_helpful"],
                )
                self.assertEqual(
                    request["paired_comm_result"], expected["request_result"]
                )
                self.assertEqual(response["paired_comm_role"], "responder")
                self.assertEqual(
                    response["paired_comm_reward"], expected["response_reward"]
                )
                self.assertEqual(
                    response["paired_comm_result"], expected["response_result"]
                )
                self.assertEqual(
                    tracker.pending_paired_comm_requests[0], [],
                    "response 后不应残留 Chef 队列中的未闭合 request",
                )
                self.assertEqual(
                    tracker.pending_paired_comm_requests[1], [],
                    "response 后不应残留 Assistant 队列中的未闭合 request",
                )

    def test_responder_other_collab_reply_is_treated_as_failed_response(self):
        tracker = self.build_paired_tracker()

        request = self._register(
            tracker,
            0,
            "Collab(request(Assistant,place_obj_on_counter()))",
            timestamp=0,
            agent_name="Chef",
            call_index=0,
        )
        response = self._register(
            tracker,
            1,
            "Collab(seek(Chef,pickup(onion)))",
            timestamp=0,
            agent_name="Assistant",
            call_index=1,
        )

        self.assertEqual(request["paired_comm_reward"], 0.5)
        self.assertEqual(response["paired_comm_reward"], -0.1)
        self.assertEqual(response["paired_comm_result"], "helpful_request_rejected_or_missed")

    def test_pending_requests_are_consumed_in_fifo_order(self):
        tracker = self.build_paired_tracker()

        first_request = self._register(
            tracker,
            0,
            "Collab(request(Assistant,place_obj_on_counter()))",
            timestamp=0,
            agent_name="Chef",
            call_index=0,
        )
        second_request = self._register(
            tracker,
            0,
            "Collab(request(Assistant,drop(onion)))",
            timestamp=1,
            agent_name="Chef",
            call_index=1,
        )
        first_response = self._register(
            tracker,
            1,
            "Collab(ack(Chef))",
            timestamp=1,
            agent_name="Assistant",
            call_index=2,
        )
        second_response = self._register(
            tracker,
            1,
            "Collab(deny(Chef))",
            timestamp=2,
            agent_name="Assistant",
            call_index=3,
        )

        self.assertEqual(first_request["paired_comm_reward"], 0.5)
        self.assertEqual(second_request["paired_comm_reward"], -0.1)
        self.assertEqual(first_response["paired_comm_result"], "helpful_request_accepted")
        self.assertEqual(first_response["paired_comm_reward"], 0.5)
        self.assertEqual(second_response["paired_comm_result"], "bad_request_denied")
        self.assertEqual(second_response["paired_comm_reward"], 1.0)
        self.assertEqual(tracker.pending_paired_comm_requests, [[], []])


if __name__ == "__main__":
    unittest.main()
