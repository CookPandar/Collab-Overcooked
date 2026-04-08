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


if __name__ == "__main__":
    unittest.main()
