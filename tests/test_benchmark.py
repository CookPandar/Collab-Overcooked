"""End-to-end benchmark test with a rule-based mock LLM (no API calls)."""
import contextlib
import glob
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collab_overcooked import main as run  # noqa: E402
from collab_overcooked.agents.modules import Module  # noqa: E402
from collab_overcooked.evaluation import evaluation  # noqa: E402
from collab_overcooked.evaluation.eval_utils import Evaluation  # noqa: E402


class MockLLM:
    """Plays boiled_egg with the Think / Recent Goal / Action protocol.

    The first actions are invalid on purpose to exercise validation, self-correction and reflection.
    """

    def __init__(self, action_only=False):
        self.chef_calls = 0
        self.assistant_calls = 0
        self.action_only = action_only
        self.reflections = 0

    def __call__(self, module, *args, **kwargs):
        instruction = module.instruction_head_list[0]["content"]
        user = module.current_user_message["content"]
        if "Rethink:" in instruction:
            self.reflections += 1
            return "Rethink: When the counter is empty, wait for the assistant.", 10
        obs = user[user.rfind("Current Observation:"):]
        conversation = re.search(r"Past conversation:\n(.*?)(?:\nRecent Goal:|\Z)", obs, re.S).group(1)
        holds = lambda who: re.search(rf"<{who}> holds (.*?)\.", obs).group(1)
        if "Suppose you are a Chef" in instruction:
            self.chef_calls += 1
            if self.chef_calls == 1:
                action = "pickup(egg, counter)"
            elif "boiled_egg" in holds("Chef"):
                action = "deliver_soup()"
            elif "<pot0> has boiled_egg" in obs:
                action = "pickup(boiled_egg, pot0)"
            elif "<pot0> starts processing" in obs:
                action = "wait(2)"
            elif "<pot0> has egg" in obs:
                action = "cook(pot0)"
            elif "egg" in holds("Chef"):
                action = "put_obj_in_utensil(pot0)"
            elif "counters have egg" in obs:
                action = "pickup(egg, counter)"
            elif "[EMPTY]" in conversation and "egg" not in holds("Assistant") and "raised an error" not in user:
                action = "Collab(request(Assistant,pickup(egg, ingredient_dispenser));request(Assistant,place_obj_on_counter()))"
            else:
                action = "wait(1)"
        else:
            self.assistant_calls += 1
            if self.assistant_calls == 1:
                action = "put_obj_in_utensil(dish)"  # format error: a dish is not a utensil
            elif "egg" in holds("Assistant"):
                action = "place_obj_on_counter()"
            elif "Chef: Collab(request" in conversation:
                action = "pickup(egg, ingredient_dispenser)"
            else:
                action = "wait(1)"
        if self.action_only:
            return f"Action: {action}", 10
        return f"Think: next step.\nRecent Goal: boiled_egg\nAction: {action}", 10


class BenchmarkTest(unittest.TestCase):
    def test_episode_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Module, "query", new=lambda self, *a, **k: llm(self)):
            llm = MockLLM()
            variant = run.parse_args(["--order", "boiled_egg", "--horizon", "30", "--gpt_model", "mock",
                                      "--statistics_save_dir", os.path.join(tmp, "data")])
            with contextlib.redirect_stdout(io.StringIO()):
                run.main(variant)
            log_file = glob.glob(os.path.join(tmp, "data", "mock", "boiled_egg", "experiment_*.json"))[0]
            with open(log_file) as f:
                log = json.load(f)

            self.assertEqual(log["total_score"], 20)
            self.assertEqual(log["total_order_finished"], ["boiled_egg"])
            chef_actions = [a["action"] for a in log["total_action_list"][0]]
            self.assertEqual(chef_actions[-1], "deliver_soup()")
            first = log["content"][0]
            # invalid first action -> validator error, one correction and a reflection
            self.assertEqual(first["statistical_data"]["error"][0]["validator_error"]["error_num"], 1)
            self.assertEqual(first["statistical_data"]["error_correction"][0]["validator_correction"]["correction_num"], 1)
            self.assertEqual(len(first["content"]["reflection"][0]), 1)
            chef_observations = [step["content"]["observation"][0] for step in log["content"][1:]]
            self.assertTrue(any("Lessons from Past Failures" in str(obs) for obs in chef_observations))
            # invalid action format -> format error and one correction
            assistant_error = first["statistical_data"]["error"][1]["format_error"]
            self.assertEqual(assistant_error["error_num"], 1)
            self.assertIn("A dish is not a utensil", assistant_error["error_message"][0])
            self.assertEqual(first["statistical_data"]["error_correction"][1]["validator_correction"]["correction_num"], 1)
            self.assertTrue(first["content"]["original_log"][0])
            self.assertIn("Chef", log["prompt_templates"])

            os.chdir(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                evaluation.main(dict(models=["mock"], test_mode="fix_task", order="boiled_egg",
                                     log_dir="data", save_dir="eval_result"))
            with open(os.path.join(tmp, "eval_result", "mock", "boiled_egg", "evaluation_result.json")) as f:
                result = json.load(f)
            self.assertEqual(result["boiled_egg"]["task_metrics"]["success_rate"], 1.0)
            self.assertEqual(result["boiled_egg"]["statistic"]["initiate_collaboration"], 1.0)
            os.chdir(ROOT)

    def test_action_only_protocol(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(Module, "query", new=lambda self, *a, **k: llm(self)):
            llm = MockLLM(action_only=True)
            variant = run.parse_args(["--order", "boiled_egg", "--horizon", "30", "--gpt_model", "mock",
                                      "--response_protocol", "action_only", "--reflection", "false",
                                      "--history_window", "0", "--statistics_save_dir", os.path.join(tmp, "data")])
            with contextlib.redirect_stdout(io.StringIO()):
                run.main(variant)
            log_file = glob.glob(os.path.join(tmp, "data", "mock", "boiled_egg", "experiment_*.json"))[0]
            with open(log_file) as f:
                log = json.load(f)
            self.assertEqual(log["total_score"], 20)
            self.assertIn("Return exactly one field on one line", log["prompt_templates"]["Chef"]["prompt"])
            self.assertEqual(llm.reflections, 0)
            observation = log["content"][0]["content"]["observation"][0]
            self.assertNotIn("Recent Goal:", observation)
            # missing Think / Recent Goal is not a format error in action_only
            for step in log["content"]:
                for agent in range(2):
                    for message in step["statistical_data"]["error"][agent]["format_error"]["error_message"]:
                        self.assertNotIn("Think part", message)
                        self.assertNotIn("Recent Goal part", message)

    def test_extract_requests(self):
        text = "Action: Collab(request(Assistant,pickup(egg, ingredient_dispenser));request(Assistant,place_obj_on_counter()))"
        self.assertEqual(Evaluation.extract_requests(text), ["pickup(egg,ingredient_dispenser)", "place_obj_on_counter()"])
        self.assertEqual(Evaluation.extract_requests("Collab(ack(Chef))"), [])


if __name__ == "__main__":
    unittest.main()
