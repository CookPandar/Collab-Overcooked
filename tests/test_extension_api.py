"""The extension interface: a custom LLM backend and an observer receiving agent events."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from collab_overcooked import main as run  # noqa: E402
from collab_overcooked import agent_factory  # noqa: E402
from test_benchmark import MockLLM  # noqa: E402


class Recorder:
    def __init__(self):
        self.events = []

    def __getattr__(self, name):
        if not name.startswith("on_"):
            raise AttributeError(name)
        return lambda agent, **payload: self.events.append((name[3:], agent.name, payload))


class MockBackend:
    """backend(messages, call_context) -> (text, metadata), reusing the mock policy of test_benchmark."""

    def __init__(self):
        self.llm = MockLLM()
        self.contexts = []

    def __call__(self, messages, ctx):
        self.contexts.append(ctx)
        instruction, user = messages[1]["content"].split("<input>\n", 1)
        fake = type("M", (), {"instruction_head_list": [{"content": instruction}],
                              "current_user_message": {"content": user}})
        text, tokens = self.llm(fake)
        return text, {"token_count": tokens, "logprob": -1.0}


class ExtensionApiTest(unittest.TestCase):
    def test_backend_and_observer(self):
        backend, recorder = MockBackend(), Recorder()
        original = agent_factory.make_agent

        def make_agent(mdp, layout, **kwargs):
            return original(mdp, layout, observer=recorder, backend=backend, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, \
                unittest.mock.patch.object(run, "make_agent", make_agent):
            variant = run.parse_args(["--order", "boiled_egg", "--horizon", "30", "--gpt_model", "mock",
                                      "--statistics_save_dir", os.path.join(tmp, "data")])
            with contextlib.redirect_stdout(io.StringIO()):
                run.main(variant)

        calls = [(agent, p["call"]) for kind, agent, p in recorder.events if kind == "llm_call"]
        self.assertEqual(len(calls), len(backend.contexts))
        keys = [(c["agent_index"], c["timestep"], c["call_index"]) for _, c in calls]
        self.assertEqual(len(keys), len(set(keys)), "call keys are unique")
        for ctx in backend.contexts:
            self.assertIn(ctx["call_type"], {"planner_main", "validator_correction", "communication",
                                             "format_correction", "rethink"})
        self.assertTrue(all(p["metadata"]["logprob"] == -1.0 for kind, _, p in recorder.events if kind == "llm_call"))

        known = set(keys)
        bound = [p for kind, _, p in recorder.events if kind == "action_bound"]
        self.assertTrue(bound)
        for p in bound:
            s = p["source"]
            self.assertIn((s["agent_index"], s["timestep"], s["call_index"]), known)

        validations = [(agent, p) for kind, agent, p in recorder.events if kind == "validation"]
        first_fail = next(p for agent, p in validations if agent == "Chef" and not p["ok"])
        self.assertEqual(first_fail["action"], "pickup(egg,counter)")
        self.assertEqual(first_fail["source"]["call_type"], "planner_main")
        corrected = next(p for agent, p in validations if agent == "Chef" and p["ok"]
                         and p["source"]["timestep"] == first_fail["source"]["timestep"])
        self.assertEqual(corrected["source"]["call_type"], "validator_correction")

        format_errors = [p for kind, agent, p in recorder.events if kind == "format_error" and agent == "Assistant"]
        self.assertTrue(any("A dish is not a utensil" in p["detail"] for p in format_errors))

        executed = [p for kind, _, p in recorder.events if kind == "action_executed"]
        self.assertTrue(executed)
        self.assertTrue(all(p["source"] is not None for p in executed))


if __name__ == "__main__":
    unittest.main()
