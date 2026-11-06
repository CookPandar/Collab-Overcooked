"""The CollabRM collaboration reward model as the think judge.

A LoRA adapter on Qwen3-8B (trained in ~/CollabRM on LLM-judge labels of the collaboration rubric) is served by vLLM
(OpenAI API, `--lora-modules rm=<adapter>`). After an episode ends, every judgeable call becomes a CollabRM unit (task,
history of both agents, target step, hindsight window of 5 timesteps, outcome, reference) and the model scores the four
rubric dimensions D1-D4; the call's score is their mean / 2 in [-1, 1] (0 when no dimension applies).

Config: judge: {type: collab_rm, url: http://localhost:PORT, model: rm, workers: 32}
"""
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .think_judge import judgeable

MAX_CHARS = 56000   # ~16K tokens: longer inputs keep the head (rubric + task) and the tail (recent history ...)
HEAD_CHARS = 14000


class CollabRMJudge:
    def __init__(self, url, model="rm", workers=32, timeout=600, retries=2, events=True, reference=True,
                 collab_rm="~/CollabRM", log_path=None, type=None):
        sys.path.insert(0, os.path.expanduser(collab_rm))
        from collab_rm import overcooked, rubric
        self.oc, self.rubric = overcooked, rubric
        self.base_url = url.rstrip("/") + ("" if url.rstrip("/").endswith("/v1") else "/v1")
        self.model = model
        self.workers = workers
        self.timeout = timeout
        self.retries = retries
        self.events = events
        self.reference = reference
        self.log_path = os.path.expanduser(log_path) if log_path else None
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "failures": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.context = {}

    def _request(self, text):
        instruction = self.rubric.RM_INSTRUCTION_EVENTS if self.events else self.rubric.RM_INSTRUCTION
        if len(text) > MAX_CHARS:
            text = text[:HEAD_CHARS] + "\n...\n" + text[-(MAX_CHARS - HEAD_CHARS):]
        payload = {"model": self.model, "temperature": 0.0, "max_tokens": 2048,
                   "messages": [{"role": "system", "content": self.rubric.RM_SYSTEM},
                                {"role": "user", "content": instruction + text}],
                   "chat_template_kwargs": {"enable_thinking": False}}
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode())
        usage = data.get("usage") or {}
        with self.lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        return data["choices"][0]["message"]["content"] or ""

    def score_unit(self, unit):
        for attempt in range(self.retries + 1):
            raw = None
            try:
                raw = self._request(unit["input"])
                parsed = self.rubric.parse_compact(raw)
                result = {"score": self.rubric.reward(parsed["scores"]), "dims": parsed["scores"],
                          "analysis": raw[:2000]}
                self.log(unit, raw, result)
                return result
            except Exception as error:  # noqa: BLE001
                with self.lock:
                    self.usage["failures"] += 1
                self.log(unit, raw, None, error=f"{type(error).__name__}: {str(error)[:300]}")
                time.sleep(2 * (attempt + 1))
        return {"score": None}

    def log(self, unit, raw, result, error=None):
        if not self.log_path:
            return
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "model": self.model, **self.context, "id": unit["id"],
                 "raw_output": raw, "parsed": result, "rule": unit.get("rule"), "error": error}
        with self.lock:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def units(self, episode, episode_id):
        """CollabRM units of the judgeable calls of one episode: [(transition, unit)]."""
        transitions = episode["transitions"]
        calls = []
        for sid, t in enumerate(transitions):
            record = t["judge_record"]
            calls.append({"sid": sid, "agent_index": t["agent_index"], "timestep": t["timestep"],
                          "call_index": t["call_index"], "order": episode["order"], "response": t["response"],
                          "observation": record["observation"], "validations": record["validations"],
                          "format_reward": t["format_reward"], "validator_reward": t["validator_reward"],
                          "ites": t["sequence_reward"]})
        ep = {"run": "rl", "update": self.context.get("update", 0), "episode": episode_id, "order": episode["order"],
              "calls": calls, "success": bool(episode["success"])}
        return [(t, self.oc.make_unit(ep, i, with_reference=self.reference))
                for i, t in enumerate(transitions) if judgeable(t)]

    def judge_episodes(self, episodes):
        """Adds `judge` ({"score", "dims"} or {"score": None}) to every judgeable transition, None to the others."""
        jobs = []
        for k, e in enumerate(episodes):
            for t in e["transitions"]:
                t["judge"] = None
            jobs.extend(self.units(e, e.get("episode_id", k)))
        with ThreadPoolExecutor(self.workers) as pool:
            for (t, _), result in zip(jobs, pool.map(lambda job: self.score_unit(job[1]), jobs)):
                t["judge"] = result
        return len(jobs)
