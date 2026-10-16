"""A persistent vLLM OpenAI-compatible server for rollouts; updated LoRA adapters are loaded at runtime.

Each update is served under new adapter names (e.g. Chef_u12): vLLM's prefix cache is keyed by adapter, so
reusing a name for new weights could reuse key/value blocks computed with the old weights."""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request


class VLLMServer:
    def __init__(self, python, model, port, gpu, kv_cache_gib=8, max_model_len=8192, max_lora_rank=16,
                 log_path=None):
        self.url = f"http://127.0.0.1:{port}"
        self.served = {}  # agent_index -> served adapter name
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_ALLOW_RUNTIME_LORA_UPDATING="True",
                   VLLM_SERVER_DEV_MODE="1")  # enables /sleep and /wake_up
        command = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", model, "--port", str(port),
                   "--host", "127.0.0.1", "--enable-lora", "--max-loras", "2", "--max-lora-rank", str(max_lora_rank),
                   "--max-model-len", str(max_model_len), "--enable-prefix-caching", "--enable-sleep-mode",
                   "--kv-cache-memory-bytes", str(int(kv_cache_gib * 2**30)),
                   # the KV cache size is set above; vLLM still refuses to start unless this fraction is free
                   "--gpu-memory-utilization", "0.3",
                   # CUDA graphs for decoding without torch.compile (fast decoding, no compile dependency)
                   "--compilation-config", json.dumps({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})]
        self.log = open(log_path, "a") if log_path else subprocess.DEVNULL
        self.process = subprocess.Popen(command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self._wait()

    def _wait(self, timeout=900):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {self.process.returncode}")
            try:
                urllib.request.urlopen(self.url + "/health", timeout=5)
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(3)
        raise TimeoutError("vLLM server did not become healthy")

    def _post(self, path, payload):
        request = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()

    def load_adapters(self, paths, tag):
        """`paths`: {agent_index: LoRA directory}. Serves them as <role>_<tag> and unloads the previous ones."""
        roles = {0: "Chef", 1: "Assistant"}
        previous = dict(self.served)
        for agent_index, path in paths.items():
            name = f"{roles[agent_index]}_{tag}"
            self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
            self.served[agent_index] = name
        for name in previous.values():
            if name not in self.served.values():
                self._post("/v1/unload_lora_adapter", {"lora_name": name})
        return dict(self.served)

    def sleep(self):
        """Free the GPU memory of the server (weights to CPU, KV cache dropped) while the policy is trained."""
        self._post("/sleep?level=1", {})

    def wake_up(self):
        self._post("/wake_up", {})

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
