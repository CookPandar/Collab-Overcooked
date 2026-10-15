import re
import time
from functools import lru_cache
from typing import Optional, Union

import tiktoken
from openai import OpenAI
from rich import print as rprint

from ..config import LLAMA_TOKENIZER_DIR, LOCAL_SERVER_API_KEY, get_openai_api_key
from ..human_eval.web_util import output_to_port
from .retry import retry_with_exponential_backoff

# global statistics of one episode
statistics_dict = {
    "total_timestamp": [],
    "total_order_finished": [],
    "total_score": 0,
    "total_action_list": [[], []],
    "content": [],
    "prompt_templates": {},
}

# turn statistics
turn_statistics_dict = {
    "timestamp": 0,
    "order_list": [],
    "actions": [],
    "map": "",
    "statistical_data": {
        "score": 0,
        "communication": [
            {"call": 0, "turn": [], "token": []},
            {"call": 0, "turn": [], "token": []},
        ],
        "error": [
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
            {
                "format_error": {"error_num": 0, "error_message": []},
                "validator_error": {"error_num": 0, "error_message": []},
            },
        ],
        "error_correction": [
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
            {
                "format_correction": {"correction_num": 0, "correction_tokens": []},
                "validator_correction": {
                    "correction_num": 0,
                    "reflection_obtain": [],
                    "correction_tokens": [],
                },
            },
        ],
    },
    "content": {
        "observation": [[], []],
        "reflection": [[], []],
        "content": [[], []],
        "action_list": [[], []],
        # raw input/output of every LLM call of each agent in this timestep
        "original_log": [[], []],
        # the LLM call that produced the action executed in this timestep
        "action_source": [None, None],
    },
}

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove the reasoning of thinking models (e.g. Qwen3, DeepSeek-R1) before parsing."""
    text = THINK_BLOCK.sub("", text or "")
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]
    return text.strip()


API_MODEL_PATTERN = re.compile(r"gpt|deepseek|claude|gemini|^o\d", re.IGNORECASE)


def is_api_model(model: str) -> bool:
    """API models: GPT / o-series (OpenAI), DeepSeek, Claude, Gemini. Other names are served by vLLM."""
    return bool(API_MODEL_PATTERN.search(model))


def is_reasoning_model(model: str) -> bool:
    """OpenAI o-series models only accept the default temperature."""
    return bool(re.match(r"o\d", model.lower()))


@lru_cache(maxsize=1)
def load_llama_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(LLAMA_TOKENIZER_DIR, local_files_only=True)


def count_tokens(text: str, model: str) -> int:
    if "gpt-3.5" in model or "human" in model:
        return len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text))
    if is_api_model(model):
        return len(tiktoken.encoding_for_model("gpt-4").encode(text))
    return len(load_llama_tokenizer().encode(text))


class Module(object):
    """
    This module is responsible for communicating with LLMs.

    API models (GPT / o-series, DeepSeek, Claude, Gemini) are queried through the OpenAI API, or
    an OpenAI-compatible endpoint set by `base_url` / OPENAI_BASE_URL. Other models are served
    by vLLM at `local_server_api` under the name `model_dirname + model`.
    """

    def __init__(
        self,
        role_messages,
        model="gpt-3.5-turbo-0125",
        api_key=None,
        base_url=None,
        model_dirname="",
        local_server_api="http://localhost:8000/v1",
        temperature=0.7,
        timeout: Optional[float] = None,
        enable_thinking=None,
        backend=None,
    ):
        self.model = model
        # optional callable backend(messages, call_context) -> (text, metadata) replacing the API request,
        # e.g. a local policy for training; `call_context` describes the call (see LLMAgents._query)
        self.backend = backend
        self.call_context = {}
        self.last_metadata = None
        self.api_key = api_key
        self.base_url = base_url
        self.model_dirname = model_dirname
        self.local_server_api = local_server_api
        self.temperature = temperature
        self.timeout = timeout
        # vLLM chat template switch of thinking models (e.g. Qwen3); None keeps the template default
        self.enable_thinking = enable_thinking
        # set when the server rejects `temperature` (e.g. GPT-5 / reasoning models)
        self.default_temperature_only = False

        self.instruction_head_list = role_messages
        # a dynamic changed dialog_history used for generating  different input for each failure
        self.dialog_history_list = []
        # save the dialog_history of meetting first failture
        self.dialog_history_list_storage = []
        self.current_user_message = None
        self.current_timestep = None

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)

    def query_messages(self, rethink=False) -> list:
        system_message = {
            "role": "system",
            "content": "You are an intelligent agent planner, you need to generate output and plan in the specified format according to the game rules and environmental status.",
        }
        user_content = (
            self.instruction_head_list[0]["content"]
            + "<input>\n"
            + self.current_user_message["content"]
        )
        return [system_message, {"role": "user", "content": user_content}]

    def client_and_model(self):
        """Return the OpenAI client and the request model name of this agent."""
        kwargs = {"timeout": float(self.timeout)} if self.timeout is not None else {}
        if self.base_url or is_api_model(self.model):
            api_key = self.api_key or (
                get_openai_api_key() if is_api_model(self.model) else LOCAL_SERVER_API_KEY
            )
            return OpenAI(api_key=api_key, base_url=self.base_url, **kwargs), self.model
        client = OpenAI(
            api_key=self.api_key or LOCAL_SERVER_API_KEY, base_url=self.local_server_api, **kwargs
        )
        return client, self.model_dirname + self.model

    @retry_with_exponential_backoff
    def query(
        self,
        stop=None,
        temperature=None,
        debug_mode="Y",
        trace=True,
        rethink=False,
        map="",
    ):
        messages = self.query_messages(rethink)

        if not trace and not rethink:
            messages[-1]["content"] += " Based on the failure explanation and scene description, analyze and plan again."

        temperature = self.temperature if temperature is None else temperature
        if self.backend is not None:
            # errors of a custom backend are raised, not retried
            response_text, self.last_metadata = self.backend(messages, dict(self.call_context))
            token_count = (self.last_metadata or {}).get("token_count")
            if token_count is None:
                token_count = count_tokens(response_text, self.model)
            return response_text, token_count

        max_retries = 3
        for retry_count in range(1, max_retries + 1):
            try:
                if "human" in self.model:
                    response_text = self.parse_human_response(self.query_human(messages, map))
                else:
                    client, model_name = self.client_and_model()
                    request = {"model": model_name, "messages": messages}
                    if is_reasoning_model(self.model):
                        if self.model.lower().startswith(("o1-mini", "o1-preview")):
                            # these models do not accept system messages
                            request["messages"] = [{
                                "role": "user",
                                "content": messages[0]["content"] + "\n\n" + messages[1]["content"],
                            }]
                    elif not self.default_temperature_only:
                        request["temperature"] = temperature
                    if self.enable_thinking is not None and not (self.base_url or is_api_model(self.model)):
                        request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(self.enable_thinking)}}
                    response = client.chat.completions.create(**request)
                    response_text = strip_reasoning(response.choices[0].message.content)
                    if is_api_model(self.model):
                        time.sleep(0.5)  # rate limiting
                break
            except Exception as e:
                rprint(f"[red][LLM ERROR][/red]: {e}")
                if "temperature" in str(e).lower() and not self.default_temperature_only:
                    # e.g. GPT-5 / reasoning models only accept the default temperature
                    rprint(f"[yellow]{self.model} rejects the temperature, retry with the default one[/yellow]")
                    self.default_temperature_only = True
                if retry_count >= max_retries:
                    rprint("[red][ERROR][/red]: Query failed after maximum retries!")
                    return "", 0
                time.sleep(1)

        return response_text, count_tokens(response_text, self.model)

    def query_human(self, messages, map):
        """Send the observation to the human-evaluation web page and wait for the reply."""
        content = messages[-1]["content"]
        if "Suppose you are a Chef" in content:
            receiver = "agent0"
        elif "Suppose you are a Assistant" in content:
            receiver = "agent1"
        else:
            raise ValueError("Invalid role in human interface")

        human_message = content[content.find("<input>\n") + len("<input>\n") :]
        recipe = None
        if receiver == "agent0":
            recipe_start = content.find("<Recipe need to know>:\n")
            if recipe_start != -1:
                recipe_start += len("<Recipe need to know>:\n")
                recipe = content[recipe_start : content.find("**Skill**")]

        error = None
        error_start = human_message.find("DO NOT COMMUNICATE WITH YOUR TEAMMATE:\n")
        if error_start != -1:
            error = human_message[error_start + len("DO NOT COMMUNICATE WITH YOUR TEAMMATE:\n") :]
            human_message = human_message[: human_message.find("Below are the failed and think history")]
        return output_to_port(receiver, human_message, map=map, recipe=recipe, error=error)

    def parse_human_response(self, response):
        if response is None:
            raise RuntimeError("No reply from the human-evaluation page")
        action = (response.get("action") or "").strip() or "wait(1)"
        return f"Think: [NOTHING]\nRecent Goal: [EMPTY]\nAction: {action}"

    def reset(self):
        self.dialog_history_list = []
