<div align="center">
  <h1> Collab-Overcooked </h1>
</div>

We propose a new LLM-powered Multi-Agent System (LLM-MAS) benchmark, Collab-Overcooked, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks from two novel perspectives. First, it provides a multi-agent framework supporting diverse tasks and objectives and encourages collaboration through natural language communication. Second, it introduces a spectrum of process-oriented evaluation metrics to assess the fine-grained collaboration capabilities of different LLM agents, a dimension often overlooked in prior work.

## 📢 v2

Collab-Overcooked v2 is a major update:

- **New agent protocol.** Each timestep an agent outputs `Think / Recent Goal / Action` and performs **one** action. It communicates through structured collaboration primitives (`Collab(request|seek|deny|ack)`) instead of free-form `say` messages. Agents also see their recent decisions and the lessons they learned from corrected mistakes. See [Agent Protocol](#agent-protocol).
- **Restructured code.** `src/` became the installable package `collab_overcooked`, with a single configuration for the 30 tasks and command-line tools for running and evaluating.
- **Fixes.** Several benchmark bugs were fixed, see [Repository History](#repository-history).

⚠️ Results of v2 are **not directly comparable** with v1 or the paper (EMNLP 2025), because the agent protocol changed. To reproduce the paper, use [v1](#repository-history).

## Getting Started

### Install
We recommend using conda. Python 3.8 is recommended for this project.

```bash
git clone https://github.com/YusaeMeow/Collab-Overcooked.git
cd Collab-Overcooked

conda create -n collab-overcooked python=3.8
conda activate collab-overcooked

pip install -e lib/overcooked_ai   # the modified Overcooked-AI environment
pip install -e .                   # collab_overcooked and its requirements
```

Notes: [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai) is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game Overcooked. We made certain modifications based on Overcooked-AI (see `lib/overcooked_ai/README.md`).

### API Keys
An API key is only needed for API models (names containing `gpt`, `deepseek`, `claude` or `gemini`, and OpenAI o-series models such as `o4-mini`). Open-source models and the evaluation do not need one.

```bash
export OPENAI_API_KEY=sk-...
# optional: an OpenAI-compatible endpoint, e.g. for DeepSeek models
export OPENAI_BASE_URL=https://api.deepseek.com
```

For compatibility with v1, the key can also be put on the first line of `openai_key.txt` in the working directory. Models of other providers are used through their OpenAI-compatible endpoints, with per-role `base_url` and `api_key_env` in a `--config` file, e.g. for Claude:
```yaml
agents:
  chef: {model: claude-sonnet-4-20250514, base_url: "https://api.anthropic.com/v1/", api_key_env: ANTHROPIC_API_KEY}
  assistant: {model: claude-sonnet-4-20250514, base_url: "https://api.anthropic.com/v1/", api_key_env: ANTHROPIC_API_KEY}
```

### Quick Test
```bash
python -m collab_overcooked.main --horizon 3 --order boiled_egg --gpt_model gpt-4o
```
If the environment map, the agents' outputs and 3 time steps are printed without errors, the environment and agent configuration are working. `collab-overcooked` is installed as a shortcut for `python -m collab_overcooked.main`.

The test suite runs complete episodes with a rule-based mock LLM and needs no API key:
```bash
pip install -e ".[human]"   # aiohttp, needed by the human-play tests
python -m unittest discover tests
```

### Run Experiments
```bash
# one task, logs are saved to ./data/<model>/<order>/experiment_*.json
python -m collab_overcooked.main --gpt_model gpt-4o --order boiled_egg --episode 10

# a mixed team, logs are saved to ./data/chef-gpt-4o_assistant-gpt-4o-mini/<order>/
python -m collab_overcooked.main --chef_model gpt-4o --assistant_model gpt-4o-mini --order boiled_egg

# all 30 tasks + evaluation
bash scripts/run_benchmark.sh gpt-4o 10
```

Main arguments of `collab_overcooked.main`:

| Argument | Default | Description |
| --- | --- | --- |
| `--gpt_model` | `gpt-3.5-turbo-0125` | Model of both agents. API models (see [API Keys](#api-keys)) use the OpenAI API or a compatible endpoint; `human` starts human evaluation; other names are served by vLLM. |
| `--chef_model`, `--assistant_model` | | Model of one role, overriding `--gpt_model`. |
| `--order` | | Task name, e.g. `boiled_egg` (see `collab_overcooked/config.py` for all 30 tasks). |
| `--horizon` | `120` | Max time steps of an episode. |
| `--episode` | `1` | Number of episodes. |
| `--test_mode` | `fix_task` | `fix_task` stops when the order is delivered; `fix_time` always runs `horizon` steps. |
| `--response_protocol` | `think_action` | `think_action`: the agent outputs Think / Recent Goal / Action; `action_only`: a single Action field (e.g. for small models and RL). |
| `--reflection` | `true` | Write a reflection after a successful correction (shown as *Lessons from Past Failures*). |
| `--history_window` | `3` | Number of past decision records shown to the agent (`0` disables history). |
| `--temperature` | `0.7` | Sampling temperature. Models that only accept the default temperature (OpenAI o-series, GPT-5, ...) automatically use the default. |
| `--timeout` | | Timeout (s) of one LLM request. |
| `--enable_thinking` | | `true` / `false`: switch the thinking mode of vLLM-served thinking models such as Qwen3 (default: the chat template default). |
| `--local_server_api` | `http://localhost:8000/v1` | vLLM server address (or the websocket port for human evaluation). |
| `--model_dirname` | | Prefix prepended to the model name when querying vLLM. |
| `--statistics_save_dir` | `data` | Output directory of the experiment logs. |
| `--config` | | YAML file setting any of the arguments above, plus per-role `model`, `base_url`, `api_key` / `api_key_env`, `temperature`, etc. See [configs/example.yaml](configs/example.yaml). |

### Configure the Open-source LLMs
We recommend using [vLLM](https://github.com/vllm-project/vllm) for local deployment of open-source LLMs. The model is queried by the name `<model_dirname><model>`, which must match the name served by vLLM:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
python -m collab_overcooked.main --gpt_model Qwen2.5-7B-Instruct --model_dirname Qwen/ \
    --local_server_api http://localhost:8000/v1 --order boiled_egg
```
The `<think>...</think>` reasoning of thinking models (e.g. Qwen3, DeepSeek-R1) is removed before the reply is parsed.

## Agent Protocol
The Chef knows the recipe; the Assistant does not, and each of them can only reach the utensils on its own side of the kitchen. So the order can only be completed through collaboration.

Every timestep, each agent receives:
- **History**: its last `history_window` decision records (observation and its Think / Recent Goal / Action at that time).
- **Current Observation**: the order, the kitchen scene, both agents' held items and current actions, the collaboration messages of this timestep, its Recent Goal, and *Lessons from Past Failures* (when available).

It replies with three fields:
```
Think: reasoning about the scene, the recipe and the teammate's messages
Recent Goal: the multi-step plan, kept until it is finished or no longer appropriate
Action: exactly one action
```
With `--response_protocol action_only`, the agent replies with the `Action` field only (prompt: `communication_rule_action_only.txt`), which is convenient for small models and RL training.

The action is either one embodied action from the agent's skill set (`pickup(obj, place)`, `put_obj_in_utensil(utensil)`, `cook(pot0)`, `fill_dish_with_food(utensil)`, `deliver_soup()`, `wait(n)`, ...) or one collaboration action:

| Primitive | Meaning |
| --- | --- |
| `Collab(request(receiver, action); request(receiver, action))` | ask the teammate to perform actions |
| `Collab(seek(receiver, message))` | ask the teammate for help (e.g. the Assistant asks what to do) |
| `Collab(deny(receiver, message))` | refuse a request that cannot be done |
| `Collab(ack(receiver, message))` | accept a request |

A `Collab` triggers a dialogue within the same timestep: the teammate answers immediately with a `Collab` or an embodied action, for at most 3 exchanges. The initiator acts again from the next timestep.

Every action is checked by a validator. If the action is invalid (wrong format or not executable in the current state), the agent receives the error and gets one chance to correct it; otherwise it waits for one step. After a successful correction, the agent writes a short reflection, which is shown in its later observations as *Lessons from Past Failures*.

To drive the agents with your own model (e.g. for SFT or RL) and observe their LLM calls, validations and executed actions, see the [Extension API](docs/extension_api.md).

SFT and RL baselines for small open models (AgentGym-RL and MARSHAL objectives, action-only or with Think) are in [baselines/](baselines/README.md).

The prompts are in `collab_overcooked/prompts/gpt/`: `environment_rule.txt` (game rules), `chef_skill.txt` / `assistant_skill.txt` (actions), `communication_rule.txt` (input/output format) and `reflection_rule.txt`.

## Evaluation
The evaluation consists of three sequential steps:
- **evaluation**: evaluates the environment's output, calculates the metrics for each task, and stores the results in the corresponding task folder.
- **organize_result**: summarizes the metrics of each task into the `statistics_data.csv` file.
- **convert_result**: computes the metrics for each complexity level and stores the results in the `converted_data.csv` file.

```bash
# evaluate a single task: reads logs from ./data and writes results to ./eval_result
python -m collab_overcooked.evaluation.evaluation --model gpt-4o --order boiled_egg

# evaluate all 30 tasks of one or more models
python -m collab_overcooked.evaluation.evaluation --test_mode build_in --model gpt-4o

# aggregate per-task results into a CSV, then compute per-level averages
python -m collab_overcooked.evaluation.organize_result --model gpt-4o
python -m collab_overcooked.evaluation.convert_result
```
`bash scripts/run_evaluation.sh gpt-4o` runs all three steps.

The final evaluation results are stored in `eval_result/converted_data.csv`, with the following key fields:
- **mean_f1_agent_*:** The F1 score for agent * computed using the TES function, considering both correctly matched actions and redundant actions.
- **mean_similarity_agent_*:** The similarity between the actions generated by agent * and the RATs (Reference Action Templates).
- **mean_redundancy_agent_*:** The redundancy between the actions generated by agent * and the RATs.
- **initiate_collaboration:** The ability of the LLM-MAS (Multi-Agent System) to initiate collaboration, i.e. whether the actions the Chef requests with `Collab(request(...))` are useful.
- **respond_collaboration:** The ability of the LLM-MAS to respond to collaboration, i.e. whether the Assistant performs the requested actions.

To evaluate under a time limit of `multiple` x the reference steps of each task, truncate the logs first:
```bash
python -m collab_overcooked.evaluation.truncate_logs --model gpt-4o --multiple 1.5
python -m collab_overcooked.evaluation.evaluation --test_mode build_in --model gpt-4o --log_dir data/truncate_1_5
```

Each log `experiment_*.json` also keeps, for every timestep, the observations, the parsed replies, the errors and corrections, and the raw input/output of every LLM call (`original_log`), as well as the full prompt of each role (`prompt_templates`).

## Human Players

### Human Evaluation
A human can replace an LLM agent through a web page and play with the same protocol:
```bash
python -m collab_overcooked.human_eval.server --port 9999
# open collab_overcooked/human_eval/index0.html (Chef) and index1.html (Assistant) in a browser
python -m collab_overcooked.main --gpt_model human --local_server_api 9999 --order boiled_egg
```

### Play with an Agent Teammate
The optional browser mode lets a human control either chef with the keyboard while an agent operates the other chef, in real time and without a chat channel. It uses the same game engine and does not need API keys.

```bash
pip install -e ".[human]"
python -m collab_overcooked.human_play --order boiled_egg
```

Open <http://127.0.0.1:8080>, then press **Start / Resume**. Use **WASD / arrows** to move, **Space** to interact, and **E** to start a loaded appliance. Select the ingredient before using its dispenser. Items can be passed on shared counters. The default teammate is a deterministic recipe baseline; a chat-completions model server or a custom HTTP policy can also be used. See [human-agent play](docs/human-play.md) for setup, protocol, recording, and validation.

## Project Structure
```
Collab-Overcooked/
├── collab_overcooked/
│   ├── main.py              # run episodes (entry point)
│   ├── config.py            # paths, the 30 tasks and their levels, API key
│   ├── agent_factory.py     # build agents and merge per-step statistics
│   ├── agents/
│   │   ├── collab.py        # LLM agent: prompting, parsing, communication, validation, reflection, motion control
│   │   ├── modules.py       # LLM query module (OpenAI API / vLLM / human) and statistics templates
│   │   └── retry.py
│   ├── evaluation/          # evaluation → organize_result → convert_result, truncate_logs
│   ├── human_eval/          # websocket server and web pages for human evaluation
│   ├── human_coop/          # real-time human-agent play (human_play.py)
│   ├── prompts/             # agent prompts (gpt/), recipes (recipe/) and reference action trajectories (reference/)
│   └── llama_tokenizer/     # tokenizer used to count tokens of open-source models
├── lib/overcooked_ai/       # modified Overcooked-AI environment
├── configs/example.yaml     # example configuration (mixed team)
├── scripts/                 # run_benchmark.sh, run_evaluation.sh
├── tests/                   # mock-LLM benchmark tests, extension API and human-play tests
└── docs/                    # extension_api.md, human-play.md
```

## Modify the Environment
The environment settings and logic are stored in the `lib/overcooked_ai` folder. Layout files are stored in the `overcooked_ai_py/data/layouts` folder, while the environment logic is in the `overcooked_ai_py/mdp` folder.

- To add more tasks (such as additional recipes or ingredients), you only need to modify the corresponding layout files, and add the recipe and reference files to `collab_overcooked/prompts/recipe` and `collab_overcooked/prompts/reference`.
- If you need to add new interactive elements, ensure that you update the environment logic accordingly.

## Repository History
- **v2 (current)**
  - Agent protocol: `Think / Recent Goal / Action` output (or `action_only`), one action per timestep, `Collab(request|seek|deny|ack)` communication with at most 3 exchanges per timestep, decision history (`--history_window`), recent goals, and reflections shown as *Lessons from Past Failures*.
  - Different models for the Chef and the Assistant (`--chef_model`, `--assistant_model`, `--config`); `temperature` and `timeout` options; OpenAI-compatible endpoints; reasoning of thinking models is removed before parsing.
  - Extension API: a custom LLM backend and an observer of agent events (LLM calls, validations, executed actions with the call that produced them), for training code such as SFT and RL.
  - Evaluation reads the new protocol (requested actions in `Collab(request(receiver, action))`) and needs no API calls. Logs keep the raw input/output of every LLM call and the prompts.
  - Fixes:
    - Reflection never ran in v1 because of a wrong condition, so *Lessons from Past Failures* were always empty. It now runs after every successful correction.
    - Invalid actions are counted as format errors and corrected.
    - Actions are matched by their parsed name instead of substrings.
    - The contents of a cooking utensil were always shown as `chopped_onion`.
    - A wrong utensil name could crash the episode.
    - In single-task mode, delivered orders were refilled.
    - Multi-episode runs mixed the statistics of different episodes.
    - Human-evaluation status messages.
  - Code: `src/` became the package `collab_overcooked/`; the 30 tasks are defined once in `config.py`; unused code (example retrieval, unused prompts, the Overcooked-AI JavaScript demo, visualization and original tests) was removed.
- **v1**: the original release used for the paper (EMNLP 2025), with the `src/` + `lib/` layout and the `analysis / plan / say` protocol. It is available at commit [`f7b66f0`](https://github.com/YusaeMeow/Collab-Overcooked/tree/f7b66f0082b2a7c950ff7ee9b25ad588b5751197) (tag `v1.0`).

| v1 (`src/`) | v2 (`collab_overcooked/`) |
| --- | --- |
| `main.py` | `main.py` |
| `utils.py` | `agent_factory.py` |
| `collab/collab.py`, `collab/modules.py`, `collab/utils.py` | `agents/collab.py`, `agents/modules.py`, `agents/retry.py` |
| `collab/web_util.py`, `service.py`, `../index0.html`, `../index1.html` | `human_eval/` |
| `human_coop/`, `human_play.py` | `human_coop/`, `human_play.py` |
| `evaluation.py`, `eval_utils.py`, `organize_result.py`, `convert_result.py`, `evaluation.sh` | `evaluation/`, `scripts/run_evaluation.sh` |
| `time_step.py` | `evaluation/truncate_logs.py` |
| `prompts/` | `prompts/` |
| `../lib/llama_tokenizer` | `llama_tokenizer/` |

## Reference
```bibtex
@inproceedings{sun-etal-2025-collab,
  title={Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents},
  author={Sun, Haochen and Zhang, Shuwen and Niu, Lujie and Ren, Lei and Xu, Hao and Fu, Hao and Zhao, Fangkun and Yuan, Caixia and Wang, Xiaojie},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year= {2025},
  url= {https://aclanthology.org/2025.emnlp-main.249/},
  doi= {10.18653/v1/2025.emnlp-main.249}
}


@inproceedings{zhang2024proagent,
  title={Proagent: building proactive cooperative agents with large language models},
  author={Zhang, Ceyao and Yang, Kaijie and Hu, Siyi and Wang, Zihao and Li, Guanghe and Sun, Yihang and Zhang, Cheng and Zhang, Zhaowei and Liu, Anji and Zhu, Song-Chun and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17591--17599},
  year={2024}
}

@inproceedings{carroll2019utility,
 title={On the Utility of Learning About Humans for Human-AI Coordination},
 author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
 booktitle={Advances in Neural Information Processing Systems},
 pages={},
 volume={32},
 year={2019},
}
```
