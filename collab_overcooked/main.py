"""Run Collab-Overcooked episodes with two LLM agents (Chef and Assistant)."""

import copy
import datetime
import json
import os
import time
import warnings
from argparse import ArgumentParser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*")  # ignore "Unable to register cuBLAS factory" due to use tf-CPU

from rich import print as rprint
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action

from .agent_factory import make_agent, combine_statistic_dict
from .agents.modules import statistics_dict
from .config import RECIPE_DIR

EMPTY_STATISTICS = copy.deepcopy(statistics_dict)
from .human_eval.web_util import output_to_port, check_port_in_use, change_port


def check_recipe_parse(variant):
    for r in os.listdir(RECIPE_DIR):
        if variant["order"] in r.lower():
            return True
    raise ValueError("Not valid order name!")


def print_finished_skills(obs):
    skills = ""
    for i, ml_action in enumerate(obs.ml_actions):
        if ml_action is None:
            continue
        skills += f"P{i} finished <{ml_action}>. "
    print(skills)


def run_develop(env, horizon):
    """
    You can customize the 'action_list' and 'parm' to test the environment
    """
    action_list = []
    parm = []

    env.reset()
    r_total = 0
    for t in range(min(horizon, len(action_list))):
        s_t = env.state
        print(f"\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n")
        print(env.mdp.state_string(s_t).replace("ø", "o"))

        obs, reward, done, env_info = env.step(action_list[t], parm[t])
        print(env.mdp.get_utensil_states(s_t))
        print_finished_skills(obs)

        r_total += reward
        rprint("[red]" + f"r: {reward} | total: {r_total}\n\n")


ROLES = ["chef", "assistant"]
AGENT_KEYS = ["model", "model_dirname", "local_server_api", "base_url", "api_key", "temperature", "timeout", "enable_thinking"]


def agent_settings(variant, role):
    """LLM settings of one role: --<role>_model and the `agents.<role>` section of --config override the shared ones."""
    settings = {
        "model": variant.get(f"{role}_model") or variant["gpt_model"],
        "model_dirname": variant["model_dirname"],
        "local_server_api": variant["local_server_api"],
        "base_url": None,
        "api_key": None,
        "temperature": variant["temperature"],
        "timeout": variant["timeout"],
        "enable_thinking": variant["enable_thinking"],
    }
    role_config = (variant.get("agents") or {}).get(role) or {}
    settings.update({k: v for k, v in role_config.items() if k in AGENT_KEYS})
    if role_config.get("api_key_env"):
        settings["api_key"] = os.environ[role_config["api_key_env"]]
    if variant.get(f"{role}_model"):
        settings["model"] = variant[f"{role}_model"]
    return settings


def team_name(variant):
    """Name of the log directory: the model name, or chef/assistant model names of a mixed team."""
    chef, assistant = (agent_settings(variant, role)["model"] for role in ROLES)
    return chef if chef == assistant else f"chef-{chef}_assistant-{assistant}"


def build_team(variant, mdp, layout):
    agents_list = []
    for actor in ROLES:
        settings = agent_settings(variant, actor)
        print(f"\n----Use {settings['model']} as {actor}----\n")
        if settings["model"] == "human":
            assert check_port_in_use(variant["local_server_api"]) is True, f"port {variant['local_server_api']} is busy"
            change_port(variant["local_server_api"])
        agent = make_agent(
            mdp,
            layout,
            history_window=variant["history_window"],
            response_protocol=variant["response_protocol"],
            reflection=variant["reflection"],
            actor=actor,
            **settings,
        )
        agents_list.append(agent)
    return AgentGroup(*agents_list)


def reset_statistics():
    statistics_dict.update(copy.deepcopy(EMPTY_STATISTICS))


def notify_human(team, message, mission, port):
    for a in range(len(team.agents)):
        output_to_port(f"agent{a}", message, mission=mission, port=port)


def run_episode(variant, env, team, filename):
    """Run one episode step by step and save the statistics to `filename` after every step."""
    is_human = any(agent_settings(variant, role)["model"] == "human" for role in ROLES)
    episode_statistics = statistics_dict
    r_total = 0

    for t in range(variant["horizon"]):
        s_t = env.state
        print(f"\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n")
        map = env.mdp.state_string(s_t).replace("ø", "o")
        print(map)
        a_t, ingredient_for_pickup = team.joint_action(s_t)
        print(a_t)
        team.reset_dialogue()
        print(f"\n-----------Controller-----------\n")
        print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}")

        obs, reward, done, env_info = env.step(a_t, ingredient_for_pickup)
        print_finished_skills(obs)

        r_total += reward
        if reward > 0:
            episode_statistics["total_order_finished"].append(s_t.current_k_order[0])
            team.agents[1].teammate_ml_actions.append({"timestamp": t, "action": "deliver_soup()"})
        rprint("[red]" + f"r: {reward} | total: {r_total}\n\n")
        print(f"P0's real behavior: {team.agents[1].teammate_ml_actions}")
        print(f"P1's real behavior: {team.agents[0].teammate_ml_actions}")

        # save statistics
        turn_statistics_dict_both = combine_statistic_dict(
            team.agents[0].turn_statistics_dict,
            team.agents[1].turn_statistics_dict,
            map,
            reward,
        )
        episode_statistics["total_timestamp"].append(t)
        episode_statistics["total_score"] = r_total
        episode_statistics["total_action_list"][0] = team.agents[1].teammate_ml_actions
        episode_statistics["total_action_list"][1] = team.agents[0].teammate_ml_actions
        episode_statistics["content"].append(turn_statistics_dict_both)
        with open(filename, "w") as f:
            json.dump(episode_statistics, f, indent=4)

        if variant["test_mode"] == "fix_task" and reward != 0:
            print("Task successed!")
            # Human-eval: set task success message
            if is_human:
                notify_human(team, "Success!", "success", variant["local_server_api"])
            break

    # Human-eval: set task failed message
    if is_human:
        notify_human(team, "Fail to finish task in time!", "fail", variant["local_server_api"])
    return r_total


def main(variant):
    layout = variant["layout"]
    horizon = variant["horizon"]

    mdp = OvercookedGridworld.from_layout_name(layout)

    # set order according to parser
    if variant["order"] != "" and check_recipe_parse(variant):
        mdp.start_order_list = [variant["order"]]
        # 1 task mode
        mdp.one_task_mode = True

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    if variant["mode"] == "develop":
        run_develop(env, horizon)
        return []

    save_dir = os.path.join(variant["statistics_save_dir"], team_name(variant), variant["order"])
    os.makedirs(save_dir, exist_ok=True)

    results = []
    for i in range(variant["episode"]):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{save_dir}/experiment_{current_time}_{variant['order']}.json"

        reset_statistics()
        team = build_team(variant, mdp, layout)
        team.reset()
        env.reset()

        r_total = run_episode(variant, env, team, filename)
        print(f"Episode {i+1}/{variant['episode']}: {r_total}\n====\n\n")
        results.append(r_total)
    return results


def get_parser():
    parser = ArgumentParser(description="Collab-Overcooked Experiment")
    parser.add_argument("--config", "-c", type=str, default=None, help="YAML file whose keys set the defaults of the arguments below; `agents.chef` / `agents.assistant` set per-role LLM settings")

    # these are basis parses
    parser.add_argument("--layout", "-l", type=str, default="new_env", choices=["new_env"])
    parser.add_argument("--order", type=str, default="", help="1 task order name")
    parser.add_argument("--horizon", type=int, default=120, help="Horizon steps in one game")
    parser.add_argument("--episode", type=int, default=1, help="Number of episodes")
    parser.add_argument("--mode", type=str, default="exp", choices=["exp", "develop"], help="exp mode runs LLM agents step-by-step, develop mode replays a hand-written action list")
    parser.add_argument("--test_mode", type=str, default="fix_task", choices=["fix_task", "fix_time"], help="fix_task stops when the order is delivered; fix_time always runs `horizon` steps")
    parser.add_argument("--statistics_save_dir", type=str, default="data", help="save directory of LLM statistics")

    # LLM settings shared by both agents
    # model: API models such as 'gpt-4o', 'deepseek-chat'; 'human' for human evaluation;
    # any other name is treated as an open-source model served by vLLM at --local_server_api.
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo-0125", help="Model of both agents")
    parser.add_argument("--chef_model", type=str, default=None, help="Model of the chef (overrides --gpt_model)")
    parser.add_argument("--assistant_model", type=str, default=None, help="Model of the assistant (overrides --gpt_model)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=None, help="Timeout (s) of one LLM request")
    parser.add_argument("--enable_thinking", type=lambda v: str(v).lower() in ("1", "true", "yes", "on"), default=None, help="Switch the thinking mode of vLLM-served thinking models such as Qwen3 (true/false); default keeps the chat template default")
    parser.add_argument("--model_dirname", type=str, default="", help="Prefix prepended to an open-source model name; '<model_dirname><model>' must equal the vLLM served model name")
    parser.add_argument("--local_server_api", type=str, default="http://localhost:8000/v1", help="vLLM server address (port number for human evaluation)")

    # agent protocol and memory
    parser.add_argument("--response_protocol", type=str, default="think_action", choices=["think_action", "action_only"], help="think_action: Think / Recent Goal / Action; action_only: a single Action field")
    parser.add_argument("--reflection", type=lambda v: str(v).lower() in ("1", "true", "yes", "on"), default=True, help="Write a reflection after a successful correction (true/false)")
    parser.add_argument("--history_window", type=int, default=3, help="Number of past decision records shown to the agent (0 disables history)")
    return parser


def parse_args(argv=None):
    parser = get_parser()
    args, _ = parser.parse_known_args(argv)
    config = {}
    if args.config:
        import yaml

        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}
        unknown = set(config) - {action.dest for action in parser._actions} - {"agents"}
        if unknown:
            parser.error(f"unknown keys in {args.config}: {sorted(unknown)}")
        parser.set_defaults(**{k: v for k, v in config.items() if k != "agents"})
    variant = vars(parser.parse_args(argv))
    variant["agents"] = config.get("agents") or {}
    return variant


def cli():
    variant = parse_args()
    start_time = time.time()
    main(variant)
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {time.time() - start_time:.3f}s-----\n\n")


if __name__ == "__main__":
    cli()
