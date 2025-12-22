import time
import datetime
import os
import json
import datetime
import uuid
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from rich import print as rprint
import copy
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
work_dir = os.getcwd()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*")

from distutils.util import strtobool

def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

def check_recipe_parse(variant):
    recipe_name_list = os.listdir(PROMPT_DIR+'/recipe/') 
    recipe_filename = ""
    for r in recipe_name_list:
        if variant['order'] in r.lower():
            recipe_filename = r
            break
    if recipe_filename == "":
        raise ValueError("Not valid order name!")
    else:
        return True

# Load YAML for new configuration system
try:
    import yaml
except ImportError:
    print("PyYAML not installed. Install with: pip install PyYAML")
    yaml = None

cwd = os.getcwd()
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action
from .reward import ProcessRewardTracker

# Import from new modular system
try:
    from .agents import statistics_dict, turn_statistics_dict
    from .agents.web_util import output_to_port, check_port_in_use, change_port
    from .utils import make_agent, get_example_embedding, combine_statistic_dict
    
    # Define make_agent_from_config for new system
    def make_agent_from_config(agent_config, mdp, layout, history_window=3, reward_tracker=None):
        """Create agent from YAML configuration using existing LLMAgents"""
        from .agents.collab import LLMAgents
        from overcooked_ai_py.planning.planners import MediumLevelPlanner

        # Prepare MLAM parameters just like legacy make_agent
        mlam_params = {
            "start_orientations": False,
            "wait_allowed": True,
            "counter_goals": [],
            "counter_drop": [],
            "counter_pickup": [],
            "same_motion_goals": True,
        }
        counter_locations = mdp.get_counter_locations()
        mlam_params["counter_goals"] = counter_locations
        mlam_params["counter_drop"] = counter_locations
        mlam_params["counter_pickup"] = counter_locations

        # Build planner with proper counter awareness
        mlam = MediumLevelPlanner.from_pickle_or_compute(
            mdp, mlam_params, force_compute=True
        )

        # Map role to actor name
        role = agent_config.get("role", "Chef")
        actor = "chef" if role.lower() == "chef" else "assistant"

        # Backward-compatible config fields
        retrival_method = agent_config.get(
            "retrieval_method", agent_config.get("retrival_method", "recent_k")
        )
        history_k = int(agent_config.get("history_k", agent_config.get("K", 1)))
        local_server_api = agent_config.get(
            "base_url", agent_config.get("local_server_api", "http://localhost:8000/v1")
        )

        agent_history_window = agent_config.get("history_window", history_window)

        agent = LLMAgents(
            mlam,
            layout,
            model=agent_config.get("model", "gpt-3.5-turbo"),
            model_dirname=agent_config.get("model_dirname", "~/"),
            local_server_api=local_server_api,
            retrival_method=retrival_method,
            K=history_k,
            actor=actor,
            auto_unstuck=agent_config.get("auto_unstuck", False),
            controller_mode=agent_config.get("controller_mode", "new"),
            debug_mode=agent_config.get("debug_mode", "Y"),
            outdir=agent_config.get("outdir"),
            history_window=agent_history_window,
            reward_tracker=reward_tracker,
        )

        if agent_config.get("api_key"):
            agent.api_key = agent_config["api_key"]

        agent.set_mdp(mdp)
        return agent
except ImportError:
    # Fallback to old system  
    from .agents.modules import statistics_dict, turn_statistics_dict
    from .agents.web_util import output_to_port, check_port_in_use, change_port
    from .utils import make_agent, get_example_embedding, combine_statistic_dict
    make_agent_from_config = None

import socket


def load_config_from_yaml(config_path):
    """Load configuration from YAML file"""
    if not yaml:
        raise ImportError("PyYAML is required for YAML configuration. Install with: pip install PyYAML")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def convert_yaml_to_variant(config):
    """Convert YAML config to old-style variant dict"""
    env_config = config.get('environment', {})
    agents_config = config.get('agents', {})
    run_config = config.get('run', {})
    
    variant = {
        'layout': env_config.get('layout', 'cramped_room'),
        'horizon': env_config.get('horizon', 10),
        'order': env_config.get('order', 'boiled_egg'),
        'episode': config.get('episode', run_config.get('episode', 1)),
        'mode': config.get('mode', run_config.get('mode', 'exp')),
        'test_mode': config.get('test_mode', run_config.get('test_mode', 'single_task')),
        'p0': config.get('p0', run_config.get('p0', 'LLMPair')),
        'p1': config.get('p1', run_config.get('p1', 'LLMPair')),
        'collab_mode': config.get('collab_mode', run_config.get('collab_mode', 'llm')),
        'llm_model': config.get(
            'llm_model',
            run_config.get(
                'llm_model',
                config.get('gpt_model', 'gpt-3.5-turbo'),
            ),
        ),
        'reward': config.get('reward', run_config.get('reward', {})),
        'history_window': config.get('history_window', run_config.get('history_window', 3)),
        'agent_configs': agents_config,
        'use_new_system': True,
        'run_id': run_config.get('run_id', config.get('run_id')),
        'results_root': run_config.get('results_root', config.get('results_root', 'results')),
    }
    
    return variant


def main(variant=None, config_path=None):
    """
    Main function supporting both old variant dict and new YAML config
    """
    
    # Handle new YAML configuration
    if config_path:
        config = load_config_from_yaml(config_path)
        variant = convert_yaml_to_variant(config)
        variant['yaml_config'] = config
    
    if variant is None:
        raise ValueError("Either variant dict or config_path must be provided")

    statistics_dict.setdefault("process_rewards", [])
    statistics_dict["process_rewards"].clear()
    statistics_dict["prompt_templates"] = {}

    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    order_name = variant.get('order', 'task')

    mode = variant.get('mode', 'exp')
    collab_mode = variant.get('collab_mode', 'llm').lower()
    llm_model_name = variant.get('llm_model', 'gpt-3.5-turbo')

    run_id = variant.get('run_id')
    if not run_id:
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_id = f"{timestamp}-{uuid.uuid4().hex[:6]}"
        variant['run_id'] = run_id
    print(f"[Run] 使用 run_id: {run_id}")

    results_root = Path(variant.get('results_root', variant.get('statistics_save_dir', 'results')))
    try:
        history_window = max(0, int(variant.get('history_window', 3)))
    except (TypeError, ValueError):
        history_window = 0
    
    mdp = OvercookedGridworld.from_layout_name(layout)

    reward_tracker = None
    reward_reference_dir = Path(PROMPT_DIR) / "reference"
    reward_settings = variant.get('reward', {})
    try:
        reward_tracker = ProcessRewardTracker(
            order=variant['order'],
            mdp=mdp,
            reference_dir=reward_reference_dir,
            settings=reward_settings,
        )
    except Exception as exc:
        print(f"[ProcessRewardTracker] disabled: {exc}")
        reward_tracker = None

    #set order according to parser
    if variant['order'] !="" and check_recipe_parse(variant):
        mdp.start_order_list = [variant['order']]
        # 1 task mode
        mdp.one_task_mode = True

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    
    p0_algo = variant.get('p0', 'LLMPair')
    p1_algo = variant.get('p1', 'LLMPair')
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")

    start_time = time.time()
    results = []

    actor_num = 0
    actor_list = ['chef','assistant']
    for i in range(episode):  
        if reward_tracker:
            reward_tracker.reset()

        agents_list = []

        episode_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        
        # Handle save directory for new config system
        if variant.get('use_new_system'):
            save_dir = results_root / f"{run_id}_{order_name}"
        else:
            stats_dir = Path(variant.get('statistics_save_dir', 'data'))
            save_dir = stats_dir / llm_model_name / order_name
        
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / f"experiment_{episode_stamp}_{order_name}.json"

        if mode == 'develop':
            """
            You can customize the 'action_list' and 'parm' to test the environment
            """
            action_list = []
            parm = []

            env.reset()
            r_total = 0
            for t in range(horizon):
                s_t = env.state
                # print(s_t.timestep, env.t)
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                print(env.mdp.state_string(s_t).replace('ø', 'o'))


                obs, reward, done, env_info = env.step(action_list[t], parm[t])
                print(env.mdp.get_utensil_states(s_t))
                ml_actions = obs.ml_actions
                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')
            break

        
        # Create agents - support both old and new systems
        if variant.get('use_new_system') and make_agent_from_config:
            # Use new configuration system
            agent_configs = variant.get('agent_configs', {})
            for i, (agent_id, agent_config) in enumerate(agent_configs.items()):
                if agent_id.startswith('agent_'):
                    print(f"\n----Use {agent_config.get('model', 'unknown')} ({agent_config.get('type', 'unknown')})----\n")
                    agent = make_agent_from_config(
                        agent_config,
                        mdp,
                        layout,
                        history_window=history_window,
                        reward_tracker=reward_tracker,
                    )
                    agents_list.append(agent)
        else:
            # Use old system
            for alg in [p0_algo, p1_algo]:
                if alg == "LLMPair":
                    if collab_mode != "human":
                        assert llm_model_name is not None, print('you should choose a llm model')
                    if mode == "OpenSource":
                        assert os.path.exists(variant.get('model_dirname', '')), print(f"you should input right open-source model absolute path")
                    display_name = "Human" if collab_mode == "human" else llm_model_name
                    print(f"\n----Use {display_name} ({collab_mode})----\n")
                    if collab_mode == "human":
                        assert check_port_in_use(variant.get("local_server_api", "http://localhost:8080")), print(f"port {variant.get('local_server_api', 'http://localhost:8080')} is busy")
                        change_port(variant.get("local_server_api", "http://localhost:8080"))
                    
                    gpt_model = "human" if collab_mode == "human" else llm_model_name
                    model_dirname = variant.get('model_dirname', '~/')
                    local_server_api = variant.get('local_server_api', 'http://localhost:8000/v1')
                    retrival_method = variant.get('retrival_method', 'recent_k')
                    K = variant.get('K', 3)
                    
                    agent = make_agent(
                        alg,
                        mdp,
                        layout,
                        model=gpt_model,
                        model_dirname=model_dirname,
                        local_server_api=local_server_api,
                        retrival_method=retrival_method,
                        K=K,
                        actor=actor_list[actor_num],
                        history_window=history_window,
                        reward_tracker=reward_tracker,
                    )
                else:
                    agent = make_agent(alg, mdp, layout)
                agents_list.append(agent)
                actor_num += 1

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0

        
        if mode == 'exp':
            for t in range(horizon):
                s_t = env.state
                # print(s_t.timestep, env.t)
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                map = env.mdp.state_string(s_t).replace('ø', 'o')
                print(map)   
                a_t, ingredient_for_pickup = team.joint_action(s_t) 
                print(a_t)
                dialogue_t = team.reset_dialogue()
                print(f"\n-----------Controller-----------\n")    
                print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}")
                parm = ingredient_for_pickup

                obs, reward, done, env_info = env.step(a_t,parm)

                ml_actions = obs.ml_actions
                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills)

                reward_info = None
                if reward_tracker:
                    # 记录过程奖励，便于日志与可视化分析
                    reward_info = reward_tracker.after_step(t, ml_actions, env.state)
                    statistics_dict["process_rewards"].append(reward_info)

                r_total += reward
                if reward>0:
                    statistics_dict['total_order_finished'].append(s_t.current_k_order[0])
                    team.agents[1].teammate_ml_actions.append({'timestamp':t,'action':"deliver_soup()"})
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')
                print(f"P0's real behavior: {team.agents[1].teammate_ml_actions}")
                print(f"P1's real behavior: {team.agents[0].teammate_ml_actions}")


                #save statistics 
                turn_statistics_dict_agent0 = team.agents[0].turn_statistics_dict
                turn_statistics_dict_agent1 = team.agents[1].turn_statistics_dict

                turn_statistics_dict_both = combine_statistic_dict(turn_statistics_dict_agent0,turn_statistics_dict_agent1,map,reward)
                if reward_info:
                    turn_statistics_dict_both["statistical_data"]["process_reward"] = reward_info

                statistics_dict['total_timestamp'].append(t)
                statistics_dict['total_score'] = r_total
                statistics_dict['total_action_list'][0] = team.agents[1].teammate_ml_actions
                statistics_dict['total_action_list'][1] = team.agents[0].teammate_ml_actions
                statistics_dict['content'].append(turn_statistics_dict_both)
                #statistics_dict['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(filename, 'w') as f:
                    json.dump(statistics_dict,f,indent=4)
                
                if variant['test_mode'] == 'fix_task':
                    if reward != 0:
                        print("Task successed!")
                        #Human-eval: set task success message
                        if collab_mode == "human":
                            for a in range(len(team.agents)):
                                output_to_port(
                                    f"agent{a}",
                                    "Success!",
                                    mission="success",
                                    port=variant.get('local_server_api', "http://localhost:8080"),
                                )
                        break
            #Human-eval: set task failed message
            if collab_mode == "human":
                for a in range(len(team.agents)):
                    output_to_port(
                        f"agent{a}",
                        "Fail to finish task in time!",
                        mission="fail",
                        port=variant.get('local_server_api', "http://localhost:8080"),
                    )
        print(f"Episode {i+1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)
   
    end_time = time.time()
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")


    
if __name__ == '__main__':

    parser = ArgumentParser(description='OvercookedAI Experiment')

    # these are basis parses
    parser.add_argument('--layout', '-l', type=str, default='new_env', choices=['new_env'])
    parser.add_argument('--p0',  type=str, default='LLMPair', choices=['LLMPair', 'Human'], help='Algorithm for P0 agent 0')
    parser.add_argument('--p1', type=str, default='LLMPair', choices=['LLMPair', 'Human'], help='Algorithm for P1 agent 1')
    parser.add_argument('--horizon', type=int, default=120, help='Horizon steps in one game')
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')

    # these parsers are only required when using LLMPair.

    parser.add_argument('--collab_mode', type=str, default='llm', choices=['llm', 'human'], help='Whether collaborators are LLMS or humans')
    parser.add_argument('--llm_model', '--gpt_model', dest='llm_model', type=str, default='gpt-3.5-turbo-0125',
                        help='LLM identifier when collab_mode=llm')
    
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k', 'bert_topk'], help='Use similarity-based(BERT, CLIP) retrieval or retrieve recent K history in dialog.')
    parser.add_argument('--K', type=int, default=0, help="The number of dialogues you want to retrieve.")
    parser.add_argument('--history_window', type=int, default=3, help='Number of past decision snippets to include (0 disables history)')

    # 
    parser.add_argument('--model_dirname', type=str, default='.', help='absolute path of open-source model')      
    parser.add_argument('--local_server_api', type=str, default= "http://localhost:8000/v1", help='IP and port address to connect with local open source llm')     
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'debug_validator', 'develop'], help='exp mode run step-by-step, demo mode run via traj')                                
    parser.add_argument('--test_mode', type=str, default='fix_task', choices=['fix_task', 'fix_time'])
    parser.add_argument('--save', type=boolean_argument, default=True, help='Whether save the result')
    parser.add_argument('--log_dir', type=str, default=None, help='dir to save result')
    parser.add_argument('--debug', type=boolean_argument, default=True, help='debug mode')
    parser.add_argument('--order', type=str, default="", help='1 task order name')
    parser.add_argument('--run_id', type=str, default=None, help='Unique run identifier (optional)')

    #
    parser.add_argument('--statistics_save_dir', type=str, default='data', help='save directory of LLM statistics')
    parser.add_argument('--config_path', type=str, default=None, help='Path to YAML config (overrides CLI arguments)')


    args = parser.parse_args()

    start_time = time.time()
    if args.config_path:
        main(config_path=args.config_path)
    else:
        variant = vars(args)
        variant.pop('config_path', None)
        main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
