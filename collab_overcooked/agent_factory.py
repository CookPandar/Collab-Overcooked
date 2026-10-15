from overcooked_ai_py.planning.planners import MediumLevelPlanner

from .agents.collab import LLMAgents


def make_agent(mdp, layout, observer=None, **llm_args):
    """Build an LLM agent; `llm_args` are passed to LLMAgents (model, actor, history_window, backend, ...).
    `observer` receives the agent events (see docs/extension_api.md)."""
    counter_locations = mdp.get_counter_locations()
    mlam_params = {
        "start_orientations": False,
        "wait_allowed": True,
        "counter_goals": counter_locations,
        "counter_drop": counter_locations,
        "counter_pickup": counter_locations,
        "same_motion_goals": True,
    }
    mlam = MediumLevelPlanner.from_pickle_or_compute(mdp, mlam_params, force_compute=True)
    agent = LLMAgents(mlam, layout, **llm_args)
    agent.observer = observer
    agent.set_mdp(mdp)
    return agent


def combine_statistic_dict(dict1, dict2, map, score):
    """Merge the turn statistics of agent 0 (dict1) and agent 1 (dict2) into one timestep record."""
    rs = dict1
    rs["actions"].append(dict2["actions"][0])
    rs["map"] = map
    rs["statistical_data"]["score"] = score
    rs["statistical_data"]["communication"][1] = dict2["statistical_data"]["communication"][1]
    rs["statistical_data"]["error"][1] = dict2["statistical_data"]["error"][1]
    rs["statistical_data"]["error_correction"][1] = dict2["statistical_data"]["error_correction"][1]

    for key in ["observation", "reflection", "content", "action_list", "original_log", "action_source"]:
        rs["content"][key][1] = dict2["content"][key][1]
    return rs
