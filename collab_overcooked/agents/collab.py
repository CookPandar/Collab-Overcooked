import copy
import itertools
import os
import queue
import re
import warnings
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from rich import print as rprint
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.search import find_path, get_intersect_counter, query_counter_states

from ..config import PROMPT_DIR
from .modules import Module, statistics_dict, turn_statistics_dict

NAME_TO_ACTION = {
    "NORTH": Direction.NORTH,
    "SOUTH": Direction.SOUTH,
    "EAST": Direction.EAST,
    "WEST": Direction.WEST,
    "INTERACT": Action.INTERACT,
    "STAY": Action.STAY,
}


class LLMPair(object):

    def __init__(
        self,
        model="gpt-3.5-turbo-0125",
        model_dirname="",
        local_server_api="http://localhost:8000/v1",
    ):
        self.agent_index = None
        self.model = model
        self.model_dirname = model_dirname
        self.local_server_api = local_server_api

    def set_agent_index(self, agent_index):
        raise NotImplementedError

    def action(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class LLMAgents(LLMPair):

    def __init__(
        self,
        mlam,
        layout,
        model="gpt-3.5-turbo-0125",
        model_dirname="",
        local_server_api="http://localhost:8000/v1",
        api_key=None,
        base_url=None,
        temperature=0.7,
        timeout: Optional[float] = None,
        enable_thinking=None,
        backend=None,
        actor="",
        auto_unstuck=False,
        controller_mode="new",  # the default overcooked-ai Greedy controller
        debug_mode="N",
        agent_index=None,
        outdir=None,
        history_window=3,
        response_language=None,
        response_protocol="think_action",
        reflection=True,
    ):
        super().__init__(
            model=model, model_dirname=model_dirname, local_server_api=local_server_api
        )

        if response_protocol not in ("think_action", "action_only"):
            raise ValueError(f"Unknown response protocol: {response_protocol}")
        self.response_protocol = response_protocol
        # think_action: Think / Recent Goal / Action; action_only: a single Action field
        self.requires_reasoning_fields = response_protocol == "think_action"
        self.output_format_name = (
            "Think / Recent Goal / Action" if self.requires_reasoning_fields else "single Action field"
        )
        self.reflection = reflection
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.backend = backend
        self.trace = True
        self.debug_mode = "Y"
        self.controller_mode = controller_mode
        self.mlam = mlam
        self.layout = layout
        self.mdp = self.mlam.mdp
        self.test_mode = False
        self.test_ml_action = deque([])

        self.out_dir = outdir
        self.agent_index = agent_index

        self.timeout = timeout

        self.prev_state = None
        self.auto_unstuck = auto_unstuck

        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.pot_id_to_pos = []
        self.action_wait_parse = queue.Queue()
        self.end_talk = False
        self.actor = actor
        self.name = "Chef" if self.actor == "chef" else "Assistant"
        self.communication_role = "ask"
        self.recipe = {}
        self.order = ""
        self.failed_history = []
        self.state = None
        self.response_language = response_language
        # dict to record if the error in T timestamp  was corrected, or just 'wait(1)'
        self.error_correct = {}
        self.history_records = []
        self.conversation_history = []
        self.current_turn_conversation = []
        self.conversation_history_timestamp = None
        self.current_observation_snapshot = None
        self.current_recent_goal_text = "[EMPTY]"
        self.pending_collab_reply = False
        self.pending_llm_logs = []
        # extension interface
        self.observer = None
        self.last_call = None
        self.current_action_source = None
        self._queued_sources = deque()
        self._call_timestep = None
        self._next_call_index = 0
        self._last_validation_failed_format = False
        self.communication_turn_limit = 3
        self._communication_turn_counter = 0
        self._communication_turn_timestamp = None
        self._collab_ack_consumed = False
        if history_window is None:
            window_value = 0
        else:
            try:
                window_value = int(history_window)
            except (TypeError, ValueError):
                window_value = 0
        self.history_window = max(0, window_value)
        turn_statistics_dict_cp = copy.deepcopy(turn_statistics_dict)
        self.turn_statistics_dict = turn_statistics_dict_cp
        # self.generate_layout_prompt()

    def set_mdp(self, mdp: OvercookedGridworld):
        self.mdp = mdp

    def create_gptmodule(self, module_name):
        print(f"\n--->Initializing GPT {module_name}<---\n")
        if module_name != "planner":
            raise Exception(f"Module {module_name} not supported.")
        messages = [{"role": "system", "content": ""}]
        return Module(
            messages,
            self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            model_dirname=self.model_dirname,
            local_server_api=self.local_server_api,
            temperature=self.temperature,
            timeout=self.timeout,
            enable_thinking=self.enable_thinking,
            backend=self.backend,
        )

    # 	return messages
    def history_rule(self):
        window = self.history_window
        if window <= 0:
            return "1. `History`: disabled, you only see the current observation."
        records = "record" if window == 1 else f"{window} records"
        return (
            f"1. `History (latest decisions)` (when available): your last {records}, each with the "
            "Order, Scene, Agent State and Past conversation of that timestep and the Think / Recent Goal / Action you produced."
        )

    def load_prompt_file(self, mode="origin"):
        prompt_dir = os.path.join(PROMPT_DIR, "gpt")
        with open(os.path.join(prompt_dir, "prompt.txt"), "r") as g:
            prompt = g.read()
        if mode == "reflection":
            rule_file = "reflection_rule.txt"
        elif self.response_protocol == "action_only":
            rule_file = "communication_rule_action_only.txt"
        else:
            rule_file = "communication_rule.txt"
        with open(os.path.join(prompt_dir, rule_file), "r") as g:
            communication_rule = g.read().replace("{history_rule}", self.history_rule())
        prompt = prompt.replace("{communication_rule}", communication_rule)
        prompt = prompt.replace("{role}", self.name)
        prompt = prompt.replace("{teammate}", self.teammate.name)

        with open(os.path.join(prompt_dir, "environment_rule.txt"), "r") as g:
            environment_rule = g.read().replace("{role}", self.name)
        prompt = prompt.replace("{environment_rule}", environment_rule)
        character_player = "Suppose you are a player who is proficient in the overcooked_ai game. Your goal is to cooperate with your teammate who is also a LLM agent in order to get a high score."
        character_reflection = "Suppose you are a reflector who is proficient in the overcooked_ai game. Your goal is to analyzes the self-correction process of an LLM player after making a wrong action, and summarize the experience to help LLM players to reference and avoid making mistakes again."
        prompt = prompt.replace(
            "{character}", character_reflection if mode == "reflection" else character_player
        )
        with open(os.path.join(prompt_dir, f"{self.actor}_skill.txt"), "r") as g:
            prompt = prompt.replace("{skill}", g.read())

        if self.response_language:
            prompt += (
                "\n\nLanguage\n--------\n"
                f"- Respond only in {self.response_language}.\n"
                "- Do not switch languages based on teammate messages.\n"
            )

        # only the chef knows the recipe
        if self.actor == "chef":
            self.load_recipe()
            prompt = prompt.replace(
                "{has_recipe}",
                "You have recipe, so you need to direct yourself and your teammates to complete the order.",
            )
        else:
            self.teammate.load_recipe()
            prompt = prompt.replace(
                "{has_recipe}",
                "You do not have recipe, and you should always ask the chef for guidance, rather than making bad decisions on your own.",
            )
        recipe_content = "".join(value + "\n\n\n" for value in self.recipe.values())
        prompt = prompt.replace(
            "{recipe}", recipe_content if recipe_content != "" else "You do not have the recipe\n"
        )
        self.planner.instruction_head_list = [{"role": "system", "content": prompt}]
        if mode == "origin":
            statistics_dict.setdefault("prompt_templates", {})[self.name] = {
                "role": self.name,
                "order": self.order,
                "prompt": prompt,
            }
        return prompt

    def reset(self, teammate: LLMPair):
        self.planner.reset()
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.current_timestep = 0
        self.teammate_ml_actions = []
        self.teammate = teammate
        self._clear_action_queue()
        self.last_call = None
        self.current_action_source = None

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index
        self.planner = self.create_gptmodule("planner")
        self.planner.name = self.name

        # print(self.planner.instruction_head_list[0]['content'])

    def generate_layout_prompt(self):
        layout_prompt = f"Chef space:"
        if self.name == "Chef":
            utensil_list_chef = self.mdp.utensil_list_chef
            utensil_list_assistant = self.teammate.mdp.utensil_list_assist
        elif self.name == "Assistant":
            utensil_list_chef = self.teammate.mdp.utensil_list_chef
            utensil_list_assistant = self.mdp.utensil_list_assist

        for u in utensil_list_chef:
            layout_prompt += u + "  "
        layout_prompt += "counter \n"

        layout_prompt += "Assistant space:"
        for u in utensil_list_assistant:
            layout_prompt += u + "  "
        layout_prompt += "dish_dispenser  ingredient_dispenser\n"

        return layout_prompt

    def generate_state_prompt(self, state):
        # Ensure both agents have populated utensil accessibility lists before describing layout
        self.build_access_utensil(state)
        if getattr(self, "teammate", None):
            self.teammate.build_access_utensil(state)
        self._ensure_conversation_timestamp(state.timestep)
        self.layout_prompt = self.generate_layout_prompt()
        self.current_timestep = state.timestep
        ego = state.players[self.agent_index]
        teammate = state.players[1 - self.agent_index]

        time_prompt = f"Scene {state.timestep}: "
        ego_object = ego.held_object.name if ego.held_object else "nothing"
        teammate_object = (
            teammate.held_object.name if teammate.held_object else "nothing"
        )
        ego_state_prompt = f"<{self.name}> holds "

        chef = self if self.actor == "chef" else self.teammate
        if ego_object in list(chef.recipe.keys()):
            ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
        elif ego_object == "nothing":
            ego_state_prompt += f"{ego_object}. "
        else:
            ego_state_prompt += f"one {ego_object}. "

        current_action_text = self.current_ml_action if self.current_ml_action else "[EMPTY]"
        action_state_prompt = (
            f"The current action being executed by {self.name} is [{current_action_text}] "
        )

        teammate_state_prompt = f"<{self.teammate.name}> holds "
        if teammate_object == "soup":
            teammate_state_prompt += f"a dish with {teammate_object}. "
        elif teammate_object == "nothing":
            teammate_state_prompt += f"{teammate_object}. "
        else:
            teammate_state_prompt += f"one {teammate_object}. "

        teammate_current_action = (
            self.teammate.current_ml_action if self.teammate.current_ml_action else "[EMPTY]"
        )
        teammate_action_state_prompt = (
            f"The current action being executed by {self.teammate.name} is [{teammate_current_action}] "
        )

        kitchen_state_prompt = "Kitchen states: "
        prompt_dict = {
            "empty": "<{utensil_name}> is empty; ",
            "cooking": "<{utensil_name}> starts processing {food},it will be ready after {t} timesteps; ",
            "ready": "<{utensil_name}> has {food}; ",
            "full": "<{utensil_name}> has {food}, which are all the ingredients you need to make {transition}; ",
            "partially_full": "<{utensil_name}> has {food}, which is part of {transition}; ",
            "wrong": "<{utensil_name}> has {food},which seems do not belong to any recipe;",
        }
        utensil_states_dict = self.mdp.get_utensil_states(state)

        for key in utensil_states_dict.keys():
            for utensil in utensil_states_dict[key]:
                if key == "empty":
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil
                    )
                elif key == "cooking":
                    order = self.mdp.utensil_state_dict[utensil]["order"]
                    utensil_type = utensil[:-1]
                    # The cook time for all kinds ingredient of one certain order in an utensil should be the same
                    middle_ingredient_dict = self.mdp.recipe_config["recipes"][
                        utensil_type
                    ]
                    middle_ingredient_dict_list = list(middle_ingredient_dict.keys())
                    ingredient_in_utensil_first = self.mdp.utensil_state_dict[utensil][
                        "soup"
                    ].state[0]
                    middle_ingredient = ""
                    for m in middle_ingredient_dict_list:
                        if ingredient_in_utensil_first in m:
                            middle_ingredient = m
                    if middle_ingredient == "":
                        raise ValueError(
                            f"No operation time for {ingredient_in_utensil_first} in {utensil_type}"
                        )
                    recipe_cook_time = middle_ingredient_dict[middle_ingredient][
                        "cook_time"
                    ]
                    now_cook_time = self.mdp.utensil_state_dict[utensil]["soup"].state[
                        2
                    ]
                    food_desc = self._format_food_description(
                        self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                    )

                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=food_desc,
                        t=recipe_cook_time - now_cook_time,
                    )
                elif key == "ready":
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=self._format_food_description(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ),
                    )
                elif key == "full":
                    for rpe_name, rpe in self.mdp.recipes[utensil[:-1]].items():
                        if set(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ) == set(rpe["recipe"]):
                            transition = rpe_name
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=self._format_food_description(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ),
                        transition=transition,
                    )
                elif key == "partially_full":
                    for rpe_name, rpe in self.mdp.recipes[utensil[:-1]].items():
                        if (
                            len(
                                set(
                                    self.mdp.utensil_state_dict[utensil]["soup"].state[
                                        0
                                    ]
                                ).intersection(set(rpe["recipe"]))
                            )
                            > 0
                        ):
                            transition = rpe_name
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=self._format_food_description(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ),
                        transition=transition,
                    )
                elif key == "wrong":
                    kitchen_state_prompt += prompt_dict[key].format(
                        utensil_name=utensil,
                        food=self._format_food_description(
                            self.mdp.utensil_state_dict[utensil]["soup"].state[0]
                        ),
                    )
        intersect_counters = get_intersect_counter(
            state.players_pos_and_or[self.agent_index],
            state.players_pos_and_or[1 - self.agent_index],
            self.mdp,
            self.mlam,
        )
        counter_states = query_counter_states(self.mdp, state)

        kitchen_state_prompt += (
            "{} counters can be visited by <{}>. Their states are as follows: ".format(
                len(intersect_counters), self.name
            )
        )
        count_states = {}
        for i in intersect_counters:
            obj_i = "nothing"
            if counter_states[i] != " ":
                obj_i = counter_states[i]
            if obj_i in count_states:
                count_states[obj_i] += 1
            else:
                count_states[obj_i] = 1
        total_obj = self.mdp.default_ingredients
        for i in count_states:
            if i == "nothing":
                continue
            kitchen_state_prompt += (
                f"{count_states[i]} counters have {i} which you can pick it up. "
            )
        if len(list(count_states.keys())) == 1:
            kitchen_state_prompt += "counters have nothing."
        # for i in total_obj:
        # 	if i not in count_states:
        # 		kitchen_state_prompt += f'No counters have {i}. '
        kitchen_state_prompt += "\n"

        wong_message_prompt = (
            state.error_message[0] if len(state.error_message) > 0 else ""
        )
        self.planner.layout = self.layout_prompt
        self.planner.wong_message_prompt = wong_message_prompt

        order_text = (self.order or "").strip()
        if order_text.lower().startswith("order:"):
            order_text = order_text.split(":", 1)[1].strip()
        if not self.mdp.one_task_mode and len(state.current_k_order) > 1:
            tail_orders = [
                o.strip()
                for o in state.current_k_order[1:]
                if isinstance(o, str) and o.strip()
            ]
            if tail_orders:
                order_text += "".join(f"<<{o}" for o in tail_orders)

        layout_section = self.layout_prompt.strip()
        kitchen_section = kitchen_state_prompt.strip()
        scene_components = [f"Scene {state.timestep}:"]
        if layout_section:
            scene_components.append(layout_section)
        if kitchen_section:
            scene_components.append(kitchen_section)
        scene_text = "\n".join(scene_components)
        agent_state_text = (
            ego_state_prompt
            + action_state_prompt
            + teammate_state_prompt
            + teammate_action_state_prompt
        ).strip()
        conversation_text = self.format_conversation_history()
        self.current_observation_snapshot = {
            "timestamp": self.current_timestep,
            "order": order_text,
            "scene": scene_text,
            "agent_state": agent_state_text,
            "past_conversation": conversation_text,
        }
        current_block = self.format_current_observation_block(
            self.current_timestep,
            order_text,
            scene_text,
            agent_state_text,
            conversation_text,
            self.current_recent_goal_text,
        ) + self.format_lessons()
        history_prompt = self.build_history_prompt()
        sections = []
        if history_prompt:
            sections.append("History (latest decisions):\n" + history_prompt + "\n")
        sections.append("Current Observation:\n" + current_block)
        return "".join(sections)

    def build_access_utensil(self, state):
        am = self.mlam
        index_list = [0, 1]
        if self.mdp.utensil_list_chef == [] or self.mdp.utensil_list_assist == []:
            for i in index_list:
                player = state.players[i]
                for utensil in self.mdp.utensil_list:
                    motion_goals = am.ml_action_manager.go_to_utensil_actions(state, utensil, i)
                    motion_goals = [
                        mg
                        for mg in motion_goals
                        if self.mlam.mp.is_valid_motion_start_goal_pair(
                            player.pos_and_or, mg
                        )
                    ]
                    if len(motion_goals) > 0:
                        if i == 0:
                            self.mdp.utensil_list_chef.append(utensil)
                        else:
                            self.mdp.utensil_list_assist.append(utensil)
        else:
            return

    ##################
    """
	The followings are the Planner part
	"""
    ##################

    def action(self, state):
        self.state = state
        self.build_access_utensil(state)
        turn_statistics_dict_cp = copy.deepcopy(turn_statistics_dict)
        self.turn_statistics_dict = turn_statistics_dict_cp
        start_pos_and_or = state.players_pos_and_or[self.agent_index]

        # only use to record the teammate ml_action,
        # if teammate finish ml_action in t-1, it will record in s_t,
        # otherwise, s_t will just record None,
        # and we here check this information and store it
        self.current_timestep = state.timestep
        self.planner.current_timestep = state.timestep
        self._reset_comm_turn_counter()
        self._collab_ack_consumed = False
        self.teammate.order = state.current_k_order[0]
        self.order = state.current_k_order[0]
        self.change_communication_role("ask", "answer")
        self.planner.dialog_history_list = []
        self.teammate.planner.dialog_history_list = []
        # check if the teammate has finsihed his action
        if self.teammate.current_ml_action_steps > 0:
            current_ml_action_done = self.teammate.check_current_ml_action_done(state)
            if current_ml_action_done:
                self.teammate.current_ml_action = None

        if state.ml_actions[1 - self.agent_index] != None:
            self.teammate_ml_actions.append(
                {
                    "timestamp": self.current_timestep,
                    "action": state.ml_actions[1 - self.agent_index],
                }
            )

        # if current ml action does not exist, generate a new one
        if self.current_ml_action is None:
            self.current_ml_action = self.generate_ml_action(state)

        # when "wait" and has other action in action_wait_parse ,replace wait as the action
        if "wait" in self.current_ml_action and not self.action_wait_parse.empty():
            self.current_ml_action = self.generate_ml_action(state)

        # if the current ml action is in process, Player{self.agent_index} done, else generate a new one
        if self.current_ml_action_steps > 0:
            current_ml_action_done = self.check_current_ml_action_done(state)
            if current_ml_action_done:
                # generate a new ml action
                self.generate_success_feedback(state)
                self.current_ml_action = None
                self.current_ml_action = self.generate_ml_action(state)
        count = 0
        if self.current_ml_action_steps == 0:
            self.failed_message = self.validate_current_ml_action(state)
            self._emit_validation()
            if "success" in self.failed_message and self.test_mode:
                self.test_ml_action.popleft()
            # only try to self-correct 1 time
            while "success" not in self.failed_message:
                if self.test_mode:
                    self.current_ml_action = "wait(1)"
                    self.time_to_wait = 1
                    print(self.failed_message)
                else:
                    # del all wait parse action
                    self._clear_action_queue()
                    if count >= 1:
                        self.current_ml_action = "wait(1)"
                        self.time_to_wait = 1
                        break
                    self.trace = False
                    self.generate_failure_feedback(
                        self.current_ml_action, self.failed_message
                    )
                    self.change_correct_prompt()
                    if not getattr(self, "_last_validation_failed_format", False):
                        self.turn_statistics_dict["statistical_data"]["error"][
                            self.agent_index
                        ]["validator_error"]["error_num"] += 1
                        self.turn_statistics_dict["statistical_data"]["error"][
                            self.agent_index
                        ]["validator_error"]["error_message"].append(self.failed_message)
                    self.current_ml_action = self.generate_ml_action(state)
                    count += 1
                self.failed_message = self.validate_current_ml_action(state)
                self._emit_validation()
        # generate rethink if the problem is solved by the correction and not just 'wait(1)'
        if not self.trace and "success" in self.failed_message:
            self.turn_statistics_dict["statistical_data"]["error_correction"][
                self.agent_index
            ]["validator_correction"]["correction_num"] += 1
            if self.reflection and self.current_timestep not in self.error_correct.keys():
                self.generate_rethink(state)

        # print(self.current_ml_action_steps)
        # print(self.time_to_wait)
        # print(self.check_current_ml_action_done(state))
        # print(f"{self.name} current action:{self.current_ml_action}")
        # assert(self.current_ml_action_steps<5)
        # statistic
        self.turn_statistics_dict["content"]["reflection"][self.agent_index] = (
            copy.deepcopy(self.failed_history)
        )
        queue_snapshot = list(self.action_wait_parse.queue)
        action_list = self._gather_pending_actions(
            self.current_ml_action, queue_snapshot
        )
        self.turn_statistics_dict["content"]["action_list"][self.agent_index] = (
            copy.deepcopy(action_list)
        )
        self.turn_statistics_dict["timestamp"] = self.current_timestep
        self.turn_statistics_dict["order_list"] = state.current_k_order
        self.turn_statistics_dict["actions"].append(self.current_ml_action)
        # save statistic data
        # del history
        self.planner.dialog_history_list = []
        self.teammate.planner.dialog_history_list = []

        self.trace = True

        # if "deliver_soup()" in self.current_ml_action:
        # 	self.teammate.teammate_ml_actions.append({'timestamp':self.current_timestep,'action':"deliver_soup()"})

        if self.parse_params_in_action(self.current_ml_action)[0] in ("wait", "check_recipe"):
            self.current_ml_action_steps += 1
            self.time_to_wait -= 1
            lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
            # chosen_action =lis_actions[np.random.randint(0,len(lis_actions))]
            chosen_action = (0, 0)
            # Use version 0.0.1 logic (from dependencies/overcooked_ai)
            self.prev_state = state
            return self._finalize_action_return(chosen_action, "")
        else:
            possible_motion_goals = self.find_motion_goals(state)
            current_motion_goal, chosen_action = self.choose_motion_goal(
                start_pos_and_or, possible_motion_goals, state
            )
        # if "wait" in self.current_ml_action:
        # 	print(f'current motion goal for P{self.agent_index} is wait')
        # else:
        # 	if current_motion_goal is None:
        # 		current_motion_goal = 'None'
        # 	print(f'current motion goal for P{self.agent_index} is {current_motion_goal}')

        if self.auto_unstuck and chosen_action != Action.INTERACT:
            if self.prev_state is not None and state.players == self.prev_state.players:
                if self.agent_index == 0:
                    joint_actions = list(
                        itertools.product(Action.ALL_ACTIONS, [Action.STAY])
                    )
                elif self.agent_index == 1:
                    joint_actions = list(
                        itertools.product([Action.STAY], Action.ALL_ACTIONS)
                    )
                else:
                    raise ValueError("Player index not recognized")

                unblocking_joint_actions = []
                for j_a in joint_actions:
                    if j_a != [Action.INTERACT, Action.STAY] and j_a != [
                        Action.STAY,
                        Action.INTERACT,
                    ]:
                        # Use version 0.0.1 logic (from dependencies/overcooked_ai)
                        new_state, _, _ = self.mlam.mdp.get_state_transition(
                            state, j_a
                        )
                        if (
                            new_state.players_pos_and_or
                            != self.prev_state.players_pos_and_or
                        ):
                            unblocking_joint_actions.append(j_a)
                unblocking_joint_actions.append([Action.STAY, Action.STAY])
                chosen_action = unblocking_joint_actions[
                    np.random.choice(len(unblocking_joint_actions))
                ][self.agent_index]

        self.prev_state = state
        if chosen_action is None:
            self.current_ml_action = "wait(1)"
            self.time_to_wait = 1
            chosen_action = Action.STAY

        if self.current_ml_action_steps == 0:
            self.current_ml_action_steps = 1

        # print(f'ml_action = {self.current_ml_action}')
        # print(f'P{self.agent_index} : {Action.to_char(chosen_action)}')
        # Use version 0.0.1 logic (from dependencies/overcooked_ai)
        primary_action, primary_params = self.parse_params_in_action(self.current_ml_action)
        if primary_action == "pickup":
            pickup_item = primary_params[0] if primary_params else ""
            return self._finalize_action_return(chosen_action, pickup_item)
        elif primary_action in self.mdp.interact_actions:
            return self._finalize_action_return(chosen_action, "[START]")
        else:
            return self._finalize_action_return(chosen_action, "")

    # Parse action as function(parms1,parms2)
    def parse_params_in_action(self, action: str, need_output=False):
        action = action.replace(" ", "")
        function_name = ""
        params = []
        pattern = (
            r"(?:\d+\.\s*)?(\w+)\s*(?:\((.*?)\))?"  # r"^\"?\d*\.?\"?\s*(\w+)\((.*)\)"
        )
        match = re.match(pattern, action)
        # function name and parmas with "()"
        if match:
            function_name = match.group(1)
            if match.group(2) == None:
                params = []
            else:
                params = match.group(2).split(",")
            for index, p in enumerate(params):
                params[index] = params[index].replace(" ", "")
                params[index] = params[index].replace("'", "")
                params[index] = params[index].replace('"', "")
            params = [p for p in params if p != ""]
            if len(params) > 1 and ("put_obj_in_utensil" in function_name):
                params = [params[1]]
        elif "NOTHING" in action or "nothing" in action:
            function_name = "wait"
            params = ["1"]
        # function name and parmas without "()"
        else:
            pattern = r"^\"?\d*\.?\"?\s*(\w+)\((.*)\)"
            if need_output:
                print(f"No match function like action in {action}")
            return function_name, params
        if need_output:
            print(f"function_name : {function_name},params_name:{params}")
        return function_name, params

    def load_recipe(self):
        if self.order in self.recipe.keys():
            warnings.warn("Prompt has load the recipe")
            return
        recipe_name_list = os.listdir(PROMPT_DIR + "/recipe/")
        recipe_filename = ""
        for r in recipe_name_list:
            r_name = r[2:-4]
            if self.order == r_name:
                recipe_filename = r
                break
        with open(PROMPT_DIR + "/recipe/" + recipe_filename) as r:
            self.recipe[self.order] = r.read()

    def _collab_requires_reply(self, text: str) -> bool:
        lowered = (text or "").lower()
        return ("collab(request" in lowered) or ("collab(seek" in lowered)

    def _strip_action_prefix(self, action_text: Optional[str]) -> str:
        if not isinstance(action_text, str):
            return ""
        stripped = action_text.strip()
        if stripped.lower().startswith("action:"):
            return stripped.split(":", 1)[1].strip()
        return stripped

    def _is_collab_action(self, action_text: Optional[str]) -> bool:
        normalized = self._strip_action_prefix(action_text)
        if not normalized:
            return False
        lowered = normalized.lower()
        if lowered.startswith("collab("):
            return True
        collab_primitives = ("request(", "seek(", "ack(", "deny(")
        return any(lowered.startswith(prefix) for prefix in collab_primitives)

    def _split_action_tokens(self, action_block: Optional[str]) -> List[str]:
        body = self._strip_action_prefix(action_block or "")
        body = self._sanitize_action_text(body)
        tokens: List[str] = []
        current: List[str] = []
        depth = 0
        for char in body:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1
            if char == ";" and depth == 0:
                token = "".join(current).strip()
                if token:
                    tokens.append(token)
                current = []
                continue
            current.append(char)
        token = "".join(current).strip()
        if token:
            tokens.append(token)
        return tokens

    def _token_has_trailing_text(self, token: str) -> bool:
        stripped = (token or "").strip()
        if not stripped:
            return False
        depth = 0
        saw_open = False
        for index, char in enumerate(stripped):
            if char == "(":
                depth += 1
                saw_open = True
            elif char == ")" and depth > 0:
                depth -= 1
                if saw_open and depth == 0:
                    return bool(stripped[index + 1 :].strip())
        return False

    def _inspect_action_mode(self, action_block: Optional[str]) -> Dict[str, Any]:
        tokens = self._split_action_tokens(action_block)
        collab_tokens = [token for token in tokens if self._is_collab_action(token)]
        embodied_tokens = [token for token in tokens if not self._is_collab_action(token)]
        malformed_tokens = [
            token
            for token in tokens
            if "```" in token
            or re.search(r"(?i)\bpython\b", token)
            or self._token_has_trailing_text(token)
        ]
        if malformed_tokens:
            mode = "malformed"
        elif not tokens:
            mode = "empty"
        elif collab_tokens and embodied_tokens:
            mode = "mixed"
        elif len(embodied_tokens) > 1:
            mode = "multi_embodied"
        elif collab_tokens:
            mode = "communication"
        else:
            mode = "planner_main"
        return {
            "mode": mode,
            "tokens": tokens,
            "collab_tokens": collab_tokens,
            "embodied_tokens": embodied_tokens,
            "malformed_tokens": malformed_tokens,
        }

    @staticmethod
    def _has_nonempty_tail_after_fenced_response(response: str) -> bool:
        """A complete fenced reply must not be followed by extra narration."""
        if not isinstance(response, str):
            return False
        stripped = response.strip()
        if not stripped.startswith("```"):
            return False
        closing = stripped.find("```", 3)
        if closing == -1:
            return True
        return bool(stripped[closing + 3 :].strip())

    def _strip_code_fences(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        content = stripped[3:]
        closing = content.find("```")
        tail = ""
        if closing != -1:
            tail = content[closing + 3 :].strip()
            content = content[:closing]
        content = content.strip()
        if "\n" in content:
            first_line, remainder = content.split("\n", 1)
            # Drop a markdown fence language tag, but keep a one-line action.
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+-]*", first_line.strip()):
                content = remainder
        content = content.strip()
        if tail:
            return self._append_with_newline(content, tail)
        return content

    def _replace_pending_action_plan(self, plan_tokens: List[str], *, include_primary: bool) -> None:
        self._clear_action_queue()
        for index, action in enumerate(plan_tokens):
            if not action or self._is_collab_action(action):
                continue
            if index == 0 and not include_primary:
                continue
            self._enqueue_action(action, self.last_call)

    def _reset_comm_turn_counter(self):
        if self._communication_turn_timestamp != self.current_timestep:
            self._communication_turn_counter = 0
            self._communication_turn_timestamp = self.current_timestep

    def _register_comm_turn(self) -> int:
        self._reset_comm_turn_counter()
        self._communication_turn_counter += 1
        return self._communication_turn_counter

    def _gather_pending_actions(self, primary_action: Optional[str], queued_actions):
        pending = []
        if primary_action and not self._is_collab_action(primary_action):
            pending.append(primary_action)
        for action in queued_actions or []:
            if action and not self._is_collab_action(action):
                pending.append(action)
        return pending

    def _ensure_conversation_timestamp(self, timestamp: int):
        if self.conversation_history_timestamp != timestamp:
            self.conversation_history_timestamp = timestamp
            self.current_turn_conversation = []

    def _sanitize_action_text(self, text: Optional[str]) -> str:
        if not isinstance(text, str):
            return ""
        cleaned = self._strip_code_fences(text)
        cleaned = cleaned.replace("```]", "")
        cleaned = cleaned.replace("[```", "")
        cleaned = cleaned.strip()
        # Many LLMs emit one action per line / code fence; treat newlines as action separators.
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.replace("\n", ";")
        # Remove stray markdown fences or unmatched brackets
        cleaned = cleaned.rstrip("`")
        cleaned = cleaned.rstrip("]")
        return cleaned

    def _action_body_for_conversation(self, action_text: Optional[str]) -> str:
        text = self._strip_action_prefix(action_text or "").strip()
        text = self._sanitize_action_text(text)
        return text or "[EMPTY]"

    def _implicit_action_conversation(self, action_text: Optional[str]) -> str:
        action_body = self._action_body_for_conversation(action_text)
        return f"Your teammate did not reply to you, but directly generated an action: {action_body}"

    def append_conversation_line(
        self,
        speaker,
        message: str,
        propagate: bool = True,
        timestamp: Optional[int] = None,
    ):
        text = (message or "").strip()
        if not text or text == "[NOTHING]":
            return
        if timestamp is None:
            timestamp = self.current_timestep
        if timestamp is not None:
            self.current_timestep = int(timestamp)
            self._ensure_conversation_timestamp(int(timestamp))
        line = f"{speaker}: {text}"
        self.conversation_history.append(line)
        if len(self.conversation_history) > 30:
            self.conversation_history = self.conversation_history[-30:]
        self.current_turn_conversation.append(line)
        if len(self.current_turn_conversation) > 20:
            self.current_turn_conversation = self.current_turn_conversation[-20:]
        if (
            self.current_observation_snapshot
            and self.current_observation_snapshot.get("timestamp") == self.conversation_history_timestamp
        ):
            self.current_observation_snapshot["past_conversation"] = self.format_conversation_history()
        if propagate:
            lower_text = text.lower()
            teammate = getattr(self, "teammate", None)
            if speaker == self.name:
                self.pending_collab_reply = False
                if "collab(" in lower_text:
                    self._collab_ack_consumed = True
                if teammate is not None:
                    needs_reply = self._collab_requires_reply(lower_text)
                    teammate.pending_collab_reply = (
                        needs_reply and not getattr(teammate, "_collab_ack_consumed", False)
                    )
            elif teammate is not None and speaker == teammate.name:
                needs_reply = self._collab_requires_reply(lower_text)
                self.pending_collab_reply = needs_reply and not self._collab_ack_consumed
                teammate.pending_collab_reply = False
        if propagate and getattr(self, "teammate", None) is not None:
            teammate_history = getattr(self.teammate, "conversation_history", None)
            if teammate_history is not None:
                self.teammate.append_conversation_line(
                    speaker,
                    message,
                    propagate=False,
                    timestamp=timestamp,
                )

    def format_conversation_history(self) -> str:
        if not self.current_turn_conversation:
            return "[EMPTY]"
        return "\n".join(self.current_turn_conversation[-10:])

    def update_recent_goal_text(self, new_goal: Optional[str], reset_on_empty: bool = False):
        if new_goal is None:
            return
        cleaned = (new_goal or "").strip()
        if cleaned:
            self.current_recent_goal_text = cleaned
        elif reset_on_empty:
            self.current_recent_goal_text = "[EMPTY]"

    # ---- extension interface -------------------------------------------------------------
    # observer: optional object with any of the methods on_llm_call, on_call_update, on_format_error,
    # on_action_bound, on_validation, on_action_executed (see docs/extension_api.md).

    def _emit(self, event: str, **payload):
        handler = getattr(self.observer, "on_" + event, None) if self.observer is not None else None
        if handler is not None:
            handler(self, **payload)

    def _query(self, call_type: str, metadata: Optional[dict] = None, **query_kwargs):
        """Query the planner and record the call. `call_type` is one of planner_main,
        validator_correction, communication, format_correction, rethink."""
        if self._call_timestep != self.current_timestep:
            self._call_timestep, self._next_call_index = self.current_timestep, 0
        source = {
            "agent_index": self.agent_index,
            "agent": self.name,
            "timestep": self.current_timestep,
            "call_index": self._next_call_index,
            "call_type": call_type,
        }
        self._next_call_index += 1
        self.planner.call_context = dict(source, **(metadata or {}))
        prompt = self.planner.current_user_message["content"]
        response, tokens = self.planner.query(**query_kwargs)
        self.last_call = source
        self._log_llm_call(source, prompt, response, tokens, metadata)
        self._emit("llm_call", call=dict(source), prompt=prompt, response=response,
                   metadata=getattr(self.planner, "last_metadata", None))
        return response, tokens

    def _relabel_last_call(self, call_type: str, **fields):
        """The latest call turned out to produce an embodied action (e.g. a communication reply)."""
        if self.last_call is None:
            return
        self.last_call = dict(self.last_call, call_type=call_type)
        if self.pending_llm_logs:
            self.pending_llm_logs[-1]["call_type"] = call_type
        self._emit("call_update", call=dict(self.last_call), fields=fields)

    def _emit_validation(self):
        self._emit("validation", action=self.current_ml_action, source=self.current_action_source,
                   ok="success" in self.failed_message, message=self.failed_message,
                   format_error=self._last_validation_failed_format)

    def _set_current_action_source(self, source: Optional[dict]):
        self.current_action_source = dict(source) if source else None

    def _clear_action_queue(self):
        while not self.action_wait_parse.empty():
            self.action_wait_parse.get()
        self._queued_sources.clear()

    def _enqueue_action(self, action: str, source: Optional[dict]):
        self.action_wait_parse.put(action)
        self._queued_sources.append(dict(source) if source else None)
        self._emit("action_bound", action=action, source=source, primary=False)

    def _log_llm_call(self, source: dict, prompt_text: str, response_text: str, tokens: int = 0, metadata: Optional[dict] = None):
        """Keep the raw input/output of every LLM call; saved to `original_log` of this timestep."""
        entry = {
            "timestamp": source["timestep"],
            "agent": self.name,
            "call_index": source["call_index"],
            "call_type": source["call_type"],
            "input": (prompt_text or "").strip(),
            "output": (response_text or "").strip(),
            "tokens": tokens,
        }
        if metadata:
            entry["metadata"] = metadata
        self.pending_llm_logs.append(entry)

    def _annotate_last_log_metadata(self, **fields):
        if not self.pending_llm_logs:
            return
        metadata = self.pending_llm_logs[-1].setdefault("metadata", {})
        for key, value in fields.items():
            if key == "format_issues":
                existing = metadata.setdefault(key, [])
                existing.extend(item for item in value if item not in existing)
            else:
                metadata[key] = value

    def _report_action_format_error(self, detail: str, source: Optional[dict] = None):
        """Count an action whose name or parameters are invalid as a format error."""
        format_error = self.turn_statistics_dict["statistical_data"]["error"][self.agent_index]["format_error"]
        format_error["error_num"] += 1
        format_error["error_message"].append(detail)
        self._annotate_last_log_metadata(format_issues=[detail])
        self._emit("format_error", source=source or self.current_action_source, detail=detail)

    def _report_format_issues(self, issues: List[str]):
        """Missing reply fields or a mixed / malformed Action field of the latest call."""
        if not issues:
            return
        self._annotate_last_log_metadata(format_issues=issues)
        for issue in issues:
            self._emit("format_error", source=self.last_call, detail=issue)

    ACTION_ISSUE_HINTS = {
        "mixed": "Your previous reply mixed communication actions and embodied actions in the same Action field. ",
        "multi_embodied": "Your previous reply contained more than one embodied action in the Action field, but only one embodied action is allowed per timestep. ",
        "malformed": "Your previous reply contained narration, code or extra text in the Action field. ",
    }

    def _rewrite_mixed_action_response(self, last_response: str, mode: str = "mixed"):
        base_prompt = self.planner.current_user_message.get("content", "")
        correction_prompt = (
            base_prompt
            + "\n\n"
            + self.ACTION_ISSUE_HINTS.get(mode, self.ACTION_ISSUE_HINTS["mixed"])
            + "Rewrite the full reply so that Action is pure: either only Collab(...) communication primitives "
            "or exactly one embodied kitchen action. Never mix both kinds in one reply, and do not add narration.\n\n"
            "Your last reply was:\n"
            + last_response
            + f"\n\nReturn a corrected reply that preserves the {self.output_format_name} format."
        )
        self.planner.current_user_message = {"role": "user", "content": correction_prompt}
        response, correction_tokens = self._query(
            "format_correction", {"action_issue": mode + "_action_types"}, stop="Scene", trace=True
        )
        statistical_data = self.turn_statistics_dict["statistical_data"]
        statistical_data["error"][self.agent_index]["format_error"]["error_num"] += 1
        statistical_data["error"][self.agent_index]["format_error"]["error_message"].append(correction_prompt)
        format_correction = statistical_data["error_correction"][self.agent_index]["format_correction"]
        format_correction["correction_num"] += 1
        format_correction["correction_tokens"].append(correction_tokens)
        return response, correction_tokens

    def _ensure_pure_action_response(self, response: str):
        """One extra LLM call to rewrite a reply whose Action field is mixed, chained or malformed."""
        action_text = self.parse_response(response, "action")
        action_info = self._inspect_action_mode(action_text)
        if self._has_nonempty_tail_after_fenced_response(response):
            action_info["mode"] = "malformed"
            action_info["malformed_tokens"] = action_info.get("tokens", [])
        self._annotate_last_log_metadata(action_mode=action_info["mode"])
        if action_info["mode"] not in {"mixed", "multi_embodied", "malformed"}:
            return response, 0, action_info

        self._report_format_issues([action_info["mode"] + "_action_types"])
        response, correction_tokens = self._rewrite_mixed_action_response(response, action_info["mode"])
        corrected_info = self._inspect_action_mode(self.parse_response(response, "action"))
        self._annotate_last_log_metadata(action_mode=corrected_info["mode"])
        if corrected_info["mode"] in {"mixed", "multi_embodied", "malformed"}:
            self._report_format_issues([corrected_info["mode"] + "_action_types"])
        return response, correction_tokens, corrected_info
    
    def _format_food_description(self, food_state):
        if isinstance(food_state, str):
            return food_state
        if isinstance(food_state, (list, tuple)):
            items = [str(x) for x in food_state if str(x)]
            if not items:
                return "[EMPTY]"
            if len(items) == 1:
                return items[0]
            return ", ".join(items)
        return str(food_state)

    def record_history_entry(self, think_text: str, recent_goal_text: str, action_text: str, error_text: str = ""):
        if not self.current_observation_snapshot:
            return
        entry = {
            "timestamp": self.current_observation_snapshot["timestamp"],
            "order": self.current_observation_snapshot["order"],
            "scene": self.current_observation_snapshot["scene"],
            "agent_state": self.current_observation_snapshot["agent_state"],
            "past_conversation": self.current_observation_snapshot["past_conversation"],
            "think": (think_text or "").strip() or "[EMPTY]",
            "recent_goal": (recent_goal_text or "").strip() or "[EMPTY]",
            "action": (action_text or "").strip() or "[EMPTY]",
        }
        if error_text:
            entry["error"] = error_text.strip()
        window = max(0, getattr(self, "history_window", 0))
        if window <= 0:
            self.history_records = []
            self.current_recent_goal_text = entry["recent_goal"]
            return
        self.history_records = [h for h in self.history_records if h["timestamp"] != entry["timestamp"]]
        self.history_records.append(entry)
        if len(self.history_records) > window:
            self.history_records = self.history_records[-window:]
        self.current_recent_goal_text = entry["recent_goal"]

    def build_history_prompt(self) -> str:
        window = max(0, getattr(self, "history_window", 0))
        if window <= 0 or not self.history_records:
            return ""
        current_ts = getattr(self, "current_timestep", None)
        usable_entries = (
            [h for h in self.history_records if current_ts is None or h["timestamp"] < current_ts]
            if current_ts is not None
            else self.history_records
        )
        if not usable_entries:
            return ""
        blocks = []
        for entry in usable_entries[-window:]:
            blocks.append(self.format_history_entry(entry))
        return "\n".join(blocks)

    def format_history_entry(self, entry: dict) -> str:
        block = [
            f"timestep {entry['timestamp']}:",
            f"Order: {entry['order']}",
            entry["scene"],
            f"Agent State: {entry['agent_state']}",
            "Past conversation:",
            entry.get("past_conversation", "[EMPTY]") or "[EMPTY]",
        ]
        if self.requires_reasoning_fields:
            block += [
                f"Think: {entry.get('think', '[EMPTY]')}",
                f"Recent Goal: {entry.get('recent_goal', '[EMPTY]')}",
            ]
        block += [f"Action: {entry.get('action', '[EMPTY]')}"]
        if entry.get("error"):
            block.append(f"Error: {entry['error']}")
        return "\n".join(block)

    def format_current_observation_block(self, timestamp: int, order_text: str, scene_text: str, agent_state_text: str, conversation_text: str, recent_goal_text: str) -> str:
        return (
            f"timestep {timestamp}:\n"
            f"Order: {order_text}\n"
            f"{scene_text}\n"
            f"Agent State: {agent_state_text}\n"
            "Past conversation:\n"
            f"{conversation_text}\n"
            + (f"Recent Goal: {recent_goal_text or '[EMPTY]'}\n" if self.requires_reasoning_fields else "")
        )

    def format_lessons(self) -> str:
        """Reflections obtained after corrected failures (latest 3), shown only when available."""
        lessons = [
            item["reflection_content"].strip()
            for item in self.failed_history[-3:]
            if item.get("reflection_content", "").strip()
        ]
        if not lessons:
            return ""
        return "Lessons from Past Failures:\n" + "".join(
            f"{index + 1}. {lesson}\n" for index, lesson in enumerate(lessons)
        )

    def _append_with_newline(self, base: str, extra: str, blank_line: bool = False) -> str:
        base = base or ""
        extra = extra or ""
        if not extra:
            return base
        if not base:
            return extra
        if not base.endswith("\n"):
            base += "\n"
        if blank_line and not base.endswith("\n\n"):
            base += "\n"
        return base + extra

    def _finalize_action_return(self, chosen_action, interaction_param):
        action_name = self.parse_params_in_action(self.current_ml_action or "")[0]
        if chosen_action == Action.INTERACT or action_name in self.mdp.interact_actions:
            self._emit("action_executed", action=self.current_ml_action, source=self.current_action_source,
                       env_action=chosen_action, interaction_param=interaction_param)
        self.turn_statistics_dict["content"]["action_source"][self.agent_index] = (
            dict(self.current_action_source) if self.current_action_source else None
        )
        original_log = self.turn_statistics_dict["content"].get("original_log")
        if not isinstance(original_log, list):
            original_log = [[], []]
            self.turn_statistics_dict["content"]["original_log"] = original_log
        original_log[self.agent_index] = copy.deepcopy(self.pending_llm_logs)
        self.pending_llm_logs = []
        return chosen_action, interaction_param

    def parse_ml_action(self, action_string, report_error=True):
        """Check the action name and parameters. Invalid actions are counted as format errors when `report_error`."""

        def invalid(detail):
            if report_error:
                self._report_action_format_error(detail)
            return False, detail

        ml_action = ""
        action, params = self.parse_params_in_action(action_string)
        self.parse_action = action
        self.parse_action_params = params
        # compare params with lower objects list
        for index, p in enumerate(self.parse_action_params):
            for u in self.mdp.utensil_list:
                if p == u.lower():
                    self.parse_action_params[index] = u
            for i in self.mdp.default_ingredients:
                if p == i.lower():
                    self.parse_action_params[index] = i
        # parse function like return false,meaning the return action list is invalid.
        if self.parse_action == "":
            detail = "Please ensure the Action field lists semicolon-separated function calls without extra narration."
            return invalid(detail)
        if self.parse_action == "place_obj_on_counter":
            if len(params) != 0:
                detail = "Wrong place_obj_on_counter() params. It should have 0 params."
                return invalid(detail)
            ml_action = "place_obj_on_counter()"
        elif self.parse_action == "pickup":
            if len(params) != 2:
                detail = "Wrong pickup() params. It should have 2 params: obj and distination."
                return invalid(detail)
            # check pickup from dispenser, and pickup from utensil, pick up from counter
            # check whether the item is in the recipe list
            if params[0] not in self.mdp.all_ingredients + ["dish"]:
                detail = f"Wrong pickup() params {params[0]}. It does not belong to any recipe ingredients or dish."
                return invalid(detail)
            if (
                (params[1] not in self.mdp.utensil_list)
                and ("counter" not in params[1])
                and ("dispenser" not in params[1])
            ):
                detail = f"Wrong pickup() params {params[1]}. It does not belong to any utensils, dispenser or counter."
                return invalid(detail)
            if ("dispenser" in params[1]) and (
                params[0] not in self.mdp.default_ingredients + ["dish"]
            ):
                detail = f"Wrong pickup() params {params[0]},{params[1]}. You can only get raw ingredients and dish from dispenser."
                return invalid(detail)
            # in case of pickup(dish,ingredient_dispenser) , pickup(ingredient,dish_dispenser)
            if ("dish" in params[0]) and ("ingredient_dispenser" in params[1]):
                detail = f"Wrong pickup() params {params[0]},{params[1]}. You can only get dish from dish_dispenser."
                return invalid(detail)
            if "dish" not in params[0] and "dish_dispenser" in params[1]:
                detail = f"Wrong pickup() params {params[0]},{params[1]}. You can only get ingredient from ingredient_dispenser."
                return invalid(detail)
            ml_action = f"pickup({params[0]},{params[1]})"
        elif self.parse_action == "put_obj_in_utensil":
            if len(params) != 1:
                detail = "Wrong put_obj_in_utensil() params. It should have 1 params: utensil."
                return invalid(detail)
            if params[0] in self.mdp.utensil_list:
                ml_action = f"put_obj_in_utensil({params[0]})"
            else:
                if params[0] == "dish":
                    detail = (
                        "Wrong put_obj_in_utensil() params: dish. A dish is not a utensil, so it cannot be used in put_obj_in_utensil(). "
                        "To serve food on a dish, first pickup(dish, dish_dispenser) or pickup(dish, counter), then fill_dish_with_food(utensil_name). "
                        "If the order does not need a dish, pickup(finished_food, utensil_name) and then deliver_soup()."
                    )
                else:
                    detail = f"Wrong put_obj_in_utensil() parmas:{params[0]}"
                return invalid(detail)
        elif self.parse_action == "fill_dish_with_food":
            if len(params) != 1:
                detail = "Wrong fill_dish_with_food() params. It should have 1 params: utensil."
                return invalid(detail)
            if params[0] in self.mdp.utensil_list:
                ml_action = f"fill_dish_with_food({params[0]})"
            else:
                detail = f"Wrong fill_dish_with_food() parmas:{params[0]}"
                return invalid(detail)
        elif self.parse_action == "deliver_soup":
            ml_action = "deliver_soup()"
        elif self.parse_action == "check_recipe":
            ml_action = "check_recipe()"
        # 	check all the action need to interact with utensils
        elif self.parse_action in self.mdp.interact_actions:
            if len(params) != 1:
                detail = "Please ensure the Action field is a semicolon-separated list of function calls without narration."
                return invalid(detail)
            utensils = self.mdp.interact_actions.get(self.parse_action, [])
            if params[0] in utensils:
                ml_action = f"{self.parse_action}({params[0]})"
            else:
                detail = f"Wrong {self.parse_action}() parmas:{params[0]}"
                return invalid(detail)
        elif self.parse_action == "wait":
            time_to_wait = self.parse_wait_string(action_string)
            if time_to_wait > 5:
                detail = "Too long wait time. It should be less than 5."
                return invalid(detail)
            else:
                self.time_to_wait = time_to_wait
                ml_action = action_string
        else:
            detail = "Please ensure the Action field is a semicolon-separated list of function calls without narration."
            return invalid(detail)

        return True, ml_action

    def parse_wait_string(self, s):
        # Check if it's just "wait"
        if s == "wait":
            return 1

        # Remove 'wait' and other characters from the string
        s = (
            s.replace("wait", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace(".", "")
        )

        # If it's a number, return it as an integer
        if s.isdigit():
            return int(s)

        # If it's not a number, return a default value or raise an exception
        return 1

    def change_communication_role(self, role, team_role):
        # self
        self.communication_role = role
        a = self.load_prompt_file()
        b = self.teammate.load_prompt_file()

    def build_correction_suffix(self):
        failure_notes = [
            dialog["content"]
            for dialog in self.planner.dialog_history_list
            if dialog.get("role") == "failure_explanation"
        ]
        suffix_lines = ["\n\n### Correction Guidance"]
        if failure_notes:
            suffix_lines.append("Controller feedback from the previous attempt:")
            suffix_lines.extend(failure_notes)
        suffix_lines.append(
            f"Regenerate a complete response that strictly follows the {self.output_format_name} format and resolves the issue above."
        )
        return "\n".join(suffix_lines)

    def change_correct_prompt(self):
        base_prompt = self.load_prompt_file(mode="origin")
        correction_suffix = self.build_correction_suffix()
        self.planner.instruction_head_list[0]["content"] = base_prompt + correction_suffix

    def message_formate_control(
        self, role: str, turn_dialog_self: list, turn_dialog_team: list
    ):
        pre_message = ""
        answer_num = 1
        think_turn = 1
        flag = False
        # turn 0
        # pre_message += f'''{self.name} think history turn 0 : [EMPTY]\n'''
        ##As asker, generate message for self
        # <example>:
        # Order:Onion_soup(No recipe)
        # Scene 0: Chef holds nothing. Assistant holds nothing. Assistant's action:None.Kitchen states: Chef space:Pot0 empty.Pot1 empty.chopping_board1 empty.Stirrer empty. Assistant space:chopping_board0 empty.Cooker empty.Counter:empty.
        # Assistant think history turn 0 : [EMPTY]
        # Assistant think history turn 1 : I do not know what to do. I should ask chef.
        # Assistant say turn 1 : Can you give me advice?
        # Chef  say turn 1 : You should wait.<END>
        # Assistant think : I need to wait.
        # Assistant say : [NOTHING]
        # Assistant plan: wait(1)
        # </example>
        if role == "asker":
            for t in turn_dialog_self:
                # if t["role"] == "think":
                # 	pre_message +=  f'''{self.name} think history turn {think_turn} : {t['content']}\n'''
                if t["role"] == "talk":
                    pre_message += f"""{self.name} say history turn {think_turn} : {t['content']}\n"""
                    # search teammate response
                    for r in turn_dialog_team:
                        if r["role"] == "talk":
                            if think_turn == answer_num:
                                pre_message += f"""{self.teammate.name} say history turn {think_turn} : {r['content']}\n"""
                                break
                            else:
                                answer_num += 1
                    think_turn += 1
        ##As answer, generate message for self
        # <example>:
        # Assistant think history turn 0 : [EMPTY]
        # Chef say history turn 1 : Can you pick up an onion for me ?
        # Assistant think history turn 1: I can go the ingredient_dispenser pick an onion for chef. I do not know if it is right.
        # Assistant say history turn 1: Do you want me pick onion from ingredient_dispenser?
        # Chef say history turn 2 : YES.
        # Assistant think: I known i need to pick onion from ingredient_dispenser.
        # Assistant say : OK<END>
        # Assistant plan : pickup(onion,ingredient_dispenser)
        # </example>
        elif role == "answer":
            for t in turn_dialog_team:
                if t["role"] == "talk":
                    pre_message += f"""{self.teammate.name} say history turn {think_turn} : {t['content']}\n"""
                    # search self think and talk
                    for r in turn_dialog_self:
                        if r["role"] == "think":
                            if think_turn == answer_num:
                                # pre_message += f'''{self.name} think history turn {think_turn} : {r['content']}\n'''
                                flag = True
                            else:
                                answer_num += 1
                        if r["role"] == "talk" and flag:
                            pre_message += f"""{self.name} say history turn {think_turn} : {r['content']}\n"""
                            flag = False
                            break
                    think_turn += 1
        else:
            raise ValueError("Wrong actor for build communication message!")
        return pre_message

    def communication(self, message, state):
        # [observation,Analysis \n Player 0:talk]
        self.end_talk = False
        you_response = message
        team_response = ""
        communication_turn = 0
        max_communication_turns = 3
        last_message = ""

        print("\n\n>>>>>>>>>>>>>>>>>>Begin communication<<<<<<<<<<<<<\n")
        while self.end_talk is False:
            communication_turn += 1
            if communication_turn > max_communication_turns:
                print(
                    f"[Warning] Communication exceeded {max_communication_turns} exchanges. Forcing termination."
                )
                self.end_talk = True
                you_response = "Action: wait(1)"
                break
            print(f"Input for {self.teammate.name}" + ":\n")
            format_you_response = self.teammate.message_formate_control(
                "answer",
                self.teammate.planner.dialog_history_list,
                self.planner.dialog_history_list,
            )
            self.teammate.current_timestep = state.timestep
            self.teammate.planner.current_timestep = state.timestep
            self.teammate.state_prompt = self.teammate.generate_state_prompt(state)
            print(
                self._append_with_newline(
                    self.teammate.state_prompt,
                    self.teammate.planner.wong_message_prompt,
                    blank_line=True,
                )
            )
            # teammate must reply
            self.end_talk, team_response = self.teammate.answer(
                format_you_response, "team", state
            )
            print(f"Answer of {self.teammate.name}" + ":\n")
            print(team_response + "\n\n")
            teammate_talk, _ = self.teammate.parse_response(team_response, "talk")
            self.append_conversation_line(self.teammate.name, teammate_talk)

            # A physical teammate response is already queued by ``answer``. End
            # the exchange here instead of asking the initiator to answer again,
            # which can reopen the dialogue and repeat the same request.
            if self.end_talk:
                you_response = "Action: wait(1)"
                break

            print(f"Input for {self.name}" + ":\n")
            format_team_response = self.message_formate_control(
                "asker",
                self.planner.dialog_history_list,
                self.teammate.planner.dialog_history_list,
            )
            self.current_timestep = state.timestep
            self.planner.current_timestep = state.timestep
            self.state_prompt = self.generate_state_prompt(state)
            print(
                self._append_with_newline(
                    self.state_prompt,
                    self.planner.wong_message_prompt,
                    blank_line=True,
                )
            )
            last_message = format_team_response + "\n\n"
            self.pre_message = last_message

            self.end_talk, you_response = self.answer(
                format_team_response, "you", state
            )
            if self.end_talk is False:
                print(f"Answer of {self.name}" + ":\n")
            else:
                print(
                    f">>>>>>>>>>>>>>>>>>>>{self.name} decide to make action:<<<<<<<<<<<<<<<<<\n"
                )
            print(you_response + "\n\n\n")
            you_talk, _ = self.parse_response(you_response, "talk")
            self.append_conversation_line(self.name, you_talk)
        print("\n\n>>>>>>>>>>>>>>>>>>Finish communication<<<<<<<<<<<<<\n")
        # parse
        action_block = self.parse_response(you_response, "action")
        if action_block == "":
            print("You did not provide an action last time.\n")
            action_block, _ = self.important_part_no_create(1, "action", you_response)
        return action_block

    def important_part_no_create(self, retry_num, part_type, response):
        part_display = {
            "think": "Think",
            "action": "Action",
            "talk": "Action",
            "recent_goal": "Recent Goal",
        }.get(part_type, part_type)
        prompt = (
            "\n\nYou did not create correct "
            + part_display
            + " part last time, now please remember to add "
            + part_display
            + " according to the format of example strictly!Below is the history:<BEGAIN>\n"
        )
        retry_num_max = retry_num
        final_response = response
        while retry_num > 0:
            # retry query
            self.planner.current_user_message = {
                "role": "user",
                "content": prompt
                + self.planner.current_user_message["content"]
                + "\n\nYour last response is :"
                + response
                + "\n\n<END>Now please return correct answer with your loss part.",
            }
            # print(self.planner.current_user_message)
            response, correction_tokens = self._query(
                "format_correction", {"missing_part": part_type}, stop="Scene", trace=True
            )
            response, extra_tokens, _ = self._ensure_pure_action_response(response)
            correction_tokens += extra_tokens
            # statistic
            self.turn_statistics_dict["statistical_data"]["error"][self.agent_index][
                "format_error"
            ]["error_num"] += 1
            self.turn_statistics_dict["statistical_data"]["error"][self.agent_index][
                "format_error"
            ]["error_message"].append(self.planner.current_user_message["content"])

            # print(response)
            parse_result = self.parse_response(response, part_type)
            retry_num -= 1
            final_response = response
            response = "\n\nYour last response is :" + response + "\n\n"
            if retry_num == 0 and parse_result == "":
                warnings.warn(
                    "Failed to create "
                    + part_display
                    + " for "
                    + str(retry_num_max)
                    + " times"
                )
                parse_result = "You do not have " + part_display + " last time."
            if parse_result != "":
                # correction_num means the number of truly correct the format error ,maybe a success need several trys.
                # correction_tokens means all the tokens for trying, so len(correction_tokens)>= correction_num
                self.turn_statistics_dict["statistical_data"]["error_correction"][
                    self.agent_index
                ]["format_correction"]["correction_num"] += 1
                self.turn_statistics_dict["statistical_data"]["error_correction"][
                    self.agent_index
                ]["format_correction"]["correction_tokens"].append(correction_tokens)
                break
        return parse_result, final_response

    # It is used to give organized messages to gpt and organize the generated think, talk and other information
    def answer(self, message, role, state):
        self.state_prompt = self.generate_state_prompt(state)
        self.planner.current_user_message = {
            "role": "user",
            "content": self._append_with_newline(self.state_prompt, message, blank_line=True),
        }
        response, tokens_num = self._query("communication", {"role": role}, stop="Scene", trace=True)
        extra_tokens = 0
        response, purity_tokens, action_info = self._ensure_pure_action_response(response)
        extra_tokens += purity_tokens
        format_issues: List[str] = []
        think_text = self.parse_response(response, "think")
        if self.requires_reasoning_fields and think_text == "":
            format_issues.append("missing_think")
        # Think did not generate error handling:
        if self.requires_reasoning_fields and think_text == "":
            print("Do not create think", response)
            # print(self.planner.current_user_message["content"])
            think_text, _ = self.important_part_no_create(1, "think", message)

        # Agent1 has decided to stop talking
        # if ~self.end_talk:
        parse_talk, end_talk = self.parse_response(response, "talk")
        action_text = self.parse_response(response, "action")
        if action_text == "":
            format_issues.append("missing_action")
        elif action_info["mode"] in {"mixed", "multi_embodied", "malformed"}:
            format_issues.append(action_info["mode"] + "_action_types")
        recent_goal_text = self.parse_response(response, "recent_goal")
        if self.requires_reasoning_fields and recent_goal_text == "":
            format_issues.append("missing_recent_goal")
        self.update_recent_goal_text(recent_goal_text)
        recent_goal_entry = recent_goal_text or self.current_recent_goal_text
        implicit_talk_override = None
        collab_reply_missing = (
            role == "team"
            and self.pending_collab_reply
            and "collab(" not in (parse_talk or "").lower()
        )
        if collab_reply_missing:
            implicit_talk_override = self._implicit_action_conversation(action_text)
            parse_talk = implicit_talk_override
            end_talk = not self._is_collab_action(action_text)
            self.pending_collab_reply = False
        if role == "team" and "collab(" in (parse_talk or "").lower():
            self._collab_ack_consumed = True
            self.pending_collab_reply = False
        self.planner.dialog_history_list.append({"role": "think", "content": think_text})
        total_tokens = tokens_num + extra_tokens
        self._report_format_issues(format_issues)

        # statistic
        if role == "you":
            communication_index = self.agent_index
            self.turn_statistics_dict["statistical_data"]["communication"][
                communication_index
            ]["turn"].append(parse_talk)
            self.turn_statistics_dict["statistical_data"]["communication"][
                communication_index
            ]["token"].append(total_tokens)
            self.turn_statistics_dict["content"]["content"][communication_index].append(
                {
                    "agent": self.agent_index,
                    "think": think_text,
                    "say": parse_talk,
                    "action": action_text,
                    "recent_goal": recent_goal_entry,
                }
            )
        else:
            communication_index = self.teammate.agent_index
            self.teammate.turn_statistics_dict["statistical_data"]["communication"][
                communication_index
            ]["turn"].append(parse_talk)
            self.teammate.turn_statistics_dict["statistical_data"]["communication"][
                communication_index
            ]["token"].append(total_tokens)
            self.teammate.turn_statistics_dict["content"]["content"][
                communication_index
            ].append(
                {
                    "agent": self.agent_index,
                    "think": think_text,
                    "say": parse_talk,
                    "action": action_text,
                    "recent_goal": recent_goal_entry,
                }
            )

        if implicit_talk_override is None and parse_talk == "":
            parse_talk, response = self.important_part_no_create(1, "talk", response)
        if implicit_talk_override is None:
            parse_talk, end_talk = self.parse_response(response, "talk")
            if parse_talk == "[NOTHING]" and end_talk:
                parse_talk = "OK.<END>"
        else:
            parse_talk = implicit_talk_override
        self.planner.dialog_history_list.append({"role": "talk", "content": parse_talk})

        # check if there is action
        # nothing did not produce an action
        new_plan = self.parse_response(response, "action")
        pattern = r"(?i)nothing"
        matches = re.findall(pattern, new_plan)
        if matches:
            return end_talk, response
        if self._is_collab_action(new_plan):
            return end_talk, response

        clean_plan = [
            a for a in self.parse_ml_action_top(new_plan, False) if not self._is_collab_action(a)
        ]
        if clean_plan:
            self._relabel_last_call("planner_main", reclassified_from="communication",
                                    produced_actions=list(clean_plan))
        # When correct, new action should also replace the old action list.
        if not self.trace:
            if clean_plan:
                self._replace_pending_action_plan(clean_plan, include_primary=True)
            return end_talk, response
        # A new action is queued only when no action is pending, otherwise it is ignored.
        if clean_plan and self.action_wait_parse.qsize() == 0:
            print(f"\nGenerate new action list <{clean_plan}>\n")
            self._replace_pending_action_plan(clean_plan, include_primary=True)
        elif self.action_wait_parse.qsize() >= 1:
            rprint(
                f"[yellow][ADD][/yellow]:Current action are too much. Does not add <{clean_plan}> in queue\n"
            )

        return end_talk, response

    # Extract the think and TALK from the GPT reply.
    # mode:think,<TALK>
    def parse_response(self, response, mode, need_correct=False):
        role = "Chef" if self.agent_index == 0 else "Assistant"
        text = self._strip_code_fences(response or "")

        section_pattern = re.compile(
            r"^\s*(Think|Recent Goal|Action)\s*:\s*(.*?)(?=^\s*(?:Think|Recent Goal|Action)\s*:|\Z)",
            re.IGNORECASE | re.DOTALL | re.MULTILINE,
        )
        sections = {}
        for match in section_pattern.finditer(text):
            header = match.group(1).strip().lower()
            content = (match.group(2) or "").strip()
            sections[header] = content

        if mode == "think":
            think_block = sections.get("think")
            if think_block:
                return think_block
            pattern = r"Think\s*:\s*(.*?)(?:\n\s*Recent Goal:|\n\s*Action:|\Z)"
            match = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match[0]
            if need_correct:
                response, _ = self.important_part_no_create(1, "think", response)
            else:
                response = ""
            return response
        elif mode == "recent_goal":
            recent_block = sections.get("recent goal")
            if recent_block:
                return recent_block
            pattern = r"Recent Goal\s*:\s*(.*?)(?:\n\s*Action:|\Z)"
            match = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match[0].strip()
            return ""
        elif mode == "talk":
            action_block = sections.get("action")
            if action_block:
                stripped = self._strip_action_prefix(action_block)
                stripped = self._sanitize_action_text(stripped)
                if self._is_collab_action(stripped):
                    return stripped, False
                if "[NOTHING]" in stripped.upper():
                    return "[NOTHING]", True
                return "[NOTHING]", True
            pattern = f"(?:{role})\\s+say\\s*:?\\s*(.*?)(?:\\s+|\\n+)?$"
            match = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                if "[NOTHING]" in match[0]:
                    return "[NOTHING]", True
                end_match = (
                    True
                    if re.findall(r"(.*?)(END)", match[0], re.DOTALL | re.IGNORECASE)
                    else False
                )
                return match[0], end_match
            return "[NOTHING]", True
        elif mode == "action":
            action_block = sections.get("action")
            if action_block:
                cleaned = self._sanitize_action_text(action_block)
                return f"Action: {cleaned}"
            action_pattern = r"Action\s*:\s*(.*)"
            match = re.search(action_pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                cleaned = self._sanitize_action_text(match.group(1))
                return f"Action: {cleaned}"
            plan_pattern = rf"{role}\s+plan\s*:?\s*(.*)"
            plan_match = re.search(plan_pattern, text, re.IGNORECASE | re.DOTALL)
            if plan_match:
                plan_body = self._sanitize_action_text(plan_match.group(1))
                return f"Action: {plan_body}"
            if need_correct:
                response, _ = self.important_part_no_create(1, "action", response)
            else:
                response = ""
            return response
        else:
            raise KeyError("Not exist mode for parse_response()")

    def generate_rethink(self, state):
        extracted_text = ""
        failture = ""
        message = self.message_formate_control(
            "asker",
            self.planner.dialog_history_list_storage,
            self.teammate.planner.dialog_history_list_storage,
        )
        failure_message = ""
        for index, s in enumerate(self.planner.dialog_history_list):
            if s["role"] == "failure_explanation":
                # Multi failed message
                if failure_message != "":
                    failure_message += (
                        f"When you handile the error above, {s['content']}"
                    )
                else:
                    failure_message = "Failed Reason:" + s["content"]
                    failture = s["content"]
        success_message = f"Solution: Your action '{self.current_ml_action}' successes at solving the problem.\n"
        self.load_prompt_file("reflection")
        self.state_prompt = self.generate_state_prompt(state)
        combined_prompt = self._append_with_newline(
            self._append_with_newline(
                self._append_with_newline(self.state_prompt, message, blank_line=True),
                failure_message,
                blank_line=True,
            ),
            success_message,
            blank_line=True,
        )
        self.planner.current_user_message = {
            "role": "user",
            "content": combined_prompt,
        }
        print(f"rethink input content: {self.planner.current_user_message['content']}")
        response, correction_tokens = self._query(
            "rethink", {"failure": failture}, stop="Scene", trace=self.trace, rethink=True
        )
        print(f"response of rethink: {response}")
        pattern = r"Rethink:(.*?)\."
        matches = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if matches:
            extracted_text = matches.group(1).strip()
            print(f"success extract rethink content.\n{extracted_text}")
            # count
            self.turn_statistics_dict["statistical_data"]["error_correction"][
                self.agent_index
            ]["validator_correction"]["reflection_obtain"] = self.current_timestep
        else:
            print("\nFailed to create rethink.\n")
        # problem + solution
        flag = 0
        for index, reflection in enumerate(self.failed_history):
            if failture in reflection["error"]:
                self.failed_history[index]["reflection_content"] += extracted_text
                flag = 1
        # If the reflection is not duplicated and reflection successes
        if flag == 0 and matches:
            self.failed_history.append(
                {
                    "error": failture,
                    "reflection_content": extracted_text,
                    "obtain_timestamp": self.current_timestep,
                }
            )
        return

    def generate_ml_action(self, state):
        """
        Selects a medium level action for the current state.
        Motion goals can be thought of instructions of the form:
                [do X] at location [Y]

        In this method, X (e.g. deliver the soup, pick up an onion, etc) is chosen based on
        a simple set of  heuristics based on the current state.

        Effectively, will return a list of all possible locations Y in which the selected
        medium level action X can be performed.
        """
        if self.test_mode:
            self.current_ml_action_steps = 0
            action = ""
            if len(self.test_ml_action) == 0:
                action = "wait(1)"
                self.time_to_wait = 1
                self.test_ml_action.append(action)
            else:
                action = self.test_ml_action[0]
                numbers = re.findall(r"\d+", action)
                if "wait" in action and numbers != []:
                    self.time_to_wait = int(numbers[0])
            return action

        # At most one error at a time
        failure_message = ""
        ml_action = ""
        selected_source = None
        think_output = ""
        recent_goal_entry = self.current_recent_goal_text
        if self.action_wait_parse.empty() or (not self.trace):
            state_prompt = self.generate_state_prompt(state)
            is_validator_correction = False
            # Error checking
            for index, s in enumerate(self.planner.dialog_history_list):
                if s["role"] == "failure_explanation":
                    is_validator_correction = True
                    # Multi failed message
                    if failure_message != "":
                        failure_message += (
                            f"When you handile the error above, {s['content']}\n"
                        )
                    else:
                        failure_message = (
                            "\nBelow are the failed and think history about your last chosen action. Use this information to reach a correct action alone AND DO NOT COMMUNICATE WITH YOUR TEAMMATE:\n"
                            + s["content"]
                            + "\n"
                        )
            state_prompt += failure_message

            print(f"\n\n### Observation module to " + self.name + "\n")

            self.state_prompt = state_prompt
            state_message = {
                "role": "user",
                "content": self._append_with_newline(
                    state_prompt,
                    self.planner.wong_message_prompt,
                    blank_line=True,
                ),
            }
            self.state_prompt = state_prompt
            self.planner.current_user_message = state_message
            # print("history information")
            print(f"{state_message['content']}")
            # statistic
            self.turn_statistics_dict["content"]["observation"][self.agent_index] = (
                state_message["content"]
            )

            print(f"\n\n\n### GPT Planner module\n")
            print("====== GPT Query ======")
            query_call_type = (
                "validator_correction" if is_validator_correction else "planner_main"
            )
            map_text = self.mdp.state_string(self.state).replace("ø", "o")
            response, tokens_num = self._query(
                query_call_type, None, stop="Scene", trace=self.trace, map=map_text
            )
            response, _, action_info = self._ensure_pure_action_response(response)
            print(response)
            # check whether need communication
            # check whether has the action
            format_issues: List[str] = []
            communicate_response, _ = self.parse_response(response, "talk")
            think_text = self.parse_response(response, "think")
            if self.requires_reasoning_fields and think_text == "":
                format_issues.append("missing_think")
            recent_goal_text = self.parse_response(response, "recent_goal")
            if self.requires_reasoning_fields and recent_goal_text == "":
                format_issues.append("missing_recent_goal")
            action_text_block = self.parse_response(response, "action")
            print(
                f"[LLMAgents] agent={self.agent_index} parsed think_len={len(think_text)} "
                f"talk={communicate_response!r} action_block={action_text_block!r}"
            )
            if action_text_block == "":
                format_issues.append("missing_action")
            elif action_info["mode"] in {"mixed", "multi_embodied", "malformed"}:
                format_issues.append(action_info["mode"] + "_action_types")
            think_output = think_text
            if self.requires_reasoning_fields and think_text == "":
                print("\n\n\n******No Think Part, Correcting*********\n\n\n")
                think_text, response = self.important_part_no_create(1, "think", response)
                think_output = think_text
            if self.requires_reasoning_fields and recent_goal_text == "":
                print("\n\n\n******No Recent Goal Part, Correcting*********\n\n\n")
                recent_goal_text, response = self.important_part_no_create(
                    1, "recent_goal", response
                )
            if action_text_block == "":
                print("\n\n\n******No Action Part, Correcting**********\n\n\n")
                action_text_block, response = self.important_part_no_create(1, "action", response)
            self._report_format_issues(format_issues)
            self.update_recent_goal_text(recent_goal_text)
            recent_goal_entry = self.current_recent_goal_text
            # If important keyword still miss, this timestamp is failed.
            if (self.requires_reasoning_fields and think_text == "") or action_text_block == "":
                print(
                    "\n\n\nMiss important part after several retry. This timstep is failed!\n\n\n"
                )
                ml_action = "wait(1)"
                self._set_current_action_source(self.last_call)
                return ml_action

            self.planner.add_msg_to_dialog_history(
                {"role": "scene", "content": state_message["content"]}
            )
            self.planner.add_msg_to_dialog_history(
                {"role": "think", "content": think_text}
            )

            # statistic
            self.turn_statistics_dict["content"]["content"][self.agent_index].append(
                {
                    "agent": self.agent_index,
                    "think": think_text,
                    "say": communicate_response,
                    "action": action_text_block,
                    "recent_goal": recent_goal_entry,
                }
            )
            # If this function call is to correct action, the tokens should count into the number of correction_tokens.
            if failure_message != "":
                self.turn_statistics_dict["statistical_data"]["error_correction"][
                    self.agent_index
                ]["validator_correction"]["correction_tokens"].append(tokens_num)
                # check if the action is empty
                if action_text_block == "":
                    action_parse, action_text_block = self.important_part_no_create(1, "action", response)
                    if action_parse == "":
                        self.error_correct[self.current_timestep] = False
                        ml_action = "wait(1)"
                    else:
                        ml_action = self.parse_ml_action_top(action_text_block, True)
                else:
                    ml_action = self.parse_ml_action_top(action_text_block, True)
            elif ("[NOTHING]" not in communicate_response) and (
                "[EMPTY]" not in communicate_response
            ):
                turn_count = self._register_comm_turn()
                if turn_count >= self.communication_turn_limit:
                    # too many collaboration attempts in this timestep
                    ml_action = "wait(1)"
                    self.time_to_wait = 1
                    self._relabel_last_call("planner_main", forced_action=ml_action,
                                            forced_action_reason="communication_turn_limit")
                else:
                    self.planner.add_msg_to_dialog_history(
                        {"role": "talk", "content": communicate_response}
                    )
                    self.append_conversation_line(self.name, communicate_response)
                    # statistic
                    self.turn_statistics_dict["statistical_data"]["communication"][
                        self.agent_index
                    ]["turn"].append(communicate_response)
                    self.turn_statistics_dict["statistical_data"]["communication"][
                        self.agent_index
                    ]["token"].append(tokens_num)
                    self.turn_statistics_dict["statistical_data"]["communication"][
                        self.agent_index
                    ]["call"] += 1
                    response = self.communication(communicate_response, state)
            elif ("[NOTHING]" not in action_text_block) and (action_text_block != ""):
                ml_action = self.parse_ml_action_top(action_text_block, True)
            else:
                # No action and no communication content, wait
                ml_action = "wait(1)"
                self.time_to_wait = 1
            print(f"[LLMAgents] agent={self.agent_index} selected ml_action={ml_action!r}")
        else:
            # execute the next queued action (e.g. the action agreed in communication)
            queued = []
            while not self.action_wait_parse.empty():
                queued.append((self.action_wait_parse.get(), self._queued_sources.popleft() if self._queued_sources else None))
            self._queued_sources.clear()
            actionable = [(act, src) for act, src in queued if not self._is_collab_action(act)]
            for act, src in actionable[1:]:
                self.action_wait_parse.put(act)
                self._queued_sources.append(src)
            if actionable:
                ml_action, selected_source = actionable[0]
                actionable = [act for act, _ in actionable]
                print(f"\n\n\n### Already have action sequence in pre-communication:\n {actionable}")
                response = f"Action: {ml_action}"
            else:
                response = ""
            recent_goal_entry = self.current_recent_goal_text
        # when handle several failure, the precious communication history should be jumped
        if self.planner.dialog_history_list_storage == [] and not self.trace:
            self.planner.dialog_history_list_storage = self.planner.dialog_history_list
            self.teammate.planner.dialog_history_list_storage = (
                self.teammate.planner.dialog_history_list
            )
        self.del_dialog_history()
        if ml_action == "":
            ml_action = self.parse_ml_action_top(response, True)
        if selected_source is None:
            selected_source = self.last_call
        self._set_current_action_source(selected_source)
        self._emit("action_bound", action=ml_action, source=selected_source, primary=True)
        self.current_ml_action_steps = 0
        failed_message_value = getattr(self, "failed_message", "success")
        cleaned_failure = ""
        if isinstance(failed_message_value, str) and failed_message_value.strip().lower() != "success":
            cleaned_failure = failed_message_value.strip()
        self.record_history_entry(
            think_output, self.current_recent_goal_text, ml_action, cleaned_failure
        )
        return ml_action

    # parse ml_action from  response and self correct
    def parse_ml_action_top(self, response, add_to_queue):
        """Extract the embodied action from an Action field.

        Collab(...) tokens are ignored here. More than one embodied action is a format error and
        becomes wait(1). An invalid action is kept as is, so that the validator reports it and
        the agent gets one chance to correct it.
        add_to_queue: replace the pending action queue and return the primary action;
        otherwise return the list of embodied actions.
        """
        print("\n===== Parser =====\n")
        action_string = (response or "").strip()
        if not action_string.lower().startswith("action"):
            extracted = self.parse_response(response, "action")
            if extracted:
                action_string = extracted.strip()
        if action_string == "":
            print(f"Response does not follow the format rules:{response}")
            return "wait(1)" if add_to_queue else ["wait(1)"]
        action_body = self._sanitize_action_text(self._strip_action_prefix(action_string))
        action_tokens = [
            token.strip() for token in self._split_action_tokens(action_body) if token.strip()
        ]
        raw_non_collab_tokens = [
            token for token in action_tokens if not self._is_collab_action(token)
        ]
        if len(raw_non_collab_tokens) > 1:
            detail = (
                "Only one embodied action is allowed in one Action field. "
                "Multiple Collab(...) actions are allowed, but embodied actions must not be chained."
            )
            print(f"[Parser] invalid action format: {detail}")
            self._report_action_format_error(detail, source=self.last_call)
            if add_to_queue:
                self._clear_action_queue()
                self.time_to_wait = 1
                return "wait(1)"
            return []
        non_collab_tokens = []
        for token in raw_non_collab_tokens:
            if re.search(r"NOTHING|EMPTY", token, re.IGNORECASE):
                non_collab_tokens.append("wait(1)")
                continue
            format_valid, parsed_action = self.parse_ml_action(token, report_error=False)
            # an invalid action is reported and corrected by validate_current_ml_action
            non_collab_tokens.append(parsed_action if format_valid and parsed_action else token)
        ml_action = non_collab_tokens[0] if non_collab_tokens else "wait(1)"
        if add_to_queue:
            self._clear_action_queue()
            for a in non_collab_tokens[1:]:
                self._enqueue_action(a, self.last_call)
            if "wait" in ml_action:
                self.time_to_wait = self.parse_wait_string(ml_action)
        if "wait" not in ml_action:
            self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
        return ml_action if add_to_queue else non_collab_tokens

    ##################
    """
	The followings are the Verificator part
	"""
    ##################

    def check_current_ml_action_done(self, state):
        """
        checks if the current ml action is done
        :return: True or False
        """
        action, params = self.parse_params_in_action(self.current_ml_action)
        self.parse_action_params = params
        player = state.players[self.agent_index]
        if action == "pickup":
            return player.has_object() and player.get_object().name == params[0]
        elif action in self.mdp.interact_actions:
            return (
                params[0] in self.mdp.get_utensil_states(state)["cooking"]
                or state.error_message != []
            )
        elif action == "fill_dish_with_food":
            return player.held_object != None and self.order in player.held_object.name
        elif action in ("place_obj_on_counter", "put_obj_in_utensil", "deliver_soup"):
            return not player.has_object()
        elif action == "check_recipe":
            if self.actor == "chef":
                chef = self
                if chef.time_to_wait == 0:
                    self.load_recipe()
                else:
                    self.time_to_wait -= 1
            else:
                chef = self.teammate
            return self.order in chef.recipe
        elif action == "wait":
            return self.time_to_wait <= 0

    def check_pickup_food_dish(self, food, utensil):
        valide = True
        utensil = re.sub(r"\d+", "", utensil)
        if food in self.mdp.need_dish.keys():
            # check if the utensil is the finally step, which can not directly pick up without dish.
            for utensil_name in self.mdp.recipes.keys():
                u = self.mdp.recipes[utensil_name]
                for f in u.keys():
                    if (food == f) and (utensil == utensil_name):
                        valide = False
        return valide or not (self.mdp.need_dish[food] == 1)

    def validate_current_ml_action(self, state):
        """
        make sure the current_ml_action exists and is valid
        return: success, or failed reason.
        """
        self._last_validation_failed_format = False
        failed_message = "success"
        if self.current_ml_action is None:
            return f"There is no action for {self.actor}.\n"
        player = state.players[self.agent_index]
        has_object = player.has_object()
        self.parse_action, self.parse_action_params = self.parse_params_in_action(
            self.current_ml_action
        )

        def valide_obj(obj):
            return player.has_object() and obj in player.get_object().name

        empty_counter = self.mdp.get_empty_counter_locations(state)
        utensil_state = self.mdp.get_utensil_states(state)
        # Action format check
        format_valide, format_error_message = self.parse_ml_action(
            self.current_ml_action
        )
        if not format_valide:
            self._last_validation_failed_format = True
            return format_error_message
        elif "wait" in format_error_message:
            self.current_ml_action = format_error_message
            return failed_message
        # Assessing the logical Soundness of actions
        if self.parse_action == "pickup":
            if (self.parse_action_params[1] in self.mdp.utensil_list) and (
                self.parse_action_params[1] in utensil_state["cooking"]
            ):
                failed_message = f"{self.actor} can not pick up {self.parse_action_params[0]} from {self.parse_action_params[1]}. The utensil {self.parse_action_params[1]} is cooking, and you should wait for it is ready.\n"
                return failed_message
            # In case of directly picking up cooked food which need dish in the finally step
            if not self.check_pickup_food_dish(
                self.parse_action_params[0], self.parse_action_params[1]
            ):
                failed_message = f"{self.actor} can not pick up {self.parse_action_params[0]} from {self.parse_action_params[1]}. {self.order} need a dish.\n"
                return failed_message
            if self.parse_action_params[0] in self.mdp.utensil_list:
                failed_message = f"You can not move utensil. It is fixed! If you can not access utensil, you should use counter as transfer station\n"
                return failed_message
            # check if agent access the pickup destination
            flag2 = len(self.find_motion_goals(state)) == 0
            if flag2:
                failed_message = f"{self.actor} can not reach the destination.Please check if the destination is in your space.Please check if you can interact it. You can let your teammate pick the ingredient from utensil and put in on the counter, then pick up ingredient from the counter.\n"
                if "counter" in self.parse_action_params:
                    failed_message = (
                        f"There is no {self.parse_action_params[0]} on counter {self.actor} can visited."
                        + (
                            "Assistant can directly pick ingredients from dispenser.\n"
                            if self.actor == "assistant"
                            else "If assistant is already preparing ingredient for you,you should wait.\n"
                        )
                    )
                elif (
                    "dispenser" in self.parse_action_params[1] and self.actor == "chef"
                ):
                    failed_message = f"Chef can not directly access dispenser.Chef can gain ingredients with the help of assistant."
            elif has_object:
                failed_message = f"There is object in {self.actor}'s hand, so can not pick other thing.\n"
            # check if the item is in the destination
            elif (
                (self.parse_action_params[1] in utensil_state["empty"])
                or (self.parse_action_params[1] in self.mdp.utensil_list)
                and (
                    self.parse_action_params[0]
                    != self.mdp.utensil_state_dict[self.parse_action_params[1]][
                        "soup"
                    ].state[0]
                )
            ):
                failed_message = f"There is no {self.parse_action_params[0]} in {self.parse_action_params[1]}, please check the item your want to pickup.\n"
            return failed_message + "\n"
        elif self.parse_action == "put_obj_in_utensil":
            # check if the food in utensil is full
            if not self.parse_action_params:
                return "Wrong put_obj_in_utensil() params. It should have 1 params: utensil.\n"
            if self.parse_action_params[0] in utensil_state["cooking"]:
                failed_message = f"{self.actor} can not put obj into  {self.parse_action_params[0]}. The utensil {self.parse_action_params[0]} is cooking, and you should wait for it is ready.\n"
                return failed_message
            if self.parse_action_params[0] in utensil_state["ready"]:
                failed_message = f"{self.actor} can not put obj into  {self.parse_action_params[0]}. The utensil {self.parse_action_params[0]} is ready, and you should pick up it or fill with dish according to the recipe.\n"
                return failed_message
            # if  self.parse_action_params[0] in utensil_state['full']:
            # 	return f"The {self.parse_action_params[0]} is full. You can not put more ingredients in it.\n"

            # If the utensil is empty, add the order name TODO:maybe a bug here:If agent add ingredient not for current order?
            if self.parse_action_params[0] in utensil_state["empty"]:
                self.mdp.utensil_state_dict[self.parse_action_params[0]][
                    "order"
                ] = self.order
            flag2 = len(self.find_motion_goals(state)) == 0
            if flag2:
                failed_message = f"{self.actor} can not reach the {self.parse_action_params[0]}.Please check if you can interact it. You can put ingredient on the counter,then let your teammate pick the ingredient from counter and put it into utensil.\n"
            elif len(empty_counter) == 0:
                failed_message = f"There is no empty counter to place object.\n"
            elif (
                self.parse_action_params[0] in utensil_state["ready"]
                or self.parse_action_params[0] in utensil_state["cooking"]
            ):
                failed_message = f"{self.parse_action_params[0]} is busy. You can not add more ingredients in it.\n"
            elif not player.has_object():
                failed_message = f"There is no object in {self.actor}'s hand, so can not put it on {self.parse_action_params[0]}.\n"
            elif player.get_object().name == "dish":
                failed_message = f"You can not put dish into any utensil. Dish can only be placed on counter.\n"
            return failed_message
        elif self.parse_action == "place_obj_on_counter":
            if not has_object:
                failed_message = f"There is no object in {self.actor}'s hand, so can not place object on counter.\n"
            elif len(empty_counter) == 0:
                failed_message = f"There is no empty counter to place object.\n"
            return failed_message
        elif self.parse_action == "fill_dish_with_food":
            # check if the recipe need dish
            if self.mdp.need_dish[self.order] == 0:
                failed_message = f"{self.order} does not need dish. Please directly pick cooked food from utensil and deliver it to the service location.\n"
            # fill only when utensil is cooking or ready
            elif not (
                self.parse_action_params[0] in utensil_state["ready"]
                or self.parse_action_params[0] in utensil_state["cooking"]
            ):
                failed_message = f"{self.parse_action_params[0]} is not ready or cooking for filled.\n"
            return (
                f"There is no dish in hand.\n"
                if (not valide_obj("dish") and self.mdp.need_dish[self.order] == 1)
                else failed_message
            )
        elif self.parse_action == "deliver_soup":
            has_soup = False
            for s in self.mdp.need_dish.keys():
                if valide_obj(s):
                    has_soup = True
            if not has_soup:
                if has_object:
                    return f"Item in your hand is not finished food. You should put it on counter or right utensil first. Then check whether the order need dish to fill with. If it need a dish, you should ask assistant for a dish and then you fill dish with food from utensil. If not, you can directly pick up food from utensil. Finally you can deliver soup again.\n"
                else:
                    return f"Your hand is empty. You should check whether the order need dish to fill with first. If it need a dish, you should ask assistant for a dish and then you fill dish with food from utensil. If not, you can directly pick up food from utensil. Finally you can deliver soup again.\n"
            else:
                return failed_message

        elif self.parse_action in self.mdp.interact_actions:
            flag2 = len(self.find_motion_goals(state)) == 0
            if flag2:
                failed_message = f"{self.actor} can not reach the {self.parse_action_params[0]}..Please check if the utensil is in your space.\n"
                return failed_message
            # only when utensil is partially_full can be operated. when Empty、ready、cooking ,utensil can not be operated.
            if self.parse_action_params[0] in utensil_state["empty"]:
                failed_message = f"Ingredients in {self.parse_action_params[0]} are not enough to begin the operation.\n"
            # check if the utensil is already cooking or ready
            elif (
                self.parse_action_params[0] in utensil_state["ready"]
                or self.parse_action_params[0] in utensil_state["cooking"]
            ):
                failed_message = (
                    f"{self.parse_action_params[0]} is busy. You can not operate it.\n"
                )
            elif has_object:
                failed_message = f"There is object in {self.actor}'s hand, so can not interact with utensil.\n"
            return failed_message
        # the same as pickup(toast,counter)
        elif self.parse_action == "check_recipe":
            if self.actor != "chef":
                failed_message = (
                    f"Assistant can not check recipe.Only chef can do it.\n"
                )
            elif self.order in self.recipe.keys():
                failed_message = f"You have get the recipe before. Look at the <Recipe need to know> part.\n"
            else:
                self.time_to_wait = 2
            return failed_message
        elif self.parse_action == "wait":
            match = re.search(r"\d+", self.current_ml_action)
            number = 1
            if match:
                number = match.group()
            else:
                number = 30
            return (
                failed_message
                if 0 < int(number) <= 20
                else f"Wait time is not valide.\n"
            )
        else:
            raise ValueError("Wrong action")

    def generate_success_feedback(self, state):
        success_feedback = f"### Controller Validation\n {self.name} succeeded at {self.current_ml_action}. \n"
        print(success_feedback)
        if "wait" not in success_feedback:
            self.planner.add_msg_to_dialog_history(
                {
                    "role": "user",
                    "content": f"{self.name} succeeded at {self.current_ml_action}.",
                }
            )

    def del_dialog_history(self):
        del_list = []
        for index, dialog in enumerate(self.planner.dialog_history_list):
            if any(s == dialog["role"] for s in ["talk", "think"]):
                del_list.append(index)
        self.planner.dialog_history_list = [
            value
            for idx, value in enumerate(self.planner.dialog_history_list)
            if idx not in del_list
        ]
        del_list = []
        for index, dialog in enumerate(self.teammate.planner.dialog_history_list):
            if any(s == dialog["role"] for s in ["talk", "think"]):
                del_list.append(index)
        self.teammate.planner.dialog_history_list = [
            value
            for idx, value in enumerate(self.teammate.planner.dialog_history_list)
            if idx not in del_list
        ]

    def generate_failure_feedback(self, action, failed_message):
        failure_feedback = f"Your action {action} raised an error: " + failed_message
        print(f"\n~~~~~~~~ Explainer~~~~~~~~\n{failure_feedback}")
        self.del_dialog_history()
        self.planner.add_msg_to_dialog_history(
            {"role": "failure_explanation", "content": failure_feedback}
        )

    ##################
    """
	The followings are the Controller part almost inherited from GreedyHumanModel class
	"""
    ##################

    def find_shared_counters(self, state, mlam):
        counter_dicts = query_counter_states(self.mdp, state)

        counter_list = get_intersect_counter(
            state.players_pos_and_or[self.agent_index],
            state.players_pos_and_or[1 - self.agent_index],
            self.mdp,
            self.mlam,
        )

        print("counter_list = {}".format(counter_list))
        lis = []
        for i in counter_list:
            if counter_dicts[i] == " ":
                lis.append(i)
        available_plans = mlam.ml_action_manager._get_ml_actions_for_positions(lis)
        return available_plans

    def find_motion_goals(self, state):
        """
        Generates the motion goals for the given medium level action.
        :param state:
        :return:
        """
        am = self.mlam
        motion_goals = []
        player = state.players[self.agent_index]
        pot_states_dict = self.mdp.get_pot_states(state)
        counter_objects = self.mdp.get_counter_objects_dict(
            state, list(self.mdp.terrain_pos_dict["X"])
        )
        # NOTE: In RL mode, the model may accidentally emit multiple actions in one string
        # (e.g., "put_obj_in_utensil(...)\n\npickup(...)\n..."). We always parse the *first*
        # action, and branch based on the parsed function name instead of substring checks
        # on the raw string to avoid mis-routing (e.g., "pickup" appearing later).
        self.parse_action, self.parse_action_params = self.parse_params_in_action(
            self.current_ml_action
        )
        # pickup dish or ingredients
        ml_manager = am.ml_action_manager if hasattr(am, "ml_action_manager") else None
        if ml_manager is None:
            raise AttributeError("MediumLevelPlanner missing ml_action_manager")

        if self.parse_action == "pickup":
            # pickup(obj, destination) requires 2 params; otherwise treat as unreachable/invalid.
            if len(self.parse_action_params) < 2:
                return []
            motion_goals = ml_manager.pickup_obj_actions(
                state,
                self.parse_action_params[0],
                self.parse_action_params[1],
                self.agent_index,
                counter_objects,
            )
        elif self.parse_action == "add_toast":
            motion_goals = ml_manager.pickup_obj_actions(
                state, "toast", "counter", self.agent_index, counter_objects
            )
        elif self.parse_action == "put_obj_in_utensil":
            if not self.parse_action_params:
                return []
            motion_goals = ml_manager.go_to_utensil_actions(
                state, self.parse_action_params[0], self.agent_index
            )
        elif self.parse_action == "place_obj_on_counter":
            motion_goals = self.find_shared_counters(state, self.mlam)
            if len(motion_goals) == 0:
                motion_goals = ml_manager.place_obj_on_counter_actions(state)
        elif self.parse_action == "fill_dish_with_food":
            if not self.parse_action_params:
                return []
            motion_goals = ml_manager.go_to_utensil_actions(
                state, self.parse_action_params[0], self.agent_index
            )
        elif self.parse_action == "deliver_soup":
            motion_goals = ml_manager.deliver_soup_actions()
        elif any(s in (self.parse_action or "") for s in ["cook", "cut", "stir", "bake"]):
            if not self.parse_action_params:
                return []
            motion_goals = ml_manager.go_to_utensil_actions(
                state, self.parse_action_params[0], self.agent_index
            )
        elif self.parse_action == "wait":
            motion_goals = ml_manager.wait_actions(player)
        else:
            raise ValueError("Invalid action: {}".format(self.current_ml_action))

        motion_goals = [
            mg
            for mg in motion_goals
            if self.mlam.mp.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]

        return motion_goals

    def choose_motion_goal(self, start_pos_and_or, motion_goals, state=None):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """

        if self.controller_mode == "new":
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal_new(
                start_pos_and_or, motion_goals, state
            )
        else:
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        return chosen_goal, chosen_goal_action

    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal
            )
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action

    def get_lowest_cost_action_and_goal_new(
        self, start_pos_and_or, motion_goals, state
    ):
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
        min_cost = np.inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, plan_cost = self.real_time_planner(
                start_pos_and_or, goal, state
            )
            if plan_cost < min_cost:
                best_action = action_plan
                min_cost = plan_cost
                best_goal = goal
        if best_action is None:
            # print('\n\n\nBlocking Happend, executing default path\n\n\n')
            # print('current position = {}'.format(start_pos_and_or))
            # print('goal position = {}'.format(motion_goals))
            if np.random.rand() < 0.5:
                return None, Action.STAY
            else:
                return self.get_lowest_cost_action_and_goal(
                    start_pos_and_or, motion_goals
                )
        return best_goal, best_action

    def real_time_planner(self, start_pos_and_or, goal, state):
        terrain_matrix = {
            "matrix": copy.deepcopy(self.mlam.mdp.terrain_mtx),
            "height": len(self.mlam.mdp.terrain_mtx),
            "width": len(self.mlam.mdp.terrain_mtx[0]),
        }
        other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
        action_plan, plan_cost = find_path(
            start_pos_and_or, other_pos_and_or, goal, terrain_matrix
        )

        return action_plan, plan_cost
