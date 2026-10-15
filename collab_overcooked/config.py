"""Shared paths, task lists and API settings for Collab-Overcooked."""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(PACKAGE_DIR, "prompts")
RECIPE_DIR = os.path.join(PROMPT_DIR, "recipe")
REFERENCE_DIR = os.path.join(PROMPT_DIR, "reference")
LLAMA_TOKENIZER_DIR = os.path.join(PACKAGE_DIR, "llama_tokenizer")
HUMAN_EVAL_DIR = os.path.join(PACKAGE_DIR, "human_eval")

LOCAL_SERVER_API_KEY = "token-abc123"

# The 30 benchmark tasks, grouped by complexity level.
TASK_LEVELS = {
    "level_1": ["baked_bell_pepper", "baked_sweet_potato", "boiled_egg", "boiled_mushroom", "boiled_sweet_potato"],
    "level_2": ["baked_potato_slices", "baked_pumpkin_slices", "boiled_corn_slices", "boiled_green_bean_slices", "boiled_potato_slices"],
    "level_3": ["baked_bell_pepper_soup", "baked_carrot_soup", "baked_mushroom_soup", "baked_potato_soup", "baked_pumpkin_soup"],
    "level_4": ["sliced_bell_pepper_and_corn_stew", "sliced_bell_pepper_and_lentil_stew", "sliced_eggplant_and_chickpea_stew", "sliced_pumpkin_and_chickpea_stew", "sliced_zucchini_and_chickpea_stew"],
    "level_5": ["mashed_broccoli_and_bean_patty", "mashed_carrot_and_chickpea_patty", "mashed_cauliflower_and_lentil_patty", "mashed_potato_and_pea_patty", "mashed_sweet_potato_and_bean_patty"],
    "level_6": ["potato_carrot_and_onion_patty", "romaine_lettuce_pea_and_tomato_patty", "sweet_potato_spinach_and_mushroom_patty", "taro_bean_and_bell_pepper_patty", "zucchini_green_pea_and_onion_patty"],
}
ALL_TASKS = [task for tasks in TASK_LEVELS.values() for task in tasks]

# Reference completion steps of each level, used to truncate logs at a time limit.
LEVEL_REFERENCE_STEPS = {"level_1": 16, "level_2": 23, "level_3": 35, "level_4": 32, "level_5": 55, "level_6": 64}
TASK_REFERENCE_STEPS = {
    task: LEVEL_REFERENCE_STEPS[level] for level, tasks in TASK_LEVELS.items() for task in tasks
}


def get_openai_api_key():
    """Return the OpenAI API key (only needed by API models such as GPT).

    Reads ``OPENAI_API_KEY`` first and falls back to the first line of
    ``openai_key.txt`` in the working directory (the v1 convention).
    ``OPENAI_BASE_URL`` is honoured by the OpenAI client for compatible endpoints.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    key_file = os.path.join(os.getcwd(), "openai_key.txt")
    if os.path.exists(key_file):
        with open(key_file, "r") as f:
            key = f.read().split("\n")[0].strip()
        if key:
            return key
    raise RuntimeError(
        "OpenAI API key not found. Set OPENAI_API_KEY or put the key in ./openai_key.txt"
    )
