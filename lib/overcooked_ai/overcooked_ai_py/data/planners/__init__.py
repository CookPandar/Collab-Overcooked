"""Planner cache helpers expected by :mod:`overcooked_ai_py.planning.planners`."""

import os
import pickle

from overcooked_ai_py.static import PLANNERS_DIR


def load_saved_action_manager(filename):
    path = filename
    if not os.path.isabs(path):
        path = os.path.join(PLANNERS_DIR, filename)
    with open(path, "rb") as handle:
        return pickle.load(handle)
