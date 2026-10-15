# overcooked_ai (Collab-Overcooked fork)

This directory contains a modified copy of [Overcooked-AI](https://github.com/HumanCompatibleAI/overcooked_ai)
(Carroll et al., 2019), released under the MIT license (see `LICENSE`).

Collab-Overcooked extends the original environment with new ingredients, utensils
(e.g. chopping board, oven, steamer), multi-step recipes, and medium-level actions
that LLM agents call by name. Only the Python environment (`overcooked_ai_py`) is kept;
the JavaScript demo, the visualization tools and the original unit tests were removed.

- `overcooked_ai_py/data/layouts/new_env.layout`: kitchen layout and recipe definitions.
- `overcooked_ai_py/mdp/`: game logic (`overcooked_mdp.py`) and environment wrapper (`overcooked_env.py`).
- `overcooked_ai_py/planning/`: motion planners used to execute medium-level actions.

Install it with `pip install -e lib/overcooked_ai` from the repository root.
