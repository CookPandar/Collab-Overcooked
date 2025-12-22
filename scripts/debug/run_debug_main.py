#!/usr/bin/env python
"""
Helper entrypoint for debugging Collab-Overcooked using the same call
as quick_test.sh: python -c "from collab_overcooked.main import main; main(config_path='configs/test_personal.yaml')"
"""

from collab_overcooked.main import main


def run():
    main(config_path="configs/test_personal.yaml")


if __name__ == "__main__":
    run()
