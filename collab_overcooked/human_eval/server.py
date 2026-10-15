"""WebSocket relay between Collab-Overcooked and the human-evaluation web pages.

Usage:
    python -m collab_overcooked.human_eval.server --port 9999
then open index0.html (Chef) and index1.html (Assistant) in a browser and run
    python -m collab_overcooked.main --gpt_model human --local_server_api 9999 --order boiled_egg
"""

from argparse import ArgumentParser

from . import web_util


if __name__ == "__main__":
    parser = ArgumentParser(description="Collab-Overcooked human evaluation server")
    parser.add_argument("--port", type=str, default="9999")
    args = parser.parse_args()
    web_util.start_server(args.port)
