"""Truncate experiment logs at a time limit of `multiple` x the reference steps of each task.

An episode that has not delivered its order within the limit is cut at the limit and
marked as unfinished. The truncated logs are written to
<data_dir>/truncate_<multiple>/<model>/<order>/ and can be evaluated with
`python -m collab_overcooked.evaluation.evaluation --log_dir <data_dir>/truncate_<multiple>`.
"""

import copy as cp
import json
import os
from argparse import ArgumentParser

from ..config import TASK_REFERENCE_STEPS


def truncate_log(data, time_threshold):
    if data["total_order_finished"] != [] and data["total_timestamp"][-1] <= time_threshold:
        return data
    # part1: timestep
    data["total_timestamp"] = list(range(0, time_threshold))
    # part2: total_action_list
    for agent_index, agent_action_list in enumerate(data["total_action_list"]):
        data["total_action_list"][agent_index] = cp.copy(
            [a for a in agent_action_list if a["timestamp"] < time_threshold]
        )
    # part3: content
    data["content"] = data["content"][:time_threshold]
    # part4: finish order
    data["total_order_finished"] = []
    return data


def truncate(data_dir, model, multiple):
    model_dir = os.path.join(data_dir, model)
    if not os.path.exists(model_dir):
        print(f"{model} log not exist")
        return
    output_root = os.path.join(data_dir, f"truncate_{str(multiple).replace('.', '_')}", model)
    for order in sorted(os.listdir(model_dir)):
        if order not in TASK_REFERENCE_STEPS:
            continue
        time_threshold = int(multiple * TASK_REFERENCE_STEPS[order])
        output_dir = os.path.join(output_root, order)
        os.makedirs(output_dir, exist_ok=True)
        for root, _, files in os.walk(os.path.join(model_dir, order)):
            for file in files:
                if not file.endswith(".json"):
                    continue
                with open(os.path.join(root, file), "r") as f:
                    data = json.load(f)
                with open(os.path.join(output_dir, file), "w") as f:
                    json.dump(truncate_log(data, time_threshold), f, indent=4)
    print(f"Truncated logs of {model} saved to {output_root}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Truncate experiment logs at a multiple of the reference steps.")
    parser.add_argument("--model", "--models", dest="models", type=str, nargs="+", default=["gpt-3.5-turbo-0125"])
    parser.add_argument("--multiple", type=float, default=1.5, help="time limit = multiple x reference steps")
    parser.add_argument("--data_dir", type=str, default="data", help="dir of the experiment logs")
    args = parser.parse_args()
    for model in args.models:
        truncate(args.data_dir, model, args.multiple)
