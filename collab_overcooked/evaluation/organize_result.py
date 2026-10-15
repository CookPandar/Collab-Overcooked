"""Step 2 of evaluation: summarize the metrics of each task into <eval_dir>/statistics_data.csv."""

import json
import os
import time
from argparse import ArgumentParser

import pandas as pd

from ..config import ALL_TASKS

COLUMNS = [
    "model",
    "order",
    "success_rate",
    "time_avg",
    "time_var",
    "mean_f1_agent_0",
    "mean_similarity_agent_0",
    "mean_redundancy_agent_0",
    "std_f1_agent_0",
    "std_similarity_agent_0",
    "std_redundancy_agent_0",
    "mean_f1_agent_1",
    "mean_similarity_agent_1",
    "mean_redundancy_agent_1",
    "std_f1_agent_1",
    "std_similarity_agent_1",
    "std_redundancy_agent_1",
    "initiate_collaboration",
    "respond_collaboration",
    "overall_collaboration",
]


def collect_row(eval_dir, model, order):
    eval_file = os.path.join(eval_dir, model, order, "evaluation_result.json")
    if not os.path.exists(eval_file):
        print(f"[organize] skip {model}/{order}: no evaluation result")
        return None

    with open(eval_file, "r") as file:
        data = json.load(file)
    if order not in data:
        print(f"Error: Key '{order}' not found in evaluation_result.json.")
        return None

    order_data = data[order]
    average = order_data.get("average", {})
    task_metrics = order_data.get("task_metrics", {})
    statistic = order_data.get("statistic", {})
    row = {
        "model": model,
        "order": order,
        "success_rate": task_metrics["success_rate"],
        "time_avg": task_metrics["time_avg"],
        "time_var": task_metrics["time_var"],
        "initiate_collaboration": statistic["initiate_collaboration"],
        "respond_collaboration": statistic["respond_collaboration"],
    }
    for agent in ["agent_0", "agent_1"]:
        metrics = average["similarity_and_redundancy"][agent]
        for name in ["f1", "similarity", "redundancy"]:
            row[f"mean_{name}_{agent}"] = metrics[f"mean_{name}"]
            row[f"std_{name}_{agent}"] = metrics[f"std_{name}"]
    return row


def main(variant):
    csv_path = os.path.join(variant["eval_dir"], "statistics_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=COLUMNS)

    for model in variant["models"]:
        for order in ALL_TASKS:
            row = collect_row(variant["eval_dir"], model, order)
            if row is None:
                continue
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            print(f"Data for order '{order}' saved to {csv_path}.")

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = ArgumentParser(description="Process evaluation results and update statistics data.")
    parser.add_argument("--model", "--models", dest="models", type=str, nargs="+", default=["gpt-3.5-turbo-0125"], help="Model name(s) to summarize")
    parser.add_argument("--eval_dir", type=str, default="eval_result", help="dir of the evaluation results")
    start_time = time.time()
    main(vars(parser.parse_args()))
    print("\n======= Finished all =======\n")
    print(f"Cost time: {time.time() - start_time:.3f}s\n")
