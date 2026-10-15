"""Step 3 of evaluation: average the metrics of each complexity level into <eval_dir>/converted_data.csv."""

import os
from argparse import ArgumentParser

import pandas as pd

from ..config import TASK_LEVELS


def get_level(order):
    for level_name, orders in TASK_LEVELS.items():
        if order in orders:
            return level_name
    return None


def main(eval_dir):
    data = pd.read_csv(os.path.join(eval_dir, "statistics_data.csv"))
    data["level"] = data["order"].apply(get_level)

    columns_to_average = [col for col in data.columns if col not in ["model", "order", "level"]]
    data[columns_to_average] = data[columns_to_average].apply(pd.to_numeric, errors="coerce")

    grouped_data = data.groupby(["model", "level"])[columns_to_average].mean()
    grouped_data.to_csv(os.path.join(eval_dir, "converted_data.csv"), sep=",")


if __name__ == "__main__":
    parser = ArgumentParser(description="Compute the metrics of each complexity level.")
    parser.add_argument("--eval_dir", type=str, default="eval_result", help="dir of the evaluation results")
    main(parser.parse_args().eval_dir)
