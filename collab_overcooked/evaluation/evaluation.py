"""Step 1 of evaluation: compute the metrics of each task from experiment logs.

Logs are read from <log_dir>/<model>/<order>/experiment_*.json (written by
collab_overcooked.main) and results are written to <save_dir>/<model>/<order>/.
"""

import os
import time
from argparse import ArgumentParser

from ..config import ALL_TASKS
from .eval_utils import Evaluation, ExpLog


def evaluate_logs(log_dir, save_dir, order=None):
    """Evaluate the logs under `log_dir`. If `order` is None the order of each log is recognized automatically."""
    exp_log = ExpLog(log_dir)
    if exp_log.__len__() == 0:
        print(f"[evaluation] Skip: no experiment_*.json under {log_dir}")
        return False
    if order is None:
        order_name_list = [exp_log.get_secondary_order_list(idx, 0)[0] for idx in range(exp_log.__len__())]
    else:
        order_name_list = [order] * exp_log.__len__()
    evaluation = Evaluation(order_name_list=order_name_list, exp_log=exp_log)
    os.makedirs(save_dir, exist_ok=True)
    evaluation.evaluate(save_dir)
    return True


def main(variant):
    for model in variant["models"]:
        if variant["test_mode"] == "fix_task":
            if variant["order"] == "AUTO":
                evaluate_logs(os.path.join(variant["log_dir"], model), os.path.join(variant["save_dir"], model))
            else:
                evaluate_logs(
                    os.path.join(variant["log_dir"], model, variant["order"]),
                    os.path.join(variant["save_dir"], model, variant["order"]),
                    variant["order"],
                )
        elif variant["test_mode"] == "build_in":
            for order in ALL_TASKS:
                evaluate_logs(
                    os.path.join(variant["log_dir"], model, order),
                    os.path.join(variant["save_dir"], model, order),
                    order,
                )


def get_parser():
    parser = ArgumentParser(description="Collab-Overcooked evaluation")
    parser.add_argument("--model", "--models", dest="models", type=str, nargs="+", default=["gpt-3.5-turbo-0125"], help="Model name(s), i.e. the sub-directories of --log_dir")
    parser.add_argument("--test_mode", type=str, default="fix_task", choices=["fix_task", "build_in"], help="fix_task: evaluate one order (or AUTO); build_in: evaluate all 30 tasks")
    parser.add_argument("--order", type=str, default="AUTO", help='1 task order name, "AUTO" represents automatic recognition')
    parser.add_argument("--log_dir", type=str, default="data", help="dir to read experiment logs from")
    parser.add_argument("--save_dir", type=str, default="eval_result", help="dir to write evaluation results to")
    return parser


if __name__ == "__main__":
    start_time = time.time()
    main(vars(get_parser().parse_args()))
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {time.time() - start_time:.3f}s-----\n\n")
