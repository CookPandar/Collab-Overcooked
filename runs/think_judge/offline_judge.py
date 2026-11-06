"""Judge saved rollouts after training (e.g. the `none` arm, which was trained without a judge).

    python runs/think_judge/offline_judge.py --runs l2_none l2_sum_w0.5 --updates 5 10 15 20 25 30 35 40 45 50

The saved rollouts lack `history_before`, and `validated_action` is rebuilt from the response and its validations,
so every arm judged here sees the same (slightly reduced) context; compare offline scores only with each other.
Scores go to eval/offline_judge_<runs>.jsonl, the full judge requests to judge_annotations/offline_<run>.jsonl.
"""

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from baselines.rl.think_judge import ThinkJudge, _ACTION_FIELD, judge_episode, judgeable

ROOT = Path(__file__).resolve().parent


def record(row):
    action = _ACTION_FIELD.search(row["response"] or "")
    validated = None
    if row["action_mode"] == "physical" and row["validations"] and row["validations"][-1][0] and action:
        validated = action.group(1).strip().strip("`").strip() or None
    return {"agent_index": row["agent_index"], "call_type": row["call_type"], "response": row["response"],
            "observation": row["observation"], "validations": row["validations"], "validated_action": validated,
            "ites": row["ites"], "format_error": row["format_reward"] < 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--updates", type=int, nargs="+", required=True)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = open(ROOT / f"eval/offline_judge_{'_'.join(args.runs)}.jsonl", "a")
    for run in args.runs:
        judge = ThinkJudge(model=args.model, workers=args.workers, env_file="~/ParaCook/.env.shuaiapi.local",
                           log_path=ROOT / f"judge_annotations/offline_{run}.jsonl")
        judge.context = {"run": run, "offline": True}
        jobs = []
        for update in args.updates:
            rows = [json.loads(l) for l in gzip.open(ROOT / f"rl/{run}/rollouts/u{update:05d}.jsonl.gz", "rt")]
            episodes = {}
            for row in sorted((r for r in rows if "call_index" in r), key=lambda r: (r["episode"], r["call_index"])):
                if judgeable(row):  # the episode summaries at the end have no call_index
                    episodes.setdefault((update, row["episode"], row["order"]), []).append(row)
            jobs += list(episodes.items())
        with ThreadPoolExecutor(args.workers) as pool:
            results = pool.map(lambda job: judge_episode(judge, [record(r) for r in job[1]], job[0][2],
                                                         meta={"update": job[0][0], "episode": job[0][1]}), jobs)
            for ((update, episode, order), calls), scores in zip(jobs, results):
                for row, score in zip(calls, scores):
                    out.write(json.dumps({"run": run, "update": update, "episode": episode, "order": order,
                                          "agent_index": row["agent_index"], "call_index": row["call_index"],
                                          "score": score and score.get("score"),
                                          "follows": score and score.get("action_follows"),
                                          "online_score": row.get("judge_score")}) + "\n")
                out.flush()
        print(f"{run}: {sum(len(c) for _, c in jobs)} calls judged", flush=True)

if __name__ == "__main__":
    main()
