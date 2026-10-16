"""Plot the training curves of one or more RL runs (their metrics.jsonl).

    python -m baselines.rl.plot runs/rl_level2 runs/rl_level3 --output curves.png
"""

import argparse
import json
from pathlib import Path

PANELS = [("success_rate", "success rate"), ("episode_return", "team return / episode"),
          ("steps", "episode length (steps)"), ("format_error_rate", "format error rate"),
          ("update/kl", "KL to SFT reference")]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="output_dir of baselines.rl.train runs")
    parser.add_argument("--output", default="curves.png")
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.4 * len(PANELS), 3.6))
    for run in args.runs:
        records = [json.loads(line) for line in open(Path(run) / "metrics.jsonl")]
        for ax, (key, title) in zip(axes, PANELS):
            ax.plot([r["update"] for r in records], [r[key] for r in records], "o-", ms=3, label=Path(run).name)
            ax.set_title(title)
            ax.set_xlabel("update")
            ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(args.output)


if __name__ == "__main__":
    main()
