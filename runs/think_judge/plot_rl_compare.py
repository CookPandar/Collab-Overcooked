"""Training curves of several RL runs on shared axes: rollout success (10-update rolling mean), judge score, the
judge's state-grounding score (from the saved rollouts; runs judged without it are skipped) and KL.
usage: plot_rl_compare.py OUT_PNG TITLE RUN=LABEL ...  (first 4 runs get validated categorical slots 1-4)"""
import gzip
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

font_manager.fontManager.addfont(str(Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"))
plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 9, "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.edgecolor": "#d6d5d0", "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb"})
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TJ = Path(__file__).parent
MAX_UPDATE = 50


def rolling(values, k=10):
    return np.array([np.mean(values[max(0, i - k + 1):i + 1]) for i in range(len(values))])


def grounding(run, update):
    path = TJ / "rl" / run / "rollouts" / f"u{update:05d}.jsonl.gz"
    if not path.exists():
        return np.nan
    values = [r["judge_dims"].get("grounding") for r in map(json.loads, gzip.open(path, "rt"))
              if "judge_dims" in r]
    values = [v for v in values if v is not None]
    return np.mean(values) if values else np.nan


out, title, specs = sys.argv[1], sys.argv[2], [s.split("=", 1) for s in sys.argv[3:]]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
for k, (run, label) in enumerate(specs):
    rows = [json.loads(l) for l in open(TJ / "rl" / run / "metrics.jsonl")][:MAX_UPDATE]
    u = np.array([r["update"] for r in rows])
    style = {"color": COLORS[k], "lw": 2.2 if k % 2 == 0 else 1.6, "ls": "-" if k % 2 == 0 else (0, (4, 2)),
             "label": label}
    axes[0].plot(u, rolling(np.array([r["success_rate"] for r in rows])), **style)
    axes[1].plot(u, rolling(np.array([r.get("judge/score", np.nan) for r in rows])), **style)
    g = np.array([grounding(run, int(x)) for x in u])
    if not np.all(np.isnan(g)):
        axes[2].plot(u, rolling(g), **style)
    axes[3].plot(u, [max(r["update/kl"], 1e-4) for r in rows], **dict(style, lw=1.2))
names = ["训练成功率（10 次滚动平均）", "评审总分（10 次滚动平均）", "grounded 状态分（10 次滚动平均）",
         "与参考模型的 KL（对数坐标）"]
for ax, name in zip(axes, names):
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_xlabel("RL 更新次数")
    ax.yaxis.grid(True, color="#ebeae6", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylim(0, 1)
axes[3].set_yscale("log")
axes[2].text(0.02, 0.92, "旧版评审未打 grounded 分", transform=axes[2].transAxes, fontsize=8, color="#52514e")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=len(specs), frameon=False, loc="lower left", bbox_to_anchor=(0.01, -0.02))
fig.suptitle(title, x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(out, dpi=160)
