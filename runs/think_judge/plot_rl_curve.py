"""Training curves of one RL run from rl/<run>/metrics.jsonl: per-update rollout success (dots) with a 10-update
rolling mean, ITES per episode, format / validator error rates, and KL / entropy.
usage: plot_rl_curve.py RUN OUT_PNG [TITLE]"""
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
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#ebeae6"


def rolling(values, k=10):
    return np.array([np.mean(values[max(0, i - k + 1):i + 1]) for i in range(len(values))])


run, out = sys.argv[1], sys.argv[2]
title = sys.argv[3] if len(sys.argv) > 3 else run
rows = [json.loads(line) for line in open(Path(__file__).parent / "rl" / run / "metrics.jsonl")]
u = np.array([r["update"] for r in rows])
get = lambda key: np.array([r.get(key, np.nan) for r in rows], dtype=float)
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
panels = [("成功率（每次更新 8 局）", [("success_rate", BLUE, "每次更新", True)], (0, 1)),
          ("每局 ITES", [("ites_per_episode", BLUE, "每次更新", True)], None),
          ("错误率", [("format_error_rate", BLUE, "格式错误", False), ("validator_error_rate", ORANGE, "动作被拒", False)], None),
          ("与参考模型的 KL（对数坐标）", [("update/kl", BLUE, "KL", False)], None)]
for ax, (name, series, ylim) in zip(axes, panels):
    for key, color, label, dots in series:
        y = get(key)
        if dots:
            ax.scatter(u, y, s=10, color=color, alpha=0.35, label=label, zorder=2)
            ax.plot(u, rolling(y), color=color, lw=2, label="10 次滑动平均", zorder=3)
        else:
            ax.plot(u, y, color=color, lw=1.5, label=label, zorder=3)
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_xlabel("RL 更新次数")
    if ylim:
        ax.set_ylim(*ylim)
    if name.startswith("与参考模型"):
        ax.set_yscale("log")  # KL spikes span orders of magnitude
    ax.yaxis.grid(True, color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
fig.suptitle(title, x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(out, dpi=160)
s = get("success_rate")
print("rolling success at u10/20/30/40/50:", [round(rolling(s)[i - 1], 3) for i in (10, 20, 30, 40, 50)])
print("mean success u1-10 / u21-30 / u41-50:", [round(s[a:b].mean(), 3) for a, b in ((0, 10), (20, 30), (40, 50))])
