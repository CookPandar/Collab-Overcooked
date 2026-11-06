"""Think vs action-only on levels 1-6, only the RL checkpoints whose evaluation is complete on every level, plus the
Level-3 action-only RL checkpoint. Same data and statistics as plot_all_levels.py (30 episodes per point, error bar =
std over 3 groups of 10). Matched pairs share a colour: Think solid, action-only dashed.
Writes figures/pairs.png and figures/panels/pairs_{success,ites}.png."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_all_levels import BG, INK, LEVELS, load, style

# name, label, colour, line style
RUNS = [("none_u50", "Think L2 RL u50（无 think 奖励）", "#2a78d6", "-"),
        ("action_l2rl_u50", "Action L2 RL u50", "#2a78d6", "--"),
        ("sum_u50", "Think L2 RL u50（sum 奖励）", "#1baf7a", "-"),
        ("action_l3rl_u42", "Action L3 RL u42", "#b8452a", "--")]
results = {(name, level): load(name, level) for name, *_ in RUNS for level in LEVELS}
missing = [k for k, v in results.items() if v is None]
assert not missing, f"incomplete cells: {missing}"


def draw(ax, key, title):
    for i, (name, label, color, ls) in enumerate(RUNS):
        x = np.array(LEVELS) + (i - len(RUNS) / 2) * 0.03
        v = [results[(name, l)][key] for l in LEVELS]
        ax.errorbar(x, [np.mean(s) for s in v], yerr=[np.std(s) for s in v], color=color, ls=ls, lw=1.8,
                    marker="o" if ls == "-" else "s", ms=5, capsize=3, label=label)
    if key == "success":
        ax.set_ylim(-0.03, 1.03)
    style(ax, title)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)


PANELS = [("success", "成功率"), ("ites", "每局 ITES")]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=BG)
for ax, (key, title) in zip(axes, PANELS):
    draw(ax, key, f"Think（实线）vs Action-only（虚线）：{title}")
    f, a = plt.subplots(figsize=(6.8, 4.4), facecolor=BG)
    draw(a, key, f"Think（实线）vs Action-only（虚线）：{title}")
    f.tight_layout(); f.savefig(f"figures/panels/pairs_{key}.png", dpi=150, facecolor=BG); plt.close(f)
fig.suptitle("零样本跨 Level 评测（RL：Think vs Action-only）：每点 30 局，误差线 = 3 组×10 局的标准差",
             color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("figures/pairs.png", dpi=140, facecolor=BG)
for name, label, *_ in RUNS:
    print(f"{label:34s}", " ".join(f"L{l} {np.mean(results[(name, l)]['success']):.2f}/{np.mean(results[(name, l)]['ites']):.2f}"
                                    for l in LEVELS))
