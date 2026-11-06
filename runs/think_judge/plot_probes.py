"""Chart of probe_eval.py circular scores along one level's training chain (base -> SFT -> RL not converged ->
RL converged): accuracy per question type with Wilson 95% intervals and the chance level of guessing every rotation.
usage: plot_probes.py SCORES_JSONL OUT_PNG TITLE NAME=LABEL ...  (in plotting order)"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
font_manager.fontManager.addfont(str(Path.home() / ".local/share/fonts/NotoSansCJKsc-Regular.otf"))
plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

QTYPES = ["F-hold", "F-content", "F-status", "F-space", "F-next", "F-feasible",
          "S-hold", "S-doing", "S-wants", "S-profile", "S-who", "S-next", "S-canreq",
          "P-due", "P-mate-due", "P-before", "P-remaining"]
NAMES = {"F-hold": "我手里\n拿着什么", "F-content": "某厨具里\n装着什么", "F-status": "某厨具\n处于什么状态",
         "F-space": "我自己能\n操作厨具U吗", "F-next": "我下一步\n该做哪个动作", "F-feasible": "我现在\n能执行动作X吗",
         "S-hold": "队友手里\n拿着什么", "S-doing": "队友此刻\n在执行什么", "S-wants": "队友最近\n请我做什么",
         "S-profile": "队友能\n操作厨具U吗", "S-who": "某菜谱步骤\n该由谁做", "S-next": "队友下一个\n动作是什么",
         "S-canreq": "队友现在\n能执行动作X吗", "P-due": "我按计划\n下一步该完成", "P-mate-due": "队友按计划\n下一步该完成",
         "P-before": "X必须在\nY之前做吗", "P-remaining": "我的计划\n还剩几步",
         "F": "第一视角\n平均", "S": "第二视角\n平均", "P": "多步规划\n平均"}
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated categorical slots 1-5 (light surface)
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#ebeae6"


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, p - (centre - half), (centre + half) - p


def main():
    path, out, title, specs = sys.argv[1], sys.argv[2], sys.argv[3], [s.split("=", 1) for s in sys.argv[4:]]
    hits = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        for key in (r["qtype"], r["view"]):
            hits[(r["checkpoint"], key)].append(r["correct"])
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": "#d6d5d0", "figure.facecolor": SURF,
                         "axes.facecolor": SURF})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(19, 5.4), gridspec_kw={"width_ratios": [17, 2.8]})
    w = 0.8 / len(specs)
    qtypes = [q for q in QTYPES if any(hits[(n, q)] for n, _ in specs)]
    views = [v for v in "FSP" if any(hits[(n, v)] for n, _ in specs)]
    for axis, cols in ((ax, qtypes), (ax2, views)):
        x = np.arange(len(cols))
        for k, (name, label) in enumerate(specs):
            stats = [wilson(sum(hits[(name, q)]), len(hits[(name, q)])) for q in cols]
            vals = [s[0] for s in stats]
            bars = axis.bar(x + (k - (len(specs) - 1) / 2) * w, vals, w - 0.03, color=COLORS[k], label=label, zorder=2,
                            yerr=np.array([[s[1] for s in stats], [s[2] for s in stats]]),
                            error_kw={"ecolor": INK2, "elinewidth": 0.8, "capsize": 0})
            if k in (0, len(specs) - 1):  # direct labels on the two ends of the chain
                for b, v in zip(bars, vals):  # inside the bar when it is tall enough, else above it
                    inside = v > 0.12
                    axis.text(b.get_x() + b.get_width() / 2, 0.015 if inside else v + 0.04, f"{v:.2f}",
                              ha="center", va="bottom", fontsize=6.5, color="white" if inside else INK2,
                              rotation=90, zorder=3)
        axis.set_xticks(x, [NAMES[q] for q in cols], fontsize=8.5)
        axis.set_ylim(0, 1.05)
        axis.yaxis.grid(True, color=GRID, zorder=0)
        for s in ("top", "right"):
            axis.spines[s].set_visible(False)
    for j, q in enumerate(qtypes):  # chance of passing every rotation by uniform guessing = (1/n)^n
        chance = 0.25 if q in ("F-feasible", "S-canreq", "S-who", "F-space", "S-profile", "P-before") else 0.25 ** 4
        ax.hlines(chance, j - 0.42, j + 0.42, color=INK, lw=1, ls=(0, (2, 2)), zorder=4)
    sections = [("F", "第一视角：对环境与自身任务的认知"), ("S", "第二视角：对队友的建模"), ("P", "多步规划")]
    start = 0
    for view, label in sections:
        n = sum(q.startswith(view) for q in qtypes)
        if n:
            ax.text(start + (n - 1) / 2, 1.0, label, ha="center", fontsize=9.5)
            if start:
                ax.axvline(start - 0.5, color="#d6d5d0", lw=1)
        start += n
    ax.set_ylabel("准确率（选项轮换的每一遍都答对才算对）")
    ax2.set_title("按视角汇总", fontsize=9.5, color=INK2)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [plt.Line2D([], [], color=INK, ls=(0, (2, 2)))], labels + ["瞎猜每遍都对的概率"],
              ncol=len(specs) + 1, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2), fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    for name, label in specs:
        print(f"{label:28s}", " ".join(f"{q}:{np.mean(hits[(name, q)]):.2f}" for q in QTYPES + list("FSP") if hits[(name, q)]))


if __name__ == "__main__":
    main()
