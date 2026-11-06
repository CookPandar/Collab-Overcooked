"""Training curves of the rule-reward ablation (rl_l2_rules.sh): rollout success, KL, reasoning length and the three
rule-verified rewards per role, measured on each run's own training rollouts (runs without rule rewards have none).
usage: plot_rules_compare.py OUT_PNG TITLE RUN=LABEL ... [--ref RUN=LABEL]  (reference run drawn in gray)"""
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
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]  # validated slots 1-7 (light)
REF, INK2, GRID = "#898781", "#52514e", "#ebeae6"
TJ = Path(__file__).parent
PANELS = [("success_rate", "训练成功率（10 轮滚动平均）", 10), ("update/kl", "与参考模型的 KL（对数坐标）", 1),
          ("think/reasoning_tokens", "推理长度（token，5 轮滚动平均）", 5),
          ("rules/ground_chef", "第一视角 ground：Chef", 5), ("rules/ground_assistant", "第一视角 ground：Assistant", 5),
          ("rules/mate_assistant", "队友 mate（旧版，可被复述刷分）：Assistant", 5),
          ("rules/plan_chef", "长程规划 plan：Chef", 5), ("rules/plan_assistant", "长程规划 plan：Assistant", 5),
          ("rules/mate_v2_assistant", "队友 mate_v2（修复版）：Assistant", 5)]


def rolling(values, k):
    values = np.asarray(values, dtype=float)
    out = []
    for i in range(len(values)):
        window = values[max(0, i - k + 1):i + 1]
        window = window[~np.isnan(window)]
        out.append(window.mean() if len(window) else np.nan)
    return np.array(out)


args = sys.argv[1:]
out, title = args[0], args[1]
ref = None
specs = []
for a in args[2:]:
    if a.startswith("--ref="):
        ref = a[len("--ref="):].split("=", 1)
    else:
        specs.append(a.split("=", 1))
runs = [(run, label, COLORS[k], 1.8) for k, (run, label) in enumerate(specs)]
if ref:
    runs.append((ref[0], ref[1], REF, 1.4))
fig, axes = plt.subplots(3, 3, figsize=(16, 11))
ends = []
for run, label, color, lw in runs:
    rows = [json.loads(l) for l in open(TJ / "rl" / run / "metrics.jsonl")][:50]
    u = np.array([r["update"] for r in rows])
    for ax, (key, _, k) in zip(axes.flat, PANELS):
        y = np.array([r.get(key, np.nan) for r in rows], dtype=float)
        if np.all(np.isnan(y)):
            continue
        if key == "update/kl":
            y = np.maximum(y, 1e-4)
        y = rolling(y, k)
        ls = (0, (4, 2)) if color == REF else "-"
        ax.plot(u, y, color=color, lw=lw, ls=ls, label=label)
        if key == "success_rate":  # direct label at the end of each line (low-contrast slots need it)
            ends.append([y[-1], u[-1], f"{label.split('（')[0]} {y[-1]:.2f}"])
ends.sort()
for k in range(1, len(ends)):  # nudge the end labels apart so they don't overlap
    ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.045)
for y, x, text in ends:
    axes[0][0].text(51.5, y, text, color=INK2, fontsize=7.5, va="center")  # one column right of u50
for ax, (key, name, _) in zip(axes.flat, PANELS):
    if not ax.lines:
        ax.text(0.5, 0.5, "暂无数据（该组排队中或未训练）", transform=ax.transAxes, ha="center", color=INK2, fontsize=9)
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_xlabel("RL 更新轮次")
    ax.set_xlim(0, 60 if key == "success_rate" else 51)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if key.startswith("rules/"):
        ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=1)
axes[0][0].set_ylim(0, 1)
axes[0][1].set_yscale("log")
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=len(runs), frameon=False, loc="lower left", bbox_to_anchor=(0.01, -0.005))
fig.suptitle(title, x=0.01, ha="left", fontsize=11.5)
fig.text(0.01, 0.955, "规则奖励各项都是在各组自己的训练 rollout 上测的；none 组不做抽取，所以没有规则奖励曲线。"
         "前 6 轮回合上限 20–35 步，之后 50 步。", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(out, dpi=150)
