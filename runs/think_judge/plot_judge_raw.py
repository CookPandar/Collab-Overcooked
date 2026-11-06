import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

FONT = "/home/zhangshuwen/.local/share/fonts/NotoSansCJKsc-Regular.otf"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": [font_manager.FontProperties(fname=FONT).get_name(), "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10})
RUNS = [("l2_none", "none（对照）", "#2a78d6", "o"), ("l2_split_w0.5", "split", "#eb6834", "s"),
        ("l2_sum_w0.5", "sum", "#1baf7a", "^"), ("l2_decoupled_w0.5", "decoupled", "#eda100", "D"),
        ("l2_gated_w0.5", "gated", "#e87ba4", "v")]
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.array([np.nanmean(y[max(0, i - k + 1):i + 1]) for i in range(len(y))])


def style(ax, title, xlabel="更新轮次"):
    ax.set_facecolor(BG)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlim(0.5, 50.5); ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)


data = {r: [json.loads(l) for l in open(f"rl/{r}/metrics.jsonl")] for r, *_ in RUNS}
offline = {}
for run, *_ in RUNS:
    path = f"eval/offline_judge_{run}.jsonl"
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if r["score"] is not None:
                offline.setdefault((run, r["update"], r["agent_index"]), []).append(r["score"])

panels = [("judge/score", "在线：judge 原始平均分（两角色合计）"), ("judge/score_chef", "在线：Chef 原始平均分"),
          ("judge/score_assistant", "在线：Assistant 原始平均分")]
def draw_online(ax, key, title):
    for run, label, color, marker in RUNS:
        rows = [r for r in data[run] if r.get(key) is not None]
        if not rows:
            continue
        x = [r["update"] for r in rows]; y = [r[key] for r in rows]
        ax.plot(x, y, color=color, lw=1, alpha=0.25)
        ax.plot(x, smooth(y), color=color, lw=2, marker=marker, ms=5, markevery=5, label=label,
                markeredgecolor=BG, markeredgewidth=1)
    style(ax, title)


def draw_offline(ax):
  for run, label, color, marker in RUNS:
    pts = sorted({u for (n, u, _) in offline if n == run})
    if not pts:
        continue
    y = [np.mean(offline.get((run, u, 0), []) + offline.get((run, u, 1), [])) for u in pts]
    ax.plot(pts, y, color=color, lw=2, marker=marker, ms=6, label=label, markeredgecolor=BG, markeredgewidth=1)
  style(ax, "离线补评：none 与 sum 的原始平均分（同一缩减上下文，每 5 轮）")


PANELS = [(draw_online, key, title) for key, title in panels] + [(draw_offline,)]
fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=BG)
for ax, (draw, *args) in zip(axes.flat, PANELS):
    draw(ax, *args)
seen = {}
for a in axes.flat:
    for h, l in zip(*a.get_legend_handles_labels()):
        seen.setdefault(l, h)
order = [label for _, label, *_ in RUNS if label in seen]
handles, labels = [seen[l] for l in order], order
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.975))
fig.suptitle("think judge 原始分数（未中心化，0.1–1.0；gpt-5.6-terra）  粗线=5 轮滑动平均，细线=每轮原始值；none 训练时无 judge，仅离线补评",
             color=INK, fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figures/judge_raw_scores.png", dpi=130, facecolor=BG)
os.makedirs("figures/panels", exist_ok=True)
for name, (draw, *args) in zip(["online", "online_chef", "online_assistant", "offline"], PANELS):
    f, ax = plt.subplots(figsize=(6.4, 4), facecolor=BG)
    draw(ax, *args)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    f.tight_layout()
    f.savefig(f"figures/panels/judge_raw_{name}.png", dpi=150, facecolor=BG)
    plt.close(f)
print("saved")
