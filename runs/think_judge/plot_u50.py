"""u50 checkpoints of the five think arms on L2: the 20 episodes of the regular eval plus the 80 fresh-seed episodes
of eval_u50_multi.sh, pooled. One small figure per panel in figures/panels/u50_*.png."""
import collections
import glob
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
RUNS = [("l2_none", "none", "#2a78d6"), ("l2_split_w0.5", "split", "#eb6834"), ("l2_sum_w0.5", "sum", "#1baf7a"),
        ("l2_decoupled_w0.5", "decoupled", "#eda100"), ("l2_gated_w0.5", "gated", "#e87ba4")]
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"

episodes = collections.defaultdict(list)
for path in glob.glob("eval/*_l2.jsonl.episodes.jsonl"):
    for line in open(path):
        e = json.loads(line)
        if e["checkpoint"].rstrip("/").endswith("u00050") and e["checkpoint"].split("/")[-3] in dict((r, 0) for r, *_ in RUNS):
            episodes[e["checkpoint"].split("/")[-3]].append(e)


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, p - (c - h), (c + h) - p


def style(ax, title):
    ax.set_facecolor(BG); ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)


def bars(ax, values, title, fmt):
    x = np.arange(len(RUNS))
    for i, (run, label, color) in enumerate(RUNS):
        v, lo, hi = values[run]
        ax.bar(i, v, color=color, width=0.6, yerr=[[lo], [hi]] if hi is not None else None, capsize=4,
               error_kw={"ecolor": MUTED, "lw": 1})
        ax.text(i, v + (hi or 0), fmt.format(v), ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(x, [f"{label}\n(n={len(episodes[run])})" for run, label, _ in RUNS])
    style(ax, title)


def success(ax):
    vals = {}
    for run, *_ in RUNS:
        n = len(episodes[run]); k = sum(e["success"] for e in episodes[run])
        vals[run] = wilson(k, n)
    bars(ax, vals, "u50 L2 成功率（95% Wilson 置信区间）", "{:.2f}")
    ax.set_ylim(0, 1)


def per_task(ax):
    orders = sorted({e["order"] for run, *_ in RUNS for e in episodes[run]})
    w = 0.8 / len(RUNS)
    for i, (run, label, color) in enumerate(RUNS):
        rate = [np.mean([e["success"] for e in episodes[run] if e["order"] == o] or [np.nan]) for o in orders]
        ax.bar(np.arange(len(orders)) + (i - 2) * w, rate, w, color=color, label=label)
    ax.set_xticks(np.arange(len(orders)), [o.replace("_", "\n", 1) for o in orders], fontsize=8)
    ax.set_ylim(0, 1); style(ax, "u50 各菜品成功率")


def mean_bars(ax, fn, title, fmt):
    vals = {}
    for run, *_ in RUNS:
        v = [fn(e) for e in episodes[run]]; v = [x for x in v if x is not None]
        m = np.mean(v) if v else 0; se = 1.96 * np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0
        vals[run] = (m, se, se)
    bars(ax, vals, title, fmt)


PANELS = {
    "success": success,
    "per_task": per_task,
    "calls": lambda ax: mean_bars(ax, lambda e: len(e["calls"]), "u50 每局 LLM 调用数（±95% CI）", "{:.0f}"),
    "steps_success": lambda ax: mean_bars(ax, lambda e: e["steps"] if e["success"] else None,
                                          "u50 成功局的完成步数（±95% CI）", "{:.1f}"),
}
os.makedirs("figures/panels", exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=BG)
for ax, (name, draw) in zip(axes.flat, PANELS.items()):
    draw(ax)
    f, a = plt.subplots(figsize=(6.4, 4), facecolor=BG)
    draw(a)
    if name == "per_task":
        a.legend(frameon=False, fontsize=8, ncol=5)
    f.tight_layout(); f.savefig(f"figures/panels/u50_{name}.png", dpi=150, facecolor=BG); plt.close(f)
axes.flat[1].legend(frameon=False, fontsize=8, ncol=5)
fig.suptitle("L2 u50 多局评测：think 奖励分配策略对比（温度 0.7，horizon 50）", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("figures/u50_eval.png", dpi=130, facecolor=BG)
print({run: (len(v), sum(e["success"] for e in v)) for run, v in episodes.items()})
