"""Stall / chatter behaviour per update (from count_behavior.py) for the five think arms and the stall-fix runs.
Writes figures/behavior.png, figures/behavior_fix.png and one small figure per panel in figures/panels/behavior_*.png."""
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
RUNS = [("l2_none", "none", "#2a78d6", "o"), ("l2_split_w0.5", "split", "#eb6834", "s"),
        ("l2_sum_w0.5", "sum", "#1baf7a", "^"), ("l2_decoupled_w0.5", "decoupled", "#eda100", "D"),
        ("l2_gated_w0.5", "gated", "#e87ba4", "v")]
FIX = [("l2_none", "none", "#2a78d6", "-"), ("l2_none_collab", "none+协作奖励", "#2a78d6", "--"),
       ("l2_none_norm", "none+组内归一化", "#2a78d6", ":"), ("l2_none_normsign", "none+保号归一化", "#2a78d6", "-."),
       ("l2_gated_w0.5", "gated", "#e87ba4", "-"), ("l2_gated_w0.5_collab", "gated+协作奖励", "#e87ba4", "--"),
       ("l2_gated_w0.5_normsign", "gated+保号归一化", "#e87ba4", "-.")]
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
data = json.load(open("eval/behavior_counts.json"))
PANELS = [("stall_steps", "拖延：Chef 拿着未加工切片的步数（每局）"),
          ("stall_calls", "拖延：Chef 拿着未加工切片时 wait / 聊天的调用数（每局）"),
          ("repeat", "重复聊天：与上一次完全相同的 Collab 调用数（每局）"),
          ("collab", "Collab（聊天）调用数（每局）"),
          ("stall_adv", "拖延调用获得的动作优势之和（每局；>0 即被强化）"),
          ("repeat_adv", "重复聊天调用获得的动作优势之和（每局；>0 即被强化）")]


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.array([np.nanmean(y[max(0, i - k + 1):i + 1]) for i in range(len(y))])


def style(ax, title, xmax=50):
    ax.set_facecolor(BG); ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlim(0.5, xmax + 0.5)
    ax.set_xticks([1, 10, 20, 30, 40, 50] if xmax <= 50 else [1] + list(range(25, xmax + 1, 25)))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel("更新轮次", color=MUTED, fontsize=9)


def series(run, key, xmax=50):
    rows = [r for r in data.get(run, []) if r["update"] <= xmax]
    return [r["update"] for r in rows], [r[key] for r in rows]


def draw_main(ax, key, title):
    for run, label, color, marker in RUNS:
        x, y = series(run, key)
        ax.plot(x, y, color=color, lw=1, alpha=0.2)
        ax.plot(x, smooth(y), color=color, lw=2, marker=marker, ms=5, markevery=5, label=label,
                markeredgecolor=BG, markeredgewidth=1)
    if key.endswith("_adv"):
        ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, title)


def draw_gated(ax):
    """gated: success rate against stall steps, both 5-update moving averages."""
    x, s = series("l2_gated_w0.5", "success")
    _, st = series("l2_gated_w0.5", "stall_steps")
    ax.plot(x, smooth(s), color="#e87ba4", lw=2.2, label="训练成功率（左轴）")
    ax.set_ylim(0, 1)
    twin = ax.twinx()
    twin.plot(x, smooth(st), color=INK, lw=1.8, ls="--", label="拖延步数/局（右轴）")
    twin.tick_params(colors=MUTED, labelsize=9); twin.spines["top"].set_visible(False)
    for s_ in ("left", "bottom", "right"): twin.spines[s_].set_color(GRID)
    style(ax, "gated：成功率与拖延步数（5 轮滑动平均）")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")


def draw_fix(ax, key, title):
    xmax = 1
    for run, label, color, ls in FIX:
        x, y = series(run, key)
        if not x:
            continue
        xmax = max(xmax, max(x))
        ax.plot(x, smooth(y), color=color, lw=2, ls=ls, label=f"{label}（{len(x)} 轮）")
    if key.endswith("_adv"):
        ax.axhline(0, color=MUTED, lw=0.8)
    style(ax, title)


os.makedirs("figures/panels", exist_ok=True)


def save(name, draw, *args, legend=True):
    f, ax = plt.subplots(figsize=(6.4, 4), facecolor=BG)
    draw(ax, *args)
    if legend:
        ax.legend(frameon=False, fontsize=8, ncol=2)
    f.tight_layout(); f.savefig(f"figures/panels/behavior_{name}.png", dpi=150, facecolor=BG); plt.close(f)


fig, axes = plt.subplots(4, 2, figsize=(13, 16), facecolor=BG)
for ax, (key, title) in zip(axes.flat, PANELS):
    draw_main(ax, key, title)
    save(key, draw_main, key, title)
draw_gated(axes.flat[6]); save("gated_success_vs_stall", draw_gated, legend=False)
draw_main(axes.flat[7], "success", "训练成功率（每轮 4 局）")
save("success", draw_main, "success", "训练成功率（每轮 4 局）")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.985))
fig.suptitle("L2 RL 中的拖延与重复聊天行为（每轮 4 局，粗线=5 轮滑动平均，细线=原始值）", color=INK, fontsize=12, y=0.997)
fig.tight_layout(rect=(0, 0, 1, 0.975))
fig.savefig("figures/behavior.png", dpi=130, facecolor=BG)

FIX_PANELS = [("stall_adv", "拖延调用获得的动作优势之和（每局）"), ("repeat_adv", "重复聊天调用获得的动作优势之和（每局）"),
              ("stall_steps", "拖延步数（每局）"), ("repeat", "重复聊天调用数（每局）")]
fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=BG)
for ax, (key, title) in zip(axes.flat, FIX_PANELS):
    draw_fix(ax, key, title)
    save(f"fix_{key}", draw_fix, key, title)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.985))
fig.suptitle("拖延修复实验：协作奖励 vs 组内优势归一化（进行中，5 轮滑动平均）", color=INK, fontsize=12, y=0.997)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("figures/behavior_fix.png", dpi=130, facecolor=BG)
# the three RL baselines (no think reward) and sum continued to 150 updates
BASELINES = [("l2_none", "REINFORCE", "#2a78d6", "o"), ("l2_none_grpo_agentgym", "GRPO（AgentGym 官方超参）", "#eb6834", "s"),
             ("l2_none_ppo_token", "PPO（token 级 critic）", "#7b5cc4", "^")]
EXTRA_PANELS = PANELS + [("success", "训练成功率（每轮 4 局）"), ("calls", "LLM 调用数（每局）")]


def draw_runs(ax, runs, key, title, xmax):
    for run, label, color, marker in runs:
        x, y = series(run, key, xmax)
        if not x:
            continue
        ax.plot(x, y, color=color, lw=1, alpha=0.2)
        ax.plot(x, smooth(y), color=color, lw=2, marker=marker, ms=5, markevery=5 if xmax <= 50 else 10, label=label,
                markeredgecolor=BG, markeredgewidth=1)
    if key.endswith("_adv"):
        ax.axhline(0, color=MUTED, lw=0.8)
    if xmax > 50:
        ax.axvline(50.5, color=MUTED, lw=0.8, ls="--", zorder=0)
    style(ax, title, xmax)


for name, runs, xmax, title in [
        ("baselines", BASELINES, 50, "RL baseline 的拖延与重复聊天行为（think 协议，无 think 奖励；粗线=5 轮滑动平均）"),
        ("sum150", [RUNS[2], ("l2_none", "none（对照，50 轮）", "#2a78d6", "o")], 150,
         "sum 续训到 150 轮的拖延与重复聊天行为（虚线=原 50 轮终点；粗线=5 轮滑动平均）")]:
    fig, axes = plt.subplots(4, 2, figsize=(13 if xmax <= 50 else 16, 16), facecolor=BG)
    for ax, (key, t) in zip(axes.flat, EXTRA_PANELS):
        draw_runs(ax, runs, key, t, xmax)
        f, a = plt.subplots(figsize=(6.4 if xmax <= 50 else 8, 4), facecolor=BG)
        draw_runs(a, runs, key, t, xmax)
        a.legend(frameon=False, fontsize=8)
        f.tight_layout(); f.savefig(f"figures/panels/behavior_{name}_{key}.png", dpi=150, facecolor=BG); plt.close(f)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, color=INK, fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(f"figures/behavior_{name}.png", dpi=130, facecolor=BG)
    plt.close(fig)
print("saved")
