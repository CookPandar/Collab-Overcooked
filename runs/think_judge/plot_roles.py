"""Chef (solid) vs Assistant (dashed) from role_analysis.py: L2 training per update (figures/roles_training.png) and the
zero-shot levels 1-6 (figures/roles_levels.png); one small figure per panel in figures/panels/roles_*.png."""
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
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
ROLES = [("chef", "Chef", "-"), ("assistant", "Assistant", "--")]
TRAIN = [("l2_none", "none", "#2a78d6"), ("l2_sum_w0.5", "sum", "#1baf7a"), ("l2_gated_w0.5", "gated", "#e87ba4"),
         ("l2_none_ppo_token", "PPO（AgentGym）", "#7b5cc4")]
LEVEL_MODELS = [("sft_l123", "Think SFT L1-3", "#8a8984"), ("sum_u50", "Think L2 RL sum", "#1baf7a"),
                ("gated_u50", "Think L2 RL gated", "#e87ba4"), ("action_sft_l123", "Action SFT L1-3", "#7b5cc4"),
                ("action_l2rl_u50", "Action L2 RL", "#1f1f1e"), ("action_l3rl_u42", "Action L3 RL", "#b8452a")]
METRICS = [("ites", "本角色 ITES（= 自己那部分的完成比例，0–1）"), ("validator_err", "validator 错误率（每次调用）"),
           ("wait_share", "wait 占本角色调用的比例"), ("collab_share", "Collab（聊天）占本角色调用的比例"),
           ("repeat", "重复聊天（每局）"), ("adv", "动作优势均值（每次调用）")]
training = json.load(open("eval/role_counts.json"))
levels = json.load(open("eval/role_levels.json"))


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.array([np.nanmean(y[max(0, i - k + 1):i + 1]) for i in range(len(y))])


def style(ax, title, xlabel):
    ax.set_facecolor(BG); ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)


def draw_training(ax, key, title):
    for run, label, color in TRAIN:
        rows = [r for r in training.get(run, []) if r["update"] <= 50]
        for role, name, ls in ROLES:
            y = [r[f"{role}/{key}"] if r[f"{role}/{key}"] is not None else np.nan for r in rows]
            ax.plot([r["update"] for r in rows], smooth(y), color=color, lw=2, ls=ls, label=f"{label} · {name}")
    if key == "adv":
        ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlim(0.5, 50.5); ax.set_xticks([1, 10, 20, 30, 40, 50])
    style(ax, title, "更新轮次（L2 训练，5 轮滑动平均）")


def draw_levels(ax, key, title, models):
    for name, label, color in models:
        for role, rname, ls in ROLES:
            pts = [(int(l), v[f"{role}/{key}"]) for l, v in sorted(levels.get(name, {}).items())]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2, ls=ls, marker="o" if ls == "-" else "s",
                    ms=4, label=f"{label} · {rname}")
    ax.set_xticks(range(1, 7), [f"L{l}" for l in range(1, 7)])
    style(ax, title, "Level（零样本，每点 30 局）")


os.makedirs("figures/panels", exist_ok=True)


def save(name, draw, *args):
    f, ax = plt.subplots(figsize=(7, 4.2), facecolor=BG)
    draw(ax, *args)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    f.tight_layout(); f.savefig(f"figures/panels/roles_{name}.png", dpi=150, facecolor=BG); plt.close(f)


fig, axes = plt.subplots(3, 2, figsize=(14, 13), facecolor=BG)
for ax, (key, title) in zip(axes.flat, METRICS):
    draw_training(ax, key, title)
    save(f"training_{key}", draw_training, key, title)
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.985))
fig.suptitle("L2 训练中的 Chef（实线）vs Assistant（虚线）", color=INK, fontsize=12, y=0.997)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig("figures/roles_training.png", dpi=130, facecolor=BG); plt.close(fig)

THINK, ACTION = LEVEL_MODELS[:3], LEVEL_MODELS[3:]
PANELS = [("ites", "Think：本角色 ITES（完成比例）", THINK), ("ites", "Action-only：本角色 ITES（完成比例）", ACTION),
          ("validator_err", "Think：validator 错误率", THINK), ("validator_err", "Action-only：validator 错误率", ACTION),
          ("wait_share", "Think：wait 占比", THINK), ("wait_share", "Action-only：wait 占比", ACTION)]
fig, axes = plt.subplots(3, 2, figsize=(14, 13), facecolor=BG)
for ax, (key, title, models) in zip(axes.flat, PANELS):
    draw_levels(ax, key, title, models)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    save(f"levels_{key}_{'think' if models is THINK else 'action'}", draw_levels, key, title, models)
fig.suptitle("零样本跨 Level：Chef（实线）vs Assistant（虚线）", color=INK, fontsize=12, y=0.997)
fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig("figures/roles_levels.png", dpi=130, facecolor=BG)
print("saved")
