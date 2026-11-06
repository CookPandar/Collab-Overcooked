"""Training curves of the three RL baselines on L2 (think protocol, no think reward): our REINFORCE-style estimator
and the AgentGym-RL estimators GRPO and PPO (token-level critic), both with the AgentGym-RL actor settings.
Writes figures/baselines.png and figures/panels/baselines_*.png."""
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
RUNS = [("l2_none", "REINFORCE（局部奖励+终局，无 baseline；lr 1e-5，KL 0.02）", "#2a78d6", "o"),
        ("l2_none_grpo_agentgym", "GRPO（AgentGym 官方：组 4，lr 1e-6，KL 0.001，熵 0.001）", "#eb6834", "s"),
        ("l2_none_ppo_token", "PPO（verl token 级 critic+GAE，前 5 轮 critic warmup；actor 同 GRPO）", "#7b5cc4", "^")]
PANELS = [("success_rate", "训练成功率", False), ("episode_return", "每局训练回报", False),
          ("ites_per_episode", "每局 ITES 奖励", False), ("calls_per_episode", "每局 LLM 调用数", False),
          ("format_error_rate", "format 错误率（每次调用）", False), ("update/kl", "到 SFT 的 KL（对数轴）", True),
          ("update/entropy", "token 熵（仅 AgentGym 设置记录）", False),
          ("critic/explained_var", "critic explained variance（仅 PPO；≤0 = 不如常数）", False)]
data = {r: [json.loads(l) for l in open(f"rl/{r}/metrics.jsonl")] for r, *_ in RUNS}


def smooth(y, k=5):
    y = np.asarray(y, float)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def draw(ax, key, title, log):
    ax.set_facecolor(BG)
    for run, label, color, marker in RUNS:
        rows = [r for r in data[run] if r["update"] <= 50]
        y = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        if key == "update/kl" and run == "l2_none_ppo_token":  # critic warmup: the actor is not updated
            y = [np.nan if r.get("update/critic_warmup") else v for r, v in zip(rows, y)]
        if key == "update/entropy":
            y = [np.nan if v == 0 else v for v in y]
        if np.all(np.isnan(y)):
            continue
        x = [r["update"] for r in rows]
        ax.plot(x, y, color=color, lw=1, alpha=0.25)
        ax.plot(x, smooth(y), color=color, lw=2, marker=marker, ms=5, markevery=5, label=label.split("（")[0],
                markeredgecolor=BG, markeredgewidth=1)
    if log:
        ax.set_yscale("log")
    if key == "critic/explained_var":
        ax.axhline(0, color=MUTED, lw=0.8)
    if key == "update/kl":
        ax.axvspan(0.5, 5.5, color=GRID, alpha=0.5, lw=0)
        ax.text(3, 0.97, "PPO\ncritic\nwarmup", color=MUTED, fontsize=8, ha="center", va="top",
                transform=ax.get_xaxis_transform())
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlim(0.5, 50.5); ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel("更新轮次", color=MUTED, fontsize=9)


os.makedirs("figures/panels", exist_ok=True)
fig, axes = plt.subplots(4, 2, figsize=(13, 16), facecolor=BG)
for ax, (key, title, log) in zip(axes.flat, PANELS):
    draw(ax, key, title, log)
    f, a = plt.subplots(figsize=(6.4, 4), facecolor=BG)
    draw(a, key, title, log)
    a.legend(frameon=False, fontsize=8)
    f.tight_layout(); f.savefig(f"figures/panels/baselines_{key.replace('/', '_')}.png", dpi=150, facecolor=BG); plt.close(f)
handles = [plt.Line2D([], [], color=c, lw=2, marker=m, ms=6) for _, _, c, m in RUNS]
fig.legend(handles, [l for _, l, _, _ in RUNS], loc="upper center", ncol=1, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, 0.99))
fig.suptitle("L2 RL baseline 对比（think 协议，无 think 奖励；Qwen3-1.7B LoRA，每轮 4 局；粗线=5 轮滑动平均，细线=原始值）",
             color=INK, fontsize=12, y=0.997)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figures/baselines.png", dpi=130, facecolor=BG)
print("saved")
