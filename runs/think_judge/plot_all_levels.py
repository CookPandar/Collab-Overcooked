"""Zero-shot results of eval_all_levels.sh on levels 1-6: success rate and ITES per episode, mean +- std over 3 groups
of 10 episodes (2 per order). Think checkpoints solid, action-only dashed; matched starts share a colour.
Writes figures/all_levels.png and figures/panels/all_levels_*.png."""
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
# name, label, colour, line style
THINK = [("sft_l12", "Think SFT L1-2", "#8a8984", "-"), ("sft_l123", "Think SFT L1-3", "#7b5cc4", "-"),
         ("none_u50", "Think L2 RL none u50", "#2a78d6", "-"), ("split_u50", "Think L2 RL split u50", "#eb6834", "-"),
         ("sum_u50", "Think L2 RL sum u50", "#1baf7a", "-"), ("decoupled_u50", "Think L2 RL decoupled u50", "#eda100", "-"),
         ("gated_u50", "Think L2 RL gated u50", "#e87ba4", "-")]
ACTION = [("action_sft_l12", "Action SFT L1-2", "#8a8984", "--"), ("action_sft_l123", "Action SFT L1-3", "#7b5cc4", "--"),
          ("action_l2rl_u50", "Action L2 RL u50", "#1f1f1e", "--"), ("action_l3rl_u42", "Action L3 RL u42", "#b8452a", "--"),
          ("action_l4rl_from_l3rl_u30", "Action L4 RL u30（自 L3 RL）", "#2a78d6", "--"),
          ("action_l4rl_from_sft_u30", "Action L4 RL u30（自 SFT L1-3）", "#1baf7a", "--")]
LEVELS = [1, 2, 3, 4, 5, 6]
GROUPS = 3


def load(name, level):
    path = f"eval/all_levels/{name}_l{level}.jsonl.episodes.jsonl"
    if not os.path.exists(f"eval/all_levels/{name}_l{level}.jsonl"):
        return None  # not run yet or still running
    # episodes are written in job order (repeat-major over the 5 orders), so each block of 10 has 2 per order
    episodes = [json.loads(l) for l in open(path)]
    size = len(episodes) // GROUPS
    groups = [episodes[g * size:(g + 1) * size] for g in range(GROUPS)]
    return {"success": [np.mean([e["success"] for e in g]) for g in groups],
            "ites": [np.mean([sum(c["ites"] for c in e["calls"]) for e in g]) for g in groups]}


results = {(name, level): load(name, level) for name, *_ in THINK + ACTION for level in LEVELS}


def style(ax, title):
    ax.set_facecolor(BG); ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xticks(LEVELS, [f"L{l}" for l in LEVELS]); ax.set_xlim(0.7, 6.3)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel("Level（L1-4 horizon 50，L5-6 horizon 120）", color=MUTED, fontsize=9)


def draw(ax, runs, key, title):
    for i, (name, label, color, ls) in enumerate(runs):
        pts = [(l, results[(name, l)][key]) for l in LEVELS if results[(name, l)]]
        if not pts:
            continue
        x = np.array([l for l, _ in pts]) + (i - len(runs) / 2) * 0.03  # small offset so error bars do not overlap
        ax.errorbar(x, [np.mean(v) for _, v in pts], yerr=[np.std(v) for _, v in pts], color=color, ls=ls, lw=1.8,
                    marker="o" if ls == "-" else "s", ms=5, capsize=3, label=label)
    if key == "success":
        ax.set_ylim(-0.03, 1.03)
    style(ax, title)


PAIRS = [THINK[0], ACTION[0], THINK[1], ACTION[1], THINK[4], THINK[6], ACTION[2], ACTION[3]]
PANELS = [("think_success", THINK, "success", "Think checkpoints：成功率"),
          ("think_ites", THINK, "ites", "Think checkpoints：每局 ITES"),
          ("action_success", ACTION, "success", "Action-only checkpoints：成功率"),
          ("action_ites", ACTION, "ites", "Action-only checkpoints：每局 ITES"),
          ("pair_success", PAIRS, "success", "Think（实线）vs Action-only（虚线）：成功率"),
          ("pair_ites", PAIRS, "ites", "Think（实线）vs Action-only（虚线）：每局 ITES")]
os.makedirs("figures/panels", exist_ok=True)
fig, axes = plt.subplots(3, 2, figsize=(13, 14), facecolor=BG)
for ax, (name, runs, key, title) in zip(axes.flat, PANELS):
    draw(ax, runs, key, title)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    f, a = plt.subplots(figsize=(6.4, 4.2), facecolor=BG)
    draw(a, runs, key, title)
    a.legend(frameon=False, fontsize=7, ncol=2)
    f.tight_layout(); f.savefig(f"figures/panels/all_levels_{name}.png", dpi=150, facecolor=BG); plt.close(f)
fig.suptitle("零样本跨 Level 评测：每点 30 局（每道菜 6 局），误差线 = 3 组×10 局的标准差；温度 0.7", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig("figures/all_levels.png", dpi=130, facecolor=BG)
done = sum(v is not None for v in results.values())
print(f"{done}/{len(results)} (checkpoint, level) cells done")
for (name, level), v in sorted(results.items()):
    if v:
        print(f"{name:18s} L{level} success {np.mean(v['success']):.2f}±{np.std(v['success']):.2f} "
              f"ites {np.mean(v['ites']):.2f}±{np.std(v['ites']):.2f}")
