import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
font_manager.fontManager.addfont("/home/zhangshuwen/.local/share/fonts/NotoSansCJKsc-Regular.otf")
CJK = font_manager.FontProperties(fname="/home/zhangshuwen/.local/share/fonts/NotoSansCJKsc-Regular.otf").get_name()

plt.rcParams.update({"font.family": [CJK, "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10})
RUNS = [("l2_none", "none（对照）", "#2a78d6", "o"), ("l2_split_w0.5", "split", "#eb6834", "s"),
        ("l2_sum_w0.5", "sum（续训至 150 轮）", "#1baf7a", "^"), ("l2_decoupled_w0.5", "decoupled", "#eda100", "D"),
        ("l2_gated_w0.5", "gated", "#e87ba4", "v")]
METRICS = [("episode_return", "每局训练回报（format+validator+ITES+terminal）"), ("success_rate", "训练成功率"),
           ("ites_per_episode", "每局 ITES 奖励"), ("judge/score", "judge 原始分数（0.1–1.0；none 为离线补评，虚线）"),
           ("judge/action_follows_rate", "judge 判定 Action 与 Think 一致的比例（none 为离线补评）"), ("format_error_rate", "format 错误率（每次调用）"),
           ("validator_error_rate", "validator 错误率（每次调用）"), ("steps", "每局步数（越少越快）"),
           ("calls_per_episode", "每局 LLM 调用数"), ("update/kl", "到 SFT 的 KL（对数轴，粗线=5 轮滑动中位数）")]
XMAX = 150  # sum was continued to 150 updates; the other arms stop at 50
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

def smooth(y, k=5, reduce=np.mean):
    y = np.asarray(y, float); out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]; w = w[~np.isnan(w)]
        out[i] = reduce(w) if len(w) else np.nan
    return out


def style_x(ax, start):
    ax.set_xlim(start, XMAX + 0.5); ax.set_xticks(([0] if start < 0 else [1]) + list(range(25, XMAX + 1, 25)))
    ax.axvline(50.5, color=MUTED, lw=0.8, ls="--", zorder=0)
    ax.text(51.5, 0.97, "其余方法止于第 50 轮", color=MUTED, fontsize=8, va="top", transform=ax.get_xaxis_transform())

import glob
import os
data = {r: [json.loads(l) for l in open(f"rl/{r}/metrics.jsonl")] for r, *_ in RUNS}
# held-out evaluation (20 episodes per point); update 0 is the shared SFT start
evals = {}
for f in glob.glob("eval/*.jsonl"):
    if f.endswith(".episodes.jsonl") or os.path.basename(f).startswith("offline_judge"):
        continue
    for line in open(f):
        r = json.loads(line); ck = r["checkpoint"].rstrip("/")
        level = 2 if "_l2" in f else 3
        if ck.endswith("sft_think_l12"):
            for run, *_ in RUNS:
                evals[(run, level, 0)] = r
        elif "/checkpoints/u" in ck:
            evals[(ck.split("/")[-3], level, int(ck.split("/u")[-1]))] = r
EVALS = [(2, "success_rate", "评测：L2 成功率（每点 20 局）"), (3, "ites_per_episode", "评测：L3 零样本 ITES（成功率全为 0）")]
def draw_eval(ax, level, key, title):
    ax.set_facecolor("#fcfcfb")
    for run, label, color, marker in RUNS:
        pts = sorted((u, r[key]) for (n, l, u), r in evals.items() if n == run and l == level)
        ax.plot([u for u, _ in pts], [v for _, v in pts], color=color, lw=2, marker=marker, ms=6, label=label,
                markeredgecolor="#fcfcfb", markeredgewidth=1)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    style_x(ax, -1.5)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel("更新轮次（0 = SFT 起点）", color=MUTED, fontsize=9)


# none was trained without a judge: its judge curves come from judging its saved rollouts offline (every 5 updates)
offline_none = {}
if os.path.exists("eval/offline_judge_l2_none.jsonl"):
    for line in open("eval/offline_judge_l2_none.jsonl"):
        r = json.loads(line)
        if r["score"] is not None:
            offline_none.setdefault(r["update"], []).append((r["score"], r["follows"] is not False))
OFFLINE = {"judge/score": lambda v: np.mean([s for s, _ in v]),
           "judge/action_follows_rate": lambda v: np.mean([f for _, f in v])}
def draw_metric(ax, key, title):
    ax.set_facecolor("#fcfcfb")
    if key in OFFLINE and offline_none:
        run, label, color, marker = RUNS[0]
        pts = sorted(offline_none)
        ax.plot(pts, [OFFLINE[key](offline_none[u]) for u in pts], color=color, lw=2, ls="--", marker=marker, ms=6,
                label=label + "（离线补评）", markeredgecolor="#fcfcfb", markeredgewidth=1)
    for run, label, color, marker in RUNS:
        rows = data[run]
        if key.startswith("judge/") and not any(key in r for r in rows):
            continue
        x = [r["update"] for r in rows]; y = [r.get(key, np.nan) for r in rows]
        ax.plot(x, y, color=color, lw=1, alpha=0.25)
        ax.plot(x, smooth(y, reduce=np.median if key == "update/kl" else np.mean), color=color, lw=2, marker=marker,
                ms=5, markevery=10, label=label, markeredgecolor="#fcfcfb", markeredgewidth=1)
    if key == "update/kl":
        ax.set_yscale("log")
        ax.set_ylim(3e-4, 30)  # sum has single-update spikes up to 1e6 after update 110 (clipped in the loss)
        ax.text(112, 25, "sum 在 u110 后有单轮尖峰（最高约 1e6，已截出画面）", color=MUTED, fontsize=8, va="top")
    for u in (3.5, 6.5):
        ax.axvline(u, color=MUTED, lw=0.8, ls=":", zorder=0)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    style_x(ax, 0.5)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel("更新轮次", color=MUTED, fontsize=9)
    if key == "episode_return":
        top = ax.get_ylim()[1]
        ax.text(7.5, top, "点线：H=20→35→50", color=MUTED, fontsize=8, ha="left", va="top")


PANELS = [(draw_metric, key, title) for key, title in METRICS] + [(draw_eval, *e) for e in EVALS]
fig, axes = plt.subplots(6, 2, figsize=(16, 22), facecolor="#fcfcfb")
for ax, (draw, *args) in zip(axes.flat, PANELS):
    draw(ax, *args)
seen = {}
for a in axes.flat:
    for h, l in zip(*a.get_legend_handles_labels()):
        seen.setdefault(l, h)
handles, labels = list(seen.values()), list(seen)
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.985))
fig.suptitle("L2 RL 训练曲线：think 奖励分配策略对比（Qwen3-1.7B，L1-2 Think SFT 起点，每轮 4 局；粗线=5 轮滑动平均，细线=原始值；虚线=horizon 切换）",
             color=INK, fontsize=12, y=0.997)
fig.tight_layout(rect=(0, 0, 1, 0.978))
fig.savefig("figures/training_curves.png", dpi=130, facecolor=fig.get_facecolor())
# one small figure per panel (for the experiment log)
os.makedirs("figures/panels", exist_ok=True)
NAMES = [key.replace("/", "_") for key, _ in METRICS] + ["eval_l2_success", "eval_l3_ites"]
for name, (draw, *args) in zip(NAMES, PANELS):
    f, ax = plt.subplots(figsize=(8, 4), facecolor="#fcfcfb")
    draw(ax, *args)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    f.tight_layout()
    f.savefig(f"figures/panels/train_{name}.png", dpi=150, facecolor=f.get_facecolor())
    plt.close(f)
print("saved")
