"""Curves of the experiments of 2026-10-21/22, written to figures/update_20261022/:
l4_rl.png, l3_rl.png, collab_fix.png, collapse.png, sft_per_level.png (+ one png per panel in panels/).
Repeat / idle counts of collab_fix.png come from replaying the detector of CallCredit.uncooperative on the saved
rollouts of every run (same definition for the old runs, which were trained without it)."""
import collections
import gzip
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

sys.path[:0] = [os.path.expanduser("~/Collab-Overcooked-private"), os.path.expanduser("~/Collab-Overcooked-private/lib/overcooked_ai")]
from baselines.rl.credit import (REWARDABLE_CALL_TYPES, action_field, load_references, next_step_ready,  # noqa: E402
                                 next_steps, normalize_action, response_mode)

FONT = "/home/zhangshuwen/.local/share/fonts/NotoSansCJKsc-Regular.otf"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": [font_manager.FontProperties(fname=FONT).get_name(), "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10})
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, GREEN, PURPLE, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#7b5cc4", "#8a8984"
OUT = "figures/update_20261022"
os.makedirs(f"{OUT}/panels", exist_ok=True)


def metrics(run):
    path = f"rl/{run}/metrics.jsonl"
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.array([np.nanmean(y[max(0, i - k + 1):i + 1]) for i in range(len(y))])


def style(ax, title, ylabel=None):
    ax.set_facecolor(BG); ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8); ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.set_xlabel("训练轮次（update）", color=MUTED, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=MUTED, fontsize=9)


def curve(ax, x, y, color, label, ls="-", raw=True):
    if raw:
        ax.plot(x, y, color=color, lw=0.8, alpha=0.25, ls=ls)
    ax.plot(x, smooth(y), color=color, lw=2, ls=ls, label=label)


def figure(name, panels, runs, suptitle, ncols=None, series=None):
    """panels: [(key, title, ylabel, kwargs)]; runs: [(run, label, colour, ls)]; series(run, key) -> (x, y)."""
    ncols = ncols or len(panels)
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), facecolor=BG, squeeze=False)
    for ax, (key, title, ylabel, opts) in zip(axes.flat, panels):
        for target in (ax, None):
            if target is None:
                f, target = plt.subplots(figsize=(6.4, 4.2), facecolor=BG)
            for run, label, color, ls in runs:
                x, y = series(run, key)
                if len(x):
                    curve(target, x, y, color, label, ls)
            for ref in opts.get("refs", []):
                target.axhline(ref[0], color=GREY, lw=1, ls=":")
                target.text(target.get_xlim()[1], ref[0], ref[1], color=MUTED, fontsize=8, ha="right", va="bottom")
            if opts.get("log"): target.set_yscale("log")
            if opts.get("ylim"): target.set_ylim(*opts["ylim"])
            style(target, title, ylabel)
            target.legend(frameon=False, fontsize=8)
            if target is not ax:
                f.tight_layout(); f.savefig(f"{OUT}/panels/{name}_{key.replace('/', '_')}.png", dpi=150, facecolor=BG); plt.close(f)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(suptitle, color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/{name}.png", dpi=130, facecolor=BG); plt.close(fig)


def metric_series(run, key):
    m = metrics(run)
    return np.array([x["update"] for x in m]), np.array([x.get(key, np.nan) for x in m], float)


# ---- 1. L4 RL from the Level-4-only SFT ------------------------------------------------------------------
L4 = [("l4_none", "Think，无 think 奖励", BLUE, "-"), ("l4_sum_w0.5", "Think，sum 奖励 w0.5", GREEN, "-"),
      ("l4_action_only", "Action-only", ORANGE, "--")]
OLD = "旧 L4 RL（起点 L1-3 SFT / L3 RL）：成功率全程 0"
figure("l4_rl", [("success_rate", "训练成功率", None, {"ylim": (-0.03, 1.03), "refs": [(0.0, OLD)]}),
                 ("ites_per_episode", "每局 ITES", None, {}),
                 ("update/kl", "KL（相对 SFT 起点）", None, {"log": True})],
       L4, "L4 RL：起点为只用 L4 数据训练的 SFT（每轮 8 局，horizon 50，5 轮滑动平均，浅色为原始值）",
       series=metric_series)

# ---- 2. L3 Think RL --------------------------------------------------------------------------------------
L3 = [("l3_none", "Think，无 think 奖励", BLUE, "-"), ("l3_sum_w0.5", "Think，sum 奖励 w0.5", GREEN, "-")]
figure("l3_rl", [("success_rate", "训练成功率", None, {"ylim": (-0.03, 1.03),
                                                     "refs": [(0.8, "Action L3 RL u42 在 L3 的评测成功率 0.80")]}),
                 ("ites_per_episode", "每局 ITES", None, {}),
                 ("update/kl", "KL（相对 SFT 起点）", None, {"log": True})],
       L3, "L3 Think RL：起点为 Think SFT L1-3（每轮 8 局，horizon 50）", series=metric_series)

# ---- 3. fixed collaboration reward -----------------------------------------------------------------------
_cache = {}


def replay(run):
    """{update: (repeat per episode, idle per episode)} with the detector of CallCredit.uncooperative."""
    if run in _cache:
        return _cache[run]
    out = {}
    for name in sorted(os.listdir(f"rl/{run}/rollouts")):
        u = int(name[1:6])
        by = collections.defaultdict(list)
        for line in gzip.open(f"rl/{run}/rollouts/{name}", "rt"):
            r = json.loads(line)
            if "response" in r:
                by[r["episode"]].append(r)
        counts = collections.Counter()
        for calls in by.values():
            refs = load_references(calls[0]["order"])
            history, last = {0: [], 1: []}, {0: None, 1: None}
            for c in sorted(calls, key=lambda c: (c["timestep"], c["agent_index"], c["call_index"])):
                i, mode = c["agent_index"], response_mode(c["response"])
                ok = c["format_reward"] >= 0 and c["validator_reward"] >= 0
                if mode == "communication":
                    message = normalize_action(action_field(c["response"]))
                    counts["repeat"] += ok and message == last[i]
                    last[i] = message
                elif mode == "wait" and c["call_type"] in REWARDABLE_CALL_TYPES and ok:
                    steps = next_steps(history[i], refs[i])
                    counts["idle"] += bool(steps) and all(next_step_ready(s, c["observation"], i) for s in steps)
                elif mode == "planner_main" and ok:
                    history[i].append(normalize_action(action_field(c["response"])))
                    last = {0: None, 1: None}
        n = max(1, len(by))
        out[u] = (counts["repeat"] / n, counts["idle"] / n)
    _cache[run] = out
    return out


def collab_series(run, key):
    if key in ("repeat", "idle"):
        r = replay(run)
        x = np.array(sorted(r))
        return x, np.array([r[u][0 if key == "repeat" else 1] for u in x])
    return metric_series(run, key)


COLLAB_PANELS = [("success_rate", "训练成功率", None, {"ylim": (-0.03, 1.03)}),
                 ("ites_per_episode", "每局 ITES", None, {}),
                 ("repeat", "每局重复消息", "次/局", {}),
                 ("idle", "每局怠工（有事可做却 wait）", "次/局", {})]
for family, base, label in (("none", "l2_none", "无 think 奖励"), ("gated", "l2_gated_w0.5", "gated w0.5")):
    runs = [(base, f"{label}（原版）", GREY, "-"), (f"{base}_collab", f"{label} + 旧协作奖励（只奖不罚）", BLUE, "--"),
            (f"{base}_collabfix", f"{label} + 修复版协作奖励（罚重复/怠工）", ORANGE, "-")]
    figure(f"collab_fix_{family}", COLLAB_PANELS, runs,
           f"L2 协作奖励修复（{label}）：重复/怠工按同一检测器在各 run 的 rollout 上回放统计", series=collab_series)

# ---- 4. collapse of the one-sided fixes ------------------------------------------------------------------
COLLAPSE = [("l2_none", "REINFORCE（参照）", GREY, "-"),
            ("_collapsed_l2_none_grpo_gated_20261021", "GRPO + invalid/failure gate", BLUE, "-"),
            ("_collapsed_l2_none_ppo_token_gated_20261022", "PPO(token) + invalid/failure gate", ORANGE, "-"),
            ("l2_none_normsign", "REINFORCE + 保号标准化（normsign）", PURPLE, "--")]
figure("collapse", [("success_rate", "训练成功率", None, {"ylim": (-0.03, 1.03)}),
                    ("format_error_rate", "格式错误率", None, {"ylim": (-0.02, 1.0)}),
                    ("update/kl", "KL（相对 SFT 起点）", None, {"log": True})],
       COLLAPSE, "单边截断类修正在成功稀少时崩溃：正 advantage 消失 → 格式错误与 KL 暴涨", series=metric_series)

# ---- 5. per-level SFT matrix -----------------------------------------------------------------------------
for proto, prefix in (("action", "action_sft_l"), ("think", "think_sft_l")):
    S, I = np.full((6, 6), np.nan), np.full((6, 6), np.nan)
    for t in range(6):
        for l in range(6):
            path = f"eval/all_levels/{prefix}{t + 1}_l{l + 1}.jsonl"
            if os.path.exists(path):
                e = [json.loads(x) for x in open(path + ".episodes.jsonl")]
                S[t, l] = np.mean([x["success"] for x in e])
                I[t, l] = np.mean([sum(c["ites"] for c in x["calls"]) for x in e])
    if np.isnan(S).all():
        continue
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=BG)
    for ax, M, title, vmax in ((axes[0], S, "成功率", 1.0), (axes[1], I, "每局 ITES", 2.0)):
        ax.imshow(np.nan_to_num(M, nan=0), cmap="Blues", vmin=0, vmax=vmax)
        for t in range(6):
            for l in range(6):
                text = "—" if np.isnan(M[t, l]) else f"{M[t, l]:.2f}"
                ax.text(l, t, text, ha="center", va="center", fontsize=9,
                        color="white" if not np.isnan(M[t, l]) and M[t, l] > 0.6 * vmax else INK)
        ax.set_xticks(range(6), [f"L{i}" for i in range(1, 7)]); ax.set_yticks(range(6), [f"L{i}" for i in range(1, 7)])
        ax.set_xlabel("评测 Level", color=MUTED); ax.set_ylabel("SFT 训练 Level", color=MUTED)
        ax.set_title(title, color=INK, loc="left"); ax.tick_params(colors=MUTED)
    fig.suptitle(f"{'Action-only' if proto == 'action' else 'Think'} 单 Level SFT 的跨 Level 评测（每格 30 局，L5-6 horizon 120）",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{OUT}/sft_per_level_{proto}.png", dpi=130, facecolor=BG); plt.close(fig)
print("written", sorted(os.listdir(OUT)))
