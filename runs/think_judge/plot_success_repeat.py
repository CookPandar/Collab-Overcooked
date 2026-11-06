"""Rule-reward ablation: training success rate and repeated-communication share, side by side.
Repeat share (per update, from rl/RUN/rollouts/uNNNNN.jsonl.gz) = among an agent's Collab messages, the fraction that the
same agent already sent (identical text) since the last executed kitchen action of either agent in that episode.
Executed kitchen action = non-Collab/non-wait action with validator_reward == 0 and format_reward == 0.
Only updates whose rollouts were saved are shown (after 10-27 ~19:40 only u1, every 25th and the last are kept).
usage: plot_success_repeat.py [OUT_PNG]   (per-update shares cached in eval/rule_rewards/repeat_share_cache.json)
       plot_success_repeat.py l2s [OUT_PNG]   reward-shaping arms (docs/reward_shaping.md): success, repeat share and
       calls per agent-step; l2s runs use the per-update metrics comm/* of metrics.jsonl (every update), the other
       runs their saved rollouts (cache eval/rule_rewards/comm_stats_cache.json)"""
import gzip
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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
OUT_DIR = TJ / "eval" / "rule_rewards"
CACHE = OUT_DIR / "repeat_share_cache.json"
RUNS = [("l2r_none", "none（无思考奖励）"), ("l2r_mate", "mate（旧版）"), ("l2r_mate_v2", "mate_v2"),
        ("l2r_plan", "plan"), ("l2r_full", "full"), ("l2r_ground_split", "ground_split"),
        ("l2r_full_v2_split", "full_v2_split")]
REF_RUN = ("l2t_sum_matchfloor_g1", "LLM 评委 sum（参考）")
MAX_U = 50
ACTION_RE = re.compile(r"Action:\s*(.*)")


def repeat_counts(path):
    """Return (repeats, messages) for one rollout file."""
    calls = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "episode_summary" in r:
                continue
            calls.append(r)
    calls.sort(key=lambda r: (r["episode"], r["timestep"], r["call_index"]))
    sent = {}  # (episode, agent) -> set of messages since last executed action
    reps = msgs = 0
    for r in calls:
        m = ACTION_RE.findall(r.get("response") or "")
        if not m:
            continue
        act = m[-1].strip()
        ep, ag = r["episode"], r["agent_index"]
        low = act.lower()
        if low.startswith("collab"):
            s = sent.setdefault((ep, ag), set())
            msgs += 1
            reps += act in s
            s.add(act)
        elif not low.startswith("wait"):
            try:
                ok = float(r.get("validator_reward")) == 0 and float(r.get("format_reward")) == 0
            except (TypeError, ValueError):
                ok = False
            if ok:
                for k in [k for k in sent if k[0] == ep]:
                    sent[k] = set()
    return reps, msgs


def rolling(values, k):
    values = np.asarray(values, dtype=float)
    out = []
    for i in range(len(values)):
        w = values[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        out.append(w.mean() if len(w) else np.nan)
    return np.array(out)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT_DIR / "success_repeat_l2r.png"
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    allruns = RUNS + [REF_RUN]
    todo = []
    for run, _ in allruns:
        for p in sorted((TJ / "rl" / run / "rollouts").glob("u*.jsonl.gz")):
            u = int(p.name[1:6])
            key = f"{run}/{u}"
            if u <= MAX_U and key not in cache:
                todo.append((key, p))
    if todo:
        with ProcessPoolExecutor(32) as ex:
            for (key, _), res in zip(todo, ex.map(repeat_counts, [p for _, p in todo])):
                cache[key] = {"repeats": res[0], "messages": res[1]}
        CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for k, (run, label) in enumerate(allruns):
        is_ref = run == REF_RUN[0]
        color, lw, ls = (REF, 1.4, (0, (4, 2))) if is_ref else (COLORS[k], 1.8, "-")
        rows = [json.loads(l) for l in open(TJ / "rl" / run / "metrics.jsonl")][:MAX_U]
        if len(rows) < MAX_U:
            label = f"{label}（至 u{len(rows)}）"
        u = np.array([r["update"] for r in rows])
        y = rolling([r.get("success_rate", np.nan) for r in rows], 10)
        ax1.plot(u, y, color=color, lw=lw, ls=ls, label=label)
        # repeat share: per-update value on 1..50 grid, NaN where rollouts weren't saved
        share = np.full(MAX_U, np.nan)
        for uu in range(1, MAX_U + 1):
            c = cache.get(f"{run}/{uu}")
            if c and c["messages"] > 0:
                share[uu - 1] = c["repeats"] / c["messages"]
        sm = rolling(share, 5)
        sm[np.isnan(share)] = np.nan  # only draw at updates that actually have data; lines break across gaps
        xs = np.arange(1, MAX_U + 1)
        ax2.plot(xs, sm, color=color, lw=lw, ls=ls)
        have = ~np.isnan(sm)
        iso = have & ~np.r_[False, have[:-1]] & ~np.r_[have[1:], False]  # isolated points (gap on both sides)
        ax2.plot(xs[iso], sm[iso], ls="none", marker="o", ms=6, color=color, mec="#fcfcfb", mew=1.5)
    ax1.set_title("训练成功率（10 轮滚动平均）", loc="left", fontsize=10)
    ax2.set_title("重复通信占比（自上次执行动作以来已发过的消息 / 全部消息）", loc="left", fontsize=10)
    for ax in (ax1, ax2):
        ax.set_xlabel("RL 更新轮次")
        ax.set_xlim(0, 51)
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    ax2.text(0.01, 0.98, "5 轮滚动平均（只用窗口内已保存的轮次）；未保存 rollout 的轮次留空不插值，孤立点用圆点标出",
             transform=ax2.transAxes, fontsize=7.5, color=INK2, va="top")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.835, 0.5), fontsize=9,
               handlelength=2.6, labelspacing=0.9)
    stamp = time.strftime("%Y-%m-%d %H:%M")
    fig.suptitle(f"L2 规则奖励消融：成功率与重复通信（Qwen3-1.7B，只用 3 道训练菜；{stamp}）", x=0.01, ha="left",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 0.83, 0.96))
    fig.savefig(out, dpi=150)
    print(out)
    for run, _ in allruns:
        vals = [(int(k.split("/")[1]), v["repeats"] / max(v["messages"], 1)) for k, v in cache.items()
                if k.split("/")[0] == run]
        vals.sort()
        print(run, " ".join(f"u{u}:{s:.2f}" for u, s in vals if u in (1, 5, 10, 15, 17, 25, 38, 50)))


L2S_RUNS = [("l2r_none", "l2r_none（基线）"), ("l2s_s0", "S0：intra_gamma+重复罚+role 归一化"),
            ("l2s_s0a", "S0a：只 role 归一化"), ("l2s_s1", "S1：S0+消息采纳奖励"),
            ("l2s_s2_rule", "S2_rule：S1+规则思考奖励"), ("l2s_s2_llm", "S2_llm：S1+LLM 评委思考奖励")]
COMM_CACHE = OUT_DIR / "comm_stats_cache.json"


def comm_stats(path):
    """Per-update communication stats of a rollout file: repeat share (as repeat_counts), calls per
    (episode, agent, timestep), message share per role."""
    reps, msgs = repeat_counts(path)
    steps, calls, mine = set(), 0, {0: [0, 0], 1: [0, 0]}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "episode_summary" in r:
                continue
            calls += 1
            steps.add((r["episode"], r["agent_index"], r["timestep"]))
            mine[r["agent_index"]][0] += r.get("action_mode") == "communication"
            mine[r["agent_index"]][1] += 1
    return {"comm/repeat_share": reps / max(1, msgs), "comm/calls_per_agent_step": calls / max(1, len(steps)),
            "comm/msg_share_chef": mine[0][0] / max(1, mine[0][1]),
            "comm/msg_share_assistant": mine[1][0] / max(1, mine[1][1])}


def series(run, key, cache):
    """Per-update values 1..MAX_U of `key` (NaN where missing): metrics.jsonl when it has the key, else rollouts."""
    out = np.full(MAX_U, np.nan)
    path = TJ / "rl" / run / "metrics.jsonl"
    rows = [json.loads(l) for l in open(path)][:MAX_U] if path.exists() else []
    for r in rows:
        if key in r:
            out[r["update"] - 1] = r[key]
    if np.isnan(out).all():
        for u in range(1, MAX_U + 1):
            c = cache.get(f"{run}/{u}")
            if c:
                out[u - 1] = c[key]
    return out


def main_l2s(out):
    cache = json.loads(COMM_CACHE.read_text()) if COMM_CACHE.exists() else {}
    allruns = L2S_RUNS + [REF_RUN]
    todo = []
    for run, _ in allruns:
        if run.startswith("l2s_"):
            continue
        for p in sorted((TJ / "rl" / run / "rollouts").glob("u*.jsonl.gz")):
            key = f"{run}/{int(p.name[1:6])}"
            if int(p.name[1:6]) <= MAX_U and key not in cache:
                todo.append((key, p))
    if todo:
        with ProcessPoolExecutor(32) as ex:
            for (key, _), res in zip(todo, ex.map(comm_stats, [p for _, p in todo])):
                cache[key] = res
        COMM_CACHE.write_text(json.dumps(cache, indent=0, sort_keys=True))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    panels = [("success_rate", "训练成功率（10 轮滚动平均）", 10, (0, 1)),
              ("comm/repeat_share", "重复通信占比（重复消息 / 全部消息，5 轮滚动）", 5, (0, 1)),
              ("comm/calls_per_agent_step", "每角色每步的调用次数（5 轮滚动）", 5, (1, None))]
    for k, (run, label) in enumerate(allruns):
        if not (TJ / "rl" / run / "metrics.jsonl").exists():
            continue
        is_ref = run == REF_RUN[0]
        color, lw, ls = (REF, 1.4, (0, (4, 2))) if is_ref else (COLORS[k], 1.8, "-")
        n = sum(1 for _ in open(TJ / "rl" / run / "metrics.jsonl"))
        if n < MAX_U:
            label = f"{label}（至 u{n}）"
        xs = np.arange(1, MAX_U + 1)
        for ax, (key, _, window, _) in zip(axes, panels):
            y = series(run, key, cache)
            smooth = rolling(y, window)
            smooth[np.isnan(y)] = np.nan  # no line past the last update with data
            ax.plot(xs, smooth, color=color, lw=lw, ls=ls, label=label)
    for ax, (_, title, _, ylim) in zip(axes, panels):
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("RL 更新轮次")
        ax.set_xlim(0, 51)
        ax.set_ylim(*ylim)
        ax.yaxis.grid(True, color=GRID, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="center left", bbox_to_anchor=(0.80, 0.5), fontsize=9,
               handlelength=2.6, labelspacing=0.9)
    fig.text(0.01, 0.01, "l2s_*：每轮的 comm/* 指标（训练时按真实调用顺序统计）；l2r_none 与 LLM 评委参考：每轮保存的 rollout"
             "（按文本统计）。成功率为训练 rollout 的成功率。", fontsize=7.5, color=INK2)
    fig.suptitle(f"L2 reward shaping：成功率与通信冗余（Qwen3-1.7B，只用 3 道训练菜；{time.strftime('%Y-%m-%d %H:%M')}）",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.03, 0.79, 0.96))
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "l2s":
        main_l2s(Path(sys.argv[2]) if len(sys.argv) > 2 else OUT_DIR / "success_repeat_l2s.png")
    else:
        main()
