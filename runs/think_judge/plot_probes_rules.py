"""Probe charts for many checkpoints at once (the L2 rule-reward arms): per role and mode, a grouped bar of the
F/S/P view averages (Wilson 95% intervals, values labeled) above a heatmap of per-question-type circular accuracy
minus a baseline checkpoint (pp, accuracy in parentheses). Delta mode: think - direct on the same probes.
Scores are read from the combined probe_eval.py outputs and split by role here.
usage: plot_probes_rules.py MODE ROLE DIRECT_JSONL THINK_JSONL OUT_PNG TITLE BASELINE NAME=LABEL[=ref] ...
  MODE: direct | think | delta;  ROLE: chef | assistant;  "=ref" draws the series in gray (reference)
  checkpoints missing from the scores are skipped (and listed on stdout)."""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plot_probes import GRID, INK, INK2, NAMES, QTYPES, SURF, wilson  # also registers the CJK font

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]  # validated slots 1-7
GRAYS = ["#c3c2bd", "#a3a29c", "#6f6e69"]  # references, light -> dark
CMAP = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#e34948", "#f0efec", "#2a78d6", "#184f95"])


def load(path, role, ids=None):
    hits = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r["role"] != role or (ids is not None and r["id"] not in ids):
            continue
        for key in (r["qtype"], r["view"]):
            hits[(r["checkpoint"], key)].append(r["correct"])
    return hits


def main():
    mode, role, direct_path, think_path, out, title, baseline = sys.argv[1:8]
    specs = []
    for s in sys.argv[8:]:
        parts = s.split("=")
        specs.append((parts[0], parts[1], len(parts) > 2 and parts[2] == "ref"))
    think = load(think_path, role)
    if mode == "delta":
        ids = {json.loads(l)["id"] for l in open(think_path) if json.loads(l)["role"] == role}
        direct = load(direct_path, role, ids)
        have = {n for n, _ in direct} & {n for n, _ in think}
    else:
        direct = load(direct_path, role)
        hits = direct if mode == "direct" else think
        have = {n for n, _ in hits}
    colors, ci, gi = [], 0, 0  # fixed by position in the full list, so a missing checkpoint does not repaint others
    for _, _, ref in specs:
        if ref:
            colors.append(GRAYS[gi % 3]); gi += 1
        else:
            colors.append(COLORS[ci]); ci += 1
    missing = [n for n, _, _ in specs if n not in have]
    colors = [c for c, s in zip(colors, specs) if s[0] in have]
    specs = [s for s in specs if s[0] in have]
    if missing:
        print("missing:", " ".join(missing))
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": "#d6d5d0", "figure.facecolor": SURF,
                         "axes.facecolor": SURF})
    src = think if mode != "direct" else direct
    qtypes = [q for q in QTYPES if any(src[(n, q)] for n, _, _ in specs)]
    cols = qtypes + ["F", "S", "P"]
    n = len(specs)
    fig = plt.figure(figsize=(19, 4.2 + 0.42 * n))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.3, 0.42 * n], width_ratios=[1.7, 3.5], hspace=0.3, wspace=0.12)
    axb = fig.add_subplot(gs[0, 0])
    axl = fig.add_subplot(gs[0, 1])
    axh = fig.add_subplot(gs[1, :])
    # top-left: F/S/P bars (accuracy, or think - direct in delta mode)
    w = 0.84 / n
    for k, (name, label, _) in enumerate(specs):
        if mode == "delta":
            vals = [(np.mean(think[(name, v)]) - np.mean(direct[(name, v)])) * 100 for v in "FSP"]
            bars = axb.bar(np.arange(3) + (k - (n - 1) / 2) * w, vals, w * 0.86, color=colors[k], label=label, zorder=2)
            for b, v in zip(bars, vals):
                axb.text(b.get_x() + b.get_width() / 2, v + (0.6 if v >= 0 else -0.6), f"{v:+.0f}", ha="center",
                         va="bottom" if v >= 0 else "top", fontsize=6.5, color=INK2, rotation=90)
        else:
            stats = [wilson(sum(src[(name, v)]), len(src[(name, v)])) for v in "FSP"]
            vals = [s[0] for s in stats]
            bars = axb.bar(np.arange(3) + (k - (n - 1) / 2) * w, vals, w * 0.86, color=colors[k], label=label,
                           zorder=2, yerr=np.array([[s[1] for s in stats], [s[2] for s in stats]]),
                           error_kw={"ecolor": INK2, "elinewidth": 0.7, "capsize": 0})
            for b, v, s in zip(bars, vals, stats):
                axb.text(b.get_x() + b.get_width() / 2, v + s[2] + 0.015, f"{v:.2f}", ha="center", va="bottom",
                         fontsize=6.5, color=INK2, rotation=90)
    axb.set_xticks(range(3), ["第一视角 F", "第二视角 S", "多步规划 P"])
    if mode == "delta":
        axb.axhline(0, color=INK2, lw=0.8)
        lo, hi = axb.get_ylim()
        axb.set_ylim(lo - 4, hi + 6)
        axb.set_ylabel("思考后 − 直接（百分点）")
    else:
        axb.set_ylim(0, 0.85)
        axb.set_ylabel("准确率（轮换每遍都对）")
    axb.yaxis.grid(True, color=GRID, zorder=0)
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)
    axb.set_title("按视角汇总", fontsize=9.5, color=INK2, loc="left")
    axl.axis("off")
    h, l = axb.get_legend_handles_labels()
    axl.legend(h, l, ncol=2, frameon=False, loc="center left", fontsize=9, title="checkpoint（彩色 = 规则奖励各组，灰 = 参考）",
               title_fontsize=9, alignment="left")
    # bottom: heatmap rows = checkpoints, cols = question types
    if mode == "delta":
        mat = np.array([[(np.mean(think[(nm, q)]) - np.mean(direct[(nm, q)])) * 100 for q in cols]
                        for nm, _, _ in specs])
        absval = np.array([[np.mean(think[(nm, q)]) for q in cols] for nm, _, _ in specs])
        vmax, cbl = 40, "思考后 − 直接作答（百分点）"
    else:
        absval = np.array([[np.mean(src[(nm, q)]) for q in cols] for nm, _, _ in specs])
        base = np.array([np.mean(src[(baseline, q)]) for q in cols])
        mat = (absval - base) * 100
        vmax, cbl = 25, f"相对 {dict((a, b) for a, b, _ in specs)[baseline]}（百分点）"
    im = axh.imshow(mat, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(n):
        for j in range(len(cols)):
            v = mat[i, j]
            axh.text(j, i, f"{v:+.0f} ({absval[i, j]:.2f})", ha="center", va="center", fontsize=7,
                     color="white" if abs(v) > vmax * 0.65 else INK)
    axh.set_xticks(range(len(cols)), [NAMES[q].replace("\n", "") if len(NAMES[q]) < 9 else NAMES[q] for q in cols],
                   fontsize=7.5)
    axh.set_yticks(range(n), [lab for _, lab, _ in specs], fontsize=8.5)
    for i, (_, _, ref) in enumerate(specs):
        axh.get_yticklabels()[i].set_color(INK2 if ref else INK)
    nf, ns = sum(q.startswith("F") for q in qtypes), sum(q.startswith("S") for q in qtypes)
    for x in (nf - 0.5, nf + ns - 0.5, len(qtypes) - 0.5):
        axh.axvline(x, color=SURF, lw=3)
    axh.tick_params(length=0)
    for s in axh.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.015, pad=0.005)
    cb.set_label(cbl, color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    for name, label, _ in specs:
        print(f"{label:28s}", " ".join(f"{q}:{mat[[s[0] for s in specs].index(name), j]:+.0f}/{absval[[s[0] for s in specs].index(name), j]:.2f}"
                                       for j, q in enumerate(cols)))


if __name__ == "__main__":
    main()
