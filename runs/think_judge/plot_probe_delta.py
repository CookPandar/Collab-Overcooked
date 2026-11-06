"""Heatmap of (think-before-answer - direct) circular accuracy on the same probes, per checkpoint x question type.
usage: plot_probe_delta.py DIRECT_JSONL THINK_JSONL OUT_PNG TITLE NAME=LABEL ..."""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from plot_probes import INK, INK2, NAMES, QTYPES, SURF  # also registers the CJK font

direct_path, think_path, out, title = sys.argv[1:5]
specs = [s.split("=", 1) for s in sys.argv[5:]]
acc = {}
for tag, path in (("direct", direct_path), ("think", think_path)):
    hits = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        for key in (r["qtype"], r["view"]):
            hits[(r["checkpoint"], key)].append(r["correct"])
    acc[tag] = {k: np.mean(v) for k, v in hits.items()}
cols = [q for q in QTYPES if (specs[0][0], q) in acc["think"]] + ["F", "S", "P"]
delta = np.array([[acc["think"][(n, q)] - acc["direct"][(n, q)] for q in cols] for n, _ in specs]) * 100
cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#e34948", "#f0efec", "#2a78d6", "#184f95"])
fig, ax = plt.subplots(figsize=(17, 3.9))
im = ax.imshow(delta, cmap=cmap, vmin=-40, vmax=40, aspect="auto")
for i in range(len(specs)):
    for j, q in enumerate(cols):
        v = delta[i, j]
        ax.text(j, i, f"{v:+.0f}\n({acc['think'][(specs[i][0], q)]:.2f})", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) > 25 else INK)
ax.set_xticks(range(len(cols)), [NAMES[q] for q in cols], fontsize=8)
ax.set_yticks(range(len(specs)), [label for _, label in specs])
for x in (5.5, 12.5, 16.5):
    ax.axvline(x, color=SURF, lw=3)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(title, loc="left", fontsize=10.5)
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
cb.set_label("思考后 − 直接作答（百分点）", color=INK2)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(out, dpi=160)
