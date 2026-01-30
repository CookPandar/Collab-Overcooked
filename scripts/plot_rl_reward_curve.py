#!/usr/bin/env python3
"""
Plot RL per-agent reward curves from `reward_curve.csv`.

The trainer writes (see `collab_overcooked/training/mappo_qwen.py`):
  - New schema (recommended): `*_sum` + `*_mean` columns for each reward component.
  - Legacy schema: `agent0_format,agent0_validator,agent0_process,...`

Notes:
- Some runs append to the same CSV and `update_idx` may restart from 1 multiple times.
  This script can either plot only the latest segment (default) or all segments.

Examples:
  python scripts/plot_rl_reward_curve.py \
    --input results/mappo_boiled_egg/reward_curve.csv \
    --output results/mappo_boiled_egg/reward_curve.png

  # Plot all segments (if the CSV contains multiple appended runs)
  python scripts/plot_rl_reward_curve.py \
    --input results/mappo_boiled_egg/reward_curve.csv \
    --output results/mappo_boiled_egg/reward_curve_all.png \
    --all-segments

  # Plot cumulative reward (running sum within each segment)
  python scripts/plot_rl_reward_curve.py \
    --input results/mappo_boiled_egg/reward_curve.csv \
    --output results/mappo_boiled_egg/reward_curve_cumsum.png \
    --metric sum \
    --cumulative

  # Plot per-episode return when each update is exactly one episode
  python scripts/plot_rl_reward_curve.py \
    --input results/mappo_boiled_egg/reward_curve.csv \
    --output results/mappo_boiled_egg/episode_return.png \
    --episode
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


@dataclass
class Row:
    segment: int
    update_idx: int
    step: int
    num_transitions: int
    agent0: Dict[str, float] = field(default_factory=dict)
    agent1: Dict[str, float] = field(default_factory=dict)


DEFAULT_LINES = "format,validator,sequence,comm,total"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot RL reward_curve.csv (per-agent).")
    p.add_argument("--input", type=Path, required=True, help="Path to reward_curve.csv")
    p.add_argument("--output", type=Path, required=True, help="Output PNG path")
    p.add_argument(
        "--x",
        choices=["step", "update_idx", "row"],
        default="row",
        help="X axis to use (default: row, i.e., a new monotonic 1..N index).",
    )
    p.add_argument(
        "--all-segments",
        action="store_true",
        help="Plot all segments (when update_idx restarts multiple times).",
    )
    p.add_argument(
        "--metric",
        choices=["mean", "sum"],
        default="mean",
        help="Plot per-transition means or sums (default: mean).",
    )
    p.add_argument(
        "--episode",
        action="store_true",
        help=(
            "Treat each row as one episode (use sum metrics and label as episode return). "
            "This is only valid when each update contains exactly one episode."
        ),
    )
    p.add_argument(
        "--cumulative",
        action="store_true",
        help="Plot cumulative (running-sum) reward within each segment.",
    )
    p.add_argument(
        "--lines",
        default="format,validator,sequence,comm,total",
        help=(
            "Comma-separated reward components to plot. Supported (best-effort): "
            "format,validator,sequence,comm,total,legacy_process,rl "
            "(default: format,validator,sequence,comm,total)."
        ),
    )
    p.add_argument(
        "--no-total",
        action="store_true",
        help="Do not plot the derived total (=format+validator+process).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Downsample if points exceed this threshold (default: 5000).",
    )
    return p.parse_args()


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def load_rows(path: Path) -> List[Row]:
    if not path.exists():
        raise FileNotFoundError(str(path))

    rows: List[Row] = []
    segment = 0
    prev_update: Optional[int] = None

    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return []
        header = [h.strip() for h in first.strip().split(",")]
        f.seek(0)

        is_v2 = "agent0_format_sum" in header
        if is_v2:
            reader = csv.DictReader(f)
            for rec in reader:
                if not rec:
                    continue
                update_idx = _safe_int(rec.get("update_idx", ""))
                step = _safe_int(rec.get("step", ""))
                num_transitions = _safe_int(rec.get("num_transitions", ""))
                if update_idx is None or step is None or num_transitions is None:
                    continue
                if prev_update is not None and update_idx < prev_update:
                    segment += 1
                prev_update = update_idx

                def gf(key: str, default: float = 0.0) -> float:
                    val = _safe_float(rec.get(key, ""))
                    return float(val) if val is not None else float(default)

                # Store both sum + mean variants; plotting will pick based on args.metric.
                rows.append(
                    Row(
                        segment=segment,
                        update_idx=update_idx,
                        step=step,
                        num_transitions=num_transitions,
                        agent0={
                            "rl_sum": gf("agent0_rl_sum"),
                            "rl_mean": gf("agent0_rl_mean"),
                            "format_sum": gf("agent0_format_sum"),
                            "format_mean": gf("agent0_format_mean"),
                            "validator_sum": gf("agent0_validator_sum"),
                            "validator_mean": gf("agent0_validator_mean"),
                            "sequence_sum": gf("agent0_sequence_sum"),
                            "sequence_mean": gf("agent0_sequence_mean"),
                            "comm_sum": gf("agent0_comm_sum"),
                            "comm_mean": gf("agent0_comm_mean"),
                            "total_sum": gf("agent0_breakdown_total_sum"),
                            "total_mean": gf("agent0_breakdown_total_mean"),
                            "legacy_process_sum": gf("agent0_legacy_process_sum"),
                            "legacy_process_mean": gf("agent0_legacy_process_mean"),
                        },
                        agent1={
                            "rl_sum": gf("agent1_rl_sum"),
                            "rl_mean": gf("agent1_rl_mean"),
                            "format_sum": gf("agent1_format_sum"),
                            "format_mean": gf("agent1_format_mean"),
                            "validator_sum": gf("agent1_validator_sum"),
                            "validator_mean": gf("agent1_validator_mean"),
                            "sequence_sum": gf("agent1_sequence_sum"),
                            "sequence_mean": gf("agent1_sequence_mean"),
                            "comm_sum": gf("agent1_comm_sum"),
                            "comm_mean": gf("agent1_comm_mean"),
                            "total_sum": gf("agent1_breakdown_total_sum"),
                            "total_mean": gf("agent1_breakdown_total_mean"),
                            "legacy_process_sum": gf("agent1_legacy_process_sum"),
                            "legacy_process_mean": gf("agent1_legacy_process_mean"),
                        },
                    )
                )
        else:
            # Legacy schema: 3 components per agent; no means.
            reader = csv.reader(f)
            for raw in reader:
                if not raw:
                    continue
                if raw[0].strip() == "update_idx":
                    continue
                if len(raw) < 9:
                    continue

                update_idx = _safe_int(raw[0])
                step = _safe_int(raw[1])
                num_transitions = _safe_int(raw[2])
                vals = [_safe_float(x) for x in raw[3:9]]
                if update_idx is None or step is None or num_transitions is None:
                    continue
                if any(v is None for v in vals):
                    continue

                if prev_update is not None and update_idx < prev_update:
                    segment += 1
                prev_update = update_idx

                a0_format = float(vals[0])
                a0_validator = float(vals[1])
                a0_process = float(vals[2])
                a1_format = float(vals[3])
                a1_validator = float(vals[4])
                a1_process = float(vals[5])
                rows.append(
                    Row(
                        segment=segment,
                        update_idx=update_idx,
                        step=step,
                        num_transitions=num_transitions,
                        agent0={
                            "format_sum": a0_format,
                            "validator_sum": a0_validator,
                            "legacy_process_sum": a0_process,
                            "total_sum": a0_format + a0_validator + a0_process,
                        },
                        agent1={
                            "format_sum": a1_format,
                            "validator_sum": a1_validator,
                            "legacy_process_sum": a1_process,
                            "total_sum": a1_format + a1_validator + a1_process,
                        },
                    )
                )

    return rows


def downsample(rows: List[Row], max_points: int) -> List[Row]:
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    stride = max(1, len(rows) // max_points)
    return rows[::stride]


def pick_segment(rows: List[Row], all_segments: bool) -> List[Row]:
    if all_segments:
        return rows
    if not rows:
        return rows
    last = max(r.segment for r in rows)
    return [r for r in rows if r.segment == last]


def get_x(rows: List[Row], mode: str) -> List[float]:
    if mode == "update_idx":
        return [float(r.update_idx) for r in rows]
    return [float(r.step) for r in rows]

def _parse_lines(spec: str) -> List[str]:
    out: List[str] = []
    for part in (spec or "").split(","):
        key = part.strip()
        if key:
            out.append(key)
    return out

def _series_key(component: str, metric: str) -> str:
    # Map logical component to stored field names.
    if component == "process":
        component = "legacy_process"
    if metric == "mean":
        return f"{component}_mean"
    return f"{component}_sum"

def _select_series(agent: Dict[str, float], components: Sequence[str], metric: str, include_total: bool) -> Dict[str, List[float]]:
    # This function is applied per-segment later; here we only validate keys.
    del agent
    del components
    del metric
    del include_total
    return {}

def _cumsum(values: Sequence[float]) -> List[float]:
    total = 0.0
    out: List[float] = []
    for v in values:
        total += float(v)
        out.append(total)
    return out


def plot(
    rows: List[Row],
    output: Path,
    x_mode: str,
    include_total: bool,
    metric: str,
    lines: List[str],
    cumulative: bool,
    episode_mode: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib 未安装：请在环境中安装 matplotlib 后再运行该脚本。") from exc

    output.parent.mkdir(parents=True, exist_ok=True)

    # Group by segment for plotting (so appended runs don't connect weirdly).
    by_seg: Dict[int, List[Row]] = {}
    for r in rows:
        by_seg.setdefault(r.segment, []).append(r)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # For x=row, we must not reset the x-axis per segment. Use a global 1..N axis
    # derived from the plotted row order.
    if x_mode == "row":
        pos = {id(r): float(i + 1) for i, r in enumerate(rows)}

        def x_for(seg_rows: List[Row]) -> List[float]:
            return [pos[id(r)] for r in seg_rows]

    else:

        def x_for(seg_rows: List[Row]) -> List[float]:
            return get_x(seg_rows, x_mode)

    # Plot each segment separately; legend only on the first segment.
    first = True
    for seg, seg_rows in sorted(by_seg.items()):
        alpha = 0.35 if len(by_seg) > 1 else 1.0
        x0 = x_for(seg_rows)

        for ax, which in zip(axes, ["agent0", "agent1"]):
            agent_series: Dict[str, List[float]] = {}
            for comp in lines:
                if comp == "total" and not include_total:
                    continue
                k = _series_key(comp, metric)
                if which == "agent0":
                    values = [r.agent0.get(k) for r in seg_rows]
                else:
                    values = [r.agent1.get(k) for r in seg_rows]
                if any(v is not None for v in values):
                    # Replace missing with 0.0 so plotting doesn't crash.
                    agent_series[comp] = [float(v) if v is not None else 0.0 for v in values]
            ys = agent_series

            for label, series in ys.items():
                if cumulative:
                    series = _cumsum(series)
                lw = 2.0 if label == "total" else 1.2
                a = min(1.0, alpha + (0.15 if label == "total" else 0.0))
                ax.plot(x0, series, label=label if first else None, linewidth=lw, alpha=a)

        first = False

    axes[0].set_title("agent0")
    axes[1].set_title("agent1")
    axes[1].set_xlabel(x_mode)
    axes[0].legend(loc="upper right", frameon=False)
    if episode_mode:
        suffix = "episode return (cumulative)" if cumulative else "episode return"
    else:
        suffix = "cumulative" if cumulative else metric
    axes[0].set_ylabel(f"reward ({suffix})")
    axes[1].set_ylabel(f"reward ({suffix})")

    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"未在 {args.input} 中解析到有效数据行。")
    rows = pick_segment(rows, all_segments=args.all_segments)
    rows = downsample(rows, max_points=args.max_points)
    lines_spec = args.lines
    x_mode = args.x
    metric = args.metric
    if args.episode:
        metric = "sum"
        if lines_spec == DEFAULT_LINES:
            lines_spec = "rl"
    lines = _parse_lines(lines_spec)
    # If legacy schema (no *_mean), forcing mean would produce empty lines; fall back to sum.
    if metric == "mean":
        has_any_mean = any(
            (("format_mean" in r.agent0) and (r.agent0.get("format_mean") not in (None, 0.0)))
            or (("format_mean" in r.agent1) and (r.agent1.get("format_mean") not in (None, 0.0)))
            for r in rows
        )
        metric = "mean" if has_any_mean else "sum"
    else:
        metric = "sum"
    plot(
        rows,
        args.output,
        x_mode=x_mode,
        include_total=(not args.no_total),
        metric=metric,
        lines=lines,
        cumulative=args.cumulative,
        episode_mode=args.episode,
    )
    print(f"[plot] Saved: {args.output}")


if __name__ == "__main__":
    main()
