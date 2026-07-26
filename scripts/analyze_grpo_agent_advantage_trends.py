#!/usr/bin/env python3
"""Summarize Chef/Assistant GRPO advantage trends across rollout epochs."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ADVANTAGES_PATH = REPO_ROOT / "Grpo" / "advantages.py"
ADVANTAGES_SPEC = importlib.util.spec_from_file_location(
    "_collab_grpo_advantages",
    ADVANTAGES_PATH,
)
if ADVANTAGES_SPEC is None or ADVANTAGES_SPEC.loader is None:
    raise ImportError(f"Cannot load GRPO advantage module from {ADVANTAGES_PATH}")
ADVANTAGES_MODULE = importlib.util.module_from_spec(ADVANTAGES_SPEC)
sys.modules[ADVANTAGES_SPEC.name] = ADVANTAGES_MODULE
ADVANTAGES_SPEC.loader.exec_module(ADVANTAGES_MODULE)
GrpoAdvantageConfig = ADVANTAGES_MODULE.GrpoAdvantageConfig
compute_grpo_advantages = ADVANTAGES_MODULE.compute_grpo_advantages

ROLLOUT_RE = re.compile(r"rollout_rank(?P<rank>\d+)_u(?P<update>\d+)\.pt$")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_rollout_file(path: Path) -> Optional[Tuple[int, int]]:
    match = ROLLOUT_RE.match(path.name)
    if not match:
        return None
    return int(match.group("rank")), int(match.group("update"))


def _find_rollout_dir(exp_root: Path, rollout_dir: Optional[str]) -> Path:
    if rollout_dir:
        path = Path(rollout_dir).expanduser()
        return path if path.is_absolute() else exp_root / path
    candidate = exp_root / "rollouts_grpo"
    if candidate.exists():
        return candidate
    return exp_root


def _group_files_by_update(rollout_dir: Path) -> Dict[int, List[Path]]:
    grouped: Dict[int, List[Path]] = {}
    for path in rollout_dir.glob("rollout_rank*_u*.pt"):
        parsed = _parse_rollout_file(path)
        if parsed is None:
            continue
        _, update = parsed
        grouped.setdefault(update, []).append(path)
    for files in grouped.values():
        files.sort()
    return dict(sorted(grouped.items()))


def _load_rows(files: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in files:
        parsed = _parse_rollout_file(path)
        if parsed is None:
            continue
        rank, update = parsed
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, list):
            continue
        for local_idx, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["_rank"] = rank
            item["_update"] = update
            item["_local_idx"] = local_idx
            item["_file"] = str(path)
            rows.append(item)
    return rows


def _reward_for_adv(row: Dict[str, Any], reward_field: str) -> float:
    if reward_field == "reward":
        return _as_float(row.get("reward"))
    if reward_field == "breakdown_total_reward":
        return _as_float(row.get("breakdown_total_reward"), _as_float(row.get("reward")))
    if reward_field == "components":
        return (
            _as_float(row.get("format_reward"))
            + _as_float(row.get("validator_reward"))
            + _as_float(row.get("sequence_reward"))
            + _as_float(row.get("communication_reward"))
            + _as_float(row.get("repeat_communication_reward"))
            + _as_float(row.get("forced_communication_reward"))
            + _as_float(row.get("paired_comm_reward"))
        )
    raise ValueError(f"Unsupported reward_field={reward_field!r}")


def _safe_stat(values: np.ndarray, fn: str) -> float:
    if values.size == 0:
        return 0.0
    if fn == "mean":
        return float(values.mean())
    if fn == "std":
        return float(values.std())
    if fn == "min":
        return float(values.min())
    if fn == "max":
        return float(values.max())
    if fn == "median":
        return float(np.median(values))
    if fn == "p25":
        return float(np.percentile(values, 25))
    if fn == "p75":
        return float(np.percentile(values, 75))
    raise ValueError(fn)


def _agent_stats(
    *,
    update: int,
    files: Sequence[Path],
    rows: Sequence[Dict[str, Any]],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    agent_idx: int,
    agent_name: str,
) -> Dict[str, Any]:
    indices = [idx for idx, row in enumerate(rows) if _as_int(row.get("agent_index"), -1) == agent_idx]
    adv = np.array([float(advantages[idx].item()) for idx in indices], dtype=float)
    ret = np.array([float(returns[idx].item()) for idx in indices], dtype=float)
    reward = np.array([_as_float(rows[idx].get("reward")) for idx in indices], dtype=float)
    process = np.array([_as_float(rows[idx].get("process_reward")) for idx in indices], dtype=float)
    sequence = np.array([_as_float(rows[idx].get("sequence_reward")) for idx in indices], dtype=float)
    paired = np.array([_as_float(rows[idx].get("paired_comm_reward")) for idx in indices], dtype=float)
    fmt = np.array([_as_float(rows[idx].get("format_reward")) for idx in indices], dtype=float)
    validator = np.array([_as_float(rows[idx].get("validator_reward")) for idx in indices], dtype=float)
    return {
        "update": update,
        "agent": agent_name,
        "agent_index": agent_idx,
        "rollout_files": len(files),
        "transitions": len(indices),
        "adv_mean": _safe_stat(adv, "mean"),
        "adv_std": _safe_stat(adv, "std"),
        "adv_min": _safe_stat(adv, "min"),
        "adv_p25": _safe_stat(adv, "p25"),
        "adv_median": _safe_stat(adv, "median"),
        "adv_p75": _safe_stat(adv, "p75"),
        "adv_max": _safe_stat(adv, "max"),
        "adv_positive_ratio": float((adv > 0).mean()) if adv.size else 0.0,
        "return_mean": _safe_stat(ret, "mean"),
        "return_std": _safe_stat(ret, "std"),
        "return_max": _safe_stat(ret, "max"),
        "reward_mean": _safe_stat(reward, "mean"),
        "reward_sum": float(reward.sum()) if reward.size else 0.0,
        "process_sum": float(process.sum()) if process.size else 0.0,
        "process_max": _safe_stat(process, "max"),
        "process_positive_count": int((process > 0).sum()) if process.size else 0,
        "sequence_sum": float(sequence.sum()) if sequence.size else 0.0,
        "sequence_positive_count": int((sequence > 0).sum()) if sequence.size else 0,
        "paired_comm_sum": float(paired.sum()) if paired.size else 0.0,
        "format_sum": float(fmt.sum()) if fmt.size else 0.0,
        "validator_sum": float(validator.sum()) if validator.size else 0.0,
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows: Sequence[Dict[str, Any]], metric: str, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for agent in ["Chef", "Assistant"]:
        agent_rows = [row for row in rows if row["agent"] == agent]
        xs = [int(row["update"]) for row in agent_rows]
        ys = [float(row[metric]) for row in agent_rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=4, label=agent)
    ax.set_xlabel("Epoch / update")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_adv_mean_with_std(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for agent in ["Chef", "Assistant"]:
        agent_rows = [row for row in rows if row["agent"] == agent]
        xs = np.array([int(row["update"]) for row in agent_rows], dtype=float)
        means = np.array([float(row["adv_mean"]) for row in agent_rows], dtype=float)
        stds = np.array([float(row["adv_std"]) for row in agent_rows], dtype=float)
        ax.plot(xs, means, marker="o", linewidth=1.8, markersize=4, label=f"{agent} mean")
        ax.fill_between(xs, means - stds, means + stds, alpha=0.12)
    ax.axhline(0, color="black", linewidth=1, alpha=0.4)
    ax.set_xlabel("Epoch / update")
    ax.set_ylabel("advantage mean +- std")
    ax.set_title("Chef vs Assistant GRPO advantage trend")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--rollout-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--start-update", type=int, default=None)
    parser.add_argument("--end-update", type=int, default=None)
    parser.add_argument("--chef-agent-idx", type=int, default=0)
    parser.add_argument("--assistant-agent-idx", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--norm-scope", default="agent")
    parser.add_argument("--normalize", default="mean_std")
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument(
        "--reward-field",
        choices=["reward", "breakdown_total_reward", "components"],
        default="reward",
    )
    args = parser.parse_args()

    exp_root = Path(args.exp_root).expanduser().resolve()
    rollout_dir = _find_rollout_dir(exp_root, args.rollout_dir)
    grouped = _group_files_by_update(rollout_dir)
    if args.start_update is not None:
        grouped = {k: v for k, v in grouped.items() if k >= args.start_update}
    if args.end_update is not None:
        grouped = {k: v for k, v in grouped.items() if k <= args.end_update}
    if not grouped:
        raise FileNotFoundError(f"No rollout files found under {rollout_dir}")

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else exp_root / "analysis" / "agent_advantage_trends"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for update, files in grouped.items():
        rows = _load_rows(files)
        if not rows:
            continue
        rewards = [_reward_for_adv(row, args.reward_field) for row in rows]
        agent_indices = [_as_int(row.get("agent_index"), 0) for row in rows]
        timesteps = [
            _as_int(row.get("timestep"), -1) if row.get("timestep") is not None else None
            for row in rows
        ]
        advantages, returns, _ = compute_grpo_advantages(
            rewards=rewards,
            agent_indices=agent_indices,
            timesteps=timesteps,
            trajectory_ids=[
                row.get("rollout_id")
                or row.get("_rollout_id")
                or f"rank{row.get('_rank', row.get('rank', 0))}_u{update:05d}"
                for row in rows
            ],
            config=GrpoAdvantageConfig(
                norm_scope=args.norm_scope,
                normalize=args.normalize,
                gamma=args.gamma,
                clip=args.clip,
            ),
        )
        summary_rows.append(
            _agent_stats(
                update=update,
                files=files,
                rows=rows,
                advantages=advantages,
                returns=returns,
                agent_idx=args.chef_agent_idx,
                agent_name="Chef",
            )
        )
        summary_rows.append(
            _agent_stats(
                update=update,
                files=files,
                rows=rows,
                advantages=advantages,
                returns=returns,
                agent_idx=args.assistant_agent_idx,
                agent_name="Assistant",
            )
        )

    summary_rows.sort(key=lambda row: (int(row["update"]), int(row["agent_index"])))
    csv_path = out_dir / "agent_advantage_trends.csv"
    _write_csv(csv_path, summary_rows)
    _plot_adv_mean_with_std(summary_rows, out_dir / "agent_advantage_mean_std_trend.png")
    _plot_metric(
        summary_rows,
        "adv_positive_ratio",
        out_dir / "agent_advantage_positive_ratio_trend.png",
        "Positive-advantage transition ratio",
    )
    _plot_metric(
        summary_rows,
        "return_mean",
        out_dir / "agent_return_mean_trend.png",
        "Mean reward-to-go return",
    )
    _plot_metric(
        summary_rows,
        "process_sum",
        out_dir / "agent_process_sum_trend.png",
        "Total process reward by agent",
    )

    print(f"updates={len(grouped)} rows={len(summary_rows)}")
    print(f"out_dir={out_dir}")
    print(f"csv={csv_path}")
    if summary_rows:
        latest_update = max(int(row["update"]) for row in summary_rows)
        latest = [row for row in summary_rows if int(row["update"]) == latest_update]
        for row in latest:
            print(
                f"u{latest_update:05d} {row['agent']}: "
                f"adv_mean={float(row['adv_mean']):.4f} "
                f"adv_std={float(row['adv_std']):.4f} "
                f"pos_ratio={float(row['adv_positive_ratio']):.4f} "
                f"return_mean={float(row['return_mean']):.4f} "
                f"process_sum={float(row['process_sum']):.4f}"
            )


if __name__ == "__main__":
    main()
