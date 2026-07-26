#!/usr/bin/env python3
"""Analyze GRPO advantages against process-reward progress per rollout.

This script recomputes the MARSHAL-style GRPO advantages for one rollout epoch
using the same reward-to-go and agent-specific unique-value normalization used
by the trainer. It then aggregates each rollout into a compact table and plots
heatmaps showing whether high-process-reward Chef trajectories also had higher
early-turn advantages.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ADVANTAGES_PATH = REPO_ROOT / "Grpo" / "advantages.py"
_ADVANTAGES_SPEC = importlib.util.spec_from_file_location(
    "_collab_grpo_advantages",
    _ADVANTAGES_PATH,
)
if _ADVANTAGES_SPEC is None or _ADVANTAGES_SPEC.loader is None:
    raise ImportError(f"Cannot load GRPO advantage module from {_ADVANTAGES_PATH}")
_ADVANTAGES_MODULE = importlib.util.module_from_spec(_ADVANTAGES_SPEC)
sys.modules[_ADVANTAGES_SPEC.name] = _ADVANTAGES_MODULE
_ADVANTAGES_SPEC.loader.exec_module(_ADVANTAGES_MODULE)
GrpoAdvantageConfig = _ADVANTAGES_MODULE.GrpoAdvantageConfig
compute_grpo_advantages = _ADVANTAGES_MODULE.compute_grpo_advantages


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


def _find_rollout_dir(exp_root: Path, rollout_dir: Optional[str]) -> Path:
    if rollout_dir:
        path = Path(rollout_dir).expanduser()
        return path if path.is_absolute() else exp_root / path
    candidate = exp_root / "rollouts_grpo"
    if candidate.exists():
        return candidate
    return exp_root


def _parse_rollout_file(path: Path) -> Optional[Tuple[int, int]]:
    match = ROLLOUT_RE.match(path.name)
    if not match:
        return None
    return int(match.group("rank")), int(match.group("update"))


def _available_updates(rollout_dir: Path) -> List[int]:
    updates = set()
    for path in rollout_dir.glob("rollout_rank*_u*.pt"):
        parsed = _parse_rollout_file(path)
        if parsed:
            updates.add(parsed[1])
    return sorted(updates)


def _select_rollout_files(rollout_dir: Path, update: Optional[int]) -> Tuple[int, List[Path]]:
    updates = _available_updates(rollout_dir)
    if not updates:
        raise FileNotFoundError(f"No rollout_rank*_u*.pt files found under {rollout_dir}")
    use_update = int(update) if update is not None else updates[-1]
    selected = []
    for path in rollout_dir.glob(f"rollout_rank*_u{use_update:05d}.pt"):
        if _parse_rollout_file(path):
            selected.append(path)
    if not selected:
        raise FileNotFoundError(f"No rollout files found for update {use_update:05d}")
    return use_update, sorted(selected)


def _load_rows(files: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    file_meta: List[Dict[str, Any]] = []
    for path in files:
        parsed = _parse_rollout_file(path)
        if not parsed:
            continue
        rank, update = parsed
        data = torch.load(path, map_location="cpu")
        if not isinstance(data, list):
            continue
        start = len(all_rows)
        for local_idx, row in enumerate(data):
            if not isinstance(row, dict):
                continue
            item = dict(row)
            item["_file"] = str(path)
            item["_rank"] = rank
            item["_update"] = update
            item["_rollout_id"] = item.get("rollout_id") or f"rank{rank}_u{update:05d}"
            item["rollout_id"] = item["_rollout_id"]
            item["_local_idx"] = local_idx
            item["_global_idx"] = len(all_rows)
            all_rows.append(item)
        file_meta.append(
            {
                "file": str(path),
                "rank": rank,
                "update": update,
                "start": start,
                "end": len(all_rows),
                "count": len(all_rows) - start,
            }
        )
    return all_rows, file_meta


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


def _prefix_indices(
    rows: Sequence[Dict[str, Any]],
    *,
    agent_idx: int,
    first_positive_idx: Optional[int],
    max_prefix_timestep: Optional[int],
) -> List[int]:
    indices = []
    for idx, row in enumerate(rows):
        if _as_int(row.get("agent_index"), -1) != agent_idx:
            continue
        if first_positive_idx is not None and idx >= first_positive_idx:
            continue
        if max_prefix_timestep is not None:
            timestep = row.get("timestep")
            if timestep is not None and _as_int(timestep, 10**9) > max_prefix_timestep:
                continue
        indices.append(idx)
    return indices


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _summarize_rollout(
    meta: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    chef_agent_idx: int,
    assistant_agent_idx: int,
    max_prefix_timestep: Optional[int],
) -> Dict[str, Any]:
    rel_adv = advantages[int(meta["start"]) : int(meta["end"])]
    rel_ret = returns[int(meta["start"]) : int(meta["end"])]
    chef_indices = [
        idx for idx, row in enumerate(rows) if _as_int(row.get("agent_index"), -1) == chef_agent_idx
    ]
    assistant_indices = [
        idx
        for idx, row in enumerate(rows)
        if _as_int(row.get("agent_index"), -1) == assistant_agent_idx
    ]
    positive_chef_indices = [
        idx for idx in chef_indices if _as_float(rows[idx].get("sequence_reward")) > 0.0
    ]
    first_positive_idx = min(positive_chef_indices) if positive_chef_indices else None
    prefix_chef_indices = _prefix_indices(
        rows,
        agent_idx=chef_agent_idx,
        first_positive_idx=first_positive_idx,
        max_prefix_timestep=max_prefix_timestep,
    )
    prefix_all_indices = [
        idx
        for idx, row in enumerate(rows)
        if (first_positive_idx is None or idx < first_positive_idx)
        and (
            max_prefix_timestep is None
            or row.get("timestep") is None
            or _as_int(row.get("timestep"), 10**9) <= max_prefix_timestep
        )
    ]

    def adv_at(indices: Sequence[int]) -> List[float]:
        return [float(rel_adv[idx].item()) for idx in indices]

    def ret_at(indices: Sequence[int]) -> List[float]:
        return [float(rel_ret[idx].item()) for idx in indices]

    chef_sequence_sum = sum(_as_float(rows[idx].get("sequence_reward")) for idx in chef_indices)
    chef_process_sum = sum(_as_float(rows[idx].get("process_reward")) for idx in chef_indices)
    chef_process_max = (
        max(_as_float(rows[idx].get("process_reward")) for idx in chef_indices)
        if chef_indices
        else 0.0
    )
    chef_positive_count = sum(
        1 for idx in chef_indices if _as_float(rows[idx].get("sequence_reward")) > 0.0
    )
    assistant_sequence_sum = sum(
        _as_float(rows[idx].get("sequence_reward")) for idx in assistant_indices
    )
    team_sequence_sum = sum(_as_float(row.get("sequence_reward")) for row in rows)
    team_reward_sum = sum(_as_float(row.get("reward")) for row in rows)
    chef_prefix_adv = adv_at(prefix_chef_indices)
    all_prefix_adv = adv_at(prefix_all_indices)
    chef_all_adv = adv_at(chef_indices)
    chef_all_returns = ret_at(chef_indices)

    return {
        "update": int(meta["update"]),
        "rank": int(meta["rank"]),
        "file": meta["file"],
        "_start": int(meta["start"]),
        "_end": int(meta["end"]),
        "num_transitions": len(rows),
        "chef_n": len(chef_indices),
        "assistant_n": len(assistant_indices),
        "chef_sequence_sum": chef_sequence_sum,
        "chef_process_sum": chef_process_sum,
        "chef_process_max": chef_process_max,
        "chef_positive_action_count": chef_positive_count,
        "assistant_sequence_sum": assistant_sequence_sum,
        "team_sequence_sum": team_sequence_sum,
        "team_reward_sum": team_reward_sum,
        "first_chef_positive_local_idx": first_positive_idx if first_positive_idx is not None else "",
        "first_chef_positive_timestep": (
            rows[first_positive_idx].get("timestep") if first_positive_idx is not None else ""
        ),
        "chef_prefix_n": len(prefix_chef_indices),
        "chef_prefix_adv_mean": _mean(chef_prefix_adv),
        "chef_prefix_adv_max": max(chef_prefix_adv) if chef_prefix_adv else 0.0,
        "chef_prefix_adv_min": min(chef_prefix_adv) if chef_prefix_adv else 0.0,
        "prefix_all_adv_mean": _mean(all_prefix_adv),
        "prefix_adv_delta_vs_all": _mean(chef_prefix_adv) - _mean(all_prefix_adv),
        "chef_all_adv_mean": _mean(chef_all_adv),
        "chef_all_adv_max": max(chef_all_adv) if chef_all_adv else 0.0,
        "chef_all_return_mean": _mean(chef_all_returns),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sort_summary_rows(rows: List[Dict[str, Any]], sort_by: str) -> None:
    if sort_by == "rank":
        rows.sort(key=lambda r: (int(r["rank"]), str(r["file"])))
        return
    if sort_by == "team_reward_desc":
        rows.sort(key=lambda r: (-float(r["team_reward_sum"]), int(r["rank"])))
        return
    if sort_by == "chef_prefix_adv_desc":
        rows.sort(key=lambda r: (-float(r["chef_prefix_adv_mean"]), int(r["rank"])))
        return
    if sort_by == "chef_process_max_desc":
        rows.sort(
            key=lambda r: (
                -float(r["chef_process_max"]),
                -float(r["chef_process_sum"]),
                -float(r["chef_sequence_sum"]),
                int(r["rank"]),
            )
        )
        return
    if sort_by == "first_positive_timestep":
        rows.sort(
            key=lambda r: (
                10**9
                if r["first_chef_positive_timestep"] == ""
                else int(r["first_chef_positive_timestep"]),
                -float(r["chef_sequence_sum"]),
                int(r["rank"]),
            )
        )
        return
    if sort_by == "chef_sequence_desc":
        rows.sort(
            key=lambda r: (
                -float(r["chef_sequence_sum"]),
                -float(r["chef_positive_action_count"]),
                -float(r["team_reward_sum"]),
                int(r["rank"]),
            )
        )
        return
    raise ValueError(f"Unsupported sort_by={sort_by!r}")


def _row_label(row: Dict[str, Any]) -> str:
    first_ts = row["first_chef_positive_timestep"]
    first_ts_text = f"t{first_ts}" if first_ts != "" else "no+"
    return (
        f"#{int(row['plot_order']):02d} "
        f"r{int(row['rank'])} "
        f"pmax={float(row['chef_process_max']):.1f} "
        f"psum={float(row['chef_process_sum']):.1f} "
        f"seq={float(row['chef_sequence_sum']):.1f} "
        f"act={int(row['chef_positive_action_count'])} "
        f"{first_ts_text}"
    )


def _plot_heatmap(
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    out_path: Path,
    cmap: str,
    ylabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(8.0, 1.4 * len(col_labels))
    fig_h = max(5.5, 0.38 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    ax.set_xlabel("Metrics")
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if math.isfinite(float(value)):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_transition_heatmap(
    matrix: np.ma.MaskedArray,
    *,
    row_labels: Sequence[str],
    title: str,
    out_path: Path,
    cmap: str,
    ylabel: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_w = max(12.0, min(28.0, 0.18 * matrix.shape[1]))
    fig_h = max(5.5, 0.42 * len(row_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#f2f2f2")
    image = ax.imshow(matrix, aspect="auto", cmap=cmap_obj)
    if matrix.shape[1] <= 80:
        tick_step = max(1, matrix.shape[1] // 20)
    else:
        tick_step = max(1, matrix.shape[1] // 30)
    xticks = np.arange(0, matrix.shape[1], tick_step)
    ax.set_xticks(xticks, labels=[str(int(x)) for x in xticks], rotation=0)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    ax.set_xlabel("Transition index within rollout")
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="GRPO advantage")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _build_transition_heatmap_rows(
    summary_rows: Sequence[Dict[str, Any]],
    all_rows: Sequence[Dict[str, Any]],
    advantages: torch.Tensor,
    *,
    agent_filter: Optional[int] = None,
    compact_agent: bool = False,
) -> Tuple[np.ma.MaskedArray, List[Dict[str, Any]]]:
    if compact_agent and agent_filter is not None:
        max_len = 0
        for summary in summary_rows:
            start = int(summary["_start"])
            end = int(summary["_end"])
            count = sum(
                1
                for global_idx in range(start, end)
                if _as_int(all_rows[global_idx].get("agent_index"), -1) == agent_filter
            )
            max_len = max(max_len, count)
    else:
        max_len = max((int(row["num_transitions"]) for row in summary_rows), default=0)
    matrix = np.full((len(summary_rows), max_len), np.nan, dtype=float)
    csv_rows: List[Dict[str, Any]] = []
    for plot_idx, summary in enumerate(summary_rows):
        start = int(summary["_start"])
        end = int(summary["_end"])
        compact_idx = 0
        for local_idx, global_idx in enumerate(range(start, end)):
            row = all_rows[global_idx]
            agent_index = _as_int(row.get("agent_index"), -1)
            if agent_filter is not None and agent_index != agent_filter:
                continue
            advantage = float(advantages[global_idx].item())
            heatmap_idx = compact_idx if compact_agent and agent_filter is not None else local_idx
            if heatmap_idx >= matrix.shape[1]:
                continue
            matrix[plot_idx, heatmap_idx] = advantage
            compact_idx += 1
            csv_rows.append(
                {
                    "plot_order": int(summary["plot_order"]),
                    "rank": int(summary["rank"]),
                    "update": int(summary["update"]),
                    "local_transition_idx": local_idx,
                    "heatmap_transition_idx": heatmap_idx,
                    "global_transition_idx": global_idx,
                    "agent_index": agent_index,
                    "timestep": row.get("timestep", ""),
                    "advantage": advantage,
                    "reward": _as_float(row.get("reward")),
                    "process_reward": _as_float(row.get("process_reward")),
                    "sequence_reward": _as_float(row.get("sequence_reward")),
                    "paired_comm_reward": _as_float(row.get("paired_comm_reward")),
                    "format_reward": _as_float(row.get("format_reward")),
                    "validator_reward": _as_float(row.get("validator_reward")),
                    "text": str(row.get("text", ""))[:300],
                    "file": summary["file"],
                }
            )
    return np.ma.masked_invalid(matrix), csv_rows


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    output = matrix.astype(float).copy()
    for col in range(output.shape[1]):
        values = output[:, col]
        mean = float(np.nanmean(values))
        std = float(np.nanstd(values))
        if std < 1e-8:
            output[:, col] = 0.0
        else:
            output[:, col] = (values - mean) / std
    return output


def _build_summary_text(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows.\n"
    chef_seq = np.array([float(r["chef_sequence_sum"]) for r in rows], dtype=float)
    chef_count = np.array([float(r["chef_positive_action_count"]) for r in rows], dtype=float)
    prefix_delta = np.array([float(r["prefix_adv_delta_vs_all"]) for r in rows], dtype=float)
    prefix_mean = np.array([float(r["chef_prefix_adv_mean"]) for r in rows], dtype=float)
    high_mask = chef_seq > float(np.mean(chef_seq))
    lines = []
    lines.append(f"num_rollouts: {len(rows)}")
    lines.append(f"chef_sequence_sum mean: {float(np.mean(chef_seq)):.4f}")
    lines.append(f"chef_positive_action_count mean: {float(np.mean(chef_count)):.4f}")
    lines.append(f"chef_prefix_adv_mean mean: {float(np.mean(prefix_mean)):.4f}")
    lines.append(f"prefix_adv_delta_vs_all mean: {float(np.mean(prefix_delta)):.4f}")
    if high_mask.any():
        lines.append(
            "high_process_rollouts prefix_adv_delta_vs_all mean: "
            f"{float(np.mean(prefix_delta[high_mask])):.4f}"
        )
        lines.append(
            "high_process_rollouts chef_prefix_adv_mean mean: "
            f"{float(np.mean(prefix_mean[high_mask])):.4f}"
        )
    if (~high_mask).any():
        lines.append(
            "low_process_rollouts prefix_adv_delta_vs_all mean: "
            f"{float(np.mean(prefix_delta[~high_mask])):.4f}"
        )
    if np.std(chef_seq) > 1e-8 and np.std(prefix_mean) > 1e-8:
        corr = float(np.corrcoef(chef_seq, prefix_mean)[0, 1])
        lines.append(f"corr(chef_sequence_sum, chef_prefix_adv_mean): {corr:.4f}")
    if np.std(chef_count) > 1e-8 and np.std(prefix_mean) > 1e-8:
        corr = float(np.corrcoef(chef_count, prefix_mean)[0, 1])
        lines.append(f"corr(chef_positive_action_count, chef_prefix_adv_mean): {corr:.4f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", required=True, help="Experiment root or rollout root.")
    parser.add_argument("--rollout-dir", default=None, help="Override rollout directory.")
    parser.add_argument("--update", type=int, default=None, help="Update/epoch to analyze; default latest.")
    parser.add_argument("--out-dir", default=None, help="Output directory.")
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
        help="Reward field used to recompute advantages; trainer uses reward.",
    )
    parser.add_argument(
        "--max-prefix-timestep",
        type=int,
        default=None,
        help="Only include prefix transitions up to this timestep.",
    )
    parser.add_argument(
        "--sort-by",
        choices=[
            "chef_sequence_desc",
            "team_reward_desc",
            "chef_prefix_adv_desc",
            "chef_process_max_desc",
            "first_positive_timestep",
            "rank",
        ],
        default="chef_process_max_desc",
        help=(
            "Row order in heatmaps. Default puts trajectories with higher Chef "
            "max process reward at the top."
        ),
    )
    parser.add_argument("--cmap", default="RdBu_r")
    args = parser.parse_args()

    exp_root = Path(args.exp_root).expanduser().resolve()
    rollout_dir = _find_rollout_dir(exp_root, args.rollout_dir)
    update, files = _select_rollout_files(rollout_dir, args.update)
    all_rows, meta_rows = _load_rows(files)
    if not all_rows:
        raise RuntimeError(f"Selected rollout files are empty for update {update:05d}")

    rewards = [_reward_for_adv(row, args.reward_field) for row in all_rows]
    agent_indices = [_as_int(row.get("agent_index"), 0) for row in all_rows]
    timesteps = [
        _as_int(row.get("timestep"), -1) if row.get("timestep") is not None else None
        for row in all_rows
    ]
    advantages, returns, metrics = compute_grpo_advantages(
        rewards=rewards,
        agent_indices=agent_indices,
        timesteps=timesteps,
        trajectory_ids=[row.get("rollout_id", row.get("_rollout_id", "__default_rollout__")) for row in all_rows],
        config=GrpoAdvantageConfig(
            norm_scope=args.norm_scope,
            normalize=args.normalize,
            gamma=args.gamma,
            clip=args.clip,
        ),
    )

    summary_rows = []
    for meta in meta_rows:
        rows = all_rows[int(meta["start"]) : int(meta["end"])]
        summary_rows.append(
            _summarize_rollout(
                meta,
                rows,
                advantages,
                returns,
                chef_agent_idx=args.chef_agent_idx,
                assistant_agent_idx=args.assistant_agent_idx,
                max_prefix_timestep=args.max_prefix_timestep,
            )
        )
    _sort_summary_rows(summary_rows, args.sort_by)
    for plot_order, row in enumerate(summary_rows, start=1):
        row["plot_order"] = plot_order
        row["sort_by"] = args.sort_by

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else exp_root / "analysis" / f"advantage_heatmap_u{update:05d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rollout_advantage_summary.csv", summary_rows)
    metrics_rows = [{"metric": key, "value": value} for key, value in sorted(metrics.items())]
    _write_csv(out_dir / "advantage_metrics.csv", metrics_rows)
    transition_matrix, transition_rows = _build_transition_heatmap_rows(
        summary_rows,
        all_rows,
        advantages,
    )
    _write_csv(out_dir / "transition_advantage_rows.csv", transition_rows)
    assistant_transition_matrix, assistant_transition_rows = _build_transition_heatmap_rows(
        summary_rows,
        all_rows,
        advantages,
        agent_filter=args.assistant_agent_idx,
        compact_agent=False,
    )
    _write_csv(out_dir / "assistant_transition_advantage_rows.csv", assistant_transition_rows)
    (
        assistant_compact_transition_matrix,
        assistant_compact_transition_rows,
    ) = _build_transition_heatmap_rows(
        summary_rows,
        all_rows,
        advantages,
        agent_filter=args.assistant_agent_idx,
        compact_agent=True,
    )
    _write_csv(
        out_dir / "assistant_transition_advantage_rows_compact.csv",
        assistant_compact_transition_rows,
    )

    columns = [
        "chef_sequence_sum",
        "chef_positive_action_count",
        "chef_prefix_adv_mean",
        "prefix_adv_delta_vs_all",
        "chef_all_adv_mean",
        "team_reward_sum",
    ]
    matrix = np.array([[float(row[col]) for col in columns] for row in summary_rows], dtype=float)
    labels = [_row_label(row) for row in summary_rows]
    ylabel = f"Rollouts sorted by {args.sort_by}"
    _plot_heatmap(
        matrix,
        row_labels=labels,
        col_labels=columns,
        title=f"GRPO rollout advantage vs process reward, update {update:05d}",
        out_path=out_dir / "advantage_process_heatmap_raw.png",
        cmap=args.cmap,
        ylabel=ylabel,
    )
    _plot_heatmap(
        _zscore_columns(matrix),
        row_labels=labels,
        col_labels=[f"z({col})" for col in columns],
        title=f"Column-normalized advantage/process heatmap, update {update:05d}",
        out_path=out_dir / "advantage_process_heatmap_zscore.png",
        cmap=args.cmap,
        ylabel=ylabel,
    )
    _plot_transition_heatmap(
        transition_matrix,
        row_labels=labels,
        title=f"Transition-level GRPO advantages, update {update:05d}",
        out_path=out_dir / "transition_advantage_heatmap.png",
        cmap=args.cmap,
        ylabel=ylabel,
    )
    _plot_transition_heatmap(
        assistant_transition_matrix,
        row_labels=labels,
        title=f"Assistant transition advantages, original positions, update {update:05d}",
        out_path=out_dir / "assistant_transition_advantage_heatmap.png",
        cmap=args.cmap,
        ylabel=ylabel,
    )
    _plot_transition_heatmap(
        assistant_compact_transition_matrix,
        row_labels=labels,
        title=f"Assistant transition advantages, compacted, update {update:05d}",
        out_path=out_dir / "assistant_transition_advantage_heatmap_compact.png",
        cmap=args.cmap,
        ylabel=ylabel,
    )
    summary_text = _build_summary_text(summary_rows)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(f"update={update:05d}")
    print(f"files={len(files)} transitions={len(all_rows)}")
    print(f"out_dir={out_dir}")
    print(summary_text, end="")


if __name__ == "__main__":
    main()
