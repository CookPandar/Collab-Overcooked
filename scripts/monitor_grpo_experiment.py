#!/usr/bin/env python
"""Monitor GRPO experiments with rollout reward audits and curve summaries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

import audit_rollout_rewards as audit


def _parse_update(path: Path) -> Optional[int]:
    match = re.search(r"_u(\d+)\.pt$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _rollout_files(root: Path, stage: str = "train") -> List[Path]:
    rollout_dir = root / ("rollouts_eval_grpo" if stage == "eval" else "rollouts_grpo")
    return sorted(rollout_dir.glob("rollout_rank*_u*.pt"))


def _updates_with_min_workers(root: Path, min_workers: int, stage: str = "train") -> List[int]:
    counts: Counter[int] = Counter()
    for path in _rollout_files(root, stage=stage):
        update = _parse_update(path)
        if update is not None:
            counts[update] += 1
    return sorted(update for update, count in counts.items() if count >= min_workers)


def _audit_update(root: Path, update: int, tokenizer, stage: str = "train") -> Tuple[int, Dict[str, int], int]:
    files = [
        path
        for path in _rollout_files(root, stage=stage)
        if _parse_update(path) == update
    ]
    rows = []
    for path in files:
        rows.extend(audit.load_rows(path, tokenizer))
    issues = audit.find_issues(rows)
    return len(rows), dict(Counter(kind for kind, _ in issues)), len(issues)


def _audit_update_with_options(
    root: Path,
    update: int,
    tokenizer,
    *,
    stage: str,
    partial_success_reward: float,
    partial_success_agent: int,
) -> Tuple[int, Dict[str, int], int]:
    files = [
        path
        for path in _rollout_files(root, stage=stage)
        if _parse_update(path) == update
    ]
    rows = []
    for path in files:
        rows.extend(audit.load_rows(path, tokenizer))
    issues = audit.find_issues(rows)
    if stage == "train":
        log_dir = root / "logs" / "rl_workers"
        if log_dir.exists():
            issues.extend(
                audit.find_missing_executed_rewards(
                    rows,
                    root / "rollouts_grpo",
                    log_dir,
                )
            )
    issues.extend(
        audit.find_partial_success_issues(
            rows,
            terminal_reward=partial_success_reward,
            agent_index=partial_success_agent,
        )
    )
    return len(rows), dict(Counter(kind for kind, _ in issues)), len(issues)


def _read_last_csv_row(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return None
    return rows[-1] if rows else None


def _float(row: Optional[Dict[str, str]], key: str) -> Optional[float]:
    if not row:
        return None
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _trend_summary(root: Path) -> Dict[str, Optional[float]]:
    train_row = _read_last_csv_row(root / "runs/grpo/train/train_curve.csv")
    reward_row = _read_last_csv_row(root / "runs/grpo/train/reward_curve.csv")
    return {
        "update_idx": _float(train_row, "update_idx"),
        "loss": _float(train_row, "loss"),
        "policy_loss": _float(train_row, "policy_loss"),
        "entropy": _float(train_row, "entropy"),
        "approx_kl": _float(train_row, "approx_kl"),
        "reward_mean": _float(train_row, "reward_mean"),
        "return_mean": _float(train_row, "return_mean"),
        "agent0_sequence_sum": _float(reward_row, "agent0_sequence_sum"),
        "agent1_sequence_sum": _float(reward_row, "agent1_sequence_sum"),
        "agent0_paired_comm_sum": _float(reward_row, "agent0_paired_comm_sum"),
        "agent1_paired_comm_sum": _float(reward_row, "agent1_paired_comm_sum"),
    }


def _stop_session(session: str, log_path: Path) -> None:
    if not session:
        return
    try:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[monitor] stopped tmux session={session}\n")
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[monitor] failed to stop session={session}: {exc}\n")


def _load_seen(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): [int(item) for item in value]
        for key, value in data.items()
        if isinstance(value, list)
    }


def _save_seen(path: Path, seen: Dict[str, List[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(seen, indent=2, sort_keys=True), encoding="utf-8")


def _format_summary(summary: Dict[str, Optional[float]]) -> str:
    parts = []
    for key in (
        "update_idx",
        "reward_mean",
        "return_mean",
        "entropy",
        "approx_kl",
        "agent0_sequence_sum",
        "agent1_sequence_sum",
        "agent0_paired_comm_sum",
        "agent1_paired_comm_sum",
    ):
        value = summary.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value:.6g}")
    return " ".join(parts)


def _parse_experiment(raw: str) -> Tuple[str, Path, str]:
    parts = raw.split(":", 2)
    if len(parts) == 1:
        root = Path(parts[0]).resolve()
        return root.name, root, ""
    if len(parts) == 2:
        label, root = parts
        return label, Path(root).resolve(), ""
    label, root, session = parts
    return label, Path(root).resolve(), session


def run_once(
    experiments: Sequence[Tuple[str, Path, str]],
    tokenizer,
    *,
    min_workers: int,
    state_path: Path,
    log_path: Path,
    stop_on_issue: bool,
    partial_success_reward: float,
    partial_success_agent: int,
) -> bool:
    seen = _load_seen(state_path)
    ok = True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[monitor] tick ts={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for label, root, session in experiments:
            summary = _trend_summary(root)
            log.write(f"[monitor] {label} root={root} {_format_summary(summary)}\n")
            for stage in ("train", "eval"):
                # Keep the train key backward-compatible so existing monitor state
                # does not re-audit old training rollouts after this change.
                key = str(root) if stage == "train" else f"{root}:eval"
                done_updates = _updates_with_min_workers(root, min_workers, stage=stage)
                seen_updates = set(seen.get(key, []))
                new_updates = [update for update in done_updates if update not in seen_updates]
                for update in new_updates:
                    transitions, issue_counts, issue_total = _audit_update_with_options(
                        root,
                        update,
                        tokenizer,
                        stage=stage,
                        partial_success_reward=partial_success_reward,
                        partial_success_agent=partial_success_agent,
                    )
                    log.write(
                        f"[audit] {label} stage={stage} update={update:05d} "
                        f"transitions={transitions} issue_counts={issue_counts}\n"
                    )
                    if issue_total:
                        ok = False
                        log.write(
                            f"[audit] ISSUE_DETECTED label={label} stage={stage} "
                            f"update={update:05d} issue_total={issue_total}\n"
                        )
                        if stop_on_issue:
                            _stop_session(session, log_path)
                    else:
                        seen_updates.add(update)
                seen[key] = sorted(seen_updates)
    _save_seen(state_path, seen)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        help="label:/path/to/experiment[:tmux_session]",
    )
    parser.add_argument("--model", default="/datacache/LLMs/Qwen2.5-7B-Instruct")
    parser.add_argument("--min-workers", type=int, default=8)
    parser.add_argument("--interval-sec", type=float, default=300.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--stop-on-issue", action="store_true")
    parser.add_argument("--partial-success-reward", type=float, default=20.0)
    parser.add_argument("--partial-success-agent", type=int, default=0)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()

    experiments = [_parse_experiment(raw) for raw in args.experiment]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    state_path = Path(args.state_file).resolve()
    log_path = Path(args.log_file).resolve()

    while True:
        ok = run_once(
            experiments,
            tokenizer,
            min_workers=args.min_workers,
            state_path=state_path,
            log_path=log_path,
            stop_on_issue=args.stop_on_issue,
            partial_success_reward=args.partial_success_reward,
            partial_success_agent=args.partial_success_agent,
        )
        if args.once:
            return 0 if ok else 1
        time.sleep(max(1.0, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
