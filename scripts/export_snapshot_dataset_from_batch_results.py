#!/usr/bin/env python3
"""
从 batch_results 的 JSON 轨迹日志重放环境，并导出 RL snapshot（jsonl）。

背景：
- Azure/批量评测日志（例如 assets/data/batch_results/.../*.json）通常只包含：
  - 每个 timestep 的文本 observation、动作字符串、统计信息等
  - 以及 total_action_list（每个 agent 的“完成动作”记录）
- 但 RL 的 snapshot 需要可被 OvercookedState.from_dict 还原的 env_state。

因此本脚本采用“重放(replay)”：
1) 用一个基础 YAML 配置构建 CollabMainSession（环境 + agent + reward tracker）。
2) 将日志中的动作序列灌入 agent 的 test_mode（不调用 LLM）。
3) 按步执行 session.step()，在每一步开始前 capture_snapshot() 写入 jsonl。

输出格式与 scripts/export_snapshot_dataset.py 一致：每行一个 JSON 对象，包含 "snapshot" 字段。
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _iter_input_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")
    files = sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found under: {path}")
    return files


def _infer_order_from_log(log: Dict[str, Any], fallback_name: str = "") -> str:
    try:
        content = log.get("content") or []
        if isinstance(content, list) and content:
            order_list = content[0].get("order_list") if isinstance(content[0], dict) else None
            if isinstance(order_list, list) and order_list and isinstance(order_list[0], str):
                return order_list[0]
    except Exception:
        pass
    # 兜底：从文件名里猜（例如 *_baked_bell_pepper.json）
    if fallback_name:
        stem = fallback_name
        if stem.endswith(".json"):
            stem = stem[:-5]
        parts = stem.split("_")
        for i in range(len(parts)):
            candidate = "_".join(parts[i:])
            if candidate:
                return candidate
    return ""


def _extract_action_sequences(
    log: Dict[str, Any],
    *,
    prefer_total_action_list: bool = True,
) -> Tuple[List[str], List[str], int]:
    """
    返回：
    - agent0_actions: List[str]
    - agent1_actions: List[str]
    - suggested_max_steps: int（通常用来控制 replay 步数上限）
    """
    content = log.get("content") or []
    suggested_max_steps = len(content) if isinstance(content, list) else 0
    if suggested_max_steps <= 0:
        timestamps = log.get("total_timestamp")
        if isinstance(timestamps, list):
            suggested_max_steps = len(timestamps)

    def _actions_from_total_action_list(idx: int) -> List[str]:
        tal = log.get("total_action_list")
        if not (isinstance(tal, list) and len(tal) >= 2 and isinstance(tal[idx], list)):
            return []
        entries = []
        for item in tal[idx]:
            if not isinstance(item, dict):
                continue
            action = item.get("action")
            ts = item.get("timestamp")
            if isinstance(action, str) and action.strip():
                entries.append((int(ts) if isinstance(ts, (int, float)) else 0, action.strip()))
        # 按 timestamp 排序，忽略 timestamp 的话也能稳定复现
        entries.sort(key=lambda x: x[0])
        return [a for _, a in entries]

    def _actions_from_per_timestep(idx: int) -> List[str]:
        actions: List[str] = []
        if not isinstance(content, list):
            return actions
        for step in content:
            if not isinstance(step, dict):
                continue
            pair = step.get("actions")
            if not (isinstance(pair, list) and len(pair) >= 2):
                continue
            a = pair[idx]
            if isinstance(a, str) and a.strip():
                actions.append(a.strip())
        # 这里返回“每步记录的动作字符串”，它不是 test_mode 所期待的“动作序列”，
        # 但可用于兜底（会偏向重复 wait(1)）。
        return actions

    if prefer_total_action_list:
        a0 = _actions_from_total_action_list(0)
        a1 = _actions_from_total_action_list(1)
        if a0 or a1:
            return a0, a1, suggested_max_steps

    return _actions_from_per_timestep(0), _actions_from_per_timestep(1), suggested_max_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        required=True,
        help="用于构建环境/agent 的 YAML 配置（会在内存里把 order/horizon 覆盖为日志对应值）。",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="batch_results 的 .json 文件或目录（目录下默认读取 *.json）。",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出的 snapshot jsonl 路径。",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="最多处理多少个输入文件（0 表示不限制）。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="每个 episode 最多 replay 多少步（0 表示使用日志长度作为上限）。",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="每隔多少步写一个 snapshot（默认每步都写）。",
    )
    parser.add_argument(
        "--prefer-total-action-list",
        action="store_true",
        help="优先使用 total_action_list 作为动作序列（默认行为）。",
    )
    parser.add_argument(
        "--use-per-timestep-actions",
        action="store_true",
        help="强制使用 content[t].actions 的逐步动作（更贴近日志，但可能导致 test_mode 行为偏差）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖输出文件（若已存在）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 允许通过 `python scripts/...py` 直接运行（此时 sys.path[0] 是 scripts/，需要把仓库根目录加入 PYTHONPATH）
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from collab_overcooked.main import convert_yaml_to_variant, load_config_from_yaml
    from collab_overcooked.training.main_session import CollabMainSession

    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    base_config_path = Path(args.base_config).expanduser()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file exists: {output_path} (use --overwrite)")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(input_path)
    if args.max_episodes and args.max_episodes > 0:
        files = files[: args.max_episodes]

    # 读取基础配置（后续会按每个 episode 的 order/horizon 做内存覆盖）
    base_config = load_config_from_yaml(str(base_config_path))
    if not isinstance(base_config, dict):
        raise ValueError(f"Invalid YAML config: {base_config_path}")

    def _dummy_policy_fn(agent_index: int, messages: List[Dict[str, str]], context: Dict[str, Any]):
        raise RuntimeError(
            "Replay 模式下不应调用 policy_fn（请确保 agents 处于 test_mode，且动作序列有效）。"
        )

    total_written = 0
    episode_idx = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for file_path in files:
            log = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(log, dict):
                continue

            order = _infer_order_from_log(log, fallback_name=file_path.name)
            if not order:
                raise ValueError(f"Cannot infer order from log: {file_path}")

            prefer_total = True
            if args.use_per_timestep_actions:
                prefer_total = False
            elif args.prefer_total_action_list:
                prefer_total = True

            a0_actions, a1_actions, suggested_steps = _extract_action_sequences(
                log, prefer_total_action_list=prefer_total
            )

            # 按日志长度控制 replay 上限，避免 test_mode 队列空后无限 wait。
            step_limit = args.max_steps if args.max_steps and args.max_steps > 0 else suggested_steps
            step_limit = max(1, int(step_limit))

            cfg = deepcopy(base_config)
            env_cfg = cfg.setdefault("environment", {})
            if isinstance(env_cfg, dict):
                env_cfg["order"] = order
                # 确保 horizon 足够覆盖 step_limit
                horizon = int(env_cfg.get("horizon", 120) or 120)
                if horizon < step_limit + 1:
                    env_cfg["horizon"] = step_limit + 1

            variant = convert_yaml_to_variant(cfg)
            variant["yaml_config"] = cfg

            session = CollabMainSession(variant, _dummy_policy_fn)

            # 进入 test_mode，并灌入动作序列
            # 约定：agent_0 是 Chef，agent_1 是 Assistant（与日志 actions 列表一致）
            for idx, agent in enumerate(session.team.agents):
                if not hasattr(agent, "test_mode"):
                    continue
                agent.test_mode = True
                seq = a0_actions if idx == 0 else a1_actions
                agent.test_ml_action = deque(seq)

            # replay 并写 snapshots（在 step 前写，方便用于 off_policy_snapshots）
            for local_step in range(step_limit):
                if args.stride <= 1 or (local_step % int(args.stride) == 0):
                    snapshot = session.capture_snapshot()
                    payload = {
                        "snapshot": snapshot,
                        "metadata": {
                            "source_file": str(file_path),
                            "episode_index": episode_idx,
                            "local_step": local_step,
                            "order": order,
                            "action_seq_lens": [len(a0_actions), len(a1_actions)],
                        },
                    }
                    writer.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    total_written += 1
                step_result = session.step()
                if step_result.done:
                    break

            episode_idx += 1

    print(f"[SnapshotReplayExport] wrote {total_written} snapshots -> {output_path}")


if __name__ == "__main__":
    main()
