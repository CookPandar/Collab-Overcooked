#!/usr/bin/env python3
"""
并行批量测试脚本：
- 支持多个模型、多温度、重复次数
- 可指定并发进程，每个进程顺序跑完整个 30 任务集
- 输出按模型/温度聚合的成功率以及每个 level 的成功率

Example:
    python scripts/run_model_suite.py \
        --models qwen2.5-7B-instruct azure-gpt-4o \
        --base-config configs/test_personal.yaml \
        --temperatures 0 0.7 \
        --repeats 3 \
        --max-workers 4 \
        --output-dir batch_results
"""

import argparse
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
import uuid

import matplotlib.pyplot as plt
import yaml

ORDER_TIME_HINT = {
    "baked_bell_pepper": 16,
    "baked_sweet_potato": 16,
    "boiled_egg": 16,
    "boiled_mushroom": 16,
    "boiled_sweet_potato": 16,
    "baked_potato_slices": 23,
    "baked_pumpkin_slices": 23,
    "boiled_corn_slices": 23,
    "boiled_green_bean_slices": 23,
    "boiled_potato_slices": 23,
    "baked_bell_pepper_soup": 35,
    "baked_carrot_soup": 35,
    "baked_mushroom_soup": 35,
    "baked_potato_soup": 35,
    "baked_pumpkin_soup": 35 #,
    # "sliced_bell_pepper_and_corn_stew": 32,
    # "sliced_bell_pepper_and_lentil_stew": 32,
    # "sliced_eggplant_and_chickpea_stew": 32,
    # "sliced_pumpkin_and_chickpea_stew": 32,
    # "sliced_zucchini_and_chickpea_stew": 32,
    # "mashed_broccoli_and_bean_patty": 55,
    # "mashed_carrot_and_chickpea_patty": 55,
    # "mashed_cauliflower_and_lentil_patty": 55,
    # "mashed_potato_and_pea_patty": 55,
    # "mashed_sweet_potato_and_bean_patty": 55,
    # "potato_carrot_and_onion_patty": 64,
    # "romaine_lettuce_pea_and_tomato_patty": 64,
    # "sweet_potato_spinach_and_mushroom_patty": 64,
    # "taro_bean_and_bell_pepper_patty": 64,
    # "zucchini_green_pea_and_onion_patty": 64,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel batch evaluation for Collab-Overcooked.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of LLM model identifiers (team labels). Use --agent-overrides to customize Chef/Assistant models.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/test_personal.yaml"),
        help="YAML config to use as the template for each run.",
    )
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=Path("collab_overcooked/prompts/recipe"),
        help="Directory containing recipe prompt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/data/batch_results"),
        help="Directory to store aggregated outputs and plots.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Default temperature when --temperatures is not provided.",
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        help="Optional list of temperatures. Overrides --temperature when supplied.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many full-suite repetitions to run per model/temperature.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel processes. Each worker sequentially runs all tasks.",
    )
    parser.add_argument(
        "--model-configs",
        type=Path,
        help="Optional JSON/YAML mapping from model name to config path. "
        "If provided, overrides --base-config per model.",
    )
    parser.add_argument(
        "--agent-overrides",
        type=Path,
        help="Optional JSON/YAML mapping from agent role (e.g., Chef/Assistant) "
        "to override fields such as model/base_url/temperature for that agent.",
    )
    return parser.parse_args()


def load_model_config_map(path: Optional[Path]) -> Dict[str, Path]:
    if not path:
        return {}
    data = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    mapping = {}
    for model_name, cfg_path in data.items():
        mapping[model_name] = Path(cfg_path)
    return mapping


def load_orders(recipe_dir: Path):
    orders = []
    missing = []
    for recipe_file in recipe_dir.glob("*.txt"):
        stem = recipe_file.stem
        parts = stem.split("_", 1)
        level = 0
        name = stem
        if len(parts) == 2 and parts[0].isdigit():
            level = int(parts[0])
            name = parts[1]
        elif len(parts) == 2:
            name = parts[1]
        if name not in ORDER_TIME_HINT:
            missing.append(name)
            continue
        orders.append({"order": name, "level": level})
    # if missing:
    #     raise ValueError(f"Missing ORDER_TIME_HINT entries for: {missing}")
    orders.sort(key=lambda item: (item["level"], item["order"]))
    return orders


def load_agent_overrides(path: Optional[Path]) -> Dict[str, Dict]:
    if not path:
        return {}
    data = json.loads(path.read_text()) if path.suffix == ".json" else yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Agent overrides must be a mapping from role to override dictionary.")
    clean = {}
    for key, value in data.items():
        if isinstance(value, dict):
            clean[str(key)] = value
    return clean


def update_config(base_cfg, order, model_name, temperature, run_id=None, agent_overrides=None):
    cfg = json.loads(json.dumps(base_cfg))
    cfg.setdefault("environment", {})
    cfg["environment"]["order"] = order
    base_time = ORDER_TIME_HINT[order]
    cfg["environment"]["horizon"] = int(base_time * 1.5)

    cfg.setdefault("run", {})
    if run_id:
        cfg["run"]["run_id"] = run_id

    overrides = agent_overrides or {}
    if overrides:
        for key, value in cfg.get("agents", {}).items():
            if not (key.startswith("agent_") and isinstance(value, dict)):
                continue
            role = value.get("role")
            override = None
            if role and role in overrides:
                override = overrides[role]
            elif key in overrides:
                override = overrides[key]
            if override:
                value.update(dict(override))

    return cfg


def invoke_main(config_path: Path, console_log_path: Path):
    cmd = [sys.executable, "-m", "collab_overcooked.main", "--config", str(config_path)]
    console_log_path.parent.mkdir(parents=True, exist_ok=True)
    with console_log_path.open("w", encoding="utf-8") as log_fh:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_fh.write(line)
            log_fh.flush()
            print(line, end="")
        retcode = process.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, cmd)


def collect_new_log(order: str, run_id: str):
    base = Path("results") / f"{run_id}_{order}"
    if not base.exists():
        return None
    json_files = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return json_files[-1] if json_files else None


def wait_for_file_stable(path: Path, timeout: float = 60.0, poll_interval: float = 0.5):
    """
    Wait until file size stops changing for at least one poll interval or timeout expires.
    Collab-Overcooked writes the JSON after every timestep, so copying too early would
    capture only a prefix. This helper avoids truncated logs when multiple workers run.
    """
    deadline = time.time() + timeout
    last_size = -1
    stable_start = None

    while time.time() < deadline:
        try:
            current_size = path.stat().st_size
        except OSError:
            time.sleep(poll_interval)
            continue

        if current_size == last_size and current_size > 0:
            if stable_start is None:
                stable_start = time.time()
            if time.time() - stable_start >= poll_interval:
                return True
        else:
            last_size = current_size
            stable_start = None
        time.sleep(poll_interval)
    return False


def run_single_task(
    order_entry,
    model,
    temperature,
    base_cfg,
    logs_dir: Path,
    json_dir: Path,
    worker_id: str,
    agent_overrides: Optional[Dict[str, Dict]] = None,
):
    order = order_entry["order"]
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    run_id = f"{worker_id}-{run_ts}-{uuid.uuid4().hex[:6]}"
    cfg = update_config(
        base_cfg,
        order,
        model,
        temperature,
        run_id=run_id,
        agent_overrides=agent_overrides,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = Path(tmp.name)

    start = time.time()
    success = False
    steps = None
    log_path = None
    base_filename = f"{run_id}_{order}"
    order_log_dir = logs_dir / order
    order_log_dir.mkdir(parents=True, exist_ok=True)
    console_log_path = order_log_dir / f"{base_filename}.log"
    try:
        invoke_main(tmp_path, console_log_path)
        log_file = collect_new_log(order, run_id)
        if log_file and log_file.exists():
            wait_for_file_stable(log_file)
            data = json.loads(log_file.read_text())
            success = bool(data.get("total_order_finished"))
            timestamps = data.get("total_timestamp") or []
            steps = timestamps[-1] if timestamps else None
            log_path = str(log_file)
        else:
            print(f"   ! No result log found for {order}")
    except subprocess.CalledProcessError as exc:
        print(f"   ! Run failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    duration = time.time() - start
    copied_log_path = None
    if log_path:
        order_json_dir = json_dir / order
        order_json_dir.mkdir(parents=True, exist_ok=True)
        dest = order_json_dir / f"{base_filename}.json"
        try:
            shutil.copy2(log_path, dest)
            copied_log_path = str(dest)
        except OSError as err:
            print(f"[Warn] Failed to copy log to {dest}: {err}")

    return {
        "model": model,
        "temperature": temperature,
        "order": order,
        "level": order_entry["level"],
        "success": success,
        "steps": steps,
        "duration_sec": duration,
        "log_path": log_path,
        "copied_log_path": copied_log_path,
        "console_log_path": str(console_log_path),
        "run_id": run_id,
    }


def run_order_job(job):
    model = job["model"]
    temperature = job["temperature"]
    repeat_idx = job["repeat"]
    entry = job["order_entry"]
    base_cfg = job["base_cfg"]
    logs_dir = job["logs_dir"]
    json_dir = job["json_dir"]
    worker_id = job["worker_id"]
    agent_overrides = job.get("agent_overrides")
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[Worker] ({model}, T={temperature}, R={repeat_idx}) -> {entry['order']} started"
    )
    result = run_single_task(
        entry,
        model,
        temperature,
        base_cfg,
        logs_dir,
        json_dir,
        worker_id,
        agent_overrides,
    )
    result["repeat"] = repeat_idx
    return result


def aggregate_summary(summary: List[Dict]):
    combo_stats = {}
    level_stats = {}
    for entry in summary:
        combo_key = (entry["model"], entry["temperature"])
        combo = combo_stats.setdefault(combo_key, {"success": 0, "total": 0})
        combo["total"] += 1
        if entry["success"]:
            combo["success"] += 1

        level_key = (combo_key, entry["level"])
        lvl = level_stats.setdefault(level_key, {"success": 0, "total": 0})
        lvl["total"] += 1
        if entry["success"]:
            lvl["success"] += 1
    return combo_stats, level_stats


def save_plots(combo_stats, output_dir: Path):
    labels = []
    rates = []
    for (model, temp), stats in sorted(combo_stats.items()):
        label = f"{model}@{temp}"
        total = stats["total"]
        rate = stats["success"] / total if total else 0.0
        labels.append(label)
        rates.append(rate)

    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(labels, rates, color="#4E79A7")
    plt.ylim(0, 1.05)
    plt.ylabel("Success Rate")
    plt.title("Overall Success Rates per Model/Temperature")
    for idx, rate in enumerate(rates):
        plt.text(idx, min(1.0, rate + 0.02), f"{rate:.2f}", ha="center")
    plt.xticks(rotation=20, ha="right")
    plot_path = output_dir / "success_rates.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Success rate plot saved to {plot_path}")


def run_suite(
    models,
    base_config_path,
    recipe_dir,
    output_dir,
    temperatures,
    repeats,
    max_workers,
    model_config_map,
    agent_overrides,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    orders = load_orders(recipe_dir)

    jobs = []
    job_counter = 0
    for model in models:
        model_dir = output_dir / model
        logs_dir = model_dir / "logs"
        json_dir = model_dir / "json"
        cfg_path = model_config_map.get(model, base_config_path)
        base_cfg = yaml.safe_load(cfg_path.read_text())
        for temp in temperatures:
            for repeat_idx in range(repeats):
                safe_temp = str(temp).replace(".", "_")
                for order_entry in orders:
                    worker_id = f"worker{repeat_idx}_{safe_temp}_{order_entry['order']}_{job_counter}"
                    job_counter += 1
                    jobs.append(
                        {
                            "model": model,
                            "temperature": temp,
                            "repeat": repeat_idx,
                            "order_entry": order_entry,
                            "base_cfg": base_cfg,
                            "logs_dir": logs_dir,
                            "json_dir": json_dir,
                            "worker_id": worker_id,
                            "agent_overrides": agent_overrides,
                        }
                    )

    summary = []
    if max_workers <= 1:
        for job in jobs:
            summary.append(run_order_job(job))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for job_result in executor.map(run_order_job, jobs):
                summary.append(job_result)

    combo_stats, level_stats = aggregate_summary(summary)

    summary_path = output_dir / "results.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDetailed results saved to {summary_path}")

    combo_report = {
        f"{model}@{temp}": {
            "success": stats["success"],
            "total": stats["total"],
            "rate": stats["success"] / stats["total"] if stats["total"] else 0.0,
        }
        for (model, temp), stats in combo_stats.items()
    }
    level_report = {}
    for (combo_key, level), stats in level_stats.items():
        label = f"{combo_key[0]}@{combo_key[1]}"
        level_entry = level_report.setdefault(label, {})
        level_entry[f"level_{level}"] = {
            "success": stats["success"],
            "total": stats["total"],
            "rate": stats["success"] / stats["total"] if stats["total"] else 0.0,
        }

    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps({"overall": combo_report, "by_level": level_report}, indent=2))
    print(f"Aggregate summary saved to {aggregate_path}")

    save_plots(combo_stats, output_dir)


def main():
    args = parse_args()
    temperatures = args.temperatures if args.temperatures else [args.temperature]
    model_config_map = load_model_config_map(args.model_configs)
    agent_overrides = load_agent_overrides(args.agent_overrides)
    run_suite(
        models=args.models,
        base_config_path=args.base_config,
        recipe_dir=args.recipe_dir,
        output_dir=args.output_dir,
        temperatures=temperatures,
        repeats=max(1, args.repeats),
        max_workers=max(1, args.max_workers),
        model_config_map=model_config_map,
        agent_overrides=agent_overrides,
    )


if __name__ == "__main__":
    main()
