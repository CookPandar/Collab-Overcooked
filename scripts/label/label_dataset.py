#!/usr/bin/env python3
"""
利用LLM对转换后的Collab-Overcooked数据集进行标注
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc]

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = REPO_ROOT / "collab_overcooked" / "prompts"
RECIPE_DIR = PROMPT_ROOT / "recipe"
REFERENCE_DIR = PROMPT_ROOT / "reference"
GPT_DIR = PROMPT_ROOT / "gpt"

LABEL_DEFINITIONS = [
    {
        "category": "规则与环境理解",
        "label": "规则理解错误",
        "feature": "误解游戏/交互规则或约束",
        "role": "-",
        "description": "例如忽略禁令、错误引用手册内容"
    },
    {
        "category": "规则与环境理解",
        "label": "任务理解错误",
        "feature": "弄错当前任务目标，例如本应该是做洋葱汤，结果智能体认为是做土豆泥",
        "role": "-",
        "description": "注意，chef对菜谱的理解错误不属于任务理解错误，而是文档理解错误。"
    },
    {
        "category": "规则与环境理解",
        "label": "文档理解错误",
        "feature": "菜谱是本任务唯一的文档，而只有chef可以接触到菜谱。因此，只有chef对菜谱的理解有错误时，可能属于文档理解错误。",
        "role": "chef",
        "description": "chef对提供的菜谱文档信息解释错误"
    },
    {
        "category": "规则与环境理解",
        "label": "环境状态理解错误",
        "feature": "错误地理解物品/自身状态",
        "role": "-",
        "description": "智能体误判手中物体、厨房器具状态等"
    },
    {
        "category": "规则与环境理解",
        "label": "自身能力理解错误",
        "feature": "对自身可执行动作的能力判断错误",
        "role": "-",
        "description": "只关注智能体对自身能力的判断错误，不关注对其他智能体能力的判断错误。后者属于队友能力理解错误"
    },
    {
        "category": "队友状态与消息理解",
        "label": "忽视队友消息",
        "feature": "无视已有回复或重复询问",
        "role": "-",
        "description": "规划/行动忽略队友反馈。例如队友已经在对话历史中表达了自己的计划，而智能体没有注意到这一点，发起冗余通信。"
    },
    {
        "category": "队友状态与消息理解",
        "label": "队友消息理解错误",
        "feature": "曲解队友消息内容",
        "role": "-",
        "description": "例如误解分工或请求"
    },
    {
        "category": "队友状态与消息理解",
        "label": "队友能力理解错误",
        "feature": "误判队友能做/不能做的事",
        "role": "-",
        "description": "认为队友拥有不存在的能力等"
    },
    {
        "category": "队友状态与消息理解",
        "label": "队友状态理解错误",
        "feature": "忽视或误判队友当前进度",
        "role": "-",
        "description": "不了解队友正在执行的任务"
    },
    {
        "category": "规划与协作",
        "label": "协作时机错误",
        "feature": "不需要协作时仍请求协作",
        "role": "发起方",
        "description": "例如可独立完成却叫队友"
    },
    {
        "category": "规划与协作",
        "label": "发起方规划错误",
        "feature": "协作时机正确但方案本身错误",
        "role": "发起方",
        "description": "未出现规则/理解类问题却规划错误"
    },
    {
        "category": "规划与协作",
        "label": "发起方规划正确",
        "feature": "提出的协作方案无误",
        "role": "发起方",
        "description": "协作必要且规划合理"
    },
    {
        "category": "规划与协作",
        "label": "响应方不遵循正确规划",
        "feature": "理解了正确方案但未执行",
        "role": "响应方",
        "description": "发起方规划正确，响应方违背"
    },
    {
        "category": "规划与协作",
        "label": "响应方不质疑错误规划",
        "feature": "明知规划有错却不提出",
        "role": "响应方",
        "description": "自身能力可识别错误仍沉默"
    },
    {
        "category": "规划与协作",
        "label": "响应方质疑错误规划",
        "feature": "指出发起方方案的问题",
        "role": "响应方",
        "description": "成功发现并提出规划错误"
    },
    {
        "category": "规划与协作",
        "label": "响应方遵循正确规划",
        "feature": "根据正确方案执行",
        "role": "响应方",
        "description": "无偏差地执行合理规划"
    },
    {
        "category": "消息生成",
        "label": "需交互时不交互",
        "feature": "需要沟通却未说话",
        "role": "-",
        "description": "错过必要的信息同步"
    },
    {
        "category": "消息生成",
        "label": "不需交互时交互",
        "feature": "不需要协作却发起通信",
        "role": "-",
        "description": "冗余或打断任务"
    },
    {
        "category": "消息生成",
        "label": "消息生成错误",
        "feature": "消息与规划内容不一致",
        "role": "-",
        "description": "言行不一致或表达错误"
    },
    {
        "category": "消息生成",
        "label": "消息生成正确",
        "feature": "消息必要且准确传达规划",
        "role": "-",
        "description": "通信内容与计划匹配"
    },
    {
        "category": "动作生成",
        "label": "动作选择错误",
        "feature": "Analysis判断正确但action错误",
        "role": "-",
        "description": "动作不符合正确规划"
    },
    {
        "category": "动作生成",
        "label": "动作格式错误",
        "feature": "动作表达不符合格式要求",
        "role": "-",
        "description": "如语法或API格式错误"
    },
    {
        "category": "动作生成",
        "label": "动作生成正确",
        "feature": "无需通信且动作正确",
        "role": "-",
        "description": "规划正确且动作与规划一致"
    },
]


def load_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def infer_task_from_dataset(dataset: List[Dict[str, str]]) -> Optional[str]:
    pattern = re.compile(r"Order:([A-Za-z0-9_]+)")
    for sample in dataset:
        text = sample.get("input", "")
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def find_prompt_file(directory: Path, task_name: str, required_suffix: Optional[str] = None) -> Path:
    candidates: List[Path] = []
    for path in directory.glob("*.txt"):
        stem = path.stem
        if required_suffix:
            if not stem.endswith(required_suffix):
                continue
            stem = stem[: -len(required_suffix)]
        if stem.endswith(task_name):
            candidates.append(path)
    if not candidates:
        for path in directory.glob("*.txt"):
            stem = path.stem
            check_stem = stem
            if required_suffix and stem.endswith(required_suffix):
                check_stem = stem[: -len(required_suffix)]
            if task_name in check_stem:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"未在{directory}中找到与任务 {task_name} 匹配的提示文件")
    return sorted(candidates)[0]


def build_label_table() -> str:
    header = "|分类|标签|特征|角色|说明|"
    sep = "|---|---|---|---|---|"
    rows = [header, sep]
    for item in LABEL_DEFINITIONS:
        rows.append(
            f"|{item['category']}|{item['label']}|{item['feature']}|{item['role']}|{item['description']}|"
        )
    return "\n".join(rows)


def build_rule_context() -> str:
    sections = []
    rule_files = ["environment_rule.txt", "communication_rule.txt"]
    for filename in rule_files:
        path = GPT_DIR / filename
        if path.exists():
            sections.append(f"{filename}:\n{load_text(path)}")
    skill_files = [
        ("Chef技能", GPT_DIR / "chef_skill.txt"),
        ("Assistant技能", GPT_DIR / "assistant_skill.txt"),
    ]
    for title, path in skill_files:
        if path.exists():
            sections.append(f"{title}:\n{load_text(path)}")
    return "\n\n".join(sections)


def build_system_prompt(task_name: str, recipe_text: str, reference_text: str) -> str:
    label_table = build_label_table()
    rule_context = build_rule_context()
    return (
        "你是Collab-Overcooked多智能体实验的高级标注员,需要判断某个时间步的智能体输出是否正确,并给出唯一标签。\n"
        f"以下规则是输入给智能体的prompt，其中'role'代表不同智能体的角色。请你也根据这些规则对智能体行为进行标注:\n"
        "## 游戏规则与技能\n"
        f"{rule_context}\n\n"
        f"## 任务({task_name})菜谱\n{recipe_text}\n\n"
        f"## 任务({task_name})标准答案\n{reference_text}\n\n"
        "## 标签定义\n"
        f"{label_table}\n\n"
        "标注原则:\n"
        "1. 每个时间步要么是行动(plan/action)要么是通信(say),如果输出中`say`字段为有意义内容(且不等于[NOTHING])则视为通信,否则判定动作。\n"
        "2. 如果存在错误,请按照时间步中出现的第一个错误原因打标签,优先级:规则/理解 -> 队友相关 -> 规划与协作 -> 消息/动作生成。\n"
        "3. 如果通信/动作完全正确,分别使用“消息生成正确”或“动作生成正确”标签。\n"
        "4. 回答需使用简体中文。"
    )


def build_user_prompt(sample: Dict[str, str], sample_id: int) -> str:
    return (
        f"样本ID: {sample_id}\n"
        f"Agent: {sample.get('agent', 'Unknown')}\n"
        f"Timestamp: {sample.get('timestamp')}\n"
        "==== 输入 ====\n"
        f"{sample.get('input', '').strip()}\n"
        "==== 输出 ====\n"
        f"{sample.get('output', '').strip()}\n\n"
        "请完成:\n"
        "1. 判断该时间步是通信还是执行动作。\n"
        "2. 结合菜谱、参考答案、规则判断输出是否正确。\n"
        "3. 如有错误,给出最先出现的错误类型,并从标签表中选取匹配项;若完全正确,根据类型选择“消息生成正确”或“动作生成正确”。\n"
        "4. 输出JSON,格式如:{\"label\":\"标签名\",\"reason\":\"一句话中文解释\"}。只返回JSON,不要额外文本。"
    )


def call_llm(
    client: "OpenAI",
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float,
    max_retries: int = 3,
) -> Dict[str, str]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM调用失败: {last_error}")


def determine_output_path(dataset_path: Path, user_arg: Optional[str]) -> Path:
    if user_arg:
        return Path(user_arg)
    label_dir = REPO_ROOT / "assets" / "data" / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    return label_dir / f"{dataset_path.stem}_labeled.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="调用LLM对Collab-Overcooked数据进行标签标注")
    parser.add_argument("dataset", type=str, help="convert_log_format.py生成的JSON数据集")
    parser.add_argument(
        "--output", type=str, default=None, help="输出路径(默认写入assets/data/label目录并追加_labeled)"
    )
    parser.add_argument("--task", type=str, default=None, help="任务名称,默认自动从数据中解析")
    parser.add_argument("--model", type=str, default="azure-gpt-4o", help="用于标注的LLM模型名")
    parser.add_argument("--api-base", type=str, default="https://zhangshuwengpt.fc.chj.cloud/agentops", help="OpenAI兼容接口的base URL")
    parser.add_argument("--api-key", type=str, default="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJHbW9oUjdNTTQ0cGpQTmIwZ2tKTjFIZ1J2bkJkcjdxQSJ9.0xbuBWNX5wKkvLrQTPo5xFMQ1t1-2MNIURnNQ4Q4KQM", help="API Key")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM温度")
    parser.add_argument("--max-samples", type=int, default=None, help="只标注前N条数据")
    parser.add_argument(
        "--overwrite-label",
        action="store_true",
        help="如果样本已存在label字段,也重新覆盖",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印首个样本的提示词,不实际调用LLM",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"找不到输入数据 {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, str]] = json.load(f)

    if args.max_samples is not None:
        dataset = dataset[: args.max_samples]

    task_name = args.task or infer_task_from_dataset(dataset)
    if not task_name:
        raise ValueError("无法从数据中推断任务名,请通过 --task 手动指定")

    recipe_path = find_prompt_file(RECIPE_DIR, task_name)
    reference_path = find_prompt_file(REFERENCE_DIR, task_name, required_suffix="_ref")
    recipe_text = load_text(recipe_path)
    reference_text = load_text(reference_path)
    system_prompt = build_system_prompt(task_name, recipe_text, reference_text)

    output_path = determine_output_path(dataset_path, args.output)

    if args.dry_run:
        preview_prompt = build_user_prompt(dataset[0], 0)
        print("=== System Prompt ===")
        print(system_prompt)
        print("\n=== User Prompt 示例 ===")
        print(preview_prompt)
        print("\n(DRY-RUN 模式,未调用LLM)")
        return

    if not args.api_key:
        raise ValueError("未提供API Key,请通过--api-key或环境变量OPENAI_API_KEY设置")
    if OpenAI is None:
        raise ImportError("未找到openai库,请先`pip install openai`再运行标注脚本")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    labeled_data: List[Dict[str, str]] = []
    for idx, sample in enumerate(dataset):
        if not args.overwrite_label and "label" in sample and sample["label"]:
            labeled_data.append(sample)
            continue

        user_prompt = build_user_prompt(sample, idx)
        label_response = call_llm(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=args.model,
            temperature=args.temperature,
        )
        enriched = dict(sample)
        enriched["label"] = label_response.get("label", "").strip()
        enriched["label_reason"] = label_response.get("reason", "").strip()
        enriched["label_model"] = args.model
        enriched["label_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        labeled_data.append(enriched)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_data, f, ensure_ascii=False, indent=2)

    print(f"标注完成,输出文件: {output_path} (样本数: {len(labeled_data)})")


if __name__ == "__main__":
    main()
