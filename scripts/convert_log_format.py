#!/usr/bin/env python3
"""
日志格式转换脚本
将Overcooked实验日志转换为训练所需的输入-输出格式
"""

import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any


def clean_observation(observation: str) -> str:
    """
    清理observation中不需要的内容
    删除action history和lessons learned部分
    """
    # 删除 "Your successful action history in the past steps are: ..." 这一行
    # 匹配到下一个换行符之前的所有内容(包括可能的列表)
    observation = re.sub(
        r'Your successful action history in the past steps are:.*?\n(?=\w|\n)',
        '',
        observation,
        flags=re.DOTALL
    )

    # 删除 "Here are some lessons you have learned from past failures..." 这一行
    observation = re.sub(
        r'Here are some lessons you have learned from past failures that you can use to make the right decisions:.*?\n(?=\w|\n)',
        '',
        observation,
        flags=re.DOTALL
    )

    # 清理多余的空行
    observation = re.sub(r'\n\n+', '\n', observation)
    observation = observation.strip()

    return observation


def format_communication_history(comm_turns: List[str], current_agent_id: int) -> str:
    """
    格式化通信历史

    Args:
        comm_turns: 通信turn列表
        current_agent_id: 当前智能体ID (0=Chef, 1=Assistant)

    通信turn的结构:
    - 偶数索引(0,2,4...): 当前智能体自己说的
    - 奇数索引(1,3,5...): 队友说的
    """
    if not comm_turns:
        return ""

    current_agent_name = "Chef" if current_agent_id == 0 else "Assistant"
    teammate_name = "Assistant" if current_agent_id == 0 else "Chef"

    comm_text = []
    for i, turn in enumerate(comm_turns):
        if i % 2 == 0:  # 偶数索引,是自己说的
            comm_text.append(f"{current_agent_name}:{turn}")
        else:  # 奇数索引,是队友说的
            comm_text.append(f"{teammate_name}:{turn}")

    return "\n".join(comm_text)


def build_history_context(all_timesteps: List[Dict], current_idx: int, agent_id: int) -> str:
    """
    构建历史上下文:最近两次有效调用的完整输入输出

    向前回溯时间步,只保留每个时间步最后一次成功的智能体调用
    """
    history_parts = []

    # 向前回溯,直到找到两个非空历史
    history_entries = []
    for hist_idx in range(current_idx - 1, -1, -1):
        if len(history_entries) >= 2:
            break

        hist_data = all_timesteps[hist_idx]
        hist_timestamp = hist_data.get("timestamp", hist_idx)

        # 直接从content.content中获取该智能体的最后一次调用
        hist_content_obj = hist_data.get("content", {})
        hist_observations = hist_content_obj.get("observation", [])
        hist_content_lists = hist_content_obj.get("content", [])
        hist_communications = hist_data.get("statistical_data", {}).get("communication", [])

        # 获取基础observation
        base_obs = ""
        if len(hist_observations) > agent_id and hist_observations[agent_id]:
            base_obs = clean_observation(hist_observations[agent_id])

        # 在所有call block中查找最后一次该智能体的完整输入输出
        last_history_entry = None
        for call_idx, call_block in enumerate(hist_content_lists):
            if not call_block:
                continue

            call_comm_history = []
            call_turns = []
            if call_idx < len(hist_communications):
                call_turns = hist_communications[call_idx].get("turn", []) or []

            for entry_idx, call in enumerate(call_block):
                entry_agent = call.get("agent")
                if entry_agent is None:
                    continue

                # 构建该条之前的通信历史
                if entry_agent == agent_id:
                    full_observation = base_obs
                    if call_comm_history:
                        comm_text = "\n".join(call_comm_history)
                        full_observation = f"{full_observation}\ncommunication history:\n{comm_text}"

                    hist_str = f"timestep {hist_timestamp}:{full_observation}\n"
                    if call.get('analysis'):
                        hist_str += f"analysis: {call['analysis']}\n"
                    if call.get('plan'):
                        hist_str += f"plan: {call['plan']}\n"
                    if call.get('say'):
                        hist_str += f"say: {call['say']}"

                    last_history_entry = hist_str

                # 更新通信历史
                history_text = call.get("say", "")
                if (not history_text or history_text == "[NOTHING]") and call_turns and entry_idx < len(call_turns):
                    history_text = call_turns[entry_idx]

                if history_text and history_text != "[NOTHING]":
                    speaker = "Chef" if entry_agent == 0 else "Assistant"
                    call_comm_history.append(f"{speaker}:{history_text}")

        if last_history_entry:
            history_entries.append(last_history_entry)

    # 保持时间顺序(旧 -> 新)
    history_entries.reverse()
    return "\n\n".join(history_entries)


def convert_log_to_training_format(input_file: str, output_file: str):
    """
    转换日志格式,自动处理两个智能体

    Args:
        input_file: 输入的JSON日志文件路径
        output_file: 输出的JSON文件路径
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    content_list = log_data.get("content", [])

    # 存储所有转换后的样本
    training_samples = []

    last_observations = {0: "", 1: ""}

    # 遍历每个时间步
    for timestep_idx, timestep_data in enumerate(content_list):
        timestamp = timestep_data.get("timestamp", timestep_idx)

        content_obj = timestep_data.get("content", {})
        observations = content_obj.get("observation", [])
        content_lists = content_obj.get("content", [])

        # 获取两个智能体的基础observation
        base_obs = {
            0: last_observations[0],
            1: last_observations[1]
        }
        if observations:
            if len(observations) > 0 and observations[0]:
                cleaned = clean_observation(observations[0])
                base_obs[0] = cleaned
                last_observations[0] = cleaned
            if len(observations) > 1 and observations[1]:
                cleaned = clean_observation(observations[1])
                base_obs[1] = cleaned
                last_observations[1] = cleaned

        base_obs_0 = base_obs[0]
        base_obs_1 = base_obs[1]

        # 构建两个智能体的历史上下文
        history_context_0 = build_history_context(content_list, timestep_idx, 0)
        history_context_1 = build_history_context(content_list, timestep_idx, 1)

        agent_configs = {
            0: {
                "agent_name": "Chef",
                "base_observation": base_obs_0,
                "history": history_context_0
            },
            1: {
                "agent_name": "Assistant",
                "base_observation": base_obs_1,
                "history": history_context_1
            }
        }

        communications = timestep_data.get("statistical_data", {}).get("communication", [])

        # 逐个对话(call)处理,确保顺序与communication一致
        for call_idx, call_entries in enumerate(content_lists):
            if not call_entries:
                continue

            call_comm_history = []
            call_turns = []
            if call_idx < len(communications):
                call_turns = communications[call_idx].get("turn", []) or []

            for entry_idx, call in enumerate(call_entries):
                agent_id = call.get("agent")
                if agent_id not in agent_configs:
                    continue

                agent_info = agent_configs[agent_id]
                agent_name = agent_info["agent_name"]
                base_observation = agent_info["base_observation"]

                # 构建完整observation: base + 当前对话历史
                full_observation = base_observation
                if call_comm_history:
                    comm_text = "\n".join(call_comm_history)
                    full_observation = f"{full_observation}\ncommunication history:\n{comm_text}"

                # 构建输入
                input_parts = []
                if agent_info["history"]:
                    input_parts.append(agent_info["history"])
                input_parts.append(f"timestep {timestamp}:{full_observation}")
                full_input = f"{agent_name}'s input: " + "\n\n".join(input_parts)

                # 构建输出
                analysis = call.get("analysis", "")
                plan = call.get("plan", "")
                say = call.get("say", "")

                output_parts = [
                    f"{agent_name}'s analysis: {analysis}" if analysis else f"{agent_name}'s analysis: ",
                    f"{agent_name}'s plan: {plan}" if plan else f"{agent_name}'s plan: ",
                    f"{agent_name} say: {say}" if say else f"{agent_name} say: "
                ]
                full_output = "\n".join(output_parts)

                training_samples.append({
                    "timestamp": timestamp,
                    "agent": agent_name,
                    "input": full_input,
                    "output": full_output
                })

                # 用say或通信turn更新对话历史,保证交替顺序
                history_text = say
                if (not history_text or history_text == "[NOTHING]") and call_turns and entry_idx < len(call_turns):
                    history_text = call_turns[entry_idx]

                if history_text and history_text != "[NOTHING]":
                    call_comm_history.append(f"{agent_name}:{history_text}")

    # 保存输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_samples, f, ensure_ascii=False, indent=2)

    print(f"转换完成!")
    print(f"- 输入文件: {input_file}")
    print(f"- 输出文件: {output_file}")
    print(f"- 生成样本数: {len(training_samples)}")

    # 统计每个智能体的样本数
    chef_count = sum(1 for s in training_samples if s['agent'] == 'Chef')
    assistant_count = sum(1 for s in training_samples if s['agent'] == 'Assistant')
    print(f"  - Chef样本数: {chef_count}")
    print(f"  - Assistant样本数: {assistant_count}")


def main():
    parser = argparse.ArgumentParser(description='转换Overcooked日志格式')
    parser.add_argument('input', type=str, help='输入的JSON日志文件路径')
    parser.add_argument('output', type=str, help='输出的JSON文件路径')

    args = parser.parse_args()

    convert_log_to_training_format(args.input, args.output)


if __name__ == "__main__":
    main()
