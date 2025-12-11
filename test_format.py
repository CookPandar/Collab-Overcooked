#!/usr/bin/env python3
"""测试通信历史格式化"""

def format_communication_history(comm_turns, current_agent_id):
    """格式化通信历史"""
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


# Agent 1 (Assistant) 的通信记录
agent1_turns = [
    "Chef, I need your guidance for the order 'baked_bell_pepper'. Can you tell me which ingredient to pick up from the ingredient_dispenser?",
    "Assistant, please pick up the bell pepper from the ingredient_dispenser.",
    "Chef, I'm holding the bell pepper. What's the next step?",
]

print("Agent 1 (Assistant) 的通信历史:")
print(format_communication_history(agent1_turns, 1))
print()

# Agent 0 (Chef) 的通信记录应该是怎样的?
# 如果两个智能体记录的是同一段对话,那么Chef的记录应该是:
agent0_turns = [
    "Assistant, please pick up the bell pepper from the ingredient_dispenser.",
    "Chef, I need your guidance for the order 'baked_bell_pepper'. Can you tell me which ingredient to pick up from the ingredient_dispenser?",
    "Please place the bell pepper on the counter.",
]

print("Agent 0 (Chef) 的通信历史:")
print(format_communication_history(agent0_turns[:2], 0))
