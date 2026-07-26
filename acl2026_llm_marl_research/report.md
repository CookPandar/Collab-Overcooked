# ACL 2026 LLM 多智能体分布式强化学习调研

范围：以 ACL 2026 Main/Findings 已公开论文为主。ACL 2026 举办时间为 2026-07-02 至 2026-07-07；本报告截至 2026-07-26。重点字段：是否协同训练、代码仓库、设备/batch size、算法、benchmark。

## 一句话结论

最贴近“LLM 多智能体 + 协同/分布式 RL 训练”的是 MHGPO、MARS2、Mixture-of-Minds、MARCH、Graph-GRPO 和 Privacy-R1。DITS 更像 MAS self-training 数据选择框架，不是 on-policy MARL；Hail to the Thief 是 decentralized GRPO 的安全/鲁棒性参考。benchmark 方面，SILO-BENCH 和 MAS-BENCH/Distributed Sorting 最适合检验分布式协作本身，TAMAS/PAC-BENCH/TraceElephant 分别补安全、隐私、失败归因。

## 核心方法对比

| Paper | 是否协同训练 | 方法/算法 | 使用 benchmark | 代码 | 设备与 batch size |
|---|---|---|---|---|---|
| MHGPO, ACL Long | 是。优化 LLM-driven MAS 内部 agent 输出，按下游依赖回传 reward。 | Heterogeneous-group-based RL；GRPO-style critic-free policy gradient；FoF/RR/IS group rollout；对比 MAPPO。 | HotpotQA, 2WikiMultiHopQA, MuSiQue；EM/F1/Accuracy。 | 论文未报告官方 repo。 | 8 x H100 80GB；batch size 512；group size 4；1 epoch/176 steps。 |
| MARS2, ACL Long | 是。多个 independently optimized policies 在 shared tree search environment 协同。 | GRPO-style RL + MCTS；path-level group advantage；tree-consistent reward shaping。 | RL 训练用 DeepCoder code prompts；评测 LiveCodeBench v6，附 MATH。 | https://github.com/TsinghuaC3I/MARTI | 论文：reference/actor 各 8 GPUs per model，vLLM engines 8 per model；train batch 256，rollout batch 512。repo 说明多 agent 常用 3 nodes x 8 H200。 |
| Mixture-of-Minds, ACL Long | 是。planning/coding/answering agents 逐步训练并用成功轨迹回溯。 | MCTS 生成 pseudo-gold trajectories；GRPO；plan/code/answer 分阶段结构化 reward。 | 训练 TableInstruct；评测 TableBench，OOD FinQA。 | https://github.com/Tonyzhou98/mixture-of-minds | global batch size 256；rollout temp 1.0；约 100 update steps；未报告 GPU 型号。 |
| Privacy-R1, ACL Long | 部分。训练 delegation/privacy policy，LLM 本体基本冻结。 | PPO；policy agent 决定 local/remote LLM 委托；reward 平衡质量与 PII 泄露。 | PUPA；新构造 Med-PCD。 | https://github.com/zackhuiiiii/Privacy-R1 | NVIDIA H200 GPUs；SFT batch 32；PPO batch 64；max 256 steps。 |
| MARCH, ACL Long | 是。Solver/Proposer/Checker 形成信息不对称协作 pipeline，joint Solver+Checker 更新。 | PPO + verifiable rewards；zero-tolerance reward；VerL + FSDP + vLLM。 | 训练 BioASQ, 2WikiMultiHopQA, MuSiQue；评测 RAGTruth, FaithBench, ContextualJudgeBench, Facts Grounding, HotpotQA/2Wiki/MuSiQue。 | https://github.com/Qwen-Applications/MARCH | multi-node multi-GPU，但未给 GPU 型号/数量；global batch/mini-batch 32；rollout number 8。 |
| Graph-GRPO, Findings | 部分。训练通信 topology controller，不训练 LLM agents。 | Edge-level Graph-GRPO；每 query 采样 K 个通信图并做相对优势。 | MMLU, GSM8K, MultiArith, SVAMP, AQUA, HumanEval。 | 未报告官方 repo。 | NVIDIA A100 GPUs；group K=16；lr 1e-4；未报告 batch size。 |
| DITS, ACL Long | 不是典型协同 RL。MAS 数据合成 + self-training，优化训练样本而非 on-policy agent 交互。 | MCTS + influence score；iSFT-DPO；LoRA 单步梯度估计 influence。 | HotpotQA, 2WikiMultiHopQA, TriviaQA, CBT, ARC-C, MMLU, WebWalker。 | https://github.com/swt-user/DITS | SFT batch 32/16；DPO batch 64；报告 GPU-hour 成本但未给 GPU 型号。 |
| Hail to the Thief, Findings | 是分布式 GRPO，但目标是攻击/防御，不是协作任务求解。 | Decentralised GRPO；horizontal/vertical dRL；恶意 completion poisoning；logit/LLM-judge filtering defense。 | GSM8K, OpenMathInstruct；math/code attacks。 | https://github.com/gensyn-ai/HTTT | H100 + InfiniBand；batch 32 prompts；12 generations/prompt；4 models，25% malicious。 |

## 次级但有参考价值

| Paper | 价值 | 算法/设置 |
|---|---|---|
| NeuralFSM, ACL Long | 学有限状态执行策略与消息权重，适合作为“训练 coordinator 而非 LLM policy”的 baseline。 | Policy-gradient style controller；API LLMs；GSM8K/MATH/HumanEval/MBPP/GPQA/HotpotQA/ALFWorld/GAIA；代码 https://github.com/DisseverYOLO/NeuralFSM。 |
| GT-PMARL, Findings | 人机科学团队任务分配，可借鉴 opportunity cost 和 Nash-Pareto 协同目标。 | Diversity-GRPO + lower-layer MARL；A800-80GB；3 actor agents；2000 epochs；未给 repo。 |
| MotifAgent, Findings | 化学垂域 CTDE/MAPPO，多 agent shared policy + centralized critic。 | MAPPO；8 x A100 80GB；transition batch 256；MoleculeSTM/PubChem, ChEBI-20, MoleculeNet, Mol-Instructions；未给 repo。 |

## Benchmark 选择建议

| Benchmark | 适合测什么 | 代码/资源 |
|---|---|---|
| SILO-BENCH, ACL Long | 分布式信息孤岛、通信复杂度、scale 到多 agent 后是否还能合成全局状态。强烈推荐。 | https://github.com/jwyjohn/acl26-silo-bench；论文报告 GH200 cluster，500+ GPU-hours equivalent per experiment。 |
| MAS-BENCH / Distributed Sorting, Findings | 低语义噪声的分布式排序，专测协调而非知识。适合做 scaling stress test。 | 论文称代码/数据在 GitHub 发布；典型 run 为 4 nodes x 4 GH200 GPUs。 |
| TAMAS, ACL Long | 多智能体安全，含 direct/indirect prompt injection、impersonation、Byzantine、colluding、contradicting agents。 | https://github.com/microsoft/TAMAS |
| PAC-BENCH, Findings | 隐私约束下的双 agent 协作，分离 task score 与 privacy score。 | https://github.com/PAC-Bench/PAC-Bench |
| TraceElephant, ACL Long | 失败归因：哪个 agent、哪一步导致失败。适合做训练后的诊断集。 | https://github.com/TraceElephant/TraceElephant |

## 对方法设计的直接启发

1. 如果目标是 ACL 2026 风格的“LLM-MAS 强化学习”，不要只做 final reward GRPO。更强的叙事是 credit assignment：agent 输出、通信边、tree path、状态转移或隐私约束都要能拿到可解释 reward。
2. benchmark 不应只用 HotpotQA/GSM8K。这些更像能力评测，不足以证明协作。建议主 benchmark 用 Overcooked/SILO-style 分布式协作任务，再辅以 HotpotQA/MuSiQue 或 code/table reasoning 做通用性。
3. 设备设置上，核心 ACL 方法普遍很重：H100/H200/A100 多卡，batch 256-512 很常见。若要复现实验，建议同时报告 low-resource 配置，比如 3B/7B LoRA + batch 32/64 + vLLM rollout。
4. 协同训练需要明确：是多个 LLM policy 一起训，还是只训 router/topology/controller。Privacy-R1 和 Graph-GRPO 的成本低很多，但论文叙事上不能说成“多 agent policy co-training”。

## Sources

- MHGPO: https://aclanthology.org/2026.acl-long.1399.pdf
- MARS2: https://aclanthology.org/2026.acl-long.1538.pdf and https://github.com/TsinghuaC3I/MARTI
- Mixture-of-Minds: https://aclanthology.org/2026.acl-long.112.pdf and https://github.com/Tonyzhou98/mixture-of-minds
- Privacy-R1: https://aclanthology.org/2026.acl-long.2130.pdf and https://github.com/zackhuiiiii/Privacy-R1
- MARCH: https://aclanthology.org/2026.acl-long.1828.pdf and https://github.com/Qwen-Applications/MARCH
- Graph-GRPO: https://aclanthology.org/2026.findings-acl.1010.pdf
- DITS: https://aclanthology.org/2026.acl-long.296.pdf and https://github.com/swt-user/DITS
- Hail to the Thief: https://aclanthology.org/2026.findings-acl.1950.pdf and https://github.com/gensyn-ai/HTTT
- NeuralFSM: https://aclanthology.org/2026.acl-long.1543.pdf and https://github.com/DisseverYOLO/NeuralFSM
- GT-PMARL: https://aclanthology.org/2026.findings-acl.1920.pdf
- MotifAgent: https://aclanthology.org/2026.findings-acl.2023.pdf
- SILO-BENCH: https://aclanthology.org/2026.acl-long.1354.pdf and https://github.com/jwyjohn/acl26-silo-bench
- Distributed Sorting: https://aclanthology.org/2026.findings-acl.1698.pdf
- TAMAS: https://aclanthology.org/2026.acl-long.1442.pdf and https://github.com/microsoft/TAMAS
- PAC-BENCH: https://aclanthology.org/2026.findings-acl.1552.pdf and https://github.com/PAC-Bench/PAC-Bench
- TraceElephant: https://aclanthology.org/2026.acl-long.912.pdf and https://github.com/TraceElephant/TraceElephant
