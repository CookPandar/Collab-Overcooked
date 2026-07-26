# ICML 2026 LLM-MAS / MARL Method and Benchmark Research

Research date: 2026-07-26

## Key Takeaway

Strictly under ICML 2026, the item that most directly matches "LLM multi-agent + reinforcement learning training" is **MAS-Orchestra**. It trains an LLM orchestrator policy with **GRPO** and evaluates on a new controlled benchmark, **MASBench**, plus public reasoning/search benchmarks.

Several other ICML 2026 papers are highly relevant to LLM multi-agent design, but they are not distributed model-parameter RL training:

- **MASPO**: joint/co-adaptive prompt optimization with evolutionary beam search and joint reward evaluation.
- **MASPOB**: bandit-based prompt-combination search with a GNN/GAT surrogate and LinUCB.
- **OMAC**: holistic supervised optimization of agent functions and collaboration structures.
- **ProtocolBench**: communication protocol benchmark/router, not a training method.

For "distributed LLM-based MARL" specifically, **FlexMARL** is the closest systems paper, but its arXiv source uses ACM SIGCOMM 2026 metadata rather than ICML 2026. **AdvEvo-MARL** is also strong for co-evolutionary attacker/defender MARL, but its source uses an ICLR 2026 template and is not ICML-confirmed.

## Main Table

| Item | ICML 2026 status | Collaborative training? | Algorithm / method | Benchmarks | Code | Device / batch notes |
|---|---:|---|---|---|---|---|
| MAS-Orchestra | Confirmed ICML 2026 | Partly: generated MAS collaborates, but training primarily updates the orchestrator; sub-agents are fixed. Includes separate vs combined training over MASBench axes. | Function-calling, one-shot holistic orchestration; **GRPO** on orchestrator policy; deterministic parser executes sub-agent graph. | **MASBench**; public: AIME24, AIME25, GPQA, HotpotQA, BrowseComp+. | https://github.com/SalesforceAIResearch/MAS-Orchestra | **8 x H200 141GB**, `train_batch_size=64`, `ppo_mini_batch_size=256`, micro per GPU LLM/RLM `2/1`, rollout/group size `32`, max concurrency `128`, verl. |
| MASPO | Confirmed ICML 2026 | Yes for prompts: co-adaptive/joint prompt optimization; not LLM weight RL. | Topological coordinate ascent, joint reward with local/lookahead/global scores, misalignment-guided evolutionary beam search, beam refresh. | MATH-500, AGIEval-MATH, AQuA, GPQA-Diamond, MBPP, HumanEval-ET. | https://github.com/wangzx1219/MASPO | API/inference style; Qwen3-8B MAS backbone, Gemini-2.5-Pro optimizer/evaluator; sample pool `50`, mini-batch `10`, beam `K=2`, candidates `K_sub=2`, rounds `T=3`, epochs/rounds `D=3`. |
| MASPOB | ICML 2026 Spotlight | Joint prompt-combination optimization; not co-training LLM weights. | Contextual bandit, **LinUCB**, topology-aware **GAT/GNN** surrogate, coordinate ascent. | HotpotQA, DROP, HumanEval, MBPP, GSM8K, MATH; complex MAS tests on HotpotQA/DROP/HumanEval. | https://github.com/HZ1008/MASPOB | API + small surrogate training; OpenAI-compatible API, PyTorch 2.0+; validation budget `50`, `max_rounds=45`, `pretrain_rounds=5`, `num_prompts=20`; no GPU batch size specified. |
| OMAC | ICML 2026 Oral | Iterative multi-dimensional MAS optimization; not RL or LLM weight co-training. | Semantic Initializer + Contrastive Comparator; optimizes Fun-1/Fun-2/Str-1/Str-2/Str-3 dimensions. | HumanEval, MMLU, MATH; appendix MBPP and GAIA. | https://github.com/xiwenchao/OMAC | API-based; GPT-3.5-Turbo-1106 default; initial collection size `3`, contrastive iterations `3`; HumanEval training about `1,400` API calls per dimension. |
| ProtocolBench / ProtocolRouter | Confirmed ICML 2026 | N/A; benchmark/router, not RL training. | Protocol comparison and constraint-aware ProtocolRouter over A2A/ACP/ANP/Agora. | GAIA, Safety Tech, Streaming Queue, Fail-Storm Recovery; axes: success/quality, latency/throughput, byte overhead, robustness. | https://github.com/ulab-uiuc/AgentProtocols | No RL batch size. Runtime scenarios include Streaming Queue with 1 coordinator + 4 workers and 1,000 MS MARCO entries; Fail-Storm uses 8 agents, killing 3 every 120s. |
| FlexMARL | Not ICML-confirmed; ACM SIGCOMM 2026 metadata | Yes: multi-agent policies are independently optimized from collaborative trajectories. | Distributed LLM-MARL systems framework; disaggregated rollout/training, experience store, micro-batch async pipeline, hierarchical load balancing, on-demand hardware binding; **GRPO**. | Private industrial Merchant Assistant and Category Assistant datasets. | No public repo found in inspected source. | **48 nodes x 16 NPUs, 64GB each**; Qwen2.5-14B/32B; batch `64`, micro batch `16`, inter-query parallelism `4`, intra-query `16`, max response `8192`. |
| AdvEvo-MARL | Not ICML-confirmed; ICLR 2026 template in source | Yes: adversarial co-evolution of attackers and defenders. | Co-evolutionary MARL; public/group baseline advantage; script uses `advantage=reinforce`, ZeRO-3, KL loss. | Chain-safety / attack-defense setting; repo includes MATH data, seed attacks, safety eval attacks. | https://github.com/PzySeere/AdvEvo-MARL | Script starts Ray with `8` GPUs; `train_batch_size=64`, `micro_train_batch_size=8`, rollout batch `16`, `n_samples_per_prompt=4`, prompt max `8000`, generation max `2048`. |

## Source Links

- MAS-Orchestra paper: https://arxiv.org/abs/2601.14652
- MAS-Orchestra code: https://github.com/SalesforceAIResearch/MAS-Orchestra
- MASBench dataset: https://huggingface.co/datasets/Salesforce/MASBench
- MASPO paper: https://arxiv.org/abs/2605.06623
- MASPO code: https://github.com/wangzx1219/MASPO
- MASPOB paper: https://arxiv.org/abs/2603.02630
- MASPOB code: https://github.com/HZ1008/MASPOB
- OMAC paper: https://arxiv.org/abs/2505.11765
- OMAC code: https://github.com/xiwenchao/OMAC
- OMAC ICML page: https://icml.cc/virtual/2026/oral/71174
- ProtocolBench paper: https://arxiv.org/abs/2510.17149
- ProtocolBench code/artifacts: https://github.com/ulab-uiuc/AgentProtocols
- FlexMARL paper: https://arxiv.org/abs/2602.09578
- AdvEvo-MARL paper: https://arxiv.org/abs/2510.01586
- AdvEvo-MARL code: https://github.com/PzySeere/AdvEvo-MARL
