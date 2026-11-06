# runs/think_judge：Think 字段训练的轨迹与日志导读

如果想了解 **Think 字段训练的难点在哪里**，按下面的顺序看。本文中的路径都相对 `~/Collab-Overcooked-private/runs/think_judge/`。

> 2026-10-26 删除了一批有 bug、训练时用了 held-out 菜、或中途崩溃的实验，清单和原因见 `DELETED_20261026.txt`。
> 如果别处引用的目录（例如 `rl/l2_none`、`rl/l2_sum_w0.5`）不存在了，就是这个原因。

## 背景

- **任务**：Collab-Overcooked 双人协作做菜。Qwen3-1.7B 的 Chef 和 Assistant 各用一个 LoRA。
- **输出格式**：每次调用都输出 `Think:`（推理）、`Recent Goal:`（多步计划）、`Action:`（规范动作）三段。
- **训练流程**：从 SFT `sft_think_l12/` 开始，在 L2 上做 RL。只用 3 道训练菜：baked_pumpkin_slices、boiled_green_bean_slices、boiled_potato_slices。
  另外 2 道菜（boiled_corn_slices、baked_potato_slices）只在评测时用（held-out）。
- **思考奖励**：给 Think / Recent Goal 打分，再加到 RL 的优势上。目前有两类：
  - LLM 评委直接打分（`l2t_*`）；
  - 规则核验（`l2r_*`），设计见 `../../docs/rule_rewards.md`。

## 1. 最先看：逐条标好对错的思考记录

**`eval/rule_rewards/offline.jsonl`**，每行是一次调用。思考里的每条说法都已经用规则判了对错，最直观。

| 字段 | 含义 |
|---|---|
| `label` | 来自哪个模型：`early`（≈SFT，成功率 0）、`newsum`（0.88）、`newgated`（0.62）；`none` / `oldsum` 的 checkpoint 已删，数据仅供参考 |
| `observation` | 模型当时看到的真实状态（Kitchen states / Agent State / 对话） |
| `response` | 模型写的 Think、Recent Goal、Action 原文 |
| `extraction` | 从思考里抽出的规范说法（LLM 只负责翻译，不判断对错） |
| `detail.F` / `detail.S` | 关于自己和厨房的说法 / 关于队友的说法，每条附规则判定的对错 |
| `detail.keys` / `detail.covered` | 下一步需要的关键事实 / 思考里正确说中了哪些 |
| `detail.plan_steps` / `detail.remaining` | 模型的计划 / 参考计划里每个角色剩下的步骤 |
| `detail.requests` | 向队友发出的请求，以及三项检查（队友够不够得到、是否在队友计划里、5 步内是否执行） |
| `ground` / `mate` / `plan` | 三个规则奖励：第一视角、队友信度、长程规划 |
| `judge_grounding` | LLM 评委给的 grounded 分（用来和规则判定对照） |
| `success` | 这一局最后有没有成功 |

按某个奖励从低到高排序，就能看到典型错误：

```python
import json
rows = [json.loads(l) for l in open("eval/rule_rewards/offline.jsonl")]
worst = sorted((r for r in rows if r["label"] == "newsum" and r["plan"] is not None), key=lambda r: r["plan"])[:10]
for r in worst:
    o = r["observation"]
    print(o[o.rfind("Kitchen states:"):o.find("Past conversation")])   # 当时的真实状态
    print(r["response"])                                               # 模型的 think
    print("F:", r["detail"]["F"])                                      # 抽出的说法及对错
    print("plan:", r["detail"]["plan_steps"], "vs", r["detail"]["remaining"])
```

## 2. 再看：训练过程中 Think 是怎么变的

**`rl/<实验>/rollouts/uNNNNN.jsonl.gz`**，每轮一个文件，记录这一轮每次调用的完整内容和奖励。
文件末尾的 `{"episode_summary": ...}` 行记录每局是否成功、用了几步。

| 字段 | 含义 |
|---|---|
| `observation` / `response` | 输入状态 / 模型输出 |
| `validations` | 动作有没有被环境拒绝 |
| `ites` | 这一步有没有推进参考进度（>0 表示推进了） |
| `judge_score` / `judge_dims` / `judge_analysis` | LLM 评委的分数、grounded 分、一句话理由（`l2t_*`） |
| `rules` / `rule_detail` | 三个规则奖励和判定细节（`l2r_*`） |
| `adv_think` | 这次思考最终拿到的优势 |

推荐对比同一个实验的 `u00001` 和 `u00050`：

| 实验 | 思考奖励 | L2 成功率（u50） |
|---|---|---|
| `rl/l2t_sum_matchfloor_g1` | LLM 评委总分 + grounded 分，作用于全部 token | 0.87 |
| `rl/l2t_gated_matchfloor_g1` | 同上，只作用于推理 token，动作出错的调用拿不到奖励 | — |
| `rl/l2r_{none,ground,mate,plan,full,mate_v2}` | 规则奖励消融（2026-10-26 开跑；`mate_v2` 为修复版队友奖励，10-27 开跑） | 进行中 |
| `rl/l2r_full_v2_split` | ground + mate_v2 + plan，只作用于推理 token（10-27 开跑，替代旧 `l2r_full`） | 进行中 |
| `rl/l2r_ground_split` | ground 只作用于推理 token（10-27 开跑；sum 版 `l2r_ground` 会压制 Chef 做动作，见 `docs/rule_rewards.md` 4.1） | 进行中 |
| `rl/l2s_s0` | reward shaping S0（`../../docs/reward_shaping.md`）：无思考奖励 + `intra_gamma 0.5` + 重复消息 −0.1 + `advantage_norm role`（10-28 开跑） | 见 reward_shaping.md 第 6 节 |
| `rl/l2s_s0a` | S0a：无思考奖励 + 只开 `advantage_norm role` | 同上 |
| `rl/l2s_s1` | S1：S0 + 消息采纳奖励（`collab_mode uptake`：队友 5 步内执行了被请求的动作才给 0.5 × 该动作 ITES） | 同上 |
| `rl/l2s_s2_rule` | S2：S1 + 规则思考奖励（ground+mate_v2+plan 取平均，split，`scale ratio 0.25`，`center role_type`） | 同上 |
| `rl/l2s_s2_llm` | S2：S1 + LLM 评委思考奖励（terra，grounding 1，split，`scale ratio 0.25`，`center role_type`） | 同上 |

其他文件：
- **`rl/<实验>/metrics.jsonl`**：每轮一行，记录训练曲线。常用字段：
  - `success_rate`：成功率；
  - `judge/score_*`、`rules/*`：思考奖励；
  - `think/reasoning_tokens`：思考长度；
  - `update/kl`：KL。

  画多组对比图用 `plot_rl_compare.py`。
- **`rl/<实验>/judge_annotations/rank0.jsonl`**：每次请求发给评委（或抽取模型）的完整 prompt 和原始回复。用来看它为什么这样打分。文件较大（约 9MB）。
- **`rl/<实验>/config.yaml`**：这个实验的完整配置。

## 3. 训练好的模型在各关卡上的表现

**`eval/all_levels/<模型>_l<关卡>.jsonl.episodes.jsonl`**，每局的完整逐步对话（`calls` 里是每一步的 response）。
- `.jsonl` 是汇总的成功率，`.vllm.log` 是服务日志，一般不用看。
- 例如 `think_l2t_sum_matchfloor_g1_u50_l3.jsonl.episodes.jsonl`：新 sum 在没训练过的 L3 上怎么失败。

## 4. 模型"知不知道"当前状态：探针

用规则出题，题目接在模型真实的 prompt 后面问它；选项会轮换，每一遍都答对才算对。分三类：第一视角 F、对队友的建模 S、多步规划 P。
- **`eval/probes/probe_l2v3_{direct,think,delta}_{chef,assistant}.png` / `.txt`**：图和表。`direct` 是直接作答，`think` 是先思考再作答，`delta` 是两者之差。
- **`eval/probes/scores_l2v3_*.jsonl`**：每道题的作答。思考模式下，`reasoning` 字段保留了模型的推理。
- 出题脚本是 `build_probes.py`，评测脚本是 `probe_eval.py`，题库是 `eval/probes/probes_l2_v3.jsonl`。

## 5. 目前已知的难点（可以带着这些去核对）

1. **Think 很少说出下一步需要的前提状态。** 关键事实覆盖率 Chef 只有 0.08–0.14，Assistant 只有 0.17–0.34。很多 think 只是在复述菜谱。
2. **想对和做对不是一回事。** 新 gated 的状态说法出错率最低（6–9%），但成功率只有 0.62；新 sum 成功率最高（0.88），出错率却有 19%。
3. **Assistant 看不到菜谱。** 它的计划经常是过时的，或者把步骤分给了错误的角色，规划奖励普遍为负。
4. **LLM 评委的 grounded 分基本不可靠。** 它和规则判定的相关系数只有 0.16。加了 grounded 奖励后思考变短了，说错的比例却没有下降。
5. **RL 主要改变"怎么行动"，对"知不知道状态"影响很小。** 成功率大幅提升，但第一视角探针基本不变，Assistant 的第一视角甚至比基座还差。
6. **规划和成功率最相关。** 按局计算的相关系数：规划 0.59，队友 0.28，第一视角 0.25。

## 其他

- **启动脚本**：
  - 训练排队：`launch_when_free.sh`，排队记录在 `launch_when_free.log`；
  - 规则奖励消融：`rl_l2_rules.sh`；
  - reward shaping（`l2s_*`）：`rl_l2_shaping.sh`；成功率 / 重复通信 / 每步调用数图：`plot_success_repeat.py l2s` →
    `eval/rule_rewards/success_repeat_l2s.png`；`l2s_*` 的 `metrics.jsonl` 每轮记录 `comm/*`（重复占比、每角色每步调用数、
    各角色消息 / wait / 动作占比、采纳率）；
  - 全关卡评测：`eval_all_levels.sh`。
- **GPU**：不要用 GPU 0/1，排除列表在 `gpu_exclude`；`free_gpu.sh` 会选剩余显存最多的卡。
