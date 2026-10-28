# 规则可核验的思考奖励：设计说明

代码：`baselines/rl/rule_rewards.py`；接入训练：`baselines/rl/train.py` 的 `think_reward.rules`；
启动脚本：`runs/think_judge/rl_l2_rules.sh`；离线校验：`runs/think_judge/rule_reward_offline.py`。

## 0. 总体思路

训练仍然是原来的游戏任务，不加任何额外题目。模型每次调用照常输出三段：

```
Think: <自由文本推理>
Recent Goal: <自由文本计划，常见为编号步骤列表>
Action: <规范动作，如 put_obj_in_utensil(pot0) 或 Collab(request(Assistant, pickup(potato,ingredient_dispenser)))>
```

我们对每次调用（只取格式正确、有 Think 的调用，与原 LLM 评委相同）计算三个奖励：

| 奖励 | 衡量什么 | 对应探针 |
|---|---|---|
| `ground` 第一视角 | 推理里关于"我自己 / 厨房当前状态"的说法是否正确，且是否说到了下一步需要的关键事实 | F-hold / F-content / F-status / F-space / F-feasible |
| `mate` 队友信度 | 对队友的判断（状态、预期动作）是否正确，向队友提的请求是否靠谱 | S-hold / S-doing / S-profile / S-canreq / S-next |
| `plan` 长程规划 | Recent Goal 的多步计划是否与剩余的参考计划一致、分工是否正确 | P-due / P-mate-due / P-before / P-remaining |

**对错全部由规则判定**，依据是三类信息：当前观测（prompt 里的 Kitchen states / Agent State / spaces）、
菜谱参考动作序列（`load_references`），以及本局真实轨迹（之后实际执行了什么）。

LLM 只负责一件事：把自由文本"翻译"成规范格式的说法（见第 1 节），它不判断对错。

## 1. 抽取：自由文本 → 规范说法

- **抽取模型**：gpt-5.6-sol，`reasoning_effort: low`。每个请求放 8 次调用；全局最多 8 个请求并发。
- **抽取原则**（写在抽取 prompt 中）：
  - 照原文翻译，不纠正、不补全、不评价；
  - 只抽取"当前时刻"被**明确陈述**的事实；
  - 意图和下一步（如"the next step is to put X in pot0"、"I will cut"）不算事实说法，只进入 `plan`。
- **防编造**：每条说法必须附原文片段 `quote`（≤8 词）。规范化空白、大小写后，片段不在原回复里的直接丢弃。
- **说法类型**（P ∈ {Chef, Assistant}）：

| 类型 | 字段 | 例子（原文 → 说法） |
|---|---|---|
| `hold` | player, item | "I am holding potato slices" → hold(Chef, potato_slices) |
| `utensil` | utensil, status∈{empty, ready_to_start, processing, finished}, contents | "The pot is currently empty" → utensil(pot0, empty) |
| `can_operate` | player, utensil, value | "I cannot reach the oven" → can_operate(Assistant, oven0, false) |
| `ready` | player, action, value | 只在原文明说"现在能 / 不能做 A"或"必须等才能做 A"时抽取 |
| `doing` | player, action 或 nothing | "The Assistant is waiting" → doing(Assistant, nothing) |
| `plan` | 有序列表 [(player, action)] | "1. Assistant places the slices on the counter. 2. Chef picks up..." |

## 2. 规则判定：每类说法怎么判对错

- **`hold`**：与观测里 `<P> holds ...` 比较。比较前去掉冠词、one / a、单复数差异；
  "a dish with X" 可以被称为 dish、X 或 dish_with_X。"nothing / empty / none" 视为同一个说法。
- **`utensil`**：与 Kitchen states 解析出的（状态, 内容）比较；给了几个字段就检查几个，全部正确才算对。
  厨具名不在观测里的说法无法核验，直接忽略。
- **`can_operate`**：厨具是否在该角色的 space 列表里。counter 两人都能用，不判。
- **`ready`**：动作现在能不能做，由规则 `next_step_ready` 加可达性共同判定：
  - pickup 要求手空，且物品确实在该位置；
  - put / place / serve 要求手里有东西；
  - 厨具步骤（cut、cook 等）要求厨具里已经备齐原料；
  - 动作用到的厨具必须在该角色的 space 里。

  例外：队友从 counter 拿东西这一类不判，因为当前观测里看不到队友那侧 counter 的状态。
- **`doing`**：与观测里 "current action being executed by P is [...]" 比较；`wait(k)` 视为 nothing。
- **视角归属**：说的是自己或厨房，计入第一视角 F；说的是队友，计入第二视角 S。

## 3. 三个奖励的计算

记号：
- 本次调用的角色记为 me，队友记为 mate；当前时间步为 t。
- **已完成进度**：每个角色已实际执行的厨房动作序列（history）。
  它与参考计划按"顺序匹配的最长前缀"对齐（与 ITES 相同）；有多条参考序列时取匹配最深的一条。
- **剩余步骤 remaining(P)**：P 的参考序列中尚未匹配的部分。
- **due**：me 的下一步参考步骤，即 remaining(me) 的第一个。
- **队友未来动作 future**：队友在 (t, t+5] 时间步内实际执行的厨房动作；窗口 k = 5。

### 3.1 `ground`：第一视角（取值约 [-2, 1]）

```
关键事实 keys = { 我手里拿着什么 }
              ∪ { due 步骤用到的厨具的状态 }
              ∪ { due 步骤现在能不能做 }
covered      = 被"正确的"F 类说法说中的关键事实
wrong        = 错误的 F 类说法数（另见下面的等待规则）
n            = F 类说法总数
ground       = |covered| / |keys|  −  2 × wrong / max(1, n)
```

- **前半项是覆盖率**：鼓励把"下一步要做的事所需的前提状态"说出来。
  - 什么都不说，覆盖率为 0。这堵住了 LLM 评委 grounded 奖励"不说就满分"的漏洞：离线发现新 sum 的思考变短了，状态说法出错率也没有下降。
  - 说了与 due 无关、但内容正确的事实，不加分也不扣分。
- **后半项是错误率**：任何一条说错都要扣分，系数 2。说得越多、错得越多，扣得越多，防止靠堆砌说法刷分。
- **等待规则**（不需要抽取）：Action 是 `wait`，但所有 due 步骤此刻都已经能做，记一次错误（wrong+1，n+1）。
  对应"明明能做却说要等"的判断错误。

**例子**：Chef 在 t=16 手里拿着 potato_slices，pot0 是空的，due = put_obj_in_utensil(pot0)。
- 思考原文："The Chef is holding potato slices … The pot is currently empty, so the Chef can proceed to place the potato slices into the pot."
- 抽到的说法：hold(Chef, potato_slices) ✓、ready(Chef, put_obj_in_utensil(pot0), true) ✓、utensil(pot0, empty) ✓。
- 3 个关键事实全部说中，没有错误，ground = 3/3 − 0 = **1.0**。
- 如果这次思考只写了"I will put the slices into the pot"，那是意图，不算说法，ground = 0。

### 3.2 `mate`：队友信度（取值 [-1, 1]）

以下各项各打一个分，然后取平均：

1. **对队友状态的说法**（S 类的 hold / doing / can_operate / ready）：说对 +1，说错 −1。
2. **计划对队友的预期**：取 Recent Goal 里分给队友的**第一步**。
   - 队友在之后 5 步内执行了这一步：+1；没执行：−1。
   - 如果离局末不足 5 步、队友又还没做，这一项不计。
3. **Action 里的每个请求** `request(mate, X)`：检查三件事，每满足一项得 1 分，总分 s∈[0,3]，本项得分为 2·s/3 − 1。
   - **可达**：X 用到的厨具或位置在队友的 space 里；
   - **符合计划**：X 是队友剩余参考步骤中的前两步之一；
   - **兑现**：队友在 5 步内确实执行了 X。

如果一项都没有（既没提队友，也没发请求），mate = 0。经过批内中心化（见第 4 节），这只相当于"中性"，不像 LLM 评委那样"不说就满分"。

**例子**：
- Chef 发出 `request(Assistant, pickup(potato, ingredient_dispenser))`：可达 ✓、符合计划 ✓、2 步后被执行 ✓，本项 **+1**。
- Chef 的思考写"The Assistant should use the pot to boil the potato slices"，计划分给 Assistant 的第一步是 put_obj_in_utensil(pot0)。Assistant 够不到 pot0，5 步内也没有执行，本项 **−1**。

### 3.2b `mate_v2`：修复版队友信度（2026-10-27）

**`l2r_mate` 训练中发现的漏洞：** mate 是对所有项取平均，所以一条错误预测可以被许多条容易说对的说法冲淡。Assistant 学会了在思考里复述观测中写明的队友状态，例如"the Chef is not holding anything"：
- 每次调用的这类说法从 0.13 条增加到约 1.3 条，正确率 99%；
- mate 分因此从 0.09 涨到 0.92；
- 但预测队友下一步的准确率只有约 0.25，没有提高。

**修复方法：**

```
state   = 关于队友当前状态的说法（hold / doing / can_operate / ready）→ 合并为 1 项：对错均值（−1..1），权重 0.25
forward = 计划分给队友的第一步（±1）+ 每个请求（2·s/3 − 1），每项权重 1
mate_v2 = clip(0.25·state + Σ forward, −1, 1)
```

复述队友状态最多只能拿 0.25 分。真正有用的是对队友未来动作的预测，以及提出的请求是否靠谱，这两部分才决定分数。

**用 `l2r_mate` 已有的轨迹重新计算 Assistant 的分数：**

| 轮次 | mate（旧） | mate_v2 |
|---|---|---|
| u1 | 0.09 | 0.02 |
| u50 | 0.92 | 0.24 |

u50 的 mate_v2 几乎全部来自那 0.25 的上限。

**新实验组与旧代码的关系：**
- 新实验组是 `l2r_mate_v2`（`rules: {mate_v2: 1}`）。
- 旧的 `mate` 保持不变，因此 `l2r_full` 和 `l2r_mate` 中途重启后，奖励仍按原定义计算。
- `l2r_full` 不重跑：这个漏洞在 full 组的主要影响是推理变冗长，问题不大。

### 3.3 `plan`：长程规划（取值 [-1, 1]）

```
H = 3（每个角色只看接下来 3 步）
cover = Σ_P LCS(计划中分给 P 的步骤, remaining(P)[:H])  /  Σ_P min(|remaining(P)|, H)
bad   = 计划里不在"该角色剩余参考步骤"中的步骤数（分错角色、已经做完、或参考里根本没有）
plan  = cover − bad / max(1, 计划步骤数)
```

- **LCS（最长公共子序列）**：同时考察覆盖和顺序。顺序颠倒的步骤只能匹配上一部分。
- 计划为空（`[EMPTY]`）时 cover = 0、bad = 0，plan = 0。
- 两个角色都已经做完时，plan 不计算（None）。

**例子**：
- Assistant 在 t=10 手里拿着 potato_slices，剩余参考步骤是 Assistant [place_obj_on_counter()]、Chef [pickup(potato_slices,counter), put_obj_in_utensil(pot0), cook(pot0)]。
  - 它的计划还是过时的："Pick up the potato slices from the chopping board and place them into the pot"。
  - 两步都不在 Assistant 的剩余步骤里（第一步已经做完，第二步是 Chef 的），cover = 0，bad = 2，plan = **−1**。
- Chef 在 t=16 的计划是 [put_obj_in_utensil(pot0), cook(pot0), deliver_soup()]，剩余参考步骤是 Chef [put_obj_in_utensil(pot0), cook(pot0), pickup(boiled_potato_slices,pot0), deliver_soup()]。
  - 前 3 步的 LCS = 2，cover = 2/3；deliver_soup() 在剩余步骤里，不算 bad。
  - plan = **0.67**。

## 4. 怎样进入训练

1. **合成**：每次调用的思考信号 = Σ wᵢ·rᵢ，权重取自 `think_reward.rules`。
   - 消融时只开一项，例如 `{ground: 1}`；完整版是 `{ground: 1, mate: 1, plan: 1}`。
   - 抽取失败的调用没有信号（记 0）。
2. **中心化**：每个角色减去本批同角色的均值（`centered_judge_scores`）。
   奖励的是"比同批平均水平好多少"，所以各项的基线水平不影响训练。
3. **量级对齐**（`scale: match`）：每个角色的思考信号乘以 max(1, std(动作优势) / std(该角色思考信号))。
   这样思考项至少与动作项同一量级，各组的思考项总量级也一致，比较的是"用哪种信号"，而不是"信号多大"。
4. **施加方式**（`mode: sum`）：优势 = A + w·j，作用在这次调用的全部 token 上，与新 sum 相同。
   A 为原动作优势（ITES、格式和验证惩罚、成功奖励），w = 1。
5. **其余设置不变**：从 SFT L1+2 开始，只用 L2 的 3 道训练菜，50 轮，每轮 4 局，KL 系数 0.02。

### 4.1 sum 模式的副作用（`l2r_ground` 中发现，2026-10-27）

sum 模式把思考奖励加到整次调用的**所有 token** 上，**也包括 Action**。所以"推理写得对不对"会直接影响"选哪个动作"。

Chef 真去做厨房动作时，Think 里常常对自己下判断，例如"I should proceed to cut…"被抽成"现在能切"。而 Chef 实际够不到 chopping_board0，这条判为错。结果做动作的调用里有 44% 说错至少一条，ground 平均只有 −0.39；发消息的调用很少对自己下判断，说错比例 11%，ground 约 −0.03。

中心化后，做动作的调用优势约 −0.2，发消息的约 +0.1。RL 于是把 Chef 推向发消息：Chef 做动作的比例从 6% 降到 3%，成功率接近 0。

修复办法是改用 split 模式：Action token 只拿环境优势 A，思考奖励只作用在推理 token 上。对应实验为 `l2r_ground_split`。

## 5. 消融设计

| 组 | 思考奖励 | 回答的问题 |
|---|---|---|
| `l2r_none` | 无（只有动作奖励） | 基线：只靠环境奖励能到哪 |
| `l2r_ground` | ground | 只奖励第一视角，第一视角探针与成功率会不会提高 |
| `l2r_mate` | mate | 只奖励队友信度，队友预测和请求质量会不会提高 |
| `l2r_plan` | plan | 只奖励规划，规划探针与成功率会不会提高 |
| `l2r_full` | ground + mate + plan | 三者是否互补 |
| `l2r_mate_v2` | mate_v2（修复版，见 3.2b） | 队友奖励去掉漏洞后是否有用 |
| `l2r_ground_split` | ground，**只加在推理 token 上**（mode split） | 去掉 sum 模式的偏差后，第一视角奖励是否有用（见 4.1） |
| `l2r_full_v2_split` | ground + mate_v2 + plan，只加在推理 token 上 | 修好的完整组合；替代旧 `l2r_full`（sum 模式 + 旧版 mate） |
| （已有）新 sum | LLM 评委总分 + LLM grounded | 对照：LLM 打分 vs 规则打分 |

**评估**：
- 游戏成功率，训练菜和 held-out 菜（boiled_corn_slices、baked_potato_slices）分开报，L1–L6 全测；
- 探针 v3，Chef 和 Assistant 分开，直接作答和先思考两种模式都测；
- 评估局上的三个规则奖励值。

## 6. 离线校验结果（在已有 rollout 上打分，每个模型 8 局）

| 模型（8 局成功率） | 角色 | ground | mate | plan | 关键事实覆盖率 | F 类说法出错率 |
|---|---|---|---|---|---|---|
| 训练早期 ≈SFT（0.00） | Chef / Asst | −0.06 / +0.06 | +0.42 / +0.26 | +0.18 / −0.39 | 0.10 / 0.25 | 14% / 12% |
| 旧 none*（0.50） | Chef / Asst | −0.16 / −0.04 | +0.10 / +0.16 | +0.25 / −0.42 | 0.08 / 0.28 | 22% / 17% |
| 旧 sum（0.62） | Chef / Asst | −0.13 / +0.12 | +0.14 / +0.14 | +0.41 / −0.43 | 0.08 / 0.34 | 13% / 13% |
| 新 sum（0.88） | Chef / Asst | −0.13 / −0.09 | +0.45 / +0.32 | +0.18 / −0.20 | 0.14 / 0.17 | 19% / 19% |
| 新 gated（0.62） | Chef / Asst | +0.01 / +0.18 | +0.25 / +0.47 | +0.19 / −0.08 | 0.09 / 0.30 | 6% / 9% |

\* 旧 `l2_none` 训练时用了 held-out 菜，这里只作参考；新基线 `l2r_none` 只用训练菜。

- **与成功率的关系**：按局计算的相关系数，plan 0.59，mate 0.28，ground 0.25。
- **与 LLM 评委的 grounded 分**：规则 ground 与它的相关系数只有 0.16，两者衡量的不是同一件事。
- **抽取稳定性**：用 sol 对同一局抽取两次，算出的奖励相关系数为 ground 0.62、mate 0.82、plan 0.85。这只基于 1 局约 39 次调用，估计较粗。

## 7. 已知局限

1. **抽取有噪声、召回不全。** 例如"我得等锅好了"有时不会被抽成 ready(…, false)。漏抽只会少得覆盖分，不会被误判为错；但它会让 ground 带噪声。
2. **参考计划只是一种可行解。** 例如游戏允许从锅里直接 deliver_soup()，但参考序列里先有 pickup，这种情况下 plan 会少给分。有多条参考序列时，取匹配最深的那条。
3. **Assistant 看不到菜谱。** 它的计划天然容易出错（离线 plan 均值为负）。plan 奖励会推动它先问 Chef 再定计划，这是想要的行为，但它的起点比 Chef 低。由于按角色中心化，这不影响两个角色各自的学习信号。
4. **队友预测有天花板。** 队友在 5 步内没做，可能是队友自己的问题，不一定是预测错。
5. **mate 只看计划给队友的第一步**，不检查后续步骤的顺序；后续步骤由 plan 负责。
6. **抽取 prompt 和关键事实的定义是我设定的**，可能带偏好：比如要求说出"手里拿着什么"。如果你希望关键事实换一种定义，可以调整。

## 8. 费用估算（按离线实测）

- 离线一共 580 个请求，共 3.50M 输入 token（约 70% 命中缓存）+ 0.96M 输出 token。
- 训练早期每局调用多，估计每组 50 轮约 2400 个请求，约 **14.5M 输入 + 4M 输出**。
- 4 组规则奖励合计约 **58M 输入 + 16M 输出**；none 组不调用 API。
- 对照：新 sum 一整组用 terra 打分，共 9.5M 输入 + 1.1M 输出。
