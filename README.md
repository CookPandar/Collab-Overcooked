<div align="center">
  <h1> Collab-Overcooked </h1>
  <p><em>A Multi-Agent Collaborative Benchmark based on Overcooked-AI</em></p>
</div>

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen.svg)](docs/)

We propose a new LLM-powered Multi-Agent System (LLM-MAS) benchmark, **Collab-Overcooked**, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks from two novel perspectives:

1. **Multi-agent Framework**: Supports diverse tasks and objectives while encouraging collaboration through natural language communication
2. **Process-oriented Evaluation**: Introduces comprehensive metrics to assess fine-grained collaboration capabilities of different LLM agents

## 🎯 Key Features

- **Multiple Cooking Tasks**: Boiled egg, soup, salad, and more
- **Diverse Kitchen Layouts**: Various configurations requiring different collaboration strategies
- **LLM Agent Support**: Works with GPT models, local LLMs, and custom agents
- **Comprehensive Evaluation**: F1 score, similarity, redundancy, and collaboration metrics
- **Easy Installation**: One-command setup with conda environment
- **Flexible Configuration**: YAML-based configuration system
- **Rich Documentation**: Complete guides and API reference

## 🚀 Quick Start

### Installation

#### Option 1: Automatic Installation (Recommended)

```bash
git clone https://github.com/your-org/Collab-Overcooked.git
cd Collab-Overcooked
bash scripts/install.sh
```

#### Option 2: Manual Installation

```bash
git clone https://github.com/your-org/Collab-Overcooked.git
cd Collab-Overcooked

# Create conda environment
conda env create -f environment.yml
conda activate collab-overcooked

# Install main package (includes all dependencies)
pip install -e .
```

### Configuration

1. **Set up API configuration**:
   Copy and edit the configuration file:

   ```bash
   cp configs/default.yaml configs/test_personal.yaml
   # Edit configs/test_personal.yaml and add your API key
   ```
2. **Customize configuration** (optional):
   Edit `configs/test_personal.yaml` to modify settings and add your API key

### Quick Test

Run a simple test to verify installation:

```bash
bash scripts/quick_test.sh
```

Or manually:

```bash
conda activate collab-overcooked
collab-overcooked --horizon 3 --order boiled_egg
```

This runs a 3-step collaboration scenario between two GPT agents making a boiled egg.

## 📖 Documentation

- **[Installation Guide](docs/installation.md)**: Detailed installation instructions
- **[Usage Guide](docs/usage.md)**: Comprehensive usage examples and tutorials
- **[API Reference](docs/api_reference.md)**: Complete API documentation

## 🔧 Usage Examples

### Basic Usage

```bash
# Run a simple experiment
collab-overcooked --horizon 10 --order soup --layout cramped_room

# Run evaluation pipeline
bash scripts/run_evaluation.sh
```

### Python API

```python
from collab_overcooked import main
from collab_overcooked.evaluation import evaluate_performance

# Run experiment programmatically
results = main()

# Custom evaluation
eval_config = {
    "tasks": ["boiled_egg", "soup"],
    "layouts": ["cramped_room"],
    "num_runs": 5,
    "metrics": ["f1_score", "collaboration_initiate"]
}

results = evaluate_performance(eval_config)
```

### Local LLM Support

Configure local LLMs using [vLLM](https://github.com/vllm-project/vllm):

```yaml
agents:
  agent_0:
    type: "local_llm"
    model_path: "/path/to/your/model"
    temperature: 0.1
```

## 📊 Evaluation

### Automated Evaluation

```bash
bash scripts/run_evaluation.sh
```

This runs the complete evaluation pipeline:

1. **evaluation.py**: Calculates metrics for each task
2. **organize_result.py**: Summarizes results into `statistics_data.csv`
3. **convert_result.py**: Computes complexity-level metrics in `converted_data.csv`

### Batch Testing Multiple Models

We provide a parallel-friendly driver to benchmark multiple LLM setups across all 30 recipe tasks:

```bash
# Optional: map each model to its own YAML template
cat > configs/model_configs.json <<'EOF'
{
  "azure-gpt-4o": "configs/azure-gpt-4o.yaml",
  "qwen2.5-7B-instruct": "configs/qwen2.5-7B-instruct.yaml"
}
EOF

python scripts/run_model_suite.py \
  --models azure-gpt-4o  \
  --model-configs configs/model_configs.json \
  --temperatures 0.7 \
  --repeats 10 \
  --max-workers 10 \
  --output-dir assets/data/batch_results
```

- `--models`: list of model identifiers; both Chef/Assistant share the same entry per run.
- `--model-configs` (optional): JSON/YAML mapping from model name to a dedicated YAML config; falls back to `--base-config` otherwise.
- `--temperatures` / `--repeats`: sweep temperatures and repeat full suites N times.
- `--max-workers`: number of parallel worker processes (each runs all tasks once).
- Outputs per-model logs under `{output_dir}/{model}/logs/{order}/` and copied JSON summaries under `{output_dir}/{model}/json/{order}/`. Aggregated statistics are stored in `results.json`, `aggregate.json`, and `success_rates.png`.
- 每次调用 `collab_overcooked.main` 都会写入 `results/<run_id>_<order>/experiment_*.json`。`run_id` 可以通过 `--run-id` 或 YAML 中的 `run.run_id` 显式指定；如果缺省，程序会自动生成一个带微秒时间戳与随机后缀的 ID。批量脚本会自动注入唯一 `run_id`，避免并发进程互相覆盖输出。
- 日志 JSON 顶层新增 `prompt_templates` 字段，内含 Chef/Assistant 的完整 system prompt（含规则与对应食谱）；做 SFT 或重现输入时可直接读取该字段，拼接 observation 即可还原原始提示。

若在无交互机群上运行，推荐使用下面四个脚本完成“环境准备 + 任务执行”的组合流程：

1. **一次性准备 conda 环境（若节点回收会被删，可在作业一开始调用）**

    ```bash
    bash scripts/cluster_env_setup.sh \
      /mnt/shared/envs/collab_overcooked \
      /mnt/shared/envs/vllm \
      3.10
    ```

    该脚本会检测目标前缀是否已存在 `conda-meta`，若缺失则创建对应的 Python 环境：`collab` 环境安装 `collab_overcooked` 及 SFT/RL 依赖，`vllm` 环境仅安装 vLLM。重复运行将自动复用已存在的路径。

2. **运行 SFT 任务（可替换任意 `train_qwen_sft.py` 参 数）**

    ```bash
    SFT_NUM_PROCS=8 \
    bash scripts/cluster_run_sft.sh /mnt/shared/envs/collab_overcooked \
      --config configs/examples/sft_qwen_level12.yaml \
      --output-dir results/sft_runs
    ```

    `cluster_run_sft.sh` 会激活 `collab` 环境并通过 `accelerate launch --num_processes ${SFT_NUM_PROCS:-1}` 调用 `scripts/train_qwen_sft.py`，额外的 `accelerate` 参数可以通过 `SFT_ACCELERATE_ARGS` 环境变量传入。

3. **运行 RL 任务**

    ```bash
    RL_NUM_PROCS=8 \
    bash scripts/cluster_run_rl.sh /mnt/shared/envs/collab_overcooked \
      --config configs/examples/rl_qwen_baked_bell_pepper.yaml
    ```

    逻辑与 SFT 脚本相同，只是入口换成 `python -m collab_overcooked.main_rl`。

4. **批量评测（会先在 vLLM 环境中托管推理服务，再调用 `run_model_suite.py`）**

    ```bash
    bash scripts/run_cluster_suite.sh \
      /mnt/shared/envs/vllm \
      /mnt/shared/envs/collab_overcooked \
      /path/to/qwen2.5-7B-instruct \
      qwen2.5-7B-instruct \
      configs/model_configs.json \
      assets/data/batch_results \
      8000 \
      0.9 \
      --max-workers 8 --repeats 1
    ```

    脚本会在本地节点启动 vLLM 服务、等待端口就绪、执行 `run_model_suite.py`，最后自动关闭服务。请确保模型配置中的 `base_url` 指向 `http://127.0.0.1:PORT/v1` 并与脚本端口保持一致。环境准备 + 任务运行均由上述脚本负责，后续切换任务时只需重复执行第 2/3/4 步即可。

### Utility / Analysis Scripts

| Script | 作用 | 备注 |
| --- | --- | --- |
| `scripts/cluster_env_setup.sh` | 在指定前缀创建/复用两个 conda 环境：`collab`（SFT/RL/批量脚本）与 `vllm`（仅托管推理服务）。 | 第 3 个参数可自定义 Python 版本，脚本会自动 `pip install -e .` 并拉取必要依赖。 |
| `scripts/cluster_run_sft.sh` | 激活 `collab` 环境，并用 `accelerate launch` 运行 `scripts/train_qwen_sft.py`。 | 通过 `SFT_NUM_PROCS` / `SFT_ACCELERATE_ARGS` 控制 `accelerate` 行为。 |
| `scripts/cluster_run_rl.sh` | 类似上面，但入口是 `python -m collab_overcooked.main_rl`。 | 支持 `RL_NUM_PROCS` / `RL_ACCELERATE_ARGS`。 |
| `scripts/run_cluster_suite.sh` | 使用 `vllm` 环境启动 vLLM OpenAI API，再切到 `collab` 环境调用 `scripts/run_model_suite.py`。 | 传入模型路径、端口、`run_model_suite.py` 额外参数即可完成整套批量评测。 |
| `scripts/run_evaluation.sh` | 执行旧版三段式评估 (`evaluation.py` → `organize_result.py` → `convert_result.py`) 并把结果放进 `results/`。 | 仅在需要兼容早期流程时使用。 |
| `scripts/run_model_suite.py` | 新版多模型基准驱动，支持并发 worker、重复次数、温度网格等；推荐使用它跑日常评测。 | 既可单独调用，也可由 `run_cluster_suite.sh` 间接触发。 |
| `scripts/summarize_split_metrics.py` | 汇总 `assets/data/batch_results/<model>/json/` 中的 per-order JSON 日志，输出每个菜品、每个 split 的成功率及 Chef/Assistant 奖励指标。支持 `--per-order-csv` / `--split-csv` 导出 CSV，并可通过 `--plot-per-order-csv label=path ... --plot-output-dir plots/` 对多个模型的 per-order 指标画对比折线图（测试/验证任务自动用淡色虚线标记）。 | 绘图阶段若缺少 matplotlib 会自动跳过。 |

> **提示**：上述脚本的组合即可覆盖“环境初始化 + SFT + RL + 批量评测”所有常见工作流，通常无需再手动维护临时 vLLM 进程或重复安装依赖。

### Aggregating Historical Runs

批量测试会把所有原始 JSON 写入 `assets/data/batch_results/<model>/json/<order>/`。若后续想统一统计成功率并绘制图表，可运行：

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/analyze_batch_results.py --models azure-gpt-4o qwen2.5-7B-instruct
```

该脚本会：

- 输出 `results.json`（按订单列出所有 run 的成功与耗时）
- 生成 `aggregate.json`（整体 + 各 level 成功率）
- 绘制 `success_rates.png`（整体柱状图）与 `level_success.png`（分 level 折线图）

如未指定 `--models`，脚本会遍历 `assets/data/batch_results` 下所有模型文件夹。若只想统计某个温度（例如 0.7），追加 `--temperature 0.7` 即可，脚本会自动过滤出对应温度的运行记录。

### Exporting SFT Data

成功日志中包含完整的 system prompt、observation 与模型输出，可直接整理成 SFT 数据：

```bash
python scripts/export_sft_dataset.py \
  --source-dir assets/data/batch_results \
  --models azure-gpt-4o \
  --levels 1 2 \
  --temperature 0.7 \
  --agents Chef Assistant \
  --output data/sft/gpt4o_level12.jsonl
```

每行 JSON 结构如下：

```json
{
  "system": "<system prompt>",
  "prompt": "<observation + 历史对话>",
  "response": "<Think/Recent Goal/Action>",
  "meta": {"model": "azure-gpt-4o", "order": "baked_bell_pepper", "level": 1, ...}
}
```

可通过 `--levels` / `--temperature` / `--agents` 控制样本筛选，`--max-samples` 则限制导出数量。

若希望按“任务级”划分训练/验证/测试（例如 Level1&2 菜谱按 7:1:2 划分，保证验证/测试订单在训练中从未出现），可运行：

```bash
python scripts/export_sft_dataset.py \
  --source-dir assets/data/batch_results \
  --models azure-gpt-4o \
  --levels 1 2 \
  --temperature 0.7 \
  --agents Chef Assistant \
  --train-output data/sft/train_level12.jsonl \
  --val-output data/sft/dev_level12.jsonl \
  --test-output data/sft/test_level12.jsonl \
  --train-ratio 0.7 --val-ratio 0.1 --test-ratio 0.2 \
  --split-seed 42
```

脚本会先收集符合条件的订单，然后按比例随机分配到 train/val/test，确保同一订单的所有轨迹只会出现在一个 split 中。若同时指定 `--output`，还会额外生成一个合并后的全集。

### Fine-tuning Qwen2.5 with Exported Data

准备好 JSONL 后，可使用 `scripts/train_qwen_sft.py` 进行（LoRA）微调：

```bash
pip install transformers datasets accelerate peft

python scripts/train_qwen_sft.py \
  --data-path data/sft/gpt4o_level12.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir runs/qwen2.5-sft-level12 \
  --epochs 1 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --learning-rate 5e-5 \
  --max-length 2048 \
  --use-lora \
  --eval-ratio 0.05
```

脚本依赖 HuggingFace Transformers + PEFT：若开启 `--use-lora`，默认在 `q_proj/k_proj/v_proj/o_proj` 上注入 LoRA；也可通过 `--lora-r/--lora-alpha/--lora-dropout` 调整。训练完成后，`--output-dir` 下会保存可直接推理的模型与 tokenizer。

如果已经通过 `export_sft_dataset.py` 生成了拆分好的 `train/dev/test` JSONL，可以直接传入：

```bash
python scripts/train_qwen_sft.py \
  --train-data data/sft/train_level12.jsonl \
  --eval-data data/sft/dev_level12.jsonl \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir runs/qwen2.5-sft-level12 \
  ...
```

`--eval-data` 不提供时，可以继续使用 `--eval-ratio` 从训练集划分验证集；`--test-data` 可留作离线评估（训练过程中不会使用）。

### RL Baseline (MAPPO + Qwen2.5)

我们提供 `python -m collab_overcooked.main_rl` 作为扩展入口：当配置文件包含 `trainer` 字段时，会自动切换到 MAPPO 训练流程，否则保持原有推理模式。例如：

```bash
python -m collab_overcooked.main_rl --config configs/examples/rl_qwen_baked_bell_pepper.yaml
```

`trainer.model_path` 需要指向本地可用的 Hugging Face 检查点（如 Qwen2.5-7B-Instruct）；脚本会加载该模型作为共享的 actor-critic，对 Collab-Overcooked 奖励进行 RL 微调。

RL 入口默认通过 `training/main_session.py` 复用 `collab_overcooked.main` 的真实 prompt / Think / Recent Goal / Action 流程。所有 planner 请求都会改由本地 HuggingFace 模型（`AutoModelForCausalLM`）生成，并在 PPO 更新时利用完整的 token 级 log-prob 与 value 估计，从而直接微调推理所用的 LLM。可额外指定 `trainer.max_new_tokens`、`trainer.generation_temperature` 等解码参数。

根据 `trainer.type` 可选择不同的实现：

- `mappo`（默认）：依赖 Accelerate/AdamW，在单 GPU 或数据并行模式下训练。
- `mappo_deepspeed`：启用 DeepSpeed ZeRO +（可选）Tensor Parallel。配置中需提供 `trainer.deepspeed_config`，并使用 `deepspeed --num_gpus N python -m collab_overcooked.main_rl --config <yaml>` 启动。示例参考 `configs/examples/rl_qwen_deepspeed.yaml`。

`trainer.output_dir` 用于指定权重与优化器状态的保存位置（默认写入 `results/mappo_<order>/` 或 `results/dsmappo_<order>/`）；每次训练结束都会把最终 checkpoint 存在 `<output_dir>/final/`，并可通过 `trainer.save_interval`（或 `trainer.checkpoint_interval`）设置按更新步数定期落盘。Accelerate 版本会调用 `Accelerator.save_state` 持久化，DeepSpeed 版本则使用 `engine.save_checkpoint`，可在相同命令下恢复训练。

### Key Metrics

- **F1 Score**: Action accuracy using TES function
- **Similarity**: Comparison with Reference Action Templates (RATs)
- **Redundancy**: Unnecessary action detection
- **Collaboration Initiate**: Ability to start collaboration
- **Collaboration Respond**: Ability to respond to collaboration

## 🛠️ Customization

### Adding New Tasks

1. Create layout files in `dependencies/overcooked_ai/overcooked_ai_py/data/layouts/`
2. Update configuration files
3. Modify evaluation scripts if needed

### Custom Agents

```python
from collab_overcooked.agents import BaseAgent

class CustomAgent(BaseAgent):
    def get_action(self, state, legal_actions):
        # Your custom logic
        return selected_action
```

### Environment Modification

The environment logic is in `dependencies/overcooked_ai/`. Modify:

- Layout files in `data/layouts/` for new recipes/ingredients
- Environment logic in `mdp/` for new interactive elements

## Reference

```bibtex
@inproceedings{zhang2024proagent,
  title={Proagent: building proactive cooperative agents with large language models},
  author={Zhang, Ceyao and Yang, Kaijie and Hu, Siyi and Wang, Zihao and Li, Guanghe and Sun, Yihang and Zhang, Cheng and Zhang, Zhaowei and Liu, Anji and Zhu, Song-Chun and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17591--17599},
  year={2024}
}

@inproceedings{carroll2019utility,
 title={On the Utility of Learning About Humans for Human-AI Coordination},
 author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
 booktitle={Advances in Neural Information Processing Systems},
 pages={},
 volume={32},
 year={2019},
}
```
