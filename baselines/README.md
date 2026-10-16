# SFT and RL baselines

Training baselines for small open models (Qwen3-1.7B) on Collab-Overcooked v2. Each role (Chef, Assistant) is a
LoRA adapter (r=16, alpha=32 on q/k/v/o) of the same base model. The agents, prompts and validation are the
benchmark's own (`collab_overcooked.agent_factory.make_agent`); the model is plugged in through the
[Extension API](../docs/extension_api.md), so training sees exactly the prompts used in evaluation.

Two response formats (`response_protocol`): `action_only` (a single `Action:` field) and `think_action`
(`Think` / `Recent Goal` / `Action`). Use the same format for the SFT data, SFT, RL and evaluation.

Extra requirements: `peft`, and vLLM for RL (tested with 0.18; it may live in a separate environment, see `vllm_python`).

## 1. SFT data

A teacher model plays the benchmark; its well-formed calls (no format or validator error) of successful episodes
become examples `{role, order, messages, response}`.

```bash
python -m baselines.sft.collect --teacher gpt-4o --protocol action_only --level 1 2 --episodes 10 \
    --output data/sft/gpt4o_action_only.jsonl
python -m baselines.sft.collect --teacher gpt-4o --protocol think_action --level 1 2 --episodes 10 \
    --output data/sft/gpt4o_think.jsonl                       # SFT with reasoning
```

`--base-url` uses a local OpenAI-compatible teacher; `--keep-failed` also keeps clean calls of failed episodes.

## 2. SFT

```bash
python -m baselines.sft.train --model /path/to/Qwen3-1.7B --data data/sft/gpt4o_action_only.jsonl \
    --output runs/sft_action_only --gradient-checkpointing
```

Writes `runs/sft_action_only/Chef` and `.../Assistant` (plus one copy per epoch under `epochs/`). Defaults: 6 epochs,
16 sequences per step, lr 5e-5 (cosine, 3% warmup), loss on the response tokens only.

## 3. RL

```bash
python -m baselines.rl.train --config baselines/configs/rl.yaml
python -m baselines.rl.train --config baselines/configs/rl.yaml level=3 objective=marshal output_dir=runs/rl_l3_marshal
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 -m baselines.rl.train --config baselines/configs/rl.yaml
```

Every update collects `episodes_per_update` episodes (the tasks of the level, mixed), computes rewards and advantages
for every LLM call, runs one PPO update of both adapters and serves the new adapters. The run resumes from
`output_dir` (`state.json`). Outputs: `metrics.jsonl` (success rate, rewards, error rates, update statistics and
timings per update), `checkpoints/uNNNNN/{Chef,Assistant}` (every `checkpoint_every` updates plus the latest two),
`config.yaml` and the vLLM log.

**Rewards** (per LLM call, `baselines/rl/credit.py`)

| | |
|---|---|
| format | -0.2 if the call had a format error or its reply is not exactly one action or one message |
| validator | -0.1 if the proposed action was rejected by the validator (and there was no format error) |
| ITES | for a validated kitchen action: TES(history + action) - TES(history) against the reference action sequences of the task (best reference, beta 0.95) |
| terminal | +2 on the last call of each role when the order is delivered |

**Advantage** (`advantages.py`): own reward + gamma^(t' - t) x terminal reward (gamma 0.99 over environment
time steps), no baseline; calls with a format or validator penalty get no terminal credit, and a positive
advantage of a call with a format penalty is set to 0; clipped to [-10, 10].

**Objectives** (`update.py`); both are PPO on response tokens with old log-probs recomputed before the update, a
KL penalty (coefficient 0.02) to the frozen SFT adapters and a fresh Adam (lr 1e-5) per update.

| | `agentgym` (AgentGym-RL / verl actor) | `marshal` (our GRPO actor) |
|---|---|---|
| batching | each role separately, minibatches of 16 calls in order | both roles shuffled, pairs of calls, 8 pairs per step |
| policy loss | PPO clip 0.2, token mean per call | dual-clip PPO (0.15, 1.3), token mean per call, mean over the pair |
| KL | low-variance KL, token mean per call | same estimator clamped to [0, 10], token mean over the pair |
| grad clip | 1.0 | 0.5 |

Optional `adaptive_lr` halves the learning rate of the following updates when the update KL exceeds a threshold.

### Speed

- One vLLM server for the whole run; each update's adapters are loaded at runtime under a new name
  (`/v1/load_lora_adapter`), with prefix caching. Rollouts run in parallel worker processes.
- The update batches calls up to `tokens_per_batch` tokens per forward pass. Calls are sorted by prompt and the
  shared prompt prefix of a batch is encoded once (the calls of an episode share the long rule prompt), and the
  LM head is applied to response positions only. Old and reference log-probs use the same batches, so the first
  PPO ratio is exactly 1.
- `torchrun --nproc_per_node N`: each rank serves its own vLLM, collects 1/N of the episodes and runs 1/N of the
  forward passes of every optimizer step; gradients are summed before each step. The update is the same as with
  one GPU (checked to 1e-7 in fp32).

On one A40, an update of 395 calls takes 64 s instead of 220 s with one call per forward pass (peak 37 GB at
`tokens_per_batch: 8192`), and vLLM is no longer restarted between collection and training.

### Relation to the earlier pipeline

The rewards and advantages are the same as those of the earlier RL code (checked on the same episodes and on 13
recorded updates; advantages agree to 2e-7). Per-sample gradients agree with the old one-call-at-a-time update to
cosine 0.9998 (agentgym) / 0.9996 (marshal); the remaining difference is bf16 rounding, which depends on the batch
composition (the same as between two batch sizes of the new code). Two intentional changes:

- the format-correction hint for a reply with several kitchen actions now asks for exactly one action;
- the format penalty of a call proposing an action with invalid parameters goes to that call (before: to call 0
  of the previous time step).
