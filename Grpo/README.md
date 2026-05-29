# Collab-Overcooked GRPO

This folder contains a MARSHAL-style GRPO implementation for Collab-Overcooked.

Core credit assignment:

- Compute agent-specific reward-to-go over collected transitions.
- Normalize distinct return values per agent, matching MARSHAL's `normalize_unique_values_by_player`.
- Do not group samples by timestep or full prompt hash.

Config example:

```yaml
trainer:
  type: grpo
  gamma: 1.0
  grpo:
    norm_scope: agent
    normalize: mean_std
    advantage_clip: 10.0
```

Run directly:

```bash
python -m Grpo.train_grpo --config configs/rl_qwen_train_grpo_t3_t13.yaml
```

The existing `scripts/cluster_run_rl.sh` also works once collect/train/eval configs set `trainer.type: grpo`.
