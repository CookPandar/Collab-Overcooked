"""GRPO trainer built on top of the existing Collab-Overcooked rollout stack."""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from collab_overcooked.training.mappo_qwen import MAPPOTrainer, QwenLMActorCritic, TextTransition

from .advantages import GrpoAdvantageConfig, compute_grpo_advantages


class GRPOTrainer(MAPPOTrainer):
    """Actor-only PPO/GRPO update with MARSHAL-style credit assignment."""

    def __init__(self, env_config, trainer_cfg, full_config=None):
        cfg = dict(trainer_cfg or {})
        cfg["critic_adapter"] = None
        cfg["compute_values_in_collect"] = False
        cfg["collect_value_backend"] = "none"
        cfg["value_head_lr"] = 0.0
        cfg["critic_lr"] = 0.0
        cfg["critic_pretrain_value_loss_threshold"] = None
        super().__init__(env_config, cfg, full_config=full_config)
        grpo_cfg = cfg.get("grpo", {}) or {}
        self.grpo_norm_scope = str(
            grpo_cfg.get("norm_scope", cfg.get("grpo_norm_scope", "agent"))
        )
        self.grpo_normalize = str(
            grpo_cfg.get("normalize", cfg.get("grpo_normalize", "mean_std"))
        )
        self.grpo_clip = grpo_cfg.get("advantage_clip", cfg.get("advantage_clip", None))
        self.grpo_gamma = float(grpo_cfg.get("gamma", cfg.get("gamma", 1.0)))
        self.grpo_eps = float(grpo_cfg.get("eps", 1e-6))
        self.dual_clip_loss = bool(cfg.get("dual_clip_loss", False))
        self.accelerator.print(
            "[GRPO] credit assignment "
            f"norm_scope={self.grpo_norm_scope} normalize={self.grpo_normalize} "
            f"gamma={self.grpo_gamma} clip={self.grpo_clip} "
            f"dual_clip_loss={self.dual_clip_loss}"
        )

    def _critic_pretrain_enabled(self) -> bool:
        return False

    def compute_advantages(self, transitions: List[TextTransition]):
        config = GrpoAdvantageConfig(
            norm_scope=self.grpo_norm_scope,
            normalize=self.grpo_normalize,
            gamma=self.grpo_gamma,
            eps=self.grpo_eps,
            clip=float(self.grpo_clip) if self.grpo_clip is not None else None,
        )
        advantages, returns, metrics = compute_grpo_advantages(
            rewards=[t.reward for t in transitions],
            agent_indices=[int(t.agent_index) for t in transitions],
            timesteps=[t.timestep for t in transitions],
            config=config,
        )
        self._last_grpo_metrics = metrics
        return advantages, returns

    def _current_group_lrs(self) -> Dict[str, float]:
        current = super()._current_group_lrs()
        current["critic_adapter_lr"] = 0.0
        current["value_head_lr"] = 0.0
        return current

    def _materialize_transition_values(self, transitions: List[TextTransition]) -> None:
        self.accelerator.print(
            f"[GRPO] skip materialize_transition_values transitions={len(transitions)}"
        )
        return

    def update_policy(self, transitions: List[TextTransition]):
        assert self.text_policy is not None
        unwrapped_model: QwenLMActorCritic = self.accelerator.unwrap_model(self.text_policy)
        rank = self.accelerator.process_index
        prompt_tensors = [t.prompt_ids for t in transitions]
        response_tensors = [t.response_ids for t in transitions]
        agent_indices = [t.agent_index for t in transitions]
        policy_temperatures = [t.policy_temperature for t in transitions]
        old_token_log_probs_list: List[torch.Tensor] = []
        for t in transitions:
            if t.response_log_probs is not None:
                old_token_log_probs_list.append(
                    t.response_log_probs.to(self.device, dtype=torch.float32)
                )
            else:
                resp_len = max(1, int(t.response_ids.numel()))
                old_token_log_probs_list.append(
                    torch.full(
                        (resp_len,),
                        float(t.log_prob) / float(resp_len),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

        advantages, returns = self.compute_advantages(transitions)
        raw_advantages = advantages.clone()
        returns_stats = returns.clone()
        advantages = advantages.to(self.device)

        batch_size = max(1, self.train_batch_size)
        num_transitions = len(transitions)
        num_minibatches = math.ceil(num_transitions / batch_size)
        total_optimization_steps = max(1, self.update_epochs * num_minibatches)
        optimizer_steps_per_update = max(
            1,
            math.ceil(total_optimization_steps / float(self.gradient_accumulation_steps)),
        )
        self.accelerator.print(
            "[GRPO] update_policy "
            f"rank={rank} transitions={num_transitions} batch_size={batch_size} "
            f"epochs={self.update_epochs} norm_scope={self.grpo_norm_scope}"
        )
        scheduler = self._build_lr_scheduler(optimizer_steps_per_update)
        param_groups = self._trainable_param_groups()
        total_group_grad_norms = {name: 0.0 for name in param_groups.keys()}
        total_group_param_deltas = {name: 0.0 for name in param_groups.keys()}

        total_loss = 0.0
        total_policy = 0.0
        total_entropy = 0.0
        total_clipfrac = 0.0
        total_approx_kl = 0.0
        total_kl_penalty = 0.0
        metric_steps = 0
        actual_optimization_steps = 0
        accum_counter = 0
        accum_approx_kl_sum = 0.0
        accum_approx_kl_count = 0
        stop_early = False

        self.optimizer.zero_grad()
        for epoch_idx in range(self.update_epochs):
            if self.shuffle_minibatches and num_transitions > 1:
                ordered_indices = torch.randperm(num_transitions).tolist()
            else:
                ordered_indices = list(range(num_transitions))

            for start in range(0, num_transitions, batch_size):
                end = min(start + batch_size, num_transitions)
                batch_indices = ordered_indices[start:end]
                batch_prompts = [prompt_tensors[i] for i in batch_indices]
                batch_responses = [response_tensors[i] for i in batch_indices]
                batch_agent_indices = [agent_indices[i] for i in batch_indices]
                batch_adv = advantages[batch_indices]

                _, entropies, token_log_probs = unwrapped_model.evaluate_policy_batch(
                    batch_prompts,
                    batch_responses,
                    batch_agent_indices,
                    policy_temperatures=[policy_temperatures[i] for i in batch_indices],
                )
                entropies = entropies.to(self.device, dtype=torch.float32)
                batch_old_token_log_probs = [
                    old_token_log_probs_list[i] for i in batch_indices
                ]
                token_logratio_parts: List[torch.Tensor] = []
                token_policy_loss_parts: List[torch.Tensor] = []
                for sample_idx, (new_lp, old_lp) in enumerate(
                    zip(token_log_probs, batch_old_token_log_probs)
                ):
                    if new_lp.numel() == 0:
                        continue
                    if old_lp.numel() != new_lp.numel():
                        old_lp = old_lp[: new_lp.numel()]
                        if old_lp.numel() < new_lp.numel():
                            pad_value = float(old_lp[-1].item()) if old_lp.numel() > 0 else 0.0
                            old_lp = torch.cat(
                                [
                                    old_lp,
                                    old_lp.new_full((new_lp.numel() - old_lp.numel(),), pad_value),
                                ],
                                dim=0,
                            )
                    sample_token_logratio = new_lp - old_lp
                    sample_token_ratios = torch.exp(sample_token_logratio)
                    sample_adv = batch_adv[sample_idx].expand_as(sample_token_logratio)
                    sample_clipped_ratios = torch.clamp(
                        sample_token_ratios,
                        1.0 - self.clip_coef,
                        1.0 + self.clip_coef,
                    )
                    sample_surr1 = sample_token_ratios * sample_adv
                    sample_surr2 = sample_clipped_ratios * sample_adv
                    sample_loss = -torch.min(sample_surr1, sample_surr2)
                    if self.dual_clip_loss:
                        dual_clip_loss = -torch.max(
                            -sample_loss,
                            (1.0 + self.clip_coef * 2.0) * sample_adv,
                        )
                        sample_loss = torch.where(sample_adv < 0, dual_clip_loss, sample_loss)
                    token_logratio_parts.append(sample_token_logratio)
                    token_policy_loss_parts.append(sample_loss.mean())

                token_logratio = (
                    torch.cat(token_logratio_parts, dim=0)
                    if token_logratio_parts
                    else torch.empty(0, dtype=torch.float32, device=self.device)
                )
                token_ratios = torch.exp(token_logratio) if token_logratio.numel() else token_logratio
                policy_loss = (
                    torch.stack(token_policy_loss_parts).mean()
                    if token_policy_loss_parts
                    else torch.zeros((), dtype=torch.float32, device=self.device)
                )
                approx_kl = (
                    ((token_ratios - 1.0) - token_logratio).mean().clamp_min(0.0)
                    if token_logratio.numel() > 0
                    else torch.zeros((), dtype=torch.float32, device=self.device)
                )
                kl_penalty = approx_kl * self.kl_penalty_coef
                entropy_loss = -entropies.mean()
                loss = policy_loss + self.entropy_coef * entropy_loss + kl_penalty
                self.accelerator.backward(loss / float(self.gradient_accumulation_steps))

                accum_counter += 1
                accum_approx_kl_sum += float(approx_kl.item())
                accum_approx_kl_count += 1
                should_step = accum_counter >= self.gradient_accumulation_steps or end >= num_transitions
                step_approx_kl = (
                    accum_approx_kl_sum / float(accum_approx_kl_count)
                    if accum_approx_kl_count > 0
                    else float(approx_kl.item())
                )
                if should_step:
                    group_snapshots = {
                        name: self._snapshot_param_group(params)
                        for name, params in param_groups.items()
                    }
                    for name, params in param_groups.items():
                        total_group_grad_norms[name] += self._param_group_grad_norm(params)
                    self.accelerator.clip_grad_norm_(
                        self.text_policy.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    unwrapped_model.clear_prefix_cache()
                    for name, params in param_groups.items():
                        total_group_param_deltas[name] += self._param_group_delta_norm(
                            params, group_snapshots[name]
                        )
                    self.optimizer.zero_grad()
                    actual_optimization_steps += 1
                    accum_counter = 0
                    accum_approx_kl_sum = 0.0
                    accum_approx_kl_count = 0

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_entropy += float(entropies.mean().item())
                total_clipfrac += (
                    ((token_ratios - 1.0).abs() > self.clip_coef).float().mean().item()
                    if token_ratios.numel() > 0
                    else 0.0
                )
                total_approx_kl += float(approx_kl.item())
                total_kl_penalty += float(kl_penalty.item())
                metric_steps += 1
                if should_step and self.target_kl is not None and step_approx_kl > self.target_kl:
                    stop_early = True
                    break
            if stop_early:
                break

        denom = metric_steps if metric_steps > 0 else 1
        reward_mean = (
            float(sum(float(t.reward) for t in transitions)) / float(len(transitions))
            if transitions
            else 0.0
        )
        returns_cpu = returns_stats.detach().cpu()
        per_agent_metrics: Dict[int, Dict[str, float]] = {}
        if transitions:
            agent_indices_cpu = torch.tensor(agent_indices, dtype=torch.long)
            for agent_idx in sorted({int(idx) for idx in agent_indices}):
                mask = agent_indices_cpu == int(agent_idx)
                if bool(mask.any().item()):
                    per_agent_metrics[int(agent_idx)] = {
                        "adv_mean": float(raw_advantages[mask].mean().item()),
                        "return_mean": float(returns_cpu[mask].mean().item()),
                    }

        result = {
            "loss": total_loss / denom,
            "policy": total_policy / denom,
            "value": 0.0,
            "entropy": total_entropy / denom,
            "reward_mean": reward_mean,
            "adv_mean": float(raw_advantages.mean().item()) if raw_advantages.numel() else 0.0,
            "return_mean": float(returns_cpu.mean().item()) if returns_cpu.numel() else 0.0,
            "value_mean": 0.0,
            "explained_var": 0.0,
            "fresh_value_mean": 0.0,
            "fresh_explained_var": 0.0,
            "clipfrac": total_clipfrac / denom,
            "approx_kl": total_approx_kl / denom,
            "policy_active_clipfrac": total_clipfrac / denom,
            "policy_active_approx_kl": total_approx_kl / denom,
            "policy_active_token_clipfrac": total_clipfrac / denom,
            "policy_active_token_approx_kl": total_approx_kl / denom,
            "kl_penalty": total_kl_penalty / denom,
            "kl_penalty_coef": float(self.kl_penalty_coef),
            "value_clipfrac": 0.0,
            "optimizer_steps": float(actual_optimization_steps),
            "stopped_early": 1.0 if stop_early else 0.0,
            "critic_pretrain_active_end": 0.0,
            "critic_pretrain_released": 0.0,
            "critic_pretrain_last_value_loss": 0.0,
        }
        for name in sorted(param_groups.keys()):
            step_denom = float(actual_optimization_steps) if actual_optimization_steps > 0 else 1.0
            result[f"{name}_grad_norm"] = total_group_grad_norms[name] / step_denom
            result[f"{name}_param_delta"] = total_group_param_deltas[name] / step_denom
        for agent_idx in range(2):
            metrics = per_agent_metrics.get(agent_idx, {})
            result[f"agent{agent_idx}_adv_mean"] = float(metrics.get("adv_mean", 0.0))
            result[f"agent{agent_idx}_return_mean"] = float(metrics.get("return_mean", 0.0))
            result[f"agent{agent_idx}_value_mean"] = 0.0
            result[f"agent{agent_idx}_explained_var"] = 0.0
            result[f"agent{agent_idx}_fresh_value_mean"] = 0.0
            result[f"agent{agent_idx}_fresh_explained_var"] = 0.0
        result.update(self._current_group_lrs())
        result.update(getattr(self, "_last_grpo_metrics", {}))
        return result

    def log_train_metrics(
        self,
        update_idx: int,
        loss_dict: Dict[str, float],
        transitions: List[TextTransition],
    ) -> None:
        super().log_train_metrics(update_idx, loss_dict, transitions)
        if not self.accelerator.is_main_process:
            return
        path = self.output_dir / "grpo_curve.csv"
        header = (
            "row_idx,update_idx,num_transitions,norm_scope,normalize,gamma,"
            "norm_bucket_count,agent0_unique_return_count,agent1_unique_return_count,"
            "return_mean,return_std,adv_mean,adv_std,"
            "agent0_adv_mean,agent0_return_mean,agent1_adv_mean,agent1_return_mean"
        )
        row_idx, _ = self._prepare_csv_log(path, header)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{row_idx},{update_idx},{len(transitions)},"
                f"{self.grpo_norm_scope},{self.grpo_normalize},{self.grpo_gamma},"
                f"{loss_dict.get('grpo/norm_bucket_count', 0.0)},"
                f"{loss_dict.get('grpo/agent0_unique_return_count', 0.0)},"
                f"{loss_dict.get('grpo/agent1_unique_return_count', 0.0)},"
                f"{loss_dict.get('grpo/return_mean', 0.0)},"
                f"{loss_dict.get('grpo/return_std', 0.0)},"
                f"{loss_dict.get('grpo/adv_mean', 0.0)},"
                f"{loss_dict.get('grpo/adv_std', 0.0)},"
                f"{loss_dict.get('agent0_adv_mean', 0.0)},"
                f"{loss_dict.get('agent0_return_mean', 0.0)},"
                f"{loss_dict.get('agent1_adv_mean', 0.0)},"
                f"{loss_dict.get('agent1_return_mean', 0.0)}\n"
            )
