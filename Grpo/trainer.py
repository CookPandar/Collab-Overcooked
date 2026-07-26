"""GRPO trainer built on top of the existing Collab-Overcooked rollout stack."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

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
        self.grpo_whiten_advantages = bool(
            grpo_cfg.get(
                "whiten_advantages",
                cfg.get("whiten_advantages", False),
            )
        )
        self.grpo_positive_adv_requires_positive_return = bool(
            grpo_cfg.get(
                "positive_advantage_requires_positive_return",
                cfg.get("positive_advantage_requires_positive_return", False),
            )
        )
        self.grpo_positive_return_threshold = float(
            grpo_cfg.get(
                "positive_return_threshold",
                cfg.get("positive_return_threshold", 0.0),
            )
        )
        self.grpo_positive_only_advantages = bool(
            grpo_cfg.get(
                "positive_only_advantages",
                cfg.get("positive_only_advantages", False),
            )
        )
        self.grpo_positive_return_source = str(
            grpo_cfg.get(
                "positive_return_source",
                cfg.get("positive_return_source", "reward"),
            )
        ).strip().lower()
        self.grpo_recompute_old_log_probs = bool(
            grpo_cfg.get(
                "recompute_old_log_probs",
                cfg.get("recompute_old_log_probs", False),
            )
        )
        self.grpo_include_rewarded_filtered_actions = bool(
            grpo_cfg.get(
                "include_rewarded_filtered_actions",
                cfg.get("grpo_include_rewarded_filtered_actions", True),
            )
        )
        train_action_modes = grpo_cfg.get(
            "train_action_modes",
            cfg.get("grpo_train_action_modes", None),
        )
        if train_action_modes is None:
            self.grpo_train_action_modes = None
        elif isinstance(train_action_modes, str):
            self.grpo_train_action_modes = {
                item.strip().lower()
                for item in train_action_modes.split(",")
                if item.strip()
            }
        else:
            self.grpo_train_action_modes = {
                str(item).strip().lower()
                for item in train_action_modes
                if str(item).strip()
            }
        self.dual_clip_loss = bool(cfg.get("dual_clip_loss", False))
        self.accelerator.print(
            "[GRPO] credit assignment "
            f"norm_scope={self.grpo_norm_scope} normalize={self.grpo_normalize} "
            f"gamma={self.grpo_gamma} clip={self.grpo_clip} "
            f"whiten_advantages={self.grpo_whiten_advantages} "
            f"dual_clip_loss={self.dual_clip_loss} "
            f"positive_adv_requires_positive_return="
            f"{self.grpo_positive_adv_requires_positive_return} "
            f"positive_only_advantages={self.grpo_positive_only_advantages} "
            f"positive_return_source={self.grpo_positive_return_source} "
            f"recompute_old_log_probs={self.grpo_recompute_old_log_probs} "
            f"train_action_modes={sorted(self.grpo_train_action_modes) if self.grpo_train_action_modes is not None else 'all'} "
            f"include_rewarded_filtered_actions={self.grpo_include_rewarded_filtered_actions}"
        )
        self._precomputed_advantages: Optional[torch.Tensor] = None
        self._precomputed_returns: Optional[torch.Tensor] = None
        self._precomputed_metrics: Dict[str, float] = {}

    def _positive_return_rewards(
        self,
        transitions: List[TextTransition],
    ) -> Optional[List[float]]:
        source = (self.grpo_positive_return_source or "reward").strip().lower()
        if source in {"", "reward", "total", "total_reward"}:
            return None
        if source in {"task", "partial", "partial_success", "partial_success_reward"}:
            return [float(getattr(t, "partial_success_reward", 0.0) or 0.0) for t in transitions]
        if source in {"env", "environment", "terminal", "terminal_reward"}:
            return [
                float(t.reward)
                - float(getattr(t, "format_reward", 0.0) or 0.0)
                - float(getattr(t, "validator_reward", 0.0) or 0.0)
                - float(getattr(t, "process_reward", 0.0) or 0.0)
                - float(getattr(t, "sequence_reward", 0.0) or 0.0)
                - float(getattr(t, "communication_reward", 0.0) or 0.0)
                - float(getattr(t, "repeat_communication_reward", 0.0) or 0.0)
                - float(getattr(t, "forced_communication_reward", 0.0) or 0.0)
                - float(getattr(t, "collab_reward", 0.0) or 0.0)
                - float(getattr(t, "paired_comm_reward", 0.0) or 0.0)
                for t in transitions
            ]
        return [float(getattr(t, source, 0.0) or 0.0) for t in transitions]

    def _critic_pretrain_enabled(self) -> bool:
        return False

    def compute_advantages(self, transitions: List[TextTransition]):
        precomputed = self._consume_precomputed_advantages(transitions)
        if precomputed is not None:
            advantages, returns = precomputed
            self._last_grpo_metrics = dict(self._precomputed_metrics)
            return advantages, returns

        config = GrpoAdvantageConfig(
            norm_scope=self.grpo_norm_scope,
            normalize=self.grpo_normalize,
            gamma=self.grpo_gamma,
            eps=self.grpo_eps,
            clip=float(self.grpo_clip) if self.grpo_clip is not None else None,
            whiten_advantages=self.grpo_whiten_advantages,
            positive_advantage_requires_positive_return=(
                self.grpo_positive_adv_requires_positive_return
            ),
            positive_return_threshold=self.grpo_positive_return_threshold,
            positive_only_advantages=self.grpo_positive_only_advantages,
        )
        advantages, returns, metrics = compute_grpo_advantages(
            rewards=[t.reward for t in transitions],
            agent_indices=[int(t.agent_index) for t in transitions],
            timesteps=[t.timestep for t in transitions],
            trajectory_ids=[
                t.rollout_id if t.rollout_id is not None else "__default_rollout__"
                for t in transitions
            ],
            positive_return_rewards=self._positive_return_rewards(transitions),
            config=config,
        )
        self._last_grpo_metrics = metrics
        return advantages, returns

    def _compute_global_grpo_advantages(self, transitions: List[TextTransition]) -> Tuple[torch.Tensor, torch.Tensor]:
        config = GrpoAdvantageConfig(
            norm_scope=self.grpo_norm_scope,
            normalize=self.grpo_normalize,
            gamma=self.grpo_gamma,
            eps=self.grpo_eps,
            clip=float(self.grpo_clip) if self.grpo_clip is not None else None,
            whiten_advantages=self.grpo_whiten_advantages,
            positive_advantage_requires_positive_return=(
                self.grpo_positive_adv_requires_positive_return
            ),
            positive_return_threshold=self.grpo_positive_return_threshold,
            positive_only_advantages=self.grpo_positive_only_advantages,
        )
        advantages, returns, metrics = compute_grpo_advantages(
            rewards=[t.reward for t in transitions],
            agent_indices=[int(t.agent_index) for t in transitions],
            timesteps=[t.timestep for t in transitions],
            trajectory_ids=[
                t.rollout_id if t.rollout_id is not None else "__default_rollout__"
                for t in transitions
            ],
            positive_return_rewards=self._positive_return_rewards(transitions),
            config=config,
        )
        self._precomputed_metrics = metrics
        return advantages, returns

    def _set_precomputed_advantages(
        self,
        transitions: List[TextTransition],
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> None:
        if advantages.numel() != len(transitions) or returns.numel() != len(transitions):
            raise ValueError("precomputed advantages/returns must match transition count.")
        self._precomputed_advantages = advantages.detach().cpu()
        self._precomputed_returns = returns.detach().cpu()

    def _consume_precomputed_advantages(
        self,
        transitions: List[TextTransition],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self._precomputed_advantages is None or self._precomputed_returns is None:
            return None
        if self._precomputed_advantages.numel() != len(transitions):
            raise ValueError(
                "precomputed advantages length does not match local transition shard: "
                f"{self._precomputed_advantages.numel()} vs {len(transitions)}"
            )
        advantages = self._precomputed_advantages
        returns = self._precomputed_returns
        self._precomputed_advantages = None
        self._precomputed_returns = None
        return advantages.clone(), returns.clone()

    def _current_group_lrs(self) -> Dict[str, float]:
        current = super()._current_group_lrs()
        current["critic_adapter_lr"] = 0.0
        current["value_head_lr"] = 0.0
        return current

    def _success_replay_bc_active(self, transition: TextTransition) -> bool:
        action_mode = (
            str(transition.action_mode or "").strip().lower()
            if transition.action_mode is not None
            else ""
        )
        if action_mode in {"malformed", "mixed", "multi_embodied"}:
            return False
        if self.grpo_train_action_modes is None:
            return True
        if action_mode in self.grpo_train_action_modes:
            return True
        return False

    def _update_success_replay_bc(
        self,
        unwrapped_model: QwenLMActorCritic,
        param_groups: Dict[str, List[torch.nn.Parameter]],
    ) -> Dict[str, float]:
        transitions = list(getattr(self, "_pending_success_replay_transitions", []) or [])
        weight = float(getattr(self, "success_replay_bc_weight", 0.0) or 0.0)
        if not transitions or weight <= 0.0:
            return {
                "success_replay/bc_transitions": float(len(transitions)),
                "success_replay/bc_active_transitions": 0.0,
                "success_replay/bc_loss": 0.0,
                "success_replay/bc_weight": weight,
                "success_replay/bc_optimizer_steps": 0.0,
            }

        active_mask = [self._success_replay_bc_active(t) for t in transitions]
        active_transition_count = int(sum(1 for item in active_mask if item))
        if active_transition_count <= 0:
            self.accelerator.print(
                "[GRPO] success replay BC skipped: no active replay transitions."
            )
            return {
                "success_replay/bc_transitions": float(len(transitions)),
                "success_replay/bc_active_transitions": 0.0,
                "success_replay/bc_loss": 0.0,
                "success_replay/bc_weight": weight,
                "success_replay/bc_optimizer_steps": 0.0,
            }

        batch_size = max(1, self.train_batch_size)
        prompt_tensors = [t.prompt_ids for t in transitions]
        response_tensors = [t.response_ids for t in transitions]
        agent_indices = [t.agent_index for t in transitions]
        policy_temperatures = [t.policy_temperature for t in transitions]

        total_loss = 0.0
        metric_steps = 0
        optimizer_steps = 0
        accum_counter = 0
        self.optimizer.zero_grad()
        self.accelerator.print(
            "[GRPO] success replay BC "
            f"transitions={len(transitions)} active={active_transition_count} "
            f"batch_size={batch_size} weight={weight}"
        )
        for start in range(0, len(transitions), batch_size):
            end = min(start + batch_size, len(transitions))
            batch_prompts = prompt_tensors[start:end]
            batch_responses = response_tensors[start:end]
            batch_agent_indices = agent_indices[start:end]
            batch_active_mask = active_mask[start:end]

            _, _, token_log_probs = unwrapped_model.evaluate_policy_batch(
                batch_prompts,
                batch_responses,
                batch_agent_indices,
                policy_temperatures=policy_temperatures[start:end],
            )
            sample_losses: List[torch.Tensor] = []
            for is_active, token_lp in zip(batch_active_mask, token_log_probs):
                if not is_active or token_lp.numel() == 0:
                    continue
                sample_losses.append(-token_lp.mean())
            zero_anchor = (
                sum((token_lp.sum() * 0.0) for token_lp in token_log_probs)
                if token_log_probs
                else torch.zeros((), dtype=torch.float32, device=self.device)
            )
            bc_loss = (
                torch.stack(sample_losses).mean()
                if sample_losses
                else zero_anchor
            )
            active_global = self._any_rank_has_active_policy_minibatch(bool(sample_losses))
            if active_global:
                self.accelerator.backward(
                    (weight * bc_loss) / float(self.gradient_accumulation_steps)
                )
                accum_counter += 1
            should_step = (
                accum_counter >= self.gradient_accumulation_steps
                or (end >= len(transitions) and accum_counter > 0)
            )
            if should_step:
                self.accelerator.clip_grad_norm_(
                    self.text_policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                unwrapped_model.clear_prefix_cache()
                self.optimizer.zero_grad()
                optimizer_steps += 1
                accum_counter = 0
            total_loss += float(bc_loss.detach().item())
            metric_steps += 1

        return {
            "success_replay/bc_transitions": float(len(transitions)),
            "success_replay/bc_active_transitions": float(active_transition_count),
            "success_replay/bc_loss": total_loss / float(max(1, metric_steps)),
            "success_replay/bc_weight": weight,
            "success_replay/bc_optimizer_steps": float(optimizer_steps),
        }

    def _materialize_transition_values(self, transitions: List[TextTransition]) -> None:
        self.accelerator.print(
            f"[GRPO] skip materialize_transition_values transitions={len(transitions)}"
        )
        return

    @staticmethod
    def _transition_has_reward_signal(transition: TextTransition) -> bool:
        fields = (
            "reward",
            "partial_success_reward",
            "sequence_reward",
            "communication_reward",
            "repeat_communication_reward",
            "forced_communication_reward",
            "collab_reward",
            "paired_comm_reward",
            "format_reward",
            "validator_reward",
        )
        for field in fields:
            if abs(float(getattr(transition, field, 0.0) or 0.0)) > 1e-9:
                return True
        return False

    def _shard_transitions_for_rank(
        self, transitions: List[TextTransition]
    ) -> List[TextTransition]:
        global_advantages, global_returns = self._compute_global_grpo_advantages(transitions)
        world_size = max(1, int(self.accelerator.num_processes))
        rank = int(self.accelerator.process_index)
        if world_size <= 1 or not transitions:
            self._set_precomputed_advantages(transitions, global_advantages, global_returns)
            return list(transitions)

        total = len(transitions)
        per_rank = int(math.ceil(total / float(world_size)))
        padded_size = per_rank * world_size
        indices = list(range(total))
        if padded_size > total:
            indices.extend(indices[: padded_size - total])

        start = rank * per_rank
        end = start + per_rank
        valid_count = max(0, min(end, total) - start)
        shard_indices = indices[start:end]
        shard = [transitions[idx] for idx in shard_indices]
        self._last_local_shard_valid_count = valid_count
        self._set_precomputed_advantages(
            shard,
            global_advantages[shard_indices],
            global_returns[shard_indices],
        )
        print(
            "[GRPO] shard after global advantage "
            f"rank={rank}/{world_size} global={total} local={len(shard)} "
            f"padded={padded_size}",
            flush=True,
        )
        self.accelerator.print(
            "[TrainOnly] GRPO rollout shard with global baseline "
            f"rank={rank}/{world_size} global={total} local={len(shard)} "
            f"padded={padded_size}"
        )
        return shard

    def update_policy(self, transitions: List[TextTransition]):
        assert self.text_policy is not None
        unwrapped_model: QwenLMActorCritic = self.accelerator.unwrap_model(self.text_policy)
        rank = self.accelerator.process_index
        prompt_tensors = [t.prompt_ids for t in transitions]
        response_tensors = [t.response_ids for t in transitions]
        agent_indices = [t.agent_index for t in transitions]
        policy_temperatures = [t.policy_temperature for t in transitions]
        action_modes = [
            str(t.action_mode or "").strip().lower() if t.action_mode is not None else ""
            for t in transitions
        ]
        loss_weights = torch.ones(len(transitions), dtype=torch.float32)
        if self.grpo_train_action_modes is not None:
            loss_weights = torch.tensor(
                [
                    1.0
                    if (
                        action_mode in self.grpo_train_action_modes
                        or (
                            self.grpo_include_rewarded_filtered_actions
                            and self._transition_has_reward_signal(transition)
                        )
                    )
                    else 0.0
                    for action_mode, transition in zip(action_modes, transitions)
                ],
                dtype=torch.float32,
            )
        active_transition_count = int(loss_weights.sum().item())
        skipped_transition_count = int(len(transitions) - active_transition_count)
        if self.grpo_train_action_modes is not None and active_transition_count == 0:
            self.accelerator.print(
                "[GRPO] train_action_modes filtered every transition in this update."
            )
        stored_old_token_log_probs_list: List[torch.Tensor] = []
        old_token_log_probs_list: List[torch.Tensor] = []
        for t in transitions:
            if t.response_log_probs is not None:
                stored_old = t.response_log_probs.to(self.device, dtype=torch.float32)
            else:
                resp_len = max(1, int(t.response_ids.numel()))
                stored_old = torch.full(
                    (resp_len,),
                    float(t.log_prob) / float(resp_len),
                    dtype=torch.float32,
                    device=self.device,
                )
            stored_old_token_log_probs_list.append(stored_old)
        old_token_log_probs_list = list(stored_old_token_log_probs_list)
        old_logprob_abs_diff_sum = 0.0
        old_logprob_abs_diff_count = 0
        if self.grpo_recompute_old_log_probs and transitions:
            recomputed_old: List[torch.Tensor] = []
            old_logprob_batch_size = max(
                1,
                int(
                    getattr(
                        unwrapped_model,
                        "eval_batch_size",
                        self.trainer_cfg.get("evaluation_batch_size", 4),
                    )
                ),
            )
            with torch.no_grad():
                for start in range(0, len(transitions), old_logprob_batch_size):
                    end = min(start + old_logprob_batch_size, len(transitions))
                    _, _, token_log_probs = unwrapped_model.evaluate_policy_batch(
                        prompt_tensors[start:end],
                        response_tensors[start:end],
                        agent_indices[start:end],
                        policy_temperatures=policy_temperatures[start:end],
                    )
                    recomputed_old.extend(
                        [item.detach().to(self.device, dtype=torch.float32) for item in token_log_probs]
                    )
            for recomputed, stored in zip(recomputed_old, stored_old_token_log_probs_list):
                lhs = recomputed
                rhs = stored
                if lhs.numel() != rhs.numel():
                    limit = min(lhs.numel(), rhs.numel())
                    lhs = lhs[:limit]
                    rhs = rhs[:limit]
                if lhs.numel() > 0:
                    old_logprob_abs_diff_sum += float((lhs - rhs).abs().sum().item())
                    old_logprob_abs_diff_count += int(lhs.numel())
            if len(recomputed_old) == len(transitions):
                old_token_log_probs_list = recomputed_old
            self.accelerator.print(
                "[GRPO] recomputed old logprobs "
                f"tokens={old_logprob_abs_diff_count} "
                f"mean_abs_diff={old_logprob_abs_diff_sum / max(1, old_logprob_abs_diff_count):.6f}"
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
                batch_loss_weights = loss_weights[batch_indices].to(self.device)

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
                    if float(batch_loss_weights[sample_idx].item()) <= 0.0:
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
                    else entropies.sum() * 0.0
                )
                active_minibatch = bool(token_policy_loss_parts)
                active_flag = torch.tensor(
                    1.0 if active_minibatch else 0.0,
                    dtype=torch.float32,
                    device=self.device,
                )
                active_sum = None
                try:
                    if hasattr(self.accelerator, "reduce"):
                        active_sum = self.accelerator.reduce(
                            active_flag,
                            reduction="sum",
                        )
                except Exception:
                    active_sum = None
                if active_sum is None:
                    gathered_active = self.accelerator.gather(active_flag)
                    active_sum = gathered_active.sum()
                active_minibatch_global = float(active_sum.item()) > 0.0
                approx_kl = (
                    ((token_ratios - 1.0) - token_logratio).mean().clamp_min(0.0)
                    if token_logratio.numel() > 0
                    else entropies.sum() * 0.0
                )
                kl_penalty = approx_kl * self.kl_penalty_coef
                active_entropy = entropies[batch_loss_weights > 0.0]
                entropy_mean = (
                    active_entropy.mean()
                    if active_entropy.numel() > 0
                    else entropies.sum() * 0.0
                )
                entropy_loss = -entropy_mean
                loss = policy_loss + self.entropy_coef * entropy_loss + kl_penalty

                if active_minibatch_global:
                    self.accelerator.backward(loss / float(self.gradient_accumulation_steps))
                    accum_counter += 1
                    accum_approx_kl_sum += float(approx_kl.item())
                    accum_approx_kl_count += 1
                should_step = (
                    accum_counter >= self.gradient_accumulation_steps
                    or (end >= num_transitions and accum_counter > 0)
                )
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
                total_entropy += float(entropy_mean.item())
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

        success_replay_bc_metrics = self._update_success_replay_bc(
            unwrapped_model,
            param_groups,
        )
        self._pending_success_replay_transitions = []

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
            "grpo/active_loss_transition_count": float(active_transition_count),
            "grpo/skipped_loss_transition_count": float(skipped_transition_count),
            "grpo/active_loss_transition_frac": (
                float(active_transition_count) / float(len(transitions))
                if transitions
                else 0.0
            ),
            "grpo/old_logprob_mean_abs_diff": (
                old_logprob_abs_diff_sum / float(old_logprob_abs_diff_count)
                if old_logprob_abs_diff_count > 0
                else 0.0
            ),
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
        result.update(success_replay_bc_metrics)
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
            "row_idx,update_idx,num_transitions,global_transition_count,trajectory_count,"
            "norm_scope,normalize,gamma,"
            "norm_bucket_count,agent0_unique_return_count,agent1_unique_return_count,"
            "return_mean,return_std,adv_mean,adv_std,"
            "agent0_adv_mean,agent0_return_mean,agent1_adv_mean,agent1_return_mean,"
            "positive_gate_return_mean,nonpositive_return_positive_adv_zeroed,"
            "negative_adv_zeroed,positive_adv_count,negative_adv_count,zero_adv_count,"
            "active_loss_transition_count,skipped_loss_transition_count,active_loss_transition_frac,"
            "old_logprob_mean_abs_diff,"
            "success_replay_bc_transitions,success_replay_bc_active_transitions,"
            "success_replay_bc_loss,success_replay_bc_weight,success_replay_bc_optimizer_steps"
        )
        row_idx, _ = self._prepare_csv_log(path, header)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{row_idx},{update_idx},{len(transitions)},"
                f"{loss_dict.get('grpo/global_transition_count', len(transitions))},"
                f"{loss_dict.get('grpo/trajectory_count', 1.0)},"
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
                f"{loss_dict.get('agent1_return_mean', 0.0)},"
                f"{loss_dict.get('grpo/positive_gate_return_mean', 0.0)},"
                f"{loss_dict.get('grpo/nonpositive_return_positive_adv_zeroed', 0.0)},"
                f"{loss_dict.get('grpo/negative_adv_zeroed', 0.0)},"
                f"{loss_dict.get('grpo/positive_adv_count', 0.0)},"
                f"{loss_dict.get('grpo/negative_adv_count', 0.0)},"
                f"{loss_dict.get('grpo/zero_adv_count', 0.0)},"
                f"{loss_dict.get('grpo/active_loss_transition_count', len(transitions))},"
                f"{loss_dict.get('grpo/skipped_loss_transition_count', 0.0)},"
                f"{loss_dict.get('grpo/active_loss_transition_frac', 1.0)},"
                f"{loss_dict.get('grpo/old_logprob_mean_abs_diff', 0.0)},"
                f"{loss_dict.get('success_replay/bc_transitions', 0.0)},"
                f"{loss_dict.get('success_replay/bc_active_transitions', 0.0)},"
                f"{loss_dict.get('success_replay/bc_loss', 0.0)},"
                f"{loss_dict.get('success_replay/bc_weight', 0.0)},"
                f"{loss_dict.get('success_replay/bc_optimizer_steps', 0.0)}\n"
            )
