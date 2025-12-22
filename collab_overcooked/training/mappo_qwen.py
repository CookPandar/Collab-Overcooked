"""MAPPO trainer tailored for Collab-Overcooked using a Qwen2.5 backbone.

This module intentionally keeps the implementation lightweight so it can serve as
an extensible baseline.  The training loop collects trajectories with the
``CollabOvercookedEnv`` wrapper, formats each agent's observation into a text
snippet, and feeds it through a shared Qwen actor-critic.  The same model
provides both policy logits and value estimates (actor/critic sharing).

The policy expects ``trainer.model_path`` in the YAML config to point to a
Hugging Face checkpoint (e.g. ``qwen/Qwen2.5-7B-Instruct``) that is available on
the current machine.  Loading semantics (LoRA, 4/8-bit, etc.) are left to the
user; the hooks below use standard ``transformers`` APIs so they can be wrapped
with PEFT if needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.distributions import Categorical

from transformers import AutoModel, AutoTokenizer

from .env_wrapper import CollabOvercookedEnv


def format_observation(obs: Dict[str, Any], agent_idx: int) -> str:
    """Convert structured state dict into a textual description for Qwen.

    This keeps the baseline simple; feel free to replace with a learned encoder.
    """

    player = obs["players"][agent_idx]
    teammate = obs["players"][1 - agent_idx]
    lines = [
        f"Timestep: {obs['timestep']}",
        f"Current orders: {', '.join(obs['orders'])}",
        "Chef" if agent_idx == 0 else "Assistant",
        f"Self position: {player['position']}, object: {player['object']}",
        f"Mate position: {teammate['position']}, object: {teammate['object']}",
        "Grid:\n" + obs["grid"],
    ]
    return "\n".join(lines)


class QwenActorCritic(nn.Module):
    """Shared actor-critic built on top of a Qwen encoder."""

    def __init__(self, model_path: str, action_dim: int, device: torch.device) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.backbone = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto" if device.type == "cuda" else None,
        )
        hidden_size = self.backbone.config.hidden_size
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.device = device
        self.to(device)

    def _encode(self, texts: Sequence[str]):
        tokens = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.backbone(**tokens)
        # Use the final token representation as a pooled feature
        hidden = outputs.last_hidden_state[:, -1, :]
        return hidden

    def forward(self, texts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(texts)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values

    def act(self, texts: Sequence[str]):
        logits, values = self.forward(texts)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values, dist

    def evaluate_actions(self, texts: Sequence[str], actions: torch.Tensor):
        logits, values = self.forward(texts)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


@dataclass
class Transition:
    obs_text: str
    action: int
    log_prob: float
    value: float
    reward: float
    done: float


class RolloutBuffer:
    def __init__(self):
        self.storage: List[Transition] = []

    def add(self, **kwargs):
        self.storage.append(Transition(**kwargs))

    def clear(self):
        self.storage.clear()


class MAPPOTrainer:
    """Minimal MAPPO loop with shared actor/critic."""

    def __init__(self, env_config: Dict[str, Any], trainer_cfg: Dict[str, Any]) -> None:
        self.accelerator = Accelerator()
        self.env = CollabOvercookedEnv(
            layout=env_config.get("layout", "cramped_room"),
            order=env_config.get("order", "boiled_egg"),
            horizon=env_config.get("horizon", 120),
            reward_settings=trainer_cfg.get("reward", {}),
        )
        # Let Accelerator decide device placement.
        self.device = self.accelerator.device
        self.gamma = trainer_cfg.get("gamma", 0.99)
        self.gae_lambda = trainer_cfg.get("gae_lambda", 0.95)
        self.steps_per_update = trainer_cfg.get("steps_per_update", 256)
        self.local_steps_per_update = max(
            1, math.ceil(self.steps_per_update / self.accelerator.num_processes)
        )
        self.total_updates = trainer_cfg.get("total_updates", 1000)
        self.clip_coef = trainer_cfg.get("clip_coef", 0.2)
        self.entropy_coef = trainer_cfg.get("entropy_coef", 0.01)
        self.value_coef = trainer_cfg.get("value_coef", 0.5)
        self.max_grad_norm = trainer_cfg.get("max_grad_norm", 0.5)

        action_dim = len(self.env.ACTION_ID)
        self.policy = QwenActorCritic(
            model_path=trainer_cfg["model_path"],
            action_dim=action_dim,
            device=self.device,
        )
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=trainer_cfg.get("lr", 1e-5)
        )
        self.policy, self.optimizer = self.accelerator.prepare(self.policy, self.optimizer)
        self.buffer = RolloutBuffer()

    # ------------------------------------------------------------------
    def train(self):
        for update_idx in range(1, self.total_updates + 1):
            self.collect_rollout()
            loss_dict = self.update_policy()
            self.accelerator.print(
                f"[MAPPO] Update {update_idx}/{self.total_updates} "
                f"loss={loss_dict['loss']:.4f} policy={loss_dict['policy']:.4f} "
                f"value={loss_dict['value']:.4f} entropy={loss_dict['entropy']:.4f}"
            )
        self.accelerator.wait_for_everyone()

    def collect_rollout(self):
        self.buffer.clear()
        obs = self.env.reset()
        step_count = 0
        done = False
        local_target = self.local_steps_per_update
        while step_count < local_target:
            texts = [format_observation(obs, idx) for idx in (0, 1)]
            actions, log_probs, values, _ = self.policy.act(texts)
            joint_action = [int(actions[0].item()), int(actions[1].item())]
            result = self.env.step(joint_action)
            reward = result.reward
            done = result.done
            for agent_idx in (0, 1):
                self.buffer.add(
                    obs_text=texts[agent_idx],
                    action=joint_action[agent_idx],
                    log_prob=log_probs[agent_idx].item(),
                    value=values[agent_idx].item(),
                    reward=reward,
                    done=float(done),
                )
            obs = result.observation if not done else self.env.reset()
            step_count += 1

    def compute_advantages(self):
        rewards = torch.tensor([t.reward for t in self.buffer.storage], dtype=torch.float32)
        values = torch.tensor([t.value for t in self.buffer.storage], dtype=torch.float32)
        dones = torch.tensor([t.done for t in self.buffer.storage], dtype=torch.float32)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0.0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update_policy(self):
        advantages, returns = self.compute_advantages()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_texts = [t.obs_text for t in self.buffer.storage]
        actions = torch.tensor([t.action for t in self.buffer.storage], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(
            [t.log_prob for t in self.buffer.storage], dtype=torch.float32, device=self.device
        )
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        log_probs, entropy, values = self.policy.evaluate_actions(obs_texts, actions)
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "entropy": entropy.mean().item(),
        }


__all__ = ["MAPPOTrainer"]
