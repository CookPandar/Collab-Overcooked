"""MARSHAL-style GRPO components for Collab-Overcooked."""

from .advantages import GrpoAdvantageConfig, compute_grpo_advantages
from .trainer import GRPOTrainer

__all__ = ["GrpoAdvantageConfig", "compute_grpo_advantages", "GRPOTrainer"]
