"""MARSHAL-style GRPO components for Collab-Overcooked."""

from .advantages import GrpoAdvantageConfig, compute_grpo_advantages

__all__ = ["GrpoAdvantageConfig", "compute_grpo_advantages", "GRPOTrainer"]


def __getattr__(name):
    if name == "GRPOTrainer":
        from .trainer import GRPOTrainer

        return GRPOTrainer
    raise AttributeError(name)
