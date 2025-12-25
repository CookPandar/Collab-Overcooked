"""
Experimental entrypoint that extends :mod:`collab_overcooked.main` with an
optional RL training mode.  When the loaded YAML config contains a ``trainer``
section (e.g. ``trainer.type = mappo``), we dispatch to the corresponding
trainer; otherwise the original evaluation pipeline is executed unchanged.

This keeps command-line usage simple: existing configs without a ``trainer``
block behave as before, while an RL-specific config can instruct the runner to
launch the MAPPO baseline powered by Qwen2.5-7B-Instruct.
"""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, Dict, Optional

from .main import load_config_from_yaml, main as legacy_main

try:
    from .training.mappo_qwen import MAPPOTrainer
except ImportError:  # pragma: no cover - optional dependency
    MAPPOTrainer = None  # type: ignore

try:
    from .training.mappo_deepspeed import DeepSpeedMAPPOTrainer
except ImportError:  # pragma: no cover - optional dependency
    DeepSpeedMAPPOTrainer = None  # type: ignore


def _maybe_run_trainer(config: Dict[str, Any]) -> bool:
    """Return True if a trainer was executed."""
    trainer_cfg = (config or {}).get("trainer")
    if not trainer_cfg:
        return False

    trainer_type = trainer_cfg.get("type", "").lower()
    trainer_cls = None
    if trainer_type in {"mappo", "mappo_accelerate"}:
        trainer_cls = MAPPOTrainer
    elif trainer_type in {"mappo_deepspeed", "deepspeed_mappo", "ds_mappo"}:
        trainer_cls = DeepSpeedMAPPOTrainer
    else:
        raise ValueError(
            f"Unsupported trainer type '{trainer_type}'. "
            "Expected 'mappo' or 'mappo_deepspeed'."
        )

    if trainer_cls is None:
        missing = "MAPPOTrainer" if trainer_type.startswith("mappo") and "deepspeed" not in trainer_type else "DeepSpeedMAPPOTrainer"
        raise ImportError(f"{missing} is unavailable. Ensure dependencies are installed.")

    env_cfg = config.get("environment", {})
    trainer = trainer_cls(env_cfg, trainer_cfg, full_config=config)
    trainer.train()
    return True


def run(variant: Optional[Dict[str, Any]] = None, config_path: Optional[str] = None):
    """
    Execute either the RL trainer (when configured) or fall back to legacy main.
    """
    base_config: Optional[Dict[str, Any]] = None
    if config_path:
        base_config = load_config_from_yaml(config_path)
    elif variant and variant.get("yaml_config"):
        base_config = variant["yaml_config"]

    if base_config and _maybe_run_trainer(base_config):
        return

    # No trainer configured -> behave like original main
    legacy_main(variant=variant, config_path=config_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Collab-Overcooked RL / Evaluation runner")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Process rank for DeepSpeed / torch.distributed launchers.",
    )
    args = parser.parse_args()
    run(config_path=args.config)
