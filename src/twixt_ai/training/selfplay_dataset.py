"""Reproducible Mini champion self-play and dataset-only generation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import torch

from twixt_ai.device import select_device
from twixt_ai.models import MINI_POLICY_VALUE_CONFIG

from .data import DatasetConfig, build_dataset
from .generations import (
    MiniGenerationConfig,
    _checkpoint,
    _game_paths,
    _run_selfplay,
    _target_distributions,
    _write_json,
)


SELFPLAY_DATASET_FORMAT = "twixt-ai-v2-selfplay-dataset-run"
SELFPLAY_DATASET_VERSION = 1
_GENERATION_DEFAULTS = MiniGenerationConfig()


@dataclass(frozen=True, slots=True)
class MiniSelfplayDatasetConfig:
    """The self-play and dataset subset of the Mini generation schedule."""

    games: int = _GENERATION_DEFAULTS.games_per_generation
    simulations: int = _GENERATION_DEFAULTS.selfplay_simulations
    exploration: float = _GENERATION_DEFAULTS.selfplay_exploration
    rollout_limit: int = _GENERATION_DEFAULTS.rollout_limit
    progressive_widening_constant: float = (
        _GENERATION_DEFAULTS.selfplay_progressive_widening_constant
    )
    progressive_widening_exponent: float = (
        _GENERATION_DEFAULTS.selfplay_progressive_widening_exponent
    )
    workers: int = _GENERATION_DEFAULTS.workers
    inference_batch_size: int = _GENERATION_DEFAULTS.inference_batch_size
    inference_max_wait_seconds: float = (
        _GENERATION_DEFAULTS.inference_max_wait_seconds
    )
    validation_fraction: float = _GENERATION_DEFAULTS.validation_fraction
    shard_size: int = _GENERATION_DEFAULTS.shard_size
    seed: int = _GENERATION_DEFAULTS.seed
    split_seed: str | None = None
    workflow_label: str = SELFPLAY_DATASET_FORMAT
    device: str = _GENERATION_DEFAULTS.device

    def __post_init__(self) -> None:
        # Delegate shared search and execution validation to the generation
        # configuration whose defaults and semantics this command preserves.
        self.generation_config()
        if self.split_seed is not None and (
            not isinstance(self.split_seed, str) or not self.split_seed.strip()
        ):
            raise ValueError("split_seed must be a non-empty string or None")
        if not isinstance(self.workflow_label, str) or not self.workflow_label.strip():
            raise ValueError("workflow_label must be a non-empty string")

    @property
    def resolved_split_seed(self) -> str:
        return self.split_seed or f"twixt-ai-mini-selfplay-dataset-{self.seed}"

    def generation_config(self) -> MiniGenerationConfig:
        """Translate without changing the existing generation search path."""

        return MiniGenerationConfig(
            generations=1,
            games_per_generation=self.games,
            selfplay_simulations=self.simulations,
            selfplay_exploration=self.exploration,
            selfplay_progressive_widening_constant=(
                self.progressive_widening_constant
            ),
            selfplay_progressive_widening_exponent=(
                self.progressive_widening_exponent
            ),
            rollout_limit=self.rollout_limit,
            workers=self.workers,
            inference_batch_size=self.inference_batch_size,
            inference_max_wait_seconds=self.inference_max_wait_seconds,
            validation_fraction=self.validation_fraction,
            shard_size=self.shard_size,
            seed=self.seed,
            device=self.device,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_mini_selfplay_dataset(
    champion_path: str | Path,
    output_dir: str | Path,
    *,
    config: MiniSelfplayDatasetConfig = MiniSelfplayDatasetConfig(),
) -> dict[str, Any]:
    """Generate Mini self-play games and their dataset without training."""

    if not isinstance(config, MiniSelfplayDatasetConfig):
        raise TypeError("config must be a MiniSelfplayDatasetConfig")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED must be 0")

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("output directory must be empty or not exist")

    generation_config = config.generation_config()
    device = select_device(config.device)
    champion = Path(champion_path)
    champion_record = _checkpoint(champion)
    if champion_record["model_config"] != MINI_POLICY_VALUE_CONFIG.to_dict():
        raise ValueError("champion must use the Mini model configuration")

    root.mkdir(parents=True, exist_ok=True)

    search = {
        "simulations": config.simulations,
        "exploration": config.exploration,
        "rollout_limit": config.rollout_limit,
        "progressive_widening_constant": config.progressive_widening_constant,
        "progressive_widening_exponent": config.progressive_widening_exponent,
        "guidance": "policy-value",
    }
    resolved_config = {
        "board": generation_config.to_dict()["board"],
        "games": config.games,
        "search": search,
        "execution": {
            "workers": config.workers,
            "worker_mode": (
                "thread" if device.resolved_device == "cuda" else "process"
            ),
            "inference_batch_size": config.inference_batch_size,
            "inference_max_wait_seconds": config.inference_max_wait_seconds,
            "device": device.to_dict(),
        },
        "dataset": {
            "validation_fraction": config.validation_fraction,
            "shard_size": config.shard_size,
        },
    }
    seeds = {
        "selfplay": config.seed,
        "dataset_split": config.resolved_split_seed,
    }
    _write_json(
        root / "config.json",
        {"resolved_config": resolved_config, "seeds": seeds},
    )

    started = perf_counter()
    selfplay_started = perf_counter()
    batch, inference = _run_selfplay(
        champion, root / "selfplay", generation_config, config.seed, device
    )
    selfplay_seconds = perf_counter() - selfplay_started
    if batch.failed:
        raise RuntimeError(f"self-play had {batch.failed} failed games")

    dataset_started = perf_counter()
    dataset = build_dataset(
        _game_paths([root / "selfplay"]),
        root / "dataset",
        config=DatasetConfig(
            shard_size=config.shard_size,
            validation_fraction=config.validation_fraction,
            split_seed=config.resolved_split_seed,
            metadata={
                "workflow": config.workflow_label,
                "champion_sha256": champion_record["sha256"],
                "mcts": search,
            },
        ),
    )
    manifest = dataset.to_dict()
    dataset_seconds = perf_counter() - dataset_started
    target_distributions = _target_distributions(root / "dataset", manifest)
    report = {
        "format": SELFPLAY_DATASET_FORMAT,
        "version": SELFPLAY_DATASET_VERSION,
        "status": "completed",
        "champion": {
            "path": str(champion),
            "sha256": champion_record["sha256"],
            "bytes": champion_record["bytes"],
        },
        "resolved_config": resolved_config,
        "seeds": seeds,
        "environment": {
            "python": platform.python_version(),
            "pytorch": str(torch.__version__),
            "device": device.to_dict(),
        },
        "selfplay": {
            "runtime_seconds": selfplay_seconds,
            "games_per_hour": batch.completed / selfplay_seconds * 3600.0,
            "summary_sha256": _sha256(root / "selfplay" / "summary.json"),
            "summary": batch.to_dict(),
            "inference": inference,
        },
        "dataset": {
            "runtime_seconds": dataset_seconds,
            "manifest": manifest,
            "manifest_sha256": _sha256(root / "dataset" / "manifest.json"),
            "target_distributions": target_distributions,
        },
        "runtime_seconds": perf_counter() - started,
    }
    _write_json(root / "report.json", report)
    return report


__all__ = [
    "SELFPLAY_DATASET_FORMAT",
    "SELFPLAY_DATASET_VERSION",
    "MiniSelfplayDatasetConfig",
    "run_mini_selfplay_dataset",
]
