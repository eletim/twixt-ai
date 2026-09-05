"""Reproducible Mini Twixt MCTS dataset generation experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from functools import partial
from pathlib import Path
import platform
from time import perf_counter

from twixt_ai.game import experiment_board
from twixt_ai.search import MCTSAgent
from twixt_ai.training import DatasetConfig, build_dataset

from .batch import BatchConfig, run_batch


EXPERIMENT_FORMAT = "twixt-ai-mini-dataset-experiment"
EXPERIMENT_VERSION = 1


@dataclass(frozen=True, slots=True)
class StageConfig:
    """The varying settings for one generation stage."""

    name: str
    games: int
    seed: int
    split_seed: str


@dataclass(frozen=True, slots=True)
class MiniDatasetExperimentConfig:
    """Fixed Issue 56 experiment settings selected from the Mini benchmark."""

    simulations: int = 100
    rollout_limit: int = 4
    workers: int = 2
    shard_size: int = 10_000
    validation_fraction: float = 0.1
    smoke: StageConfig = StageConfig("smoke", 2, 560_001, "issue-56-smoke")
    baseline: StageConfig = StageConfig(
        "baseline", 100, 560_100, "issue-56-baseline"
    )

    def to_dict(self) -> dict[str, object]:
        def stage(value: StageConfig) -> dict[str, object]:
            return {
                "games": value.games,
                "seed": value.seed,
                "split_seed": value.split_seed,
            }

        return {
            "board": experiment_board("mini").to_dict(),
            "agents": {"red": "mcts", "black": "mcts"},
            "mcts": {
                "simulations": self.simulations,
                "rollout_limit": self.rollout_limit,
                "policy_value_network": False,
            },
            "workers": self.workers,
            "worker_mode": "process",
            "dataset": {
                "shard_size": self.shard_size,
                "validation_fraction": self.validation_fraction,
            },
            "stages": {
                "smoke": stage(self.smoke),
                "baseline": stage(self.baseline),
            },
            "python_hash_seed": "0",
        }


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _environment() -> dict[str, object]:
    return {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "available_cpus": _available_cpus(),
    }


def _run_stage(
    root: Path,
    stage: StageConfig,
    config: MiniDatasetExperimentConfig,
) -> dict[str, object]:
    stage_root = root / stage.name
    selfplay_root = stage_root / "selfplay"
    dataset_root = stage_root / "dataset"
    factory = partial(
        MCTSAgent,
        simulations=config.simulations,
        rollout_limit=config.rollout_limit,
    )
    batch_config = BatchConfig(
        games=stage.games,
        workers=config.workers,
        seed=stage.seed,
        board=experiment_board("mini"),
        red_agent="mcts",
        black_agent="mcts",
        worker_mode="process",
    )

    wall_started = perf_counter()
    batch = run_batch(
        factory,
        factory,
        config=batch_config,
        output_dir=selfplay_root,
    )
    wall_seconds = perf_counter() - wall_started

    dataset_started = perf_counter()
    dataset = build_dataset(
        selfplay_root,
        dataset_root,
        config=DatasetConfig(
            shard_size=config.shard_size,
            validation_fraction=config.validation_fraction,
            split_seed=stage.split_seed,
            metadata={
                "experiment": EXPERIMENT_FORMAT,
                "experiment_version": EXPERIMENT_VERSION,
                "stage": stage.name,
                "mcts": {
                    "simulations": config.simulations,
                    "rollout_limit": config.rollout_limit,
                    "policy_value_network": False,
                },
            },
        ),
    )
    dataset_seconds = perf_counter() - dataset_started
    return {
        "selfplay": {
            "wall_seconds": wall_seconds,
            "games_per_hour": batch.completed / wall_seconds * 3600.0,
            "summary": batch.to_dict(),
        },
        "dataset": {
            "wall_seconds": dataset_seconds,
            "manifest": dataset.to_dict(),
        },
    }


def run_mini_dataset_experiment(
    output_dir: str | Path,
    *,
    config: MiniDatasetExperimentConfig = MiniDatasetExperimentConfig(),
) -> dict[str, object]:
    """Run smoke then baseline generation and persist a measured report."""

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("output directory must be empty or not exist")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED must be 0")
    environment = _environment()
    root.mkdir(parents=True, exist_ok=True)

    stages = {
        stage.name: _run_stage(root, stage, config)
        for stage in (config.smoke, config.baseline)
    }
    report = {
        "format": EXPERIMENT_FORMAT,
        "version": EXPERIMENT_VERSION,
        "benchmark_evidence": {
            "artifact": "benchmarks/mini-selfplay-performance-optimized.json",
            "selection": (
                "100 simulations was the fastest measured nontrivial MCTS budget; "
                "two workers had the highest measured games/hour"
            ),
        },
        "config": config.to_dict(),
        "environment": environment,
        "stages": stages,
    }
    _write_json(root / "report.json", report)
    return report


__all__ = [
    "EXPERIMENT_FORMAT",
    "EXPERIMENT_VERSION",
    "MiniDatasetExperimentConfig",
    "StageConfig",
    "run_mini_dataset_experiment",
]
