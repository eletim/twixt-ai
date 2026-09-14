"""Measured, staged neural self-play generation for Issue 88."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from twixt_ai.device import select_device
from twixt_ai.game import experiment_board
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralInferenceBatcher, NeuralPolicyValue
from twixt_ai.training import DatasetConfig, build_dataset

from .batch import BatchConfig, run_batch

EXPERIMENT_FORMAT = "twixt-ai-large-mini-dataset-experiment"
EXPERIMENT_VERSION = 1


@dataclass(frozen=True, slots=True)
class LargeStageConfig:
    """One independently durable generation stage."""

    name: str
    games: int
    seed: int
    split_seed: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or self.name in {"", ".", ".."}
            or any(character in self.name for character in "/\\:\0")
        ):
            raise ValueError("stage name must be a safe single path component")
        if (
            isinstance(self.games, bool)
            or not isinstance(self.games, int)
            or self.games < 1
        ):
            raise ValueError("stage games must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("stage seed must be an integer")
        if not isinstance(self.split_seed, str):
            raise TypeError("stage split_seed must be a string")


@dataclass(frozen=True, slots=True)
class LargeMiniDatasetConfig:
    """Settings selected by the checked-in Issue 87 CUDA benchmark."""

    simulations: int = 4
    rollout_limit: int = 4
    workers: int = 4
    inference_batch_size: int = 8
    inference_max_wait_seconds: float = 0.0005
    shard_size: int = 10_000
    validation_fraction: float = 0.1
    maximum_scale_runtime_hours: float = 1.0
    stages: tuple[LargeStageConfig, ...] = (
        LargeStageConfig("1k", 1_000, 880_001_000, "issue-88-1k"),
        LargeStageConfig("5k", 5_000, 880_005_000, "issue-88-5k"),
    )

    def __post_init__(self) -> None:
        for name in (
            "simulations",
            "rollout_limit",
            "workers",
            "inference_batch_size",
            "shard_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.validation_fraction <= 1:
            raise ValueError("validation_fraction must be in [0, 1]")
        if self.inference_max_wait_seconds < 0:
            raise ValueError("inference_max_wait_seconds must be non-negative")
        if self.maximum_scale_runtime_hours <= 0:
            raise ValueError("maximum_scale_runtime_hours must be positive")
        stages = tuple(self.stages)
        if not stages or (len(stages) > 1 and stages[0].games != 1_000):
            # Tests may use one tiny stage; production scale-up must start at 1k.
            raise ValueError("multi-stage experiments must start with 1,000 games")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("stage names must be distinct")
        object.__setattr__(self, "stages", stages)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _config_dict(config: LargeMiniDatasetConfig) -> dict[str, Any]:
    """Return the dataclass as the same JSON-native value used on disk."""

    value = json.loads(json.dumps(asdict(config)))
    assert isinstance(value, dict)
    return value


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _selected_benchmark(
    benchmark: Path, config: LargeMiniDatasetConfig
) -> dict[str, Any]:
    report = _load_json(benchmark)
    recommendation = report.get("recommendation")
    if not isinstance(recommendation, dict):
        raise TypeError("benchmark has no recommendation object")
    selected = recommendation.get("cuda_selfplay")
    if not isinstance(selected, dict):
        raise TypeError("benchmark has no CUDA self-play recommendation")
    expected = {
        "simulations": config.simulations,
        "workers": config.workers,
        "inference_batch_size": config.inference_batch_size,
        "inference_max_wait_seconds": config.inference_max_wait_seconds,
    }
    if any(selected.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "experiment config does not match the benchmark recommendation"
        )
    return selected


def _validate_dataset(root: Path, manifest: dict[str, Any]) -> dict[str, object]:
    outcomes = {"-1": 0, "0": 0, "1": 0}
    policy_examples = 0
    rows = 0
    verified_shards = 0
    for split in ("train", "validation"):
        for shard in manifest["splits"][split]["shards"]:
            path = root / shard["path"]
            if _sha256(path) != shard["sha256"]:
                raise ValueError(f"dataset shard hash mismatch: {path}")
            verified_shards += 1
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    example = json.loads(line)
                    outcome = example.get("outcome")
                    if outcome not in (-1, 0, 1):
                        raise ValueError("dataset contains an invalid value target")
                    outcomes[str(outcome)] += 1
                    policy = example.get("policy")
                    if not isinstance(policy, list) or not policy:
                        raise ValueError("dataset example is missing its policy target")
                    probability = sum(float(item["probability"]) for item in policy)
                    if abs(probability - 1.0) > 1e-9:
                        raise ValueError("dataset policy target does not sum to one")
                    policy_examples += 1
                    rows += 1
    if rows != manifest["examples"]:
        raise ValueError("dataset row count does not match its manifest")
    return {
        "valid": True,
        "examples": rows,
        "policy_examples": policy_examples,
        "value_target_distribution": outcomes,
        "verified_shards": verified_shards,
        "manifest_sha256": _sha256(root / "manifest.json"),
    }


def _run_stage(
    root: Path,
    stage: LargeStageConfig,
    checkpoint: Path,
    config: LargeMiniDatasetConfig,
    device: str,
) -> dict[str, object]:
    stage_root = root / stage.name
    if stage_root.exists() and any(stage_root.iterdir()):
        raise ValueError(f"incomplete stage directory already exists: {stage_root}")
    selfplay_root = stage_root / "selfplay"
    dataset_root = stage_root / "dataset"
    loaded = load_policy_value_checkpoint(checkpoint, map_location=device)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    with NeuralInferenceBatcher(
        NeuralPolicyValue(loaded.model),
        batch_size=config.inference_batch_size,
        max_wait_seconds=config.inference_max_wait_seconds,
    ) as inference:

        def factory() -> MCTSAgent:
            return MCTSAgent(
                simulations=config.simulations,
                rollout_limit=config.rollout_limit,
                policy_value=inference,
            )

        batch = run_batch(
            factory,
            factory,
            config=BatchConfig(
                games=stage.games,
                workers=config.workers,
                seed=stage.seed,
                board=experiment_board("mini"),
                red_agent="checkpoint-mcts",
                black_agent="checkpoint-mcts",
                worker_mode="thread",
            ),
            output_dir=selfplay_root,
        )
    wall_seconds = perf_counter() - started
    if batch.failed:
        raise RuntimeError(f"stage {stage.name} recorded {batch.failed} failed games")
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
                "checkpoint_sha256": _sha256(checkpoint),
                "mcts": {
                    "simulations": config.simulations,
                    "rollout_limit": config.rollout_limit,
                    "guidance": "policy-value",
                },
            },
        ),
    )
    dataset_seconds = perf_counter() - dataset_started
    manifest = dataset.to_dict()
    return {
        "selfplay": {
            "wall_seconds": wall_seconds,
            "games_per_hour": batch.completed / wall_seconds * 3600,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
            "inference": inference.statistics.to_dict(),
            "summary": batch.to_dict(),
        },
        "dataset": {
            "wall_seconds": dataset_seconds,
            "manifest": manifest,
            "validation": _validate_dataset(dataset_root, manifest),
        },
    }


def run_large_mini_dataset_experiment(
    output_dir: str | Path,
    checkpoint: str | Path,
    benchmark: str | Path,
    *,
    config: LargeMiniDatasetConfig | None = None,
) -> dict[str, object]:
    """Generate validated stages, checkpointing the report after each one."""

    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED must be 0")
    experiment_config = config or LargeMiniDatasetConfig()
    if not isinstance(experiment_config, LargeMiniDatasetConfig):
        raise TypeError("config must be a LargeMiniDatasetConfig or None")
    root, checkpoint_path, benchmark_path = map(
        Path, (output_dir, checkpoint, benchmark)
    )
    selected = _selected_benchmark(benchmark_path, experiment_config)
    device = select_device("cuda")
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "report.json"
    if report_path.exists():
        report: dict[str, Any] = _load_json(report_path)
        inputs = report.get("inputs", {})
        if (
            report.get("format") != EXPERIMENT_FORMAT
            or report.get("config") != _config_dict(experiment_config)
            or inputs.get("checkpoint_sha256") != _sha256(checkpoint_path)
            or inputs.get("benchmark_sha256") != _sha256(benchmark_path)
        ):
            raise ValueError("existing report does not match this experiment")
    elif any(root.iterdir()):
        raise ValueError("output directory has no resumable experiment report")
    else:
        report = {
            "format": EXPERIMENT_FORMAT,
            "version": EXPERIMENT_VERSION,
            "inputs": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "benchmark": str(benchmark_path),
                "benchmark_sha256": _sha256(benchmark_path),
            },
            "benchmark_evidence": {
                "selected": selected,
                "scale_runtime_threshold_hours": experiment_config.maximum_scale_runtime_hours,
            },
            "config": _config_dict(experiment_config),
            "environment": {
                "implementation": platform.python_implementation(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "available_cpus": _available_cpus(),
                "device": device.to_dict(),
            },
            "stages": {},
        }
        _write_json(report_path, report)
    stages = report["stages"]
    assert isinstance(stages, dict)
    for index, stage in enumerate(experiment_config.stages):
        if stage.name in stages:
            continue
        projection = selected["estimated_runtime"].get(f"{stage.games}_games_hours")
        if index and (
            projection is None
            or float(projection) > experiment_config.maximum_scale_runtime_hours
        ):
            report["scale_decision"] = {
                "attempted": False,
                "stage": stage.name,
                "reason": "benchmark projection is unavailable or exceeds runtime threshold",
            }
            _write_json(report_path, report)
            break
        stages[stage.name] = _run_stage(
            root,
            stage,
            checkpoint_path,
            experiment_config,
            device.resolved_device,
        )
        report["scale_decision"] = {
            "attempted": index > 0,
            "stage": stage.name,
            "benchmark_projected_hours": projection,
        }
        _write_json(report_path, report)
    return report


__all__ = [
    "EXPERIMENT_FORMAT",
    "EXPERIMENT_VERSION",
    "LargeMiniDatasetConfig",
    "LargeStageConfig",
    "run_large_mini_dataset_experiment",
]
