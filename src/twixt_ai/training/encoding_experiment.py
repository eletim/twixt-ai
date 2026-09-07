"""Matched end-to-end training comparison for Mini encoding versions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import torch

from twixt_ai.models import (
    ENCODING_VERSION,
    MINI_ENCODING_VERSION,
    MINI_NORMALIZED_POLICY_VALUE_CONFIG,
    MINI_POLICY_VALUE_CONFIG,
    PolicyValueConfig,
    load_policy_value_checkpoint,
)

from .trainer import EpochMetrics, TrainingConfig, TrainingSummary, train_model


MATCHED_ENCODING_TRAINING_FORMAT = "twixt-ai-matched-encoding-training"
MATCHED_ENCODING_TRAINING_VERSION = 1

_ENCODINGS = (
    ("22_plane_v1", "22-plane-v1", MINI_POLICY_VALUE_CONFIG),
    ("10_plane_v2", "10-plane-v2", MINI_NORMALIZED_POLICY_VALUE_CONFIG),
)


@dataclass(frozen=True, slots=True)
class MatchedEncodingTrainingConfig:
    """Optimization settings shared exactly by both encoding runs."""

    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "none"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.1
    seed: int = 750_100
    tiny_epochs: int = 300
    tiny_learning_rate: float = 1e-2
    tiny_max_loss_ratio: float = 0.25
    device: str = "cpu"
    torch_threads: int = 1

    def __post_init__(self) -> None:
        # TrainingConfig owns validation of the shared optimizer settings.
        self.training_config()
        self.tiny_training_config()
        if (
            isinstance(self.tiny_max_loss_ratio, bool)
            or not isinstance(self.tiny_max_loss_ratio, (int, float))
            or not math.isfinite(self.tiny_max_loss_ratio)
            or not 0 < self.tiny_max_loss_ratio < 1
        ):
            raise ValueError("tiny_max_loss_ratio must be between zero and one")
        if (
            isinstance(self.torch_threads, bool)
            or not isinstance(self.torch_threads, int)
            or self.torch_threads < 1
        ):
            raise ValueError("torch_threads must be a positive integer")

    def training_config(self) -> TrainingConfig:
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scheduler_step_size=self.scheduler_step_size,
            scheduler_gamma=self.scheduler_gamma,
            seed=self.seed,
            device=self.device,
        )

    def tiny_training_config(self) -> TrainingConfig:
        return TrainingConfig(
            epochs=self.tiny_epochs,
            batch_size=1,
            learning_rate=self.tiny_learning_rate,
            weight_decay=0,
            optimizer=self.optimizer,
            scheduler="none",
            seed=self.seed,
            device=self.device,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_dataset(source: Path, destination: Path) -> Path:
    """Derive the same deterministic one-position fixture for both runs."""

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    shard = manifest["splits"]["train"]["shards"][0]["path"]
    example = json.loads((source / shard).read_bytes().splitlines()[0])
    example.pop("policy", None)
    example["id"] = "issue-75-tiny-overfit"
    content = json.dumps(example, sort_keys=True).encode() + b"\n"
    (destination / "train").mkdir(parents=True)
    (destination / "validation").mkdir()
    shard_path = destination / "train" / "shard-00000.jsonl"
    shard_path.write_bytes(content)
    _write_json(destination / "manifest.json", {
        "format": manifest["format"],
        "version": manifest["version"],
        "board": manifest["board"],
        "config": {"fixture": "issue-75-tiny-overfit"},
        "source_games": 1,
        "examples": 1,
        "splits": {
            "train": {
                "examples": 1,
                "shards": [{
                    "path": "train/shard-00000.jsonl",
                    "examples": 1,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }],
            },
            "validation": {"examples": 0, "shards": []},
        },
    })
    return destination


def _checkpoint(path: Path, expected: PolicyValueConfig) -> dict[str, object]:
    loaded = load_policy_value_checkpoint(path)
    if loaded.model.config != expected:
        raise RuntimeError(f"{path} has the wrong model or encoding configuration")
    finite = all(torch.isfinite(value).all() for value in loaded.model.state_dict().values())
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "epoch": loaded.metadata["epoch"],
        "encoding_version": loaded.model.config.encoding_version,
        "model_config": loaded.model.config.to_dict(),
        "finite_weights": finite,
    }


def _history(history: tuple[EpochMetrics, ...]) -> dict[str, object]:
    best = math.inf
    checkpoints: list[dict[str, object]] = []
    finite = True
    consistent = True
    for metrics in history:
        values = metrics.to_dict()
        finite = finite and all(
            not isinstance(value, float) or math.isfinite(value)
            for value in values.values() if value is not None
        )
        consistent = consistent and math.isclose(
            metrics.train_loss,
            metrics.train_policy_loss + metrics.train_value_loss,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        selection_loss = (
            metrics.validation_loss
            if metrics.validation_loss is not None
            else metrics.train_loss
        )
        selected = selection_loss < best
        best = min(best, selection_loss)
        checkpoints.append({
            "epoch": metrics.epoch,
            "selection_loss": selection_loss,
            "selected_as_best": selected,
        })
    return {
        "finite_losses": finite,
        "loss_components_consistent": consistent,
        "checkpoint_history": checkpoints,
    }


def _measured_train(
    dataset: Path,
    output: Path,
    training_config: TrainingConfig,
    model_config: PolicyValueConfig,
) -> tuple[TrainingSummary, float]:
    started = perf_counter()
    summary = train_model(
        dataset, output, config=training_config, model_config=model_config
    )
    return summary, perf_counter() - started


def _run_encoding(
    dataset: Path,
    tiny_dataset: Path,
    output: Path,
    config: MatchedEncodingTrainingConfig,
    model_config: PolicyValueConfig,
    examples_per_epoch: int,
) -> dict[str, object]:
    tiny, tiny_seconds = _measured_train(
        tiny_dataset,
        output / "tiny",
        config.tiny_training_config(),
        model_config,
    )
    tiny_history = _history(tiny.history)
    tiny_ratio = tiny.history[-1].train_loss / tiny.history[0].train_loss

    training, wall_seconds = _measured_train(
        dataset,
        output / "training",
        config.training_config(),
        model_config,
    )
    history_checks = _history(training.history)
    checkpoint_root = output / "training"
    return {
        "encoding_version": model_config.encoding_version,
        "planes": model_config.input_channels,
        "model_config": model_config.to_dict(),
        "tiny_overfit": {
            "passed": bool(
                tiny_history["finite_losses"]
                and tiny_history["loss_components_consistent"]
                and tiny_ratio <= config.tiny_max_loss_ratio
            ),
            "threshold": config.tiny_max_loss_ratio,
            "initial_loss": tiny.history[0].train_loss,
            "final_loss": tiny.history[-1].train_loss,
            "loss_ratio": tiny_ratio,
            "wall_seconds": tiny_seconds,
            "examples_per_second": config.tiny_epochs / tiny_seconds,
            "summary": tiny.to_dict(),
            **tiny_history,
        },
        "training": {
            "wall_seconds": wall_seconds,
            "examples_per_second": examples_per_epoch * config.epochs / wall_seconds,
            "summary": training.to_dict(),
            **history_checks,
            "checkpoints": {
                "latest": _checkpoint(checkpoint_root / "latest.pt", model_config),
                "best": _checkpoint(checkpoint_root / "best.pt", model_config),
            },
        },
    }


def run_matched_encoding_training(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    config: MatchedEncodingTrainingConfig = MatchedEncodingTrainingConfig(),
) -> dict[str, Any]:
    """Train v1 and v2 under matched conditions and write a comparison report."""

    dataset = Path(dataset_dir)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("output directory must be empty or not exist")
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_examples = manifest["splits"]["train"]["examples"]
    validation_examples = manifest["splits"]["validation"]["examples"]
    examples_per_epoch = train_examples + validation_examples
    tiny = _tiny_dataset(dataset, root / "tiny-dataset")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config.torch_threads)
    try:
        results = {
            key: _run_encoding(
                dataset,
                tiny,
                root / directory,
                config,
                model_config,
                examples_per_epoch,
            )
            for key, directory, model_config in _ENCODINGS
        }
    finally:
        torch.set_num_threads(previous_threads)

    report: dict[str, Any] = {
        "format": MATCHED_ENCODING_TRAINING_FORMAT,
        "version": MATCHED_ENCODING_TRAINING_VERSION,
        "config": config.to_dict(),
        "methodology": {
            "run_order": [key for key, _, _ in _ENCODINGS],
            "matched": [
                "dataset snapshot and train/validation split",
                "trunk, policy-head, and value-head capacity",
                "optimizer and scheduler",
                "epochs, batch size, and example shuffling seed",
                "no data augmentation",
                "tiny-overfit source position and thresholds",
            ],
            "only_intended_difference": "input encoding version and plane count",
            "instability_policy": (
                "Loss histories and sanity outcomes are recorded without requiring "
                "real-dataset loss improvement."
            ),
        },
        "dataset": {
            "path": str(dataset),
            "manifest_sha256": _sha256(manifest_path),
            "train_examples": train_examples,
            "validation_examples": validation_examples,
        },
        "environment": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "available_cpus": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1,
            "torch": torch.__version__,
            "device": config.device,
            "torch_threads": config.torch_threads,
        },
        "results": results,
    }
    old = results["22_plane_v1"]["training"]
    new = results["10_plane_v2"]["training"]
    report["comparison"] = {
        "throughput_change_percent": 100 * (
            new["examples_per_second"] / old["examples_per_second"] - 1
        ),
        "wall_time_change_percent": 100 * (
            new["wall_seconds"] / old["wall_seconds"] - 1
        ),
    }
    _write_json(root / "report.json", report)
    return report


__all__ = [
    "MATCHED_ENCODING_TRAINING_FORMAT",
    "MATCHED_ENCODING_TRAINING_VERSION",
    "MatchedEncodingTrainingConfig",
    "run_matched_encoding_training",
]
