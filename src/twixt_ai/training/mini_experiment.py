"""Reproducible first-model training experiment for Mini Twixt."""

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

from twixt_ai.agents import AgentRequest
from twixt_ai.game import BoardDimensions, GameState, legal_peg_placements
from twixt_ai.models import MINI_POLICY_VALUE_CONFIG, load_policy_value_checkpoint
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralPolicyValue

from .trainer import EpochMetrics, TrainingConfig, TrainingSummary, train_model


EXPERIMENT_FORMAT = "twixt-ai-mini-training-experiment"
EXPERIMENT_VERSION = 1


@dataclass(frozen=True, slots=True)
class MiniTrainingExperimentConfig:
    """Settings and sanity thresholds for the first Mini learned model."""

    epochs: int = 20
    resume_after_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 570_100
    tiny_epochs: int = 100
    tiny_learning_rate: float = 1e-2
    tiny_max_loss_ratio: float = 0.25

    def __post_init__(self) -> None:
        if not 1 <= self.resume_after_epochs < self.epochs:
            raise ValueError("resume_after_epochs must be between 1 and epochs")
        if not 0 < self.tiny_max_loss_ratio < 1:
            raise ValueError("tiny_max_loss_ratio must be between zero and one")

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


def _make_tiny_dataset(source: Path, destination: Path) -> Path:
    """Create a deterministic one-position, one-hot overfit fixture."""

    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    shard = manifest["splits"]["train"]["shards"][0]["path"]
    first_line = (source / shard).read_bytes().splitlines()[0]
    example = json.loads(first_line)
    example.pop("policy", None)
    example["id"] = "issue-57-tiny-overfit"
    content = json.dumps(example, sort_keys=True).encode() + b"\n"
    (destination / "train").mkdir(parents=True)
    (destination / "validation").mkdir()
    train_path = destination / "train" / "shard-00000.jsonl"
    train_path.write_bytes(content)
    tiny_manifest = {
        "format": manifest["format"],
        "version": manifest["version"],
        "board": manifest["board"],
        "config": {"fixture": "issue-57-tiny-overfit"},
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
    }
    _write_json(destination / "manifest.json", tiny_manifest)
    return destination


def _training_config(
    config: MiniTrainingExperimentConfig, epochs: int
) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        seed=config.seed,
    )


def _assert_finite_history(history: tuple[EpochMetrics, ...]) -> None:
    if not history:
        raise RuntimeError("training produced no metric history")
    for metrics in history:
        values = metrics.to_dict()
        for name, value in values.items():
            if value is not None and isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError(f"epoch {metrics.epoch} produced non-finite {name}")
        if not math.isclose(
            metrics.train_loss,
            metrics.train_policy_loss + metrics.train_value_loss,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise RuntimeError("training loss does not equal policy plus value loss")


def _checkpoint(path: Path) -> dict[str, object]:
    loaded = load_policy_value_checkpoint(path)
    if any(not torch.isfinite(value).all() for value in loaded.model.state_dict().values()):
        raise RuntimeError(f"checkpoint contains non-finite weights: {path.name}")
    return {
        "path": path.name,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "epoch": loaded.metadata["epoch"],
    }


def _checkpoint_history(history: tuple[EpochMetrics, ...]) -> list[dict[str, object]]:
    best = math.inf
    result: list[dict[str, object]] = []
    for metrics in history:
        selection = (
            metrics.validation_loss
            if metrics.validation_loss is not None
            else metrics.train_loss
        )
        selected = selection < best
        best = min(best, selection)
        result.append({
            "epoch": metrics.epoch,
            "selected_as_best": selected,
            "selection_loss": selection,
        })
    return result


def _inference_smoke(checkpoint: Path) -> dict[str, object]:
    model = load_policy_value_checkpoint(checkpoint).model
    state = GameState.initial(
        BoardDimensions(model.config.board_width, model.config.board_height)
    )
    moves = legal_peg_placements(state)
    policy_value = NeuralPolicyValue(model)
    estimate = policy_value(state, moves)
    prior_total = sum(estimate.priors.values())
    if not math.isclose(prior_total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("NeuralPolicyValue returned an invalid masked policy")
    if estimate.value is None or not math.isfinite(estimate.value):
        raise RuntimeError("NeuralPolicyValue returned a non-finite value")
    result = MCTSAgent(simulations=2, policy_value=policy_value).choose_move(
        AgentRequest(state, seed=57)
    )
    if result.move not in moves:
        raise RuntimeError("learned MCTS returned an illegal move")
    return {
        "legal_moves": len(moves),
        "prior_total": prior_total,
        "value": estimate.value,
        "mcts_move": result.move.coordinate.to_dict(),
    }


def _measured_train(
    dataset: Path,
    output: Path,
    training_config: TrainingConfig,
    *,
    resume: bool = False,
) -> tuple[TrainingSummary, float]:
    started = perf_counter()
    summary = train_model(
        dataset,
        output,
        config=training_config,
        model_config=MINI_POLICY_VALUE_CONFIG,
        resume=resume,
    )
    return summary, perf_counter() - started


def run_mini_training_experiment(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    config: MiniTrainingExperimentConfig = MiniTrainingExperimentConfig(),
) -> dict[str, Any]:
    """Train, resume, and validate the first learned Mini checkpoint."""

    dataset = Path(dataset_dir)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("output directory must be empty or not exist")
    root.mkdir(parents=True, exist_ok=True)

    tiny_dataset = _make_tiny_dataset(dataset, root / "tiny" / "dataset")
    tiny_config = TrainingConfig(
        epochs=config.tiny_epochs,
        batch_size=1,
        learning_rate=config.tiny_learning_rate,
        weight_decay=0,
        seed=config.seed,
    )
    tiny, tiny_seconds = _measured_train(
        tiny_dataset, root / "tiny" / "training", tiny_config
    )
    _assert_finite_history(tiny.history)
    tiny_ratio = tiny.history[-1].train_loss / tiny.history[0].train_loss
    if tiny_ratio > config.tiny_max_loss_ratio:
        raise RuntimeError(
            f"tiny overfit loss ratio {tiny_ratio:.6g} exceeds "
            f"{config.tiny_max_loss_ratio:.6g}"
        )

    baseline_dir = root / "baseline"
    initial, initial_seconds = _measured_train(
        dataset,
        baseline_dir,
        _training_config(config, config.resume_after_epochs),
    )
    prefix = initial.history
    completed, resume_seconds = _measured_train(
        dataset,
        baseline_dir,
        _training_config(config, config.epochs),
        resume=True,
    )
    _assert_finite_history(completed.history)
    if completed.history[: len(prefix)] != prefix:
        raise RuntimeError("resume changed the committed metric history")
    for component in ("train_policy_loss", "train_value_loss", "train_loss"):
        if getattr(completed.history[-1], component) >= getattr(
            completed.history[0], component
        ):
            raise RuntimeError(f"real dataset {component} did not improve")

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    examples_per_epoch = (
        manifest["splits"]["train"]["examples"]
        + manifest["splits"]["validation"]["examples"]
    )
    wall_seconds = initial_seconds + resume_seconds
    report: dict[str, Any] = {
        "format": EXPERIMENT_FORMAT,
        "version": EXPERIMENT_VERSION,
        "config": config.to_dict(),
        "model_config": MINI_POLICY_VALUE_CONFIG.to_dict(),
        "dataset": {
            "path": str(dataset),
            "manifest_sha256": _sha256(dataset / "manifest.json"),
            "train_examples": manifest["splits"]["train"]["examples"],
            "validation_examples": manifest["splits"]["validation"]["examples"],
        },
        "environment": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "available_cpus": len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "tiny_overfit": {
            "passed": True,
            "wall_seconds": tiny_seconds,
            "examples_per_second": config.tiny_epochs / tiny_seconds,
            "initial_loss": tiny.history[0].train_loss,
            "final_loss": tiny.history[-1].train_loss,
            "loss_ratio": tiny_ratio,
            "summary": tiny.to_dict(),
        },
        "training": {
            "resume_verified": True,
            "initial_wall_seconds": initial_seconds,
            "resume_wall_seconds": resume_seconds,
            "wall_seconds": wall_seconds,
            "examples_per_second": examples_per_epoch * config.epochs / wall_seconds,
            "summary": completed.to_dict(),
            "checkpoint_history": _checkpoint_history(completed.history),
            "checkpoints": {
                "latest": _checkpoint(baseline_dir / "latest.pt"),
                "best": _checkpoint(baseline_dir / "best.pt"),
            },
        },
        "sanity_checks": {
            "finite_losses_and_weights": True,
            "loss_components_consistent": True,
            "real_policy_loss_decreased": True,
            "real_value_loss_decreased": True,
            "real_total_loss_decreased": True,
            "inference": _inference_smoke(baseline_dir / "best.pt"),
        },
    }
    _write_json(root / "report.json", report)
    return report


__all__ = [
    "EXPERIMENT_FORMAT",
    "EXPERIMENT_VERSION",
    "MiniTrainingExperimentConfig",
    "run_mini_training_experiment",
]
