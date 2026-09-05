"""Reproducible policy/value training and resumable checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from twixt_ai.game import Coordinate, GameState
from twixt_ai.models import (
    ACTION_COUNT,
    ARCHITECTURE_NAME,
    ARCHITECTURE_VERSION,
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    ENCODING_VERSION,
    PolicyValueConfig,
    PolicyValueNetwork,
    coordinate_to_action_index,
    encode_position,
)

from .data import DATASET_FORMAT, DATASET_VERSION, EXAMPLE_FORMAT, EXAMPLE_VERSION


TRAINING_FORMAT = "twixt-ai-training-checkpoint"
TRAINING_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """All settings that affect optimization and example ordering."""

    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "none"
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.1
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "scheduler_step_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in ("learning_rate", "scheduler_gamma"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if (
            isinstance(self.weight_decay, bool)
            or not isinstance(self.weight_decay, (int, float))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be a non-negative finite number")
        if self.optimizer not in {"adamw", "sgd"}:
            raise ValueError("optimizer must be 'adamw' or 'sgd'")
        if self.scheduler not in {"none", "step"}:
            raise ValueError("scheduler must be 'none' or 'step'")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))
        object.__setattr__(self, "weight_decay", float(self.weight_decay))
        object.__setattr__(self, "scheduler_gamma", float(self.scheduler_gamma))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TrainingConfig:
        if not isinstance(value, Mapping):
            raise TypeError("training config must be a mapping")
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ValueError(f"training config must contain exactly {sorted(expected)}")
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    learning_rate: float
    train_policy_loss: float
    train_value_loss: float
    train_loss: float
    validation_policy_loss: float | None
    validation_value_loss: float | None
    validation_loss: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    config: TrainingConfig
    model_config: PolicyValueConfig
    dataset_sha256: str
    completed_epochs: int
    best_epoch: int
    best_loss: float
    latest_checkpoint: str
    best_checkpoint: str
    metrics_path: str
    history: tuple[EpochMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "format": TRAINING_FORMAT,
            "version": TRAINING_VERSION,
            "config": self.config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "dataset_sha256": self.dataset_sha256,
            "completed_epochs": self.completed_epochs,
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "checkpoints": {
                "latest": self.latest_checkpoint,
                "best": self.best_checkpoint,
            },
            "metrics": self.metrics_path,
            "history": [item.to_dict() for item in self.history],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)


class _Examples:
    def __init__(self, root: Path, manifest: Mapping[str, Any], split: str) -> None:
        split_value = manifest.get("splits", {}).get(split)
        if not isinstance(split_value, Mapping) or not isinstance(split_value.get("shards"), list):
            raise ValueError(f"dataset manifest has no valid {split} split")
        self.items: list[tuple[Tensor, Tensor, Tensor]] = []
        for shard in split_value["shards"]:
            if not isinstance(shard, Mapping):
                raise ValueError(f"{split} shard entry must be an object")
            relative = shard.get("path")
            if not isinstance(relative, str):
                raise ValueError(f"{split} shard path must be a string")
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("dataset shard path must stay inside the dataset directory")
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"could not read dataset shard {path}: {exc}") from exc
            if hashlib.sha256(content).hexdigest() != shard.get("sha256"):
                raise ValueError(f"dataset shard digest mismatch: {relative}")
            lines = content.splitlines()
            if len(lines) != shard.get("examples"):
                raise ValueError(f"dataset shard example count mismatch: {relative}")
            for line in lines:
                try:
                    self.items.append(self._parse(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid training example in {relative}: {exc}") from exc
        if len(self.items) != split_value.get("examples"):
            raise ValueError(f"dataset {split} example count does not match its manifest")

    @staticmethod
    def _coordinate(value: object, name: str) -> Coordinate:
        if not isinstance(value, Mapping) or set(value) != {"x", "y"}:
            raise ValueError(f"{name} must contain exactly x and y")
        return Coordinate(value["x"], value["y"])  # type: ignore[arg-type]

    @classmethod
    def _parse(cls, value: object) -> tuple[Tensor, Tensor, Tensor]:
        if not isinstance(value, Mapping):
            raise TypeError("example must be an object")
        if value.get("format") != EXAMPLE_FORMAT or value.get("version") != EXAMPLE_VERSION:
            raise ValueError("unsupported training example format or version")
        state = GameState.from_dict(value.get("position"))  # type: ignore[arg-type]
        inputs = encode_position(state)
        target = torch.zeros(ACTION_COUNT, dtype=torch.float32)
        policy = value.get("policy")
        if policy is None:
            target[coordinate_to_action_index(cls._coordinate(value.get("action"), "action"))] = 1
        else:
            if not isinstance(policy, list) or not policy:
                raise ValueError("policy must be a non-empty array")
            seen: set[Coordinate] = set()
            total = 0.0
            for index, item in enumerate(policy):
                if not isinstance(item, Mapping):
                    raise ValueError(f"policy[{index}] must be an object")
                coordinate = cls._coordinate(item.get("coordinate"), f"policy[{index}].coordinate")
                probability = item.get("probability")
                if (
                    coordinate in seen
                    or isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not math.isfinite(probability)
                    or probability <= 0
                ):
                    raise ValueError("policy entries must be unique with positive probabilities")
                seen.add(coordinate)
                target[coordinate_to_action_index(coordinate)] = probability
                total += probability
            if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError("policy probabilities must sum to one")
        outcome = value.get("outcome")
        if isinstance(outcome, bool) or outcome not in {-1, 0, 1}:
            raise ValueError("outcome must be -1, 0, or 1")
        return inputs, target, torch.tensor(float(outcome), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.items)


def _load_dataset(root: Path) -> tuple[dict[str, Any], str, _Examples, _Examples]:
    manifest_path = root / "manifest.json"
    try:
        content = manifest_path.read_bytes()
        manifest = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read dataset manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be an object")
    if manifest.get("format") != DATASET_FORMAT or manifest.get("version") != DATASET_VERSION:
        raise ValueError("unsupported dataset format or version")
    train = _Examples(root, manifest, "train")
    validation = _Examples(root, manifest, "validation")
    if not len(train):
        raise ValueError("training split must contain at least one example")
    return manifest, hashlib.sha256(content).hexdigest(), train, validation


def _optimizer(model: PolicyValueNetwork, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    return torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def _batches(examples: _Examples, batch_size: int, order: Tensor) -> list[tuple[Tensor, Tensor, Tensor]]:
    batches = []
    for start in range(0, len(order), batch_size):
        items = [examples.items[index] for index in order[start : start + batch_size].tolist()]
        batches.append(tuple(torch.stack(values) for values in zip(*items)))  # type: ignore[arg-type]
    return batches


def _epoch(
    model: PolicyValueNetwork,
    examples: _Examples,
    config: TrainingConfig,
    *,
    optimizer: torch.optim.Optimizer | None,
    order: Tensor,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    policy_total = value_total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, policy_targets, value_targets in _batches(examples, config.batch_size, order):
            inputs = inputs.to(config.device)
            policy_targets = policy_targets.to(config.device)
            value_targets = value_targets.to(config.device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits, values = model(inputs)
            policy_loss = -(policy_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(values, value_targets)
            loss = policy_loss + value_loss
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            size = len(inputs)
            policy_total += policy_loss.item() * size
            value_total += value_loss.item() * size
            count += size
    return policy_total / count, value_total / count, (policy_total + value_total) / count


def _checkpoint_payload(
    model: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: TrainingConfig,
    dataset_sha256: str,
    epoch: int,
    best_epoch: int,
    best_loss: float,
    history: list[EpochMetrics],
) -> dict[str, object]:
    return {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_version": CHECKPOINT_VERSION,
        "architecture": ARCHITECTURE_NAME,
        "architecture_version": ARCHITECTURE_VERSION,
        "encoding_version": ENCODING_VERSION,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "metadata": {
            "training_format": TRAINING_FORMAT,
            "training_version": TRAINING_VERSION,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "training_config": config.to_dict(),
            "dataset_sha256": dataset_sha256,
        },
        "training_state": {
            "format": TRAINING_FORMAT,
            "version": TRAINING_VERSION,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "training_config": config.to_dict(),
            "dataset_sha256": dataset_sha256,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "history": [item.to_dict() for item in history],
        },
    }


def _save(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _metrics_jsonl(history: list[EpochMetrics]) -> str:
    return "".join(
        json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in history
    )


def train_model(
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    config: TrainingConfig | None = None,
    model_config: PolicyValueConfig | None = None,
    resume: bool = False,
) -> TrainingSummary:
    """Train from a versioned dataset and write ``latest.pt`` and ``best.pt``.

    Resuming restores model, optimizer, scheduler, epoch, and metric history.
    The dataset digest and all configuration except the requested total epoch
    count must match the interrupted run.
    """

    training_config = config or TrainingConfig()
    architecture_config = model_config or PolicyValueConfig()
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig or None")
    if not isinstance(architecture_config, PolicyValueConfig):
        raise TypeError("model_config must be a PolicyValueConfig or None")
    root = Path(dataset_dir)
    _, dataset_sha256, train, validation = _load_dataset(root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latest_path = output / "latest.pt"
    best_path = output / "best.pt"
    metrics_path = output / "metrics.jsonl"
    summary_path = output / "summary.json"
    if not resume and latest_path.exists():
        raise ValueError(f"training output already contains {latest_path.name}; use resume=True")

    torch.manual_seed(training_config.seed)
    try:
        device = torch.device(training_config.device)
        model = PolicyValueNetwork(architecture_config).to(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"could not initialize device {training_config.device!r}: {exc}") from exc
    optimizer = _optimizer(model, training_config)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=training_config.scheduler_step_size,
            gamma=training_config.scheduler_gamma,
        )
        if training_config.scheduler == "step"
        else None
    )
    start_epoch = 1
    best_epoch = 0
    best_loss = math.inf
    history: list[EpochMetrics] = []

    if resume:
        if not latest_path.is_file():
            raise ValueError("cannot resume without latest.pt")
        payload = torch.load(latest_path, map_location=device, weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("training_state"), Mapping):
            raise ValueError("latest.pt is not a resumable training checkpoint")
        state = payload["training_state"]
        saved_config = TrainingConfig.from_dict(state.get("training_config"))  # type: ignore[arg-type]
        comparable = training_config.to_dict()
        comparable["epochs"] = saved_config.epochs
        if comparable != saved_config.to_dict():
            raise ValueError("resume training configuration does not match latest.pt")
        if training_config.epochs < saved_config.epochs:
            raise ValueError("resumed epochs cannot be less than the original target")
        if state.get("dataset_sha256") != dataset_sha256:
            raise ValueError("resume dataset does not match latest.pt")
        if payload.get("config") != architecture_config.to_dict():
            raise ValueError("resume model configuration does not match latest.pt")
        model.load_state_dict(payload["state_dict"])  # type: ignore[arg-type]
        optimizer.load_state_dict(state["optimizer"])  # type: ignore[arg-type]
        if scheduler is not None:
            if state.get("scheduler") is None:
                raise ValueError("resume checkpoint has no scheduler state")
            scheduler.load_state_dict(state["scheduler"])  # type: ignore[arg-type]
        start_epoch = int(state["epoch"]) + 1
        best_epoch = int(state["best_epoch"])
        best_loss = float(state["best_loss"])
        raw_history = state.get("history")
        if not isinstance(raw_history, list):
            raise ValueError("resume checkpoint has invalid metric history")
        history = [EpochMetrics(**item) for item in raw_history]
        # The checkpoint is the resume source of truth. This also repairs a
        # metrics file left stale by a process interrupted after an older
        # version of the trainer saved latest.pt.
        _write_text(metrics_path, _metrics_jsonl(history))

    generator = torch.Generator().manual_seed(training_config.seed)
    # Advancing once per completed epoch recreates the exact next shuffle on resume.
    for _ in range(start_epoch - 1):
        torch.randperm(len(train), generator=generator)

    for epoch in range(start_epoch, training_config.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_values = _epoch(
            model,
            train,
            training_config,
            optimizer=optimizer,
            order=torch.randperm(len(train), generator=generator),
        )
        validation_values = (
            _epoch(
                model,
                validation,
                training_config,
                optimizer=None,
                order=torch.arange(len(validation)),
            )
            if len(validation)
            else (None, None, None)
        )
        selection_loss = validation_values[2] if validation_values[2] is not None else train_values[2]
        improved = selection_loss < best_loss
        if improved:
            best_epoch, best_loss = epoch, selection_loss
        if scheduler is not None:
            scheduler.step()
        metrics = EpochMetrics(
            epoch,
            learning_rate,
            *train_values,
            *validation_values,
        )
        history.append(metrics)
        payload = _checkpoint_payload(
            model, optimizer, scheduler, training_config, dataset_sha256,
            epoch, best_epoch, best_loss, history,
        )
        # latest.pt is the epoch commit point. Its referenced best checkpoint
        # and metric history must be durable before it advances.
        if improved:
            _save(best_path, payload)
        _write_text(metrics_path, _metrics_jsonl(history))
        _save(latest_path, payload)

    if not history:
        raise ValueError("training target is already complete; increase config.epochs")
    summary = TrainingSummary(
        training_config,
        architecture_config,
        dataset_sha256,
        history[-1].epoch,
        best_epoch,
        best_loss,
        latest_path.name,
        best_path.name,
        metrics_path.name,
        tuple(history),
    )
    _write_text(summary_path, summary.to_json(indent=2) + "\n")
    return summary


__all__ = [
    "TRAINING_FORMAT",
    "TRAINING_VERSION",
    "EpochMetrics",
    "TrainingConfig",
    "TrainingSummary",
    "train_model",
]
