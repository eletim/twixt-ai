"""Streaming value-target and checkpoint calibration diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from twixt_ai.device import select_device
from twixt_ai.game import BoardDimensions, GameState
from twixt_ai.models import encode_position_for_version, load_policy_value_checkpoint
from twixt_ai.models.policy_value import PolicyValueNetwork

from .data import DATASET_FORMAT, DATASET_VERSION, EXAMPLE_FORMAT, EXAMPLE_VERSION

VALUE_DIAGNOSTICS_FORMAT = "twixt-ai-value-diagnostics"
VALUE_DIAGNOSTICS_VERSION = 2


@dataclass(frozen=True, slots=True)
class ValueDiagnosticsConfig:
    """Stable bucketing and inference settings for a value audit."""

    ply_bucket_size: int = 16
    calibration_bins: int = 10
    difficulty_bins: int = 3
    confidence_bins: int = 5
    batch_size: int = 512
    device: str = "auto"

    def __post_init__(self) -> None:
        for name in (
            "ply_bucket_size",
            "calibration_bins",
            "difficulty_bins",
            "confidence_bins",
            "batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class _Accumulator:
    count: int = 0
    target_sum: float = 0.0
    prediction_sum: float = 0.0
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    target_counts: dict[int, int] | None = None

    def add(self, target: float, prediction: float) -> None:
        if self.target_counts is None:
            self.target_counts = {-1: 0, 0: 0, 1: 0}
        self.count += 1
        self.target_counts[int(target)] += 1
        self.target_sum += target
        self.prediction_sum += prediction
        self.squared_error_sum += (prediction - target) ** 2
        self.absolute_error_sum += abs(prediction - target)

    def to_dict(self) -> dict[str, object]:
        if not self.count:
            return {
                "examples": 0,
                "mean_target": None,
                "mean_prediction": None,
                "calibration_error": None,
                "mse": None,
                "mae": None,
                "target_balance": _distribution({-1: 0, 0: 0, 1: 0}),
            }
        mean_target = self.target_sum / self.count
        mean_prediction = self.prediction_sum / self.count
        assert self.target_counts is not None
        return {
            "examples": self.count,
            "mean_target": mean_target,
            "mean_prediction": mean_prediction,
            "calibration_error": mean_prediction - mean_target,
            "mse": self.squared_error_sum / self.count,
            "mae": self.absolute_error_sum / self.count,
            "target_balance": _distribution(self.target_counts),
        }


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _manifest(root: Path) -> tuple[dict[str, Any], bytes, BoardDimensions]:
    path = root / "manifest.json"
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read dataset manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError("dataset manifest must contain an object")
    if value.get("format") != DATASET_FORMAT or value.get("version") != DATASET_VERSION:
        raise ValueError("unsupported dataset format or version")
    board = value.get("board")
    if not isinstance(board, Mapping) or set(board) != {"width", "height"}:
        raise ValueError("dataset board must contain exactly width and height")
    return value, content, BoardDimensions(board["width"], board["height"])


def _policy_difficulty(value: Mapping[object, object]) -> float | None:
    """Return normalized search-policy entropy as a position-difficulty proxy."""

    policy = value.get("policy")
    if policy is None:
        return None
    if not isinstance(policy, list) or not policy:
        raise ValueError("policy must be a non-empty array when present")
    probabilities: list[float] = []
    for index, item in enumerate(policy):
        probability = item.get("probability") if isinstance(item, Mapping) else None
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or probability <= 0
        ):
            raise ValueError(f"policy[{index}].probability must be positive and finite")
        probabilities.append(float(probability))
    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
        raise ValueError("policy probabilities must sum to one")
    if len(probabilities) == 1:
        return 0.0
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    return entropy / math.log(len(probabilities))


def _example(
    value: object, board: BoardDimensions
) -> tuple[GameState, int, int, float | None]:
    if not isinstance(value, Mapping):
        raise TypeError("example must contain an object")
    if value.get("format") != EXAMPLE_FORMAT or value.get("version") != EXAMPLE_VERSION:
        raise ValueError("unsupported training example format or version")
    state = GameState.from_dict(value.get("position"))  # type: ignore[arg-type]
    if state.board != board:
        raise ValueError("example board dimensions do not match the manifest")
    target = value.get("outcome")
    if isinstance(target, bool) or target not in {-1, 0, 1}:
        raise ValueError("outcome must be -1, 0, or 1")
    source = value.get("source")
    ply = source.get("ply") if isinstance(source, Mapping) else None
    if isinstance(ply, bool) or not isinstance(ply, int) or ply < 0:
        raise ValueError("source.ply must be a non-negative integer")
    return state, target, ply, _policy_difficulty(value)


def _distribution(counts: Mapping[int, int]) -> dict[str, object]:
    total = sum(counts.values())
    return {
        "examples": total,
        "counts": {str(target): counts.get(target, 0) for target in (-1, 0, 1)},
        "fractions": {
            str(target): counts.get(target, 0) / total if total else None
            for target in (-1, 0, 1)
        },
        "mean_target": (
            sum(target * count for target, count in counts.items()) / total
            if total
            else None
        ),
    }


def _bucket_name(ply: int, size: int) -> str:
    start = ply // size * size
    return f"{start}-{start + size - 1}"


def _unit_bucket(value: float, bins: int) -> int:
    return min(bins - 1, max(0, int(value * bins)))


def _ranged_buckets(
    accumulators: list[_Accumulator], metric: str
) -> list[dict[str, object]]:
    bins = len(accumulators)
    return [
        {
            f"{metric}_range": [index / bins, (index + 1) / bins],
            "upper_bound_inclusive": index == bins - 1,
            **accumulator.to_dict(),
        }
        for index, accumulator in enumerate(accumulators)
    ]


def _consume_batch(
    pending: list[tuple[GameState, int, int, int | None]],
    model: PolicyValueNetwork,
    device: str,
    config: ValueDiagnosticsConfig,
    overall: _Accumulator,
    phases: dict[str, _Accumulator],
    calibration: list[_Accumulator],
    difficulties: list[_Accumulator],
    confidence: list[_Accumulator],
) -> None:
    if not pending:
        return
    inputs = torch.stack(
        [
            encode_position_for_version(
                state, model.config.encoding_version, device=device
            )
            for state, _, _, _ in pending
        ]
    )
    with torch.inference_mode():
        _, predictions = model(inputs)
    for (_, target, ply, difficulty), prediction_tensor in zip(pending, predictions):
        prediction = float(prediction_tensor.item())
        phase = _bucket_name(ply, config.ply_bucket_size)
        overall.add(target, prediction)
        phases.setdefault(phase, _Accumulator()).add(target, prediction)
        index = min(
            config.calibration_bins - 1,
            max(0, int((prediction + 1) / 2 * config.calibration_bins)),
        )
        calibration[index].add(target, prediction)
        if difficulty is not None:
            difficulties[difficulty].add(target, prediction)
        confidence[_unit_bucket(abs(prediction), config.confidence_bins)].add(
            target, prediction
        )
    pending.clear()


def diagnose_value_model(
    dataset_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    *,
    config: ValueDiagnosticsConfig | None = None,
) -> dict[str, Any]:
    """Validate a dataset and report value balance and optional calibration.

    Shards are processed one at a time so multi-gigabyte experiment datasets do
    not have to be retained in memory. Calibration is reported both globally
    and by the same ply buckets used for target distributions.
    """

    diagnostics_config = config or ValueDiagnosticsConfig()
    if not isinstance(diagnostics_config, ValueDiagnosticsConfig):
        raise TypeError("config must be a ValueDiagnosticsConfig or None")
    root = Path(dataset_dir)
    manifest, manifest_content, board = _manifest(root)
    manifest_sha256 = _sha256(manifest_content)
    device = select_device(diagnostics_config.device)
    loaded = None
    model = None
    checkpoint: Path | None = None
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
        loaded = load_policy_value_checkpoint(
            checkpoint, map_location=device.resolved_device
        )
        model = loaded.model.eval()
        if (
            model.config.board_width != board.width
            or model.config.board_height != board.height
        ):
            raise ValueError("checkpoint board dimensions do not match the dataset")
        checkpoint_dataset = loaded.metadata.get("dataset_sha256")
        if (
            checkpoint_dataset is not None
            and checkpoint_dataset != manifest_sha256
        ):
            raise ValueError("checkpoint was trained on a different dataset")

    split_reports: dict[str, Any] = {}
    for split in ("train", "validation"):
        split_value = manifest.get("splits", {}).get(split)
        if not isinstance(split_value, Mapping) or not isinstance(
            split_value.get("shards"), list
        ):
            raise TypeError(f"dataset manifest has no valid {split} split")
        targets: dict[int, int] = {-1: 0, 0: 0, 1: 0}
        phase_targets: dict[str, dict[int, int]] = {}
        overall = _Accumulator()
        phases: dict[str, _Accumulator] = {}
        calibration = [
            _Accumulator() for _ in range(diagnostics_config.calibration_bins)
        ]
        difficulties = [
            _Accumulator() for _ in range(diagnostics_config.difficulty_bins)
        ]
        confidence = [
            _Accumulator() for _ in range(diagnostics_config.confidence_bins)
        ]
        pending: list[tuple[GameState, int, int, int | None]] = []

        actual_examples = 0
        for shard in split_value["shards"]:
            if not isinstance(shard, Mapping) or not isinstance(shard.get("path"), str):
                raise TypeError(f"{split} shard entries must contain a path")
            path = root / shard["path"]
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(
                    "dataset shard path must stay inside the dataset directory"
                )
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ValueError(f"could not read dataset shard {path}: {exc}") from exc
            if _sha256(content) != shard.get("sha256"):
                raise ValueError(f"dataset shard digest mismatch: {shard['path']}")
            lines = content.splitlines()
            if len(lines) != shard.get("examples"):
                raise ValueError(
                    f"dataset shard example count mismatch: {shard['path']}"
                )
            for line in lines:
                try:
                    state, target, ply, difficulty = _example(json.loads(line), board)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid training example in {shard['path']}: {exc}"
                    ) from exc
                phase = _bucket_name(ply, diagnostics_config.ply_bucket_size)
                targets[target] += 1
                phase_targets.setdefault(phase, {-1: 0, 0: 0, 1: 0})[target] += 1
                actual_examples += 1
                if model is not None:
                    difficulty_bucket = (
                        _unit_bucket(difficulty, diagnostics_config.difficulty_bins)
                        if difficulty is not None
                        else None
                    )
                    pending.append((state, target, ply, difficulty_bucket))
                    if len(pending) == diagnostics_config.batch_size:
                        assert model is not None
                        _consume_batch(
                            pending,
                            model,
                            device.resolved_device,
                            diagnostics_config,
                            overall,
                            phases,
                            calibration,
                            difficulties,
                            confidence,
                        )
        if model is not None:
            _consume_batch(
                pending,
                model,
                device.resolved_device,
                diagnostics_config,
                overall,
                phases,
                calibration,
                difficulties,
                confidence,
            )
        if actual_examples != split_value.get("examples"):
            raise ValueError(
                f"dataset {split} example count does not match its manifest"
            )
        report: dict[str, Any] = {
            "targets": _distribution(targets),
            "ply_buckets": {
                name: _distribution(counts)
                for name, counts in sorted(
                    phase_targets.items(),
                    key=lambda item: int(item[0].split("-")[0]),
                )
            },
        }
        if model is not None:
            report["metrics"] = overall.to_dict()
            report["metrics_by_ply"] = {
                name: accumulator.to_dict()
                for name, accumulator in sorted(
                    phases.items(), key=lambda item: int(item[0].split("-")[0])
                )
            }
            report["calibration"] = [
                {
                    "prediction_range": [
                        -1 + 2 * index / diagnostics_config.calibration_bins,
                        -1 + 2 * (index + 1) / diagnostics_config.calibration_bins,
                    ],
                    **accumulator.to_dict(),
                }
                for index, accumulator in enumerate(calibration)
            ]
            report["breakdowns"] = {
                "position_difficulty": {
                    "metric": "normalized_search_policy_entropy",
                    "definition": (
                        "Entropy of the sparse MCTS visit-probability target divided "
                        "by log of its support; 0 is concentrated and 1 is uniform. "
                        "This measures teacher search ambiguity, not intrinsic game "
                        "complexity."
                    ),
                    "buckets": _ranged_buckets(
                        difficulties, "normalized_entropy"
                    ),
                    "unavailable_examples": overall.count
                    - sum(item.count for item in difficulties),
                },
                "value_head_confidence": {
                    "metric": "absolute_value_prediction",
                    "definition": (
                        "Absolute bounded value prediction; 0 is neutral and 1 is "
                        "maximal outcome confidence."
                    ),
                    "buckets": _ranged_buckets(confidence, "absolute_prediction"),
                },
            }
        split_reports[split] = report

    result: dict[str, Any] = {
        "format": VALUE_DIAGNOSTICS_FORMAT,
        "version": VALUE_DIAGNOSTICS_VERSION,
        "config": diagnostics_config.to_dict(),
        "dataset": {
            "path": str(root),
            "manifest_sha256": manifest_sha256,
            "board": board.to_dict(),
            "source_games": manifest.get("source_games"),
            "verification": {
                "manifest_sha256_computed": True,
                "shard_sha256s_verified": sum(
                    len(manifest["splits"][split]["shards"])
                    for split in ("train", "validation")
                ),
                "example_counts_verified": sum(
                    split_reports[split]["targets"]["examples"]
                    for split in ("train", "validation")
                ),
            },
        },
        "splits": split_reports,
    }
    if checkpoint is not None and loaded is not None:
        result["checkpoint"] = {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint.read_bytes()),
            "model_config": loaded.model.config.to_dict(),
            "metadata": dict(loaded.metadata),
            "dataset_sha256_matches": (
                loaded.metadata.get("dataset_sha256") == manifest_sha256
                if "dataset_sha256" in loaded.metadata
                else None
            ),
        }
        result["device"] = device.to_dict()
        train_mse = split_reports["train"]["metrics"]["mse"]
        validation_mse = split_reports["validation"]["metrics"]["mse"]
        result["generalization"] = {
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "mse_gap": (
                validation_mse - train_mse
                if train_mse is not None and validation_mse is not None
                else None
            ),
        }
    return result


__all__ = [
    "VALUE_DIAGNOSTICS_FORMAT",
    "VALUE_DIAGNOSTICS_VERSION",
    "ValueDiagnosticsConfig",
    "diagnose_value_model",
]
