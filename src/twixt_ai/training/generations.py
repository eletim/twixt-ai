"""Inspectable AlphaZero-style training generations for Mini Twixt."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from functools import partial
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

from twixt_ai import __version__
from twixt_ai.device import DeviceSelection, select_device
from twixt_ai.evaluation import AgentConfig, BenchmarkConfig, run_benchmark
from twixt_ai.game import experiment_board
from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    load_policy_value_checkpoint,
)
from twixt_ai.search import (
    DEFAULT_PROGRESSIVE_WIDENING_CONSTANT,
    DEFAULT_PROGRESSIVE_WIDENING_EXPONENT,
    MCTSAgent,
)
from twixt_ai.search.neural import NeuralInferenceBatcher, NeuralPolicyValue
from twixt_ai.selfplay import BatchConfig, BatchSummary, run_batch

from .data import DatasetConfig, build_dataset
from .trainer import TrainingConfig, train_model


GENERATIONS_FORMAT = "twixt-ai-mini-training-generations"
GENERATIONS_VERSION = 1


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class MiniGenerationConfig:
    """Complete fixed schedule for a sequence of Mini generations."""

    generations: int = 2
    games_per_generation: int = 100
    dataset_window: int = 5
    selfplay_simulations: int = 100
    selfplay_exploration: float = math.sqrt(2.0)
    selfplay_progressive_widening_constant: float = (
        DEFAULT_PROGRESSIVE_WIDENING_CONSTANT
    )
    selfplay_progressive_widening_exponent: float = (
        DEFAULT_PROGRESSIVE_WIDENING_EXPONENT
    )
    evaluation_games: int = 20
    evaluation_simulations: int = 20
    rollout_limit: int = 4
    workers: int = 2
    inference_batch_size: int = 16
    inference_max_wait_seconds: float = 0.002
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    selection_metric: str = "total"
    validation_fraction: float = 0.1
    shard_size: int = 10_000
    promotion_win_rate: float = 0.55
    seed: int = 590_100
    evaluation_seed: int | None = None
    device: str = "auto"
    artifact_uri: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "generations",
            "games_per_generation",
            "dataset_window",
            "selfplay_simulations",
            "evaluation_games",
            "evaluation_simulations",
            "rollout_limit",
            "workers",
            "inference_batch_size",
            "epochs",
            "batch_size",
            "shard_size",
        ):
            _positive_integer(getattr(self, name), name)
        if self.evaluation_games % 2:
            raise ValueError("evaluation_games must be even for paired role swaps")
        if (
            isinstance(self.inference_max_wait_seconds, bool)
            or not isinstance(self.inference_max_wait_seconds, (int, float))
            or not math.isfinite(self.inference_max_wait_seconds)
            or self.inference_max_wait_seconds < 0
        ):
            raise ValueError(
                "inference_max_wait_seconds must be a finite non-negative number"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.evaluation_seed is not None and (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
        ):
            raise TypeError("evaluation_seed must be an integer or None")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
        if self.artifact_uri is not None and (
            not isinstance(self.artifact_uri, str) or not self.artifact_uri.strip()
        ):
            raise ValueError("artifact_uri must be a non-empty string or None")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (name == "learning_rate" and value == 0)
            ):
                raise ValueError(f"{name} must be a valid finite value")
        for name in (
            "selfplay_progressive_widening_constant",
            "selfplay_progressive_widening_exponent",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number")
        if (
            isinstance(self.selfplay_exploration, bool)
            or not isinstance(self.selfplay_exploration, (int, float))
            or not math.isfinite(self.selfplay_exploration)
            or self.selfplay_exploration < 0
        ):
            raise ValueError("selfplay_exploration must be finite and non-negative")
        if self.selection_metric not in {"total", "value"}:
            raise ValueError("selection_metric must be 'total' or 'value'")
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not math.isfinite(self.validation_fraction)
            or not 0 <= self.validation_fraction < 1
        ):
            raise ValueError("validation_fraction must be in [0, 1)")
        if (
            isinstance(self.promotion_win_rate, bool)
            or not isinstance(self.promotion_win_rate, (int, float))
            or not math.isfinite(self.promotion_win_rate)
            or not 0 <= self.promotion_win_rate <= 1
        ):
            raise ValueError("promotion_win_rate must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["board"] = experiment_board("mini").to_dict()
        value["promotion_rule"] = (
            "candidate wins / all paired evaluation games >= promotion_win_rate; "
            "draws remain in the denominator"
        )
        return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory_sha256(
    categories: dict[str, object], files: int, bytes_: int
) -> str:
    payload = {"categories": categories, "files": files, "bytes": bytes_}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _target_distributions(
    dataset_root: Path, manifest: dict[str, object]
) -> dict[str, object]:
    """Summarize the policy and value targets actually consumed by training."""

    supports: dict[int, int] = {}
    examples = 0
    support_total = 0
    entropy_total = 0.0
    normalized_entropy_total = 0.0
    maximum_probability_total = 0.0
    outcomes: dict[int, int] = {}
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("dataset manifest has no splits object")
    for split in ("train", "validation"):
        split_value = splits.get(split)
        if not isinstance(split_value, dict) or not isinstance(
            split_value.get("shards"), list
        ):
            raise ValueError(f"dataset manifest has no valid {split} split")
        for shard in split_value["shards"]:
            if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
                raise ValueError("dataset manifest contains an invalid shard")
            path = dataset_root / shard["path"]
            if _sha256(path) != shard.get("sha256"):
                raise ValueError(f"dataset shard hash mismatch: {path}")
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    example = json.loads(line)
                    policy = example.get("policy")
                    if not isinstance(policy, list) or not policy:
                        raise ValueError("dataset example is missing its policy target")
                    outcome = example.get("outcome")
                    if (
                        isinstance(outcome, bool)
                        or not isinstance(outcome, (int, float))
                        or outcome not in (-1, 0, 1)
                    ):
                        raise ValueError("dataset example has an invalid value target")
                    outcomes[int(outcome)] = outcomes.get(int(outcome), 0) + 1
                    probabilities = [float(item["probability"]) for item in policy]
                    if any(
                        value <= 0 or not math.isfinite(value)
                        for value in probabilities
                    ):
                        raise ValueError("policy target probabilities must be positive")
                    if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
                        raise ValueError("policy target probabilities must sum to one")
                    support = len(probabilities)
                    entropy = -sum(value * math.log(value) for value in probabilities)
                    examples += 1
                    supports[support] = supports.get(support, 0) + 1
                    support_total += support
                    entropy_total += entropy
                    normalized_entropy_total += (
                        entropy / math.log(support) if support > 1 else 0.0
                    )
                    maximum_probability_total += max(probabilities)
    if not examples:
        raise ValueError("dataset contains no policy targets")
    return {
        "policy": {
            "examples": examples,
            "support": {
                "mean": support_total / examples,
                "minimum": min(supports),
                "maximum": max(supports),
                "histogram": {str(key): supports[key] for key in sorted(supports)},
            },
            "entropy_mean_nats": entropy_total / examples,
            "normalized_entropy_mean": normalized_entropy_total / examples,
            "maximum_probability_mean": maximum_probability_total / examples,
        },
        "value": {
            "examples": examples,
            "counts": {str(key): outcomes.get(key, 0) for key in (-1, 0, 1)},
            "fractions": {
                str(key): outcomes.get(key, 0) / examples for key in (-1, 0, 1)
            },
        },
    }


def _artifact_inventory(path: Path, root: Path) -> dict[str, object]:
    """Return a verifiable inventory for one retained artifact subtree."""

    files = sorted(item for item in path.rglob("*") if item.is_file())
    objects = [
        {
            "path": str(item.relative_to(root)),
            "sha256": _sha256(item),
            "bytes": item.stat().st_size,
        }
        for item in files
    ]
    return {
        "files": len(files),
        "bytes": sum(item["bytes"] for item in objects),
        "objects": objects,
    }


def _checkpoint(path: Path) -> dict[str, object]:
    loaded = load_policy_value_checkpoint(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "model_config": loaded.model.config.to_dict(),
    }


def _agent(
    checkpoint: str,
    simulations: int,
    rollout_limit: int,
    device: str,
    exploration: float = math.sqrt(2.0),
    progressive_widening_constant: float = DEFAULT_PROGRESSIVE_WIDENING_CONSTANT,
    progressive_widening_exponent: float = DEFAULT_PROGRESSIVE_WIDENING_EXPONENT,
) -> MCTSAgent:
    loaded = load_policy_value_checkpoint(checkpoint, map_location=device)
    return MCTSAgent(
        simulations=simulations,
        rollout_limit=rollout_limit,
        exploration=exploration,
        progressive_widening_constant=progressive_widening_constant,
        progressive_widening_exponent=progressive_widening_exponent,
        policy_value=NeuralPolicyValue(loaded.model),
    )


def _game_paths(selfplay_roots: list[Path]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for root in selfplay_roots:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        for game in summary["games"]:
            if game["status"] == "completed":
                paths.append(root / game["artifact"])
    return tuple(paths)


def _run_selfplay(
    champion: Path,
    output_dir: Path,
    config: MiniGenerationConfig,
    seed: int,
    device: DeviceSelection,
) -> tuple[BatchSummary, dict[str, object]]:
    """Run CPU orchestration with either synchronous or one shared model."""

    batch_config = BatchConfig(
        games=config.games_per_generation,
        workers=config.workers,
        seed=seed,
        board=experiment_board("mini"),
        red_agent="champion",
        black_agent="champion",
        worker_mode="thread" if device.resolved_device == "cuda" else "process",
    )
    if device.resolved_device == "cpu":
        factory = partial(
            _agent,
            str(champion),
            config.selfplay_simulations,
            config.rollout_limit,
            device.resolved_device,
            config.selfplay_exploration,
            config.selfplay_progressive_widening_constant,
            config.selfplay_progressive_widening_exponent,
        )
        batch = run_batch(
            factory, factory, config=batch_config, output_dir=output_dir
        )
        return batch, {
            "mode": "synchronous",
            "model_instances": "one per active game agent",
            "device": device.to_dict(),
        }

    # Load and place exactly one model before game threads start. Both colors in
    # every game retain CPU MCTS state and submit leaf positions to this queue.
    loaded = load_policy_value_checkpoint(
        champion, map_location=device.resolved_device
    )
    with NeuralInferenceBatcher(
        NeuralPolicyValue(loaded.model),
        batch_size=config.inference_batch_size,
        max_wait_seconds=config.inference_max_wait_seconds,
    ) as inference:
        factory = partial(
            MCTSAgent,
            simulations=config.selfplay_simulations,
            rollout_limit=config.rollout_limit,
            exploration=config.selfplay_exploration,
            progressive_widening_constant=(
                config.selfplay_progressive_widening_constant
            ),
            progressive_widening_exponent=(
                config.selfplay_progressive_widening_exponent
            ),
            policy_value=inference,
        )
        batch = run_batch(
            factory, factory, config=batch_config, output_dir=output_dir
        )
    # ``close()`` joins the inference worker. Take the snapshot after that
    # barrier because callers receive their results just before the worker
    # publishes the corresponding counters.
    statistics = inference.statistics.to_dict()
    return batch, {
        "mode": (
            "synchronous-serialized"
            if config.inference_batch_size == 1
            else "shared-batched"
        ),
        "model_instances": 1,
        "device": device.to_dict(),
        "batch_size": config.inference_batch_size,
        "max_wait_seconds": config.inference_max_wait_seconds,
        "statistics": statistics,
    }


def _evaluate(
    champion: Path,
    candidate: Path,
    config: MiniGenerationConfig,
    seed: int,
    device: DeviceSelection,
) -> dict[str, Any]:
    settings = {
        "type": "mcts",
        "simulations": config.evaluation_simulations,
        "rollout_limit": config.rollout_limit,
        "guidance": "policy-value",
    }
    benchmark_config = BenchmarkConfig(
        agents=(
            AgentConfig("champion", __version__, {
                **settings, "checkpoint_sha256": _sha256(champion)
            }),
            AgentConfig("candidate", __version__, {
                **settings, "checkpoint_sha256": _sha256(candidate)
            }),
        ),
        games_per_pair=config.evaluation_games,
        board=experiment_board("mini"),
        seed=seed,
        confidence_level=0.95,
    )
    result = run_benchmark(
        {
            "champion": partial(
                _agent,
                str(champion),
                config.evaluation_simulations,
                config.rollout_limit,
                device.resolved_device,
            ),
            "candidate": partial(
                _agent,
                str(candidate),
                config.evaluation_simulations,
                config.rollout_limit,
                device.resolved_device,
            ),
        },
        config=benchmark_config,
    ).to_dict()
    pair = result["summary"]["pairs"][0]
    wins = pair["wins"]["candidate"]
    win_rate = wins / config.evaluation_games
    result["promotion"] = {
        "candidate_wins": wins,
        "games": config.evaluation_games,
        "win_rate": win_rate,
        "required_win_rate": config.promotion_win_rate,
        "promoted": win_rate >= config.promotion_win_rate,
        "rule": "candidate wins / all games >= required_win_rate",
    }
    return result


def run_mini_training_generations(
    initial_champion: str | Path,
    output_dir: str | Path,
    *,
    config: MiniGenerationConfig = MiniGenerationConfig(),
) -> dict[str, Any]:
    """Run self-play, windowed training, evaluation, and explicit promotion."""

    if not isinstance(config, MiniGenerationConfig):
        raise TypeError("config must be a MiniGenerationConfig")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError("PYTHONHASHSEED must be 0")
    device = select_device(config.device)
    champion = Path(initial_champion)
    initial = _checkpoint(champion)
    if initial["model_config"] != MINI_POLICY_VALUE_CONFIG.to_dict():
        raise ValueError("initial champion must use the Mini model configuration")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("output directory must be empty or not exist")
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "config.json", config.to_dict())

    started = perf_counter()
    selfplay_roots: list[Path] = []
    generations: list[dict[str, Any]] = []
    lineage: list[dict[str, object]] = []
    report: dict[str, Any] = {
        "format": GENERATIONS_FORMAT,
        "version": GENERATIONS_VERSION,
        "status": "running",
        "config": config.to_dict(),
        "environment": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": device.to_dict(),
        },
        "initial_champion": initial,
        "generations": generations,
        "lineage": lineage,
    }
    _write_json(root / "report.json", report)

    for number in range(1, config.generations + 1):
        generation_root = root / f"generation-{number:04d}"
        generation_root.mkdir()
        champion_before = champion
        evaluation_seed = (
            config.evaluation_seed
            if config.evaluation_seed is not None
            else config.seed + number * 10 + 2
        )
        generation_started = perf_counter()
        generation: dict[str, Any] = {
            "generation": number,
            "status": "running",
            "champion_before": _checkpoint(champion_before),
            "resolved_config": {
                "games": config.games_per_generation,
                "dataset_window": config.dataset_window,
                "selfplay_simulations": config.selfplay_simulations,
                "selfplay_exploration": config.selfplay_exploration,
                "selfplay_progressive_widening_constant": (
                    config.selfplay_progressive_widening_constant
                ),
                "selfplay_progressive_widening_exponent": (
                    config.selfplay_progressive_widening_exponent
                ),
                "evaluation_games": config.evaluation_games,
                "evaluation_simulations": config.evaluation_simulations,
                "rollout_limit": config.rollout_limit,
                "workers": config.workers,
                "inference_batch_size": config.inference_batch_size,
                "inference_max_wait_seconds": config.inference_max_wait_seconds,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "selection_metric": config.selection_metric,
                "promotion_win_rate": config.promotion_win_rate,
                "evaluation_seed": evaluation_seed,
                "artifact_uri": config.artifact_uri,
                "device": device.to_dict(),
                "worker_mode": (
                    "thread" if device.resolved_device == "cuda" else "process"
                ),
            },
            "seeds": {
                "selfplay": config.seed + number * 10,
                "dataset_split": f"issue-59-{config.seed}-{number}",
                "training": config.seed + number * 10 + 1,
                "evaluation": evaluation_seed,
            },
        }
        generations.append(generation)
        _write_json(generation_root / "report.json", generation)
        _write_json(root / "report.json", report)
        stage = "selfplay"
        try:
            selfplay_root = generation_root / "selfplay"
            stage_started = perf_counter()
            batch, inference = _run_selfplay(
                champion_before,
                selfplay_root,
                config,
                config.seed + number * 10,
                device,
            )
            if batch.failed:
                raise RuntimeError(f"self-play had {batch.failed} failed games")
            selfplay_roots.append(selfplay_root)
            runtime_seconds = perf_counter() - stage_started
            completed_games = getattr(
                batch, "completed", config.games_per_generation - batch.failed
            )
            generation["selfplay"] = {
                "runtime_seconds": runtime_seconds,
                "games_per_hour": completed_games / runtime_seconds * 3600.0,
                "summary_sha256": (
                    _sha256(selfplay_root / "summary.json")
                    if (selfplay_root / "summary.json").is_file() else None
                ),
                "device": device.to_dict(),
                "inference": inference,
                "summary": batch.to_dict(),
            }
            _write_json(generation_root / "report.json", generation)

            stage = "dataset"
            stage_started = perf_counter()
            window = selfplay_roots[-config.dataset_window :]
            sources = _game_paths(window)
            dataset = build_dataset(
                sources,
                generation_root / "dataset",
                config=DatasetConfig(
                    shard_size=config.shard_size,
                    validation_fraction=config.validation_fraction,
                    split_seed=f"issue-59-{config.seed}-{number}",
                    metadata={
                        "workflow": GENERATIONS_FORMAT,
                        "generation": number,
                        "window_generations": list(
                            range(max(1, number - len(window) + 1), number + 1)
                        ),
                        "champion_sha256": _sha256(champion_before),
                        "mcts": {
                            "simulations": config.selfplay_simulations,
                            "exploration": config.selfplay_exploration,
                            "rollout_limit": config.rollout_limit,
                            "progressive_widening_constant": (
                                config.selfplay_progressive_widening_constant
                            ),
                            "progressive_widening_exponent": (
                                config.selfplay_progressive_widening_exponent
                            ),
                            "guidance": "policy-value",
                        },
                    },
                ),
            )
            if not dataset.train_examples:
                raise ValueError("training split must contain at least one example")
            target_distributions = _target_distributions(
                generation_root / "dataset", dataset.to_dict()
            )
            generation["dataset"] = {
                "runtime_seconds": perf_counter() - stage_started,
                "source_generations": list(
                    range(max(1, number - len(window) + 1), number + 1)
                ),
                "manifest": dataset.to_dict(),
                "manifest_sha256": _sha256(
                    generation_root / "dataset" / "manifest.json"
                ),
                # Preserve the original field for existing consumers while
                # exposing both target families under one scaling-report key.
                "policy_target_quality": target_distributions["policy"],
                "target_distributions": target_distributions,
            }
            _write_json(generation_root / "report.json", generation)

            stage = "training"
            stage_started = perf_counter()
            training_root = generation_root / "candidate"
            training = train_model(
                generation_root / "dataset",
                training_root,
                config=TrainingConfig(
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    weight_decay=config.weight_decay,
                    seed=config.seed + number * 10 + 1,
                    device=config.device,
                    selection_metric=config.selection_metric,
                ),
                model_config=MINI_POLICY_VALUE_CONFIG,
                initial_checkpoint=champion_before,
            )
            candidate = training_root / "best.pt"
            generation["training"] = {
                "runtime_seconds": perf_counter() - stage_started,
                "initialized_from_sha256": _sha256(champion_before),
                "summary": training.to_dict(),
                "candidate": _checkpoint(candidate),
            }
            _write_json(generation_root / "report.json", generation)

            stage = "evaluation"
            stage_started = perf_counter()
            evaluation = _evaluate(
                champion_before,
                candidate,
                config,
                evaluation_seed,
                device,
            )
            evaluation["runtime_seconds"] = perf_counter() - stage_started
            _write_json(generation_root / "evaluation.json", evaluation)
            evaluation_artifact = {
                "path": str(generation_root / "evaluation.json"),
                "sha256": _sha256(generation_root / "evaluation.json"),
                "bytes": (generation_root / "evaluation.json").stat().st_size,
                "opponent": "parent champion",
            }
            generation["evaluation_artifacts"] = [evaluation_artifact]
            promoted = evaluation["promotion"]["promoted"]
            if promoted:
                champion = candidate
            generation["evaluation"] = evaluation
            generation["decision"] = "promoted" if promoted else "rejected"
            generation["champion_after"] = _checkpoint(champion)
            generation["status"] = "completed"
            generation["runtime_seconds"] = perf_counter() - generation_started
            inventory = {
                "selfplay": _artifact_inventory(
                    generation_root / "selfplay", generation_root
                ),
                "dataset": _artifact_inventory(
                    generation_root / "dataset", generation_root
                ),
                "training": _artifact_inventory(
                    generation_root / "candidate", generation_root
                ),
                "evaluation": {
                    "files": 1,
                    "bytes": evaluation_artifact["bytes"],
                    "objects": [{
                        "path": "evaluation.json",
                        "sha256": evaluation_artifact["sha256"],
                        "bytes": evaluation_artifact["bytes"],
                    }],
                },
            }
            retained_files = sum(item["files"] for item in inventory.values())
            retained_bytes = sum(item["bytes"] for item in inventory.values())
            generation["retention_manifest"] = {
                "format": "twixt-ai-artifact-retention-manifest",
                "version": 1,
                "external_uri": config.artifact_uri,
                "inventory_complete": True,
                "inventory_sha256": _inventory_sha256(
                    inventory, retained_files, retained_bytes
                ),
                "storage_attestation": None,
                "categories": inventory,
                "files": retained_files,
                "bytes": retained_bytes,
            }
            generation["artifact_storage"] = {
                **{
                    name: {"files": value["files"], "bytes": value["bytes"]}
                    for name, value in inventory.items()
                },
                "files": generation["retention_manifest"]["files"],
                "bytes": generation["retention_manifest"]["bytes"],
            }
            lineage.append({
                "generation": number,
                "parent_sha256": _sha256(champion_before),
                "candidate_sha256": _sha256(candidate),
                "decision": generation["decision"],
                "champion_sha256": _sha256(champion),
            })
        except Exception as exc:
            generation["status"] = "failed"
            generation["failed_stage"] = stage
            generation["error"] = {"type": type(exc).__name__, "message": str(exc)}
            generation["runtime_seconds"] = perf_counter() - generation_started
            report["status"] = "failed"
            report["runtime_seconds"] = perf_counter() - started
            _write_json(generation_root / "report.json", generation)
            _write_json(root / "report.json", report)
            raise
        _write_json(generation_root / "report.json", generation)
        _write_json(root / "report.json", report)

    report["status"] = "completed"
    report["final_champion"] = _checkpoint(champion)
    report["runtime_seconds"] = perf_counter() - started
    _write_json(root / "report.json", report)
    return report


__all__ = [
    "GENERATIONS_FORMAT",
    "GENERATIONS_VERSION",
    "MiniGenerationConfig",
    "run_mini_training_generations",
]
