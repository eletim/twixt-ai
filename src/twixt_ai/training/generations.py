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
from twixt_ai.evaluation import AgentConfig, BenchmarkConfig, run_benchmark
from twixt_ai.game import experiment_board
from twixt_ai.models import (
    MINI_POLICY_VALUE_CONFIG,
    load_policy_value_checkpoint,
)
from twixt_ai.search import MCTSAgent
from twixt_ai.search.neural import NeuralPolicyValue
from twixt_ai.selfplay import BatchConfig, run_batch

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
    evaluation_games: int = 20
    evaluation_simulations: int = 20
    rollout_limit: int = 4
    workers: int = 2
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    shard_size: int = 10_000
    promotion_win_rate: float = 0.55
    seed: int = 590_100

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
            "epochs",
            "batch_size",
            "shard_size",
        ):
            _positive_integer(getattr(self, name), name)
        if self.evaluation_games % 2:
            raise ValueError("evaluation_games must be even for paired role swaps")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
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
        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not math.isfinite(self.validation_fraction)
            or not 0 <= self.validation_fraction <= 1
        ):
            raise ValueError("validation_fraction must be in [0, 1]")
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


def _checkpoint(path: Path) -> dict[str, object]:
    loaded = load_policy_value_checkpoint(path)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "model_config": loaded.model.config.to_dict(),
    }


def _agent(checkpoint: str, simulations: int, rollout_limit: int) -> MCTSAgent:
    loaded = load_policy_value_checkpoint(checkpoint)
    return MCTSAgent(
        simulations=simulations,
        rollout_limit=rollout_limit,
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


def _evaluate(
    champion: Path,
    candidate: Path,
    config: MiniGenerationConfig,
    seed: int,
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
            ),
            "candidate": partial(
                _agent,
                str(candidate),
                config.evaluation_simulations,
                config.rollout_limit,
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
        generation_started = perf_counter()
        generation: dict[str, Any] = {
            "generation": number,
            "status": "running",
            "champion_before": _checkpoint(champion_before),
            "resolved_config": {
                "games": config.games_per_generation,
                "dataset_window": config.dataset_window,
                "selfplay_simulations": config.selfplay_simulations,
                "evaluation_games": config.evaluation_games,
                "evaluation_simulations": config.evaluation_simulations,
                "rollout_limit": config.rollout_limit,
                "workers": config.workers,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "promotion_win_rate": config.promotion_win_rate,
            },
            "seeds": {
                "selfplay": config.seed + number * 10,
                "dataset_split": f"issue-59-{config.seed}-{number}",
                "training": config.seed + number * 10 + 1,
                "evaluation": config.seed + number * 10 + 2,
            },
        }
        generations.append(generation)
        _write_json(generation_root / "report.json", generation)
        _write_json(root / "report.json", report)
        stage = "selfplay"
        try:
            selfplay_root = generation_root / "selfplay"
            stage_started = perf_counter()
            factory = partial(
                _agent,
                str(champion_before),
                config.selfplay_simulations,
                config.rollout_limit,
            )
            batch = run_batch(
                factory,
                factory,
                config=BatchConfig(
                    games=config.games_per_generation,
                    workers=config.workers,
                    seed=config.seed + number * 10,
                    board=experiment_board("mini"),
                    red_agent="champion",
                    black_agent="champion",
                    worker_mode="process",
                ),
                output_dir=selfplay_root,
            )
            if batch.failed:
                raise RuntimeError(f"self-play had {batch.failed} failed games")
            selfplay_roots.append(selfplay_root)
            generation["selfplay"] = {
                "runtime_seconds": perf_counter() - stage_started,
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
                    },
                ),
            )
            generation["dataset"] = {
                "runtime_seconds": perf_counter() - stage_started,
                "source_generations": list(
                    range(max(1, number - len(window) + 1), number + 1)
                ),
                "manifest": dataset.to_dict(),
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
                config.seed + number * 10 + 2,
            )
            evaluation["runtime_seconds"] = perf_counter() - stage_started
            _write_json(generation_root / "evaluation.json", evaluation)
            promoted = evaluation["promotion"]["promoted"]
            if promoted:
                champion = candidate
            generation["evaluation"] = evaluation
            generation["decision"] = "promoted" if promoted else "rejected"
            generation["champion_after"] = _checkpoint(champion)
            generation["status"] = "completed"
            generation["runtime_seconds"] = perf_counter() - generation_started
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
