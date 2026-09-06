"""Reproducible strength evaluation for the first learned Mini model."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Literal

import torch

from twixt_ai import __version__
from twixt_ai.agents import RandomAgent
from twixt_ai.game import BoardDimensions, GameState, PegPlacement
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.search import (
    DEFAULT_ROLLOUT_LIMIT,
    HeuristicSearchAgent,
    MCTSAgent,
    PolicyValueEstimate,
)
from twixt_ai.search.neural import NeuralPolicyValue

from .benchmark import AgentConfig, BenchmarkConfig, run_benchmark


MINI_STRENGTH_FORMAT = "twixt-ai-mini-strength-evaluation"
MINI_STRENGTH_VERSION = 1
GuidanceMode = Literal["policy-value", "policy-only", "value-only"]
GUIDANCE_MODES: tuple[GuidanceMode, ...] = (
    "policy-value",
    "policy-only",
    "value-only",
)
BASELINES = ("random", "heuristic-search", "non-neural-mcts")


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class MiniStrengthConfig:
    """Complete schedule and agent budgets for a Mini strength experiment."""

    board: BoardDimensions = BoardDimensions(10, 10)
    games_per_matchup: int = 20
    seed: int = 580_100
    confidence_level: float = 0.95
    simulations: int = 20
    rollout_limit: int = DEFAULT_ROLLOUT_LIMIT
    search_depth: int = 1
    search_node_budget: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        for name in (
            "games_per_matchup",
            "simulations",
            "rollout_limit",
            "search_depth",
            "search_node_budget",
        ):
            _positive_integer(getattr(self, name), name)
        if self.games_per_matchup % 2:
            raise ValueError("games_per_matchup must be even so player roles can be swapped")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.confidence_level, bool)
            or not isinstance(self.confidence_level, (int, float))
            or not math.isfinite(self.confidence_level)
            or not 0 < self.confidence_level < 1
        ):
            raise ValueError("confidence_level must be between zero and one")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["board"] = self.board.to_dict()
        return value


class AblatedPolicyValue:
    """Expose a stable policy-only or value-only view of model inference."""

    def __init__(self, policy_value: NeuralPolicyValue, mode: GuidanceMode) -> None:
        if not isinstance(policy_value, NeuralPolicyValue):
            raise TypeError("policy_value must be a NeuralPolicyValue")
        if mode not in GUIDANCE_MODES:
            raise ValueError(f"mode must be one of {GUIDANCE_MODES}")
        self.policy_value = policy_value
        self.mode = mode

    def __call__(
        self, state: GameState, moves: tuple[PegPlacement, ...]
    ) -> PolicyValueEstimate:
        estimate = self.policy_value(state, moves)
        if self.mode == "policy-only":
            return PolicyValueEstimate(estimate.priors, None)
        if self.mode == "value-only":
            # An empty mapping makes MCTS retain its uniform priors.
            return PolicyValueEstimate({}, estimate.value)
        return estimate


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-POSIX fallback
        return os.cpu_count() or 1


def _package_version() -> str:
    """Return the version of the imported source checkout.

    Distribution metadata may be absent for ``PYTHONPATH=src`` execution or
    belong to a different installed checkout. In either case, the imported
    package's version describes the code that is actually being evaluated.
    """

    try:
        installed_version = version("twixt-ai")
    except PackageNotFoundError:
        return __version__
    return installed_version if installed_version == __version__ else __version__


def _json_value(value: object) -> object:
    """Copy checkpoint metadata into JSON-native containers."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def run_mini_strength_evaluation(
    checkpoint_path: str | os.PathLike[str],
    *,
    config: MiniStrengthConfig = MiniStrengthConfig(),
) -> dict[str, Any]:
    """Evaluate all learned guidance modes against three fixed baselines."""

    if not isinstance(config, MiniStrengthConfig):
        raise TypeError("config must be a MiniStrengthConfig")
    checkpoint = Path(checkpoint_path)
    loaded = load_policy_value_checkpoint(checkpoint)
    model_board = BoardDimensions(
        loaded.model.config.board_width, loaded.model.config.board_height
    )
    if model_board != config.board:
        raise ValueError("checkpoint board dimensions must match the evaluation board")

    package_version = _package_version()
    neural = NeuralPolicyValue(loaded.model)
    checkpoint_sha256 = _sha256(checkpoint)
    learned_settings = {
        "type": "mcts",
        "simulations": config.simulations,
        "rollout_limit": config.rollout_limit,
    }
    baseline_settings: dict[str, dict[str, object]] = {
        "random": {"type": "random"},
        "heuristic-search": {
            "type": "search",
            "depth": config.search_depth,
            "node_budget": config.search_node_budget,
        },
        "non-neural-mcts": dict(learned_settings),
    }
    baseline_factories = {
        "random": RandomAgent,
        "heuristic-search": lambda: HeuristicSearchAgent(
            depth=config.search_depth, node_budget=config.search_node_budget
        ),
        "non-neural-mcts": lambda: MCTSAgent(
            simulations=config.simulations, rollout_limit=config.rollout_limit
        ),
    }

    matchups: list[dict[str, object]] = []
    total_started = perf_counter()
    for mode in GUIDANCE_MODES:
        candidate = f"learned-{mode}"
        guidance = AblatedPolicyValue(neural, mode)
        for baseline in BASELINES:
            benchmark_config = BenchmarkConfig(
                agents=(
                    AgentConfig(
                        candidate,
                        package_version,
                        {
                            **learned_settings,
                            "guidance": mode,
                            "checkpoint_sha256": checkpoint_sha256,
                        },
                    ),
                    AgentConfig(
                        baseline, package_version, baseline_settings[baseline]
                    ),
                ),
                games_per_pair=config.games_per_matchup,
                board=config.board,
                seed=config.seed,
                confidence_level=config.confidence_level,
            )
            factories = {
                candidate: lambda guidance=guidance: MCTSAgent(
                    simulations=config.simulations,
                    rollout_limit=config.rollout_limit,
                    policy_value=guidance,
                ),
                baseline: baseline_factories[baseline],
            }
            started = perf_counter()
            result = run_benchmark(factories, config=benchmark_config)
            wall_seconds = perf_counter() - started
            artifact = result.to_dict()
            pair = artifact["summary"]["pairs"][0]
            total_moves = sum(game.move_count for game in result.games)
            matchups.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "strength": pair,
                    "role_splits": artifact["summary"]["agents"],
                    "first_player": artifact["summary"]["first_player"],
                    "runtime": {
                        "wall_seconds": wall_seconds,
                        "seconds_per_game": wall_seconds / config.games_per_matchup,
                        "moves": total_moves,
                        "seconds_per_move": wall_seconds / total_moves,
                    },
                    "games": artifact["games"],
                },
            )

    return {
        "format": MINI_STRENGTH_FORMAT,
        "version": MINI_STRENGTH_VERSION,
        "config": config.to_dict(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint.stat().st_size,
            "model_config": loaded.model.config.to_dict(),
            "metadata": _json_value(loaded.metadata),
        },
        "environment": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(next(loaded.model.parameters()).device),
            "available_cpus": _available_cpus(),
        },
        "methodology": {
            "guidance_modes": list(GUIDANCE_MODES),
            "baselines": list(BASELINES),
            "paired_role_swaps": True,
            "shared_pair_seed_schedule": True,
            "equal_mcts_simulation_budgets": True,
            "confidence_interval": "Wilson score interval for win rate",
        },
        "runtime": {"wall_seconds": perf_counter() - total_started},
        "matchups": matchups,
    }


__all__ = [
    "AblatedPolicyValue",
    "BASELINES",
    "GUIDANCE_MODES",
    "MINI_STRENGTH_FORMAT",
    "MINI_STRENGTH_VERSION",
    "MiniStrengthConfig",
    "run_mini_strength_evaluation",
]
