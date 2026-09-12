"""Matched search-setting diagnosis for a learned Mini checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import torch

from twixt_ai.device import select_device
from twixt_ai.game import BoardDimensions
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.search import HeuristicSearchAgent, MCTSAgent
from twixt_ai.search.neural import NeuralPolicyValue

from .benchmark import AgentConfig, BenchmarkConfig, run_benchmark
from .mini_strength import (
    AblatedPolicyValue,
    GUIDANCE_MODES,
    _available_cpus,
    _json_value,
    _package_version,
    _sha256,
)


MATCHED_SEARCH_FORMAT = "twixt-ai-matched-search-analysis"
MATCHED_SEARCH_VERSION = 1


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _positive_number(value: object, name: str, *, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not allow_zero)
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")


@dataclass(frozen=True, slots=True)
class SearchSetting:
    """One explicitly named MCTS configuration in the diagnostic screen."""

    name: str
    simulations: int
    exploration: float
    progressive_widening_constant: float
    progressive_widening_exponent: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        _positive_integer(self.simulations, "simulations")
        _positive_number(self.exploration, "exploration", allow_zero=True)
        _positive_number(
            self.progressive_widening_constant,
            "progressive_widening_constant",
        )
        _positive_number(
            self.progressive_widening_exponent,
            "progressive_widening_exponent",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_SEARCH_SETTINGS = (
    SearchSetting("standard-20", 20, math.sqrt(2.0), 1.5, 0.5),
    SearchSetting("budget-64", 64, math.sqrt(2.0), 1.5, 0.5),
    SearchSetting("budget-128", 128, math.sqrt(2.0), 1.5, 0.5),
    SearchSetting("exploration-0.7", 20, 0.7, 1.5, 0.5),
    SearchSetting("widening-3.0", 20, math.sqrt(2.0), 3.0, 0.5),
    SearchSetting("widening-exponent-0.75", 20, math.sqrt(2.0), 1.5, 0.75),
    SearchSetting("teacher-like-64", 64, 0.7, 3.0, 0.5),
)


@dataclass(frozen=True, slots=True)
class MatchedSearchConfig:
    """Complete schedule and fixed-opponent settings for the screen."""

    board: BoardDimensions = BoardDimensions(10, 10)
    games_per_matchup: int = 20
    seed: int = 1_188_100
    confidence_level: float = 0.95
    rollout_limit: int = 4
    heuristic_search_depth: int = 1
    heuristic_search_node_budget: int = 10_000
    settings: tuple[SearchSetting, ...] = DEFAULT_SEARCH_SETTINGS
    device: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        for name in (
            "games_per_matchup",
            "rollout_limit",
            "heuristic_search_depth",
            "heuristic_search_node_budget",
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
        settings = tuple(self.settings)
        if not settings or any(not isinstance(item, SearchSetting) for item in settings):
            raise TypeError("settings must contain at least one SearchSetting")
        names = [item.name for item in settings]
        if len(names) != len(set(names)):
            raise ValueError("search setting names must be unique")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
        object.__setattr__(self, "settings", settings)

    def to_dict(self) -> dict[str, object]:
        return {
            "board": self.board.to_dict(),
            "games_per_matchup": self.games_per_matchup,
            "seed": self.seed,
            "confidence_level": self.confidence_level,
            "rollout_limit": self.rollout_limit,
            "heuristic_search": {
                "depth": self.heuristic_search_depth,
                "node_budget": self.heuristic_search_node_budget,
            },
            "settings": [item.to_dict() for item in self.settings],
            "device": self.device,
        }


def _mcts_settings(setting: SearchSetting, rollout_limit: int) -> dict[str, object]:
    return {
        "type": "mcts",
        "simulations": setting.simulations,
        "exploration": setting.exploration,
        "rollout_limit": rollout_limit,
        "progressive_widening": {
            "constant": setting.progressive_widening_constant,
            "exponent": setting.progressive_widening_exponent,
        },
    }


def _mcts_agent(
    setting: SearchSetting,
    rollout_limit: int,
    *,
    policy_value: AblatedPolicyValue | None = None,
) -> MCTSAgent:
    return MCTSAgent(
        simulations=setting.simulations,
        exploration=setting.exploration,
        rollout_limit=rollout_limit,
        progressive_widening_constant=setting.progressive_widening_constant,
        progressive_widening_exponent=setting.progressive_widening_exponent,
        policy_value=policy_value,
    )


def run_matched_search_analysis(
    checkpoint_path: str | os.PathLike[str],
    *,
    config: MatchedSearchConfig = MatchedSearchConfig(),
) -> dict[str, Any]:
    """Screen every guidance mode and setting against both fixed baselines."""

    if not isinstance(config, MatchedSearchConfig):
        raise TypeError("config must be a MatchedSearchConfig")
    checkpoint = Path(checkpoint_path)
    device = select_device(config.device)
    loaded = load_policy_value_checkpoint(
        checkpoint, map_location=device.resolved_device
    )
    model_board = BoardDimensions(
        loaded.model.config.board_width, loaded.model.config.board_height
    )
    if model_board != config.board:
        raise ValueError("checkpoint board dimensions must match the evaluation board")

    package_version = _package_version()
    checkpoint_sha256 = _sha256(checkpoint)
    neural = NeuralPolicyValue(loaded.model)
    heuristic_settings = {
        "type": "search",
        "depth": config.heuristic_search_depth,
        "node_budget": config.heuristic_search_node_budget,
    }
    matchups: list[dict[str, object]] = []
    total_started = perf_counter()

    for setting in config.settings:
        search_settings = _mcts_settings(setting, config.rollout_limit)
        for mode in GUIDANCE_MODES:
            candidate = f"learned-{mode}-{setting.name}"
            guidance = AblatedPolicyValue(neural, mode)
            for baseline in ("heuristic-search", "matched-non-neural-mcts"):
                baseline_settings = (
                    heuristic_settings
                    if baseline == "heuristic-search"
                    else search_settings
                )
                benchmark_config = BenchmarkConfig(
                    agents=(
                        AgentConfig(
                            candidate,
                            package_version,
                            {
                                **search_settings,
                                "guidance": mode,
                                "checkpoint_sha256": checkpoint_sha256,
                            },
                        ),
                        AgentConfig(baseline, package_version, baseline_settings),
                    ),
                    games_per_pair=config.games_per_matchup,
                    board=config.board,
                    seed=config.seed,
                    confidence_level=config.confidence_level,
                )
                if baseline == "heuristic-search":
                    baseline_factory = lambda: HeuristicSearchAgent(
                        depth=config.heuristic_search_depth,
                        node_budget=config.heuristic_search_node_budget,
                    )
                else:
                    baseline_factory = lambda setting=setting: _mcts_agent(
                        setting, config.rollout_limit
                    )
                factories = {
                    candidate: lambda guidance=guidance, setting=setting: _mcts_agent(
                        setting,
                        config.rollout_limit,
                        policy_value=guidance,
                    ),
                    baseline: baseline_factory,
                }
                started = perf_counter()
                result = run_benchmark(factories, config=benchmark_config)
                wall_seconds = perf_counter() - started
                artifact = result.to_dict()
                matchups.append(
                    {
                        "setting": setting.name,
                        "guidance": mode,
                        "candidate": candidate,
                        "baseline": baseline,
                        "candidate_search": search_settings,
                        "baseline_search": baseline_settings,
                        "strength": artifact["summary"]["pairs"][0],
                        "role_splits": artifact["summary"]["agents"],
                        "first_player": artifact["summary"]["first_player"],
                        "runtime": {"wall_seconds": wall_seconds},
                        "games": artifact["games"],
                    }
                )

    return {
        "format": MATCHED_SEARCH_FORMAT,
        "version": MATCHED_SEARCH_VERSION,
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
            "device": device.to_dict(),
            "available_cpus": _available_cpus(),
        },
        "methodology": {
            "guidance_modes": list(GUIDANCE_MODES),
            "baselines": ["heuristic-search", "matched-non-neural-mcts"],
            "paired_role_swaps": True,
            "shared_pair_seed_schedule": True,
            "non_neural_mcts_settings_match_candidate": True,
            "heuristic_configuration_fixed_across_every_matchup": True,
            "all_scheduled_settings_recorded": True,
            "confidence_interval": "Wilson score interval for win rate",
        },
        "runtime": {"wall_seconds": perf_counter() - total_started},
        "matchups": matchups,
    }


__all__ = [
    "DEFAULT_SEARCH_SETTINGS",
    "MATCHED_SEARCH_FORMAT",
    "MATCHED_SEARCH_VERSION",
    "MatchedSearchConfig",
    "SearchSetting",
    "run_matched_search_analysis",
]
