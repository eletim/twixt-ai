"""Fixed-protocol evaluation of trained Mini value candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any, Mapping

import torch

from twixt_ai.device import select_device
from twixt_ai.game import BoardDimensions
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.search import HeuristicSearchAgent, MCTSAgent
from twixt_ai.search.neural import NeuralPolicyValue

from .benchmark import AgentConfig, BenchmarkConfig, run_benchmark
from .mini_strength import _available_cpus, _json_value, _package_version, _sha256


VALUE_CANDIDATE_FORMAT = "twixt-ai-value-candidate-evaluation"
VALUE_CANDIDATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class ValueCandidateConfig:
    """The immutable generation-2 evaluation budget and promotion rule."""

    board: BoardDimensions = BoardDimensions(10, 10)
    games_per_matchup: int = 40
    seed: int = 1_251_300
    confidence_level: float = 0.95
    simulations: int = 20
    rollout_limit: int = 4
    heuristic_search_depth: int = 1
    heuristic_search_node_budget: int = 10_000
    promotion_win_rate: float = 0.55
    device: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        for name in (
            "games_per_matchup",
            "simulations",
            "rollout_limit",
            "heuristic_search_depth",
            "heuristic_search_node_budget",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.games_per_matchup % 2:
            raise ValueError("games_per_matchup must be even so player roles can be swapped")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name, value in (
            ("confidence_level", self.confidence_level),
            ("promotion_win_rate", self.promotion_win_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between zero and one")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["board"] = self.board.to_dict()
        value["heuristic_search"] = {
            "depth": value.pop("heuristic_search_depth"),
            "node_budget": value.pop("heuristic_search_node_budget"),
        }
        return value


def _learned_agent(path: Path, device: str, simulations: int, rollout_limit: int) -> MCTSAgent:
    loaded = load_policy_value_checkpoint(path, map_location=device)
    return MCTSAgent(
        simulations=simulations,
        rollout_limit=rollout_limit,
        policy_value=NeuralPolicyValue(loaded.model),
    )


def run_value_candidate_evaluation(
    champion_path: str | os.PathLike[str],
    issue_57_path: str | os.PathLike[str],
    candidates: Mapping[str, str | os.PathLike[str]],
    *,
    config: ValueCandidateConfig = ValueCandidateConfig(),
) -> dict[str, Any]:
    """Evaluate every candidate without screening or matchup selection."""

    if not isinstance(config, ValueCandidateConfig):
        raise TypeError("config must be a ValueCandidateConfig")
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("candidates must be a non-empty mapping")
    if any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("candidate names must be non-empty strings")
    reserved_names = {
        "generation-2-champion",
        "issue-57",
        "matched-non-neural-mcts",
        "heuristic-search",
    }
    if reserved_names.intersection(candidates):
        raise ValueError("candidate names must not use reserved opponent names")

    champion = Path(champion_path)
    issue_57 = Path(issue_57_path)
    candidate_paths = {name: Path(path) for name, path in candidates.items()}
    all_checkpoints = {"generation-2-champion": champion, "issue-57": issue_57, **candidate_paths}
    device = select_device(config.device)
    checkpoint_records: dict[str, dict[str, object]] = {}
    for name, path in all_checkpoints.items():
        loaded = load_policy_value_checkpoint(path, map_location=device.resolved_device)
        model_board = BoardDimensions(
            loaded.model.config.board_width, loaded.model.config.board_height
        )
        if model_board != config.board:
            raise ValueError(f"{name} checkpoint board dimensions must match the evaluation board")
        checkpoint_records[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "model_config": loaded.model.config.to_dict(),
            "metadata": _json_value(loaded.metadata),
        }

    package_version = _package_version()
    learned_settings = {
        "type": "mcts",
        "simulations": config.simulations,
        "rollout_limit": config.rollout_limit,
        "guidance": "policy-value",
    }
    opponents = ("generation-2-champion", "issue-57", "matched-non-neural-mcts", "heuristic-search")
    matchups: list[dict[str, object]] = []
    total_started = perf_counter()

    for candidate_name, candidate_path in candidate_paths.items():
        for opponent_name in opponents:
            candidate_settings = {
                **learned_settings,
                "checkpoint_sha256": checkpoint_records[candidate_name]["sha256"],
            }
            if opponent_name in ("generation-2-champion", "issue-57"):
                opponent_path = all_checkpoints[opponent_name]
                opponent_settings = {
                    **learned_settings,
                    "checkpoint_sha256": checkpoint_records[opponent_name]["sha256"],
                }
                opponent_factory = partial(
                    _learned_agent,
                    opponent_path,
                    device.resolved_device,
                    config.simulations,
                    config.rollout_limit,
                )
            elif opponent_name == "matched-non-neural-mcts":
                opponent_settings = {
                    "type": "mcts",
                    "simulations": config.simulations,
                    "rollout_limit": config.rollout_limit,
                }
                opponent_factory = partial(
                    MCTSAgent,
                    simulations=config.simulations,
                    rollout_limit=config.rollout_limit,
                )
            else:
                opponent_settings = {
                    "type": "search",
                    "depth": config.heuristic_search_depth,
                    "node_budget": config.heuristic_search_node_budget,
                }
                opponent_factory = partial(
                    HeuristicSearchAgent,
                    depth=config.heuristic_search_depth,
                    node_budget=config.heuristic_search_node_budget,
                )
            benchmark_config = BenchmarkConfig(
                agents=(
                    AgentConfig(candidate_name, package_version, candidate_settings),
                    AgentConfig(opponent_name, package_version, opponent_settings),
                ),
                games_per_pair=config.games_per_matchup,
                board=config.board,
                seed=config.seed,
                confidence_level=config.confidence_level,
            )
            started = perf_counter()
            result = run_benchmark(
                {
                    candidate_name: partial(
                        _learned_agent,
                        candidate_path,
                        device.resolved_device,
                        config.simulations,
                        config.rollout_limit,
                    ),
                    opponent_name: opponent_factory,
                },
                config=benchmark_config,
            )
            artifact = result.to_dict()
            matchups.append({
                "candidate": candidate_name,
                "opponent": opponent_name,
                "candidate_search": candidate_settings,
                "opponent_search": opponent_settings,
                "strength": artifact["summary"]["pairs"][0],
                "role_splits": artifact["summary"]["agents"],
                "first_player": artifact["summary"]["first_player"],
                "runtime": {"wall_seconds": perf_counter() - started},
                "games": artifact["games"],
            })

    decisions = []
    for candidate_name in candidate_paths:
        champion_match = next(
            matchup for matchup in matchups
            if matchup["candidate"] == candidate_name
            and matchup["opponent"] == "generation-2-champion"
        )
        wins = champion_match["strength"]["wins"][candidate_name]
        win_rate = wins / config.games_per_matchup
        promoted = win_rate >= config.promotion_win_rate
        decisions.append({
            "candidate": candidate_name,
            "candidate_wins": wins,
            "games": config.games_per_matchup,
            "win_rate": win_rate,
            "required_win_rate": config.promotion_win_rate,
            "promoted": promoted,
            "decision": "promote" if promoted else "reject",
            "rule": "candidate wins / all champion games >= required_win_rate",
        })

    return {
        "format": VALUE_CANDIDATE_FORMAT,
        "version": VALUE_CANDIDATE_VERSION,
        "config": config.to_dict(),
        "checkpoints": checkpoint_records,
        "environment": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": device.to_dict(),
            "available_cpus": _available_cpus(),
        },
        "methodology": {
            "paired_role_swaps": True,
            "shared_pair_seed_schedule": True,
            "generation_2_budget_and_rules_unchanged": (
                config.board == BoardDimensions(10, 10)
                and config.games_per_matchup == 40
                and config.simulations == 20
                and config.rollout_limit == 4
                and config.promotion_win_rate == 0.55
            ),
            "heuristic_configuration_fixed_across_every_matchup": True,
            "all_candidates_and_opponents_recorded": True,
            "promotion_gate": "generation-2-champion",
            "promotion_rule": "candidate wins / all champion games >= 0.55",
        },
        "promotion_decisions": decisions,
        "runtime": {"wall_seconds": perf_counter() - total_started},
        "matchups": matchups,
    }


__all__ = [
    "VALUE_CANDIDATE_FORMAT",
    "VALUE_CANDIDATE_VERSION",
    "ValueCandidateConfig",
    "run_value_candidate_evaluation",
]
