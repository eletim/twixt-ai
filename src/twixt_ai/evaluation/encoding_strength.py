"""Matched learned-MCTS strength comparison for Mini input encodings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
import platform
from time import perf_counter
from typing import Any

import torch

from twixt_ai.game import BoardDimensions
from twixt_ai.models import load_policy_value_checkpoint
from twixt_ai.search import DEFAULT_ROLLOUT_LIMIT, MCTSAgent
from twixt_ai.search.neural import NeuralPolicyValue

from .benchmark import AgentConfig, BenchmarkConfig, run_benchmark
from .mini_strength import AblatedPolicyValue, GUIDANCE_MODES, _json_value, _package_version


ENCODING_STRENGTH_FORMAT = "twixt-ai-encoding-strength-comparison"
ENCODING_STRENGTH_VERSION = 1


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class EncodingStrengthConfig:
    """Complete paired schedule and shared MCTS budget."""

    board: BoardDimensions = BoardDimensions(10, 10)
    games_per_matchup: int = 20
    seed: int = 760_100
    confidence_level: float = 0.95
    simulations: int = 20
    rollout_limit: int = DEFAULT_ROLLOUT_LIMIT

    def __post_init__(self) -> None:
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        for name in ("games_per_matchup", "simulations", "rollout_limit"):
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-POSIX fallback
        return os.cpu_count() or 1


def _checkpoint_description(path: Path, loaded: Any) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "model_config": loaded.model.config.to_dict(),
        "metadata": _json_value(loaded.metadata),
    }


def _matched_checkpoint_field(metadata: object, name: str) -> object:
    return metadata.get(name) if isinstance(metadata, Mapping) else None


def _validate_checkpoints(ten: Any, twenty_two: Any, board: BoardDimensions) -> None:
    ten_config = ten.model.config
    twenty_two_config = twenty_two.model.config
    if (ten_config.encoding_version, ten_config.input_channels) != (2, 10):
        raise ValueError("10-plane checkpoint must use encoding version 2 with 10 inputs")
    if (twenty_two_config.encoding_version, twenty_two_config.input_channels) != (1, 22):
        raise ValueError("22-plane checkpoint must use encoding version 1 with 22 inputs")
    for config in (ten_config, twenty_two_config):
        if (config.board_width, config.board_height) != (board.width, board.height):
            raise ValueError("checkpoint board dimensions must match the evaluation board")

    ignored = {"encoding_version", "input_channels"}
    ten_architecture = {
        key: value for key, value in ten_config.to_dict().items() if key not in ignored
    }
    twenty_two_architecture = {
        key: value
        for key, value in twenty_two_config.to_dict().items()
        if key not in ignored
    }
    if ten_architecture != twenty_two_architecture:
        raise ValueError("checkpoint architectures must differ only by input encoding")
    for field in ("dataset_sha256", "training_config"):
        ten_value = _matched_checkpoint_field(ten.metadata, field)
        twenty_two_value = _matched_checkpoint_field(twenty_two.metadata, field)
        if ten_value is None or twenty_two_value is None:
            raise ValueError(f"checkpoint metadata field {field!r} must be recorded")
        if ten_value != twenty_two_value:
            raise ValueError(f"checkpoint metadata field {field!r} must match")


def run_encoding_strength_comparison(
    ten_plane_checkpoint: str | os.PathLike[str],
    twenty_two_plane_checkpoint: str | os.PathLike[str],
    *,
    config: EncodingStrengthConfig = EncodingStrengthConfig(),
) -> dict[str, Any]:
    """Compare matched 10- and 22-plane models directly and to one baseline."""

    if not isinstance(config, EncodingStrengthConfig):
        raise TypeError("config must be an EncodingStrengthConfig")
    paths = {
        "10-plane": Path(ten_plane_checkpoint),
        "22-plane": Path(twenty_two_plane_checkpoint),
    }
    loaded = {name: load_policy_value_checkpoint(path) for name, path in paths.items()}
    _validate_checkpoints(loaded["10-plane"], loaded["22-plane"], config.board)

    neural = {
        name: NeuralPolicyValue(checkpoint.model) for name, checkpoint in loaded.items()
    }
    checkpoint_details = {
        name: _checkpoint_description(paths[name], loaded[name]) for name in paths
    }
    package_version = _package_version()
    search_settings = {
        "type": "mcts",
        "simulations": config.simulations,
        "rollout_limit": config.rollout_limit,
    }
    matchup_specs = (
        ("10-plane", "22-plane"),
        ("10-plane", "non-neural-mcts"),
        ("22-plane", "non-neural-mcts"),
    )

    matchups: list[dict[str, object]] = []
    total_started = perf_counter()
    for mode in GUIDANCE_MODES:
        guidance = {
            name: AblatedPolicyValue(policy_value, mode)
            for name, policy_value in neural.items()
        }
        for left, right in matchup_specs:
            names = tuple(
                "non-neural-mcts" if name == "non-neural-mcts" else f"{name}-{mode}"
                for name in (left, right)
            )
            agents = []
            factories = {}
            for source_name, agent_name in zip((left, right), names):
                if source_name == "non-neural-mcts":
                    agent_settings = dict(search_settings)
                    factories[agent_name] = lambda: MCTSAgent(
                        simulations=config.simulations,
                        rollout_limit=config.rollout_limit,
                    )
                else:
                    agent_settings = {
                        **search_settings,
                        "guidance": mode,
                        "checkpoint_sha256": checkpoint_details[source_name]["sha256"],
                    }
                    policy_value = guidance[source_name]
                    factories[agent_name] = lambda policy_value=policy_value: MCTSAgent(
                        simulations=config.simulations,
                        rollout_limit=config.rollout_limit,
                        policy_value=policy_value,
                    )
                agents.append(AgentConfig(agent_name, package_version, agent_settings))

            benchmark_config = BenchmarkConfig(
                agents=tuple(agents),
                games_per_pair=config.games_per_matchup,
                board=config.board,
                seed=config.seed,
                confidence_level=config.confidence_level,
            )
            started = perf_counter()
            result = run_benchmark(factories, config=benchmark_config)
            wall_seconds = perf_counter() - started
            artifact = result.to_dict()
            total_moves = sum(game.move_count for game in result.games)
            matchups.append(
                {
                    "guidance": mode,
                    "comparison": [left, right],
                    "agents": list(names),
                    "strength": artifact["summary"]["pairs"][0],
                    "role_splits": artifact["summary"]["agents"],
                    "first_player": artifact["summary"]["first_player"],
                    "runtime": {
                        "wall_seconds": wall_seconds,
                        "seconds_per_game": wall_seconds / config.games_per_matchup,
                        "moves": total_moves,
                        "seconds_per_move": wall_seconds / total_moves,
                    },
                    "games": artifact["games"],
                }
            )

    first_parameter = next(loaded["10-plane"].model.parameters())
    return {
        "format": ENCODING_STRENGTH_FORMAT,
        "version": ENCODING_STRENGTH_VERSION,
        "config": config.to_dict(),
        "checkpoints": checkpoint_details,
        "environment": {
            "implementation": platform.python_implementation(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(first_parameter.device),
            "available_cpus": _available_cpus(),
        },
        "methodology": {
            "guidance_modes": list(GUIDANCE_MODES),
            "comparisons": [list(pair) for pair in matchup_specs],
            "paired_role_swaps": True,
            "shared_pair_seed_schedule": True,
            "equal_mcts_simulation_budgets": True,
            "matched_model_capacity_and_training_data": True,
            "confidence_interval": "Wilson score interval for win rate",
            "runtime_interpretation": (
                "Complete two-agent matchup workload; strength is determined by "
                "fixed budgets, not elapsed time."
            ),
        },
        "runtime": {"wall_seconds": perf_counter() - total_started},
        "matchups": matchups,
    }


__all__ = [
    "ENCODING_STRENGTH_FORMAT",
    "ENCODING_STRENGTH_VERSION",
    "EncodingStrengthConfig",
    "run_encoding_strength_comparison",
]
