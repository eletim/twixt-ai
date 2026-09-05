"""Reproducible microbenchmarks for the canonical game engine."""

from __future__ import annotations

from dataclasses import dataclass
import platform
from random import Random
from statistics import median
from time import perf_counter_ns
from typing import Callable

from .rules import PegPlacement, automatic_links_for_placement, legal_peg_placements
from .state import BoardDimensions, GameState, Peg
from .transitions import apply_move, create_game
from .win import has_winning_path


ENGINE_BENCHMARK_FORMAT = "twixt-ai-engine-benchmark"
ENGINE_BENCHMARK_VERSION = 1


@dataclass(frozen=True, slots=True)
class EngineBenchmarkConfig:
    """Settings that completely define an engine benchmark workload."""

    board: BoardDimensions = BoardDimensions()
    seed: int = 2401
    ply: int = 160
    iterations: int = 200
    repeats: int = 5
    warmups: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for name in ("ply", "iterations", "repeats", "warmups"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.ply < 0:
            raise ValueError("ply must be non-negative")
        if self.iterations < 1 or self.repeats < 1:
            raise ValueError("iterations and repeats must be positive")
        if self.warmups < 0:
            raise ValueError("warmups must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "board": self.board.to_dict(),
            "seed": self.seed,
            "ply": self.ply,
            "iterations": self.iterations,
            "repeats": self.repeats,
            "warmups": self.warmups,
        }


def _fixture(config: EngineBenchmarkConfig) -> tuple[GameState, PegPlacement]:
    """Build one deterministic, non-terminal position and its next move."""

    random = Random(config.seed)
    state = create_game(config.board)
    for current_ply in range(config.ply + 1):
        moves = legal_peg_placements(state)
        if not moves:
            raise ValueError(
                f"benchmark fixture became terminal before ply {config.ply}; "
                "choose a smaller ply or another seed"
            )
        move = moves[random.randrange(len(moves))]
        if current_ply == config.ply:
            return state, move
        state = apply_move(state, move)
        if state.is_terminal:
            raise ValueError(
                f"benchmark fixture became terminal before ply {config.ply}; "
                "choose a smaller ply or another seed"
            )
    raise AssertionError("unreachable")


def _measure(
    operation: Callable[[], object], config: EngineBenchmarkConfig
) -> dict[str, int | float]:
    for _ in range(config.warmups):
        for _ in range(config.iterations):
            operation()

    samples: list[float] = []
    for _ in range(config.repeats):
        started = perf_counter_ns()
        for _ in range(config.iterations):
            operation()
        elapsed = perf_counter_ns() - started
        samples.append(elapsed / config.iterations)

    median_ns = median(samples)
    return {
        "iterations_per_repeat": config.iterations,
        "median_ns_per_call": median_ns,
        "min_ns_per_call": min(samples),
        "max_ns_per_call": max(samples),
        "calls_per_second": 1_000_000_000 / median_ns,
    }


def run_engine_benchmarks(
    config: EngineBenchmarkConfig | None = None,
) -> dict[str, object]:
    """Run legal-move, link, win, and full-transition microbenchmarks."""

    if config is None:
        config = EngineBenchmarkConfig()
    if not isinstance(config, EngineBenchmarkConfig):
        raise TypeError("config must be an EngineBenchmarkConfig")

    state, move = _fixture(config)
    peg = Peg(move.player, move.coordinate)
    updated = apply_move(state, move)
    operations: tuple[tuple[str, Callable[[], object]], ...] = (
        ("legal_move_generation", lambda: legal_peg_placements(state)),
        ("automatic_link_updates", lambda: automatic_links_for_placement(state, peg)),
        ("win_check", lambda: has_winning_path(updated, move.player)),
        ("move_application", lambda: apply_move(state, move)),
    )
    results = {name: _measure(operation, config) for name, operation in operations}
    return {
        "format": ENGINE_BENCHMARK_FORMAT,
        "version": ENGINE_BENCHMARK_VERSION,
        "config": config.to_dict(),
        "fixture": {
            "pegs": len(state.pegs),
            "links": len(state.links),
            "side_to_move": state.side_to_move.value,
            "move": move.coordinate.to_dict(),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "benchmarks": results,
        "positions_per_second": results["move_application"]["calls_per_second"],
    }


__all__ = [
    "ENGINE_BENCHMARK_FORMAT",
    "ENGINE_BENCHMARK_VERSION",
    "EngineBenchmarkConfig",
    "run_engine_benchmarks",
]
