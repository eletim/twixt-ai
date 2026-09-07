"""Measured Mini Twixt MCTS and self-play throughput baselines."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
import platform
from random import Random
from statistics import median
from time import perf_counter
from typing import Callable, TypeVar

from twixt_ai.agents import AgentRequest
from twixt_ai.game import (
    BoardDimensions,
    EngineBenchmarkConfig,
    apply_move,
    create_game,
    legal_peg_placements,
    run_engine_benchmarks,
)
from twixt_ai.search import DEFAULT_ROLLOUT_LIMIT, MCTSAgent

from .match import MatchConfig, run_match


SELFPLAY_PERFORMANCE_FORMAT = "twixt-ai-selfplay-performance"
SELFPLAY_PERFORMANCE_VERSION = 1
T = TypeVar("T")


def _positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SelfPlayPerformanceConfig:
    """Settings that completely describe a throughput benchmark workload."""

    board: BoardDimensions = BoardDimensions(10, 10)
    seed: int = 2401
    fixture_ply: int = 8
    simulation_budgets: tuple[int, ...] = (100, 400, 1600)
    move_repeats: int = 3
    games_per_budget: int = 1
    scaling_games: int = 4
    worker_counts: tuple[int, ...] = (1, 2)
    scaling_simulations: int = 100
    rollout_limit: int = DEFAULT_ROLLOUT_LIMIT
    engine_iterations: int = 200
    engine_repeats: int = 5
    engine_warmups: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.board, BoardDimensions):
            raise TypeError("board must be BoardDimensions")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if (
            isinstance(self.fixture_ply, bool)
            or not isinstance(self.fixture_ply, int)
            or self.fixture_ply < 0
        ):
            raise ValueError("fixture_ply must be a non-negative integer")
        for name in (
            "move_repeats",
            "games_per_budget",
            "scaling_games",
            "scaling_simulations",
            "rollout_limit",
            "engine_iterations",
            "engine_repeats",
        ):
            _positive_integer(getattr(self, name), name)
        if (
            isinstance(self.engine_warmups, bool)
            or not isinstance(self.engine_warmups, int)
            or self.engine_warmups < 0
        ):
            raise ValueError("engine_warmups must be a non-negative integer")
        budgets = tuple(self.simulation_budgets)
        workers = tuple(self.worker_counts)
        if not budgets or not workers:
            raise ValueError("simulation_budgets and worker_counts must not be empty")
        for budget in budgets:
            _positive_integer(budget, "simulation budget")
        for worker_count in workers:
            _positive_integer(worker_count, "worker count")
        if len(set(budgets)) != len(budgets):
            raise ValueError("simulation_budgets must be unique")
        if len(set(workers)) != len(workers) or workers[0] != 1:
            raise ValueError("worker_counts must be unique and begin with 1")
        object.__setattr__(self, "simulation_budgets", budgets)
        object.__setattr__(self, "worker_counts", workers)

    def to_dict(self) -> dict[str, object]:
        return {
            "board": self.board.to_dict(),
            "seed": self.seed,
            "fixture_ply": self.fixture_ply,
            "simulation_budgets": list(self.simulation_budgets),
            "move_repeats": self.move_repeats,
            "games_per_budget": self.games_per_budget,
            "scaling_games": self.scaling_games,
            "worker_counts": list(self.worker_counts),
            "scaling_simulations": self.scaling_simulations,
            "rollout_limit": self.rollout_limit,
            "engine": {
                "iterations": self.engine_iterations,
                "repeats": self.engine_repeats,
                "warmups": self.engine_warmups,
            },
        }


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-POSIX fallback
        return os.cpu_count() or 1


def _cpu_seconds() -> float:
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


def _measure(operation: Callable[[], T]) -> tuple[T, float, float]:
    cpu_started = _cpu_seconds()
    wall_started = perf_counter()
    result = operation()
    wall_seconds = perf_counter() - wall_started
    cpu_seconds = max(0.0, _cpu_seconds() - cpu_started)
    return result, wall_seconds, cpu_seconds


def _utilization(
    wall_seconds: float, cpu_seconds: float, cpu_count: int
) -> dict[str, float]:
    core_equivalents = cpu_seconds / wall_seconds if wall_seconds else 0.0
    return {
        "cpu_seconds": cpu_seconds,
        "aggregate_percent": core_equivalents * 100.0,
        "available_cpu_percent": core_equivalents * 100.0 / cpu_count,
        "average_cores": core_equivalents,
    }


def _fixture(config: SelfPlayPerformanceConfig):
    random = Random(config.seed)
    state = create_game(config.board)
    for _ in range(config.fixture_ply):
        moves = legal_peg_placements(state)
        if not moves or state.is_terminal:
            raise ValueError("fixture became terminal; choose a smaller fixture_ply")
        state = apply_move(state, moves[random.randrange(len(moves))])
    if state.is_terminal or not legal_peg_placements(state):
        raise ValueError("fixture is terminal; choose a smaller fixture_ply")
    return state


def _play_mcts_game(arguments: tuple[BoardDimensions, int, int, int]) -> dict[str, int]:
    board, seed, simulations, rollout_limit = arguments
    result = run_match(
        MCTSAgent(simulations=simulations, rollout_limit=rollout_limit),
        MCTSAgent(simulations=simulations, rollout_limit=rollout_limit),
        config=MatchConfig(board, seed, "mcts", "mcts"),
    )
    nodes = sum(int(decision.metadata["nodes"]) for decision in result.decisions)
    return {
        "moves": len(result.moves),
        "nodes": nodes,
        "simulations": simulations * len(result.moves),
    }


def _game_workload(
    board: BoardDimensions,
    seeds: tuple[int, ...],
    simulations: int,
    rollout_limit: int,
    workers: int,
) -> list[dict[str, int]]:
    arguments = [(board, seed, simulations, rollout_limit) for seed in seeds]
    if workers == 1:
        return [_play_mcts_game(item) for item in arguments]
    with ProcessPoolExecutor(max_workers=min(workers, len(arguments))) as pool:
        return list(pool.map(_play_mcts_game, arguments))


def _game_metrics(
    games: list[dict[str, int]], wall_seconds: float, cpu_seconds: float, cpu_count: int
) -> dict[str, object]:
    game_count = len(games)
    total_nodes = sum(game["nodes"] for game in games)
    total_simulations = sum(game["simulations"] for game in games)
    total_moves = sum(game["moves"] for game in games)
    games_per_second = game_count / wall_seconds
    return {
        "games": game_count,
        "wall_seconds": wall_seconds,
        "seconds_per_game": wall_seconds / game_count,
        "games_per_hour": games_per_second * 3600.0,
        "moves": total_moves,
        "move_latency_seconds": wall_seconds / total_moves,
        "simulations_per_second": total_simulations / wall_seconds,
        "nodes_per_second": total_nodes / wall_seconds,
        "cpu_utilization": _utilization(wall_seconds, cpu_seconds, cpu_count),
    }


def run_selfplay_performance_benchmark(
    config: SelfPlayPerformanceConfig | None = None,
) -> dict[str, object]:
    """Measure NN-free engine, MCTS, game, and parallel self-play throughput."""

    config = config or SelfPlayPerformanceConfig()
    if not isinstance(config, SelfPlayPerformanceConfig):
        raise TypeError("config must be a SelfPlayPerformanceConfig")
    cpu_count = _available_cpus()
    fixture = _fixture(config)

    engine_config = EngineBenchmarkConfig(
        board=config.board,
        seed=config.seed,
        ply=config.fixture_ply,
        iterations=config.engine_iterations,
        repeats=config.engine_repeats,
        warmups=config.engine_warmups,
    )
    engine, engine_wall_seconds, engine_cpu_seconds = _measure(
        lambda: run_engine_benchmarks(engine_config)
    )
    engine["wall_seconds"] = engine_wall_seconds
    engine["cpu_utilization"] = _utilization(
        engine_wall_seconds, engine_cpu_seconds, cpu_count
    )

    searches: list[dict[str, object]] = []
    for budget in config.simulation_budgets:
        samples: list[dict[str, float | int]] = []
        for repeat in range(config.move_repeats):
            agent = MCTSAgent(simulations=budget, rollout_limit=config.rollout_limit)
            _, wall_seconds, cpu_seconds = _measure(
                lambda agent=agent, repeat=repeat: agent.choose_move(
                    AgentRequest(fixture, seed=config.seed + repeat)
                )
            )
            assert agent.last_statistics is not None
            samples.append(
                {
                    "wall_seconds": wall_seconds,
                    "cpu_seconds": cpu_seconds,
                    "nodes": agent.last_statistics.nodes,
                }
            )
        median_wall = median(float(item["wall_seconds"]) for item in samples)
        median_cpu = median(float(item["cpu_seconds"]) for item in samples)
        median_nodes = median(int(item["nodes"]) for item in samples)
        searches.append(
            {
                "simulations": budget,
                "repeats": config.move_repeats,
                "move_latency_seconds": median_wall,
                "simulations_per_second": budget / median_wall,
                "nodes_per_second": median_nodes / median_wall,
                "median_nodes": median_nodes,
                "cpu_utilization": _utilization(median_wall, median_cpu, cpu_count),
            }
        )

    seed_source = Random(config.seed)
    full_game_seeds = tuple(
        seed_source.getrandbits(64) for _ in range(config.games_per_budget)
    )
    full_games: list[dict[str, object]] = []
    for budget in config.simulation_budgets:
        games, wall_seconds, cpu_seconds = _measure(
            lambda budget=budget: _game_workload(
                config.board, full_game_seeds, budget, config.rollout_limit, 1
            )
        )
        metrics = _game_metrics(games, wall_seconds, cpu_seconds, cpu_count)
        metrics["simulations_per_move"] = budget
        metrics["estimated_wall_clock_hours"] = {
            "1000_games": 1000.0 / float(metrics["games_per_hour"]),
            "10000_games": 10000.0 / float(metrics["games_per_hour"]),
        }
        full_games.append(metrics)

    scaling_seeds = tuple(
        seed_source.getrandbits(64) for _ in range(config.scaling_games)
    )
    scaling: list[dict[str, object]] = []
    sequential_rate = 0.0
    for workers in config.worker_counts:
        games, wall_seconds, cpu_seconds = _measure(
            lambda workers=workers: _game_workload(
                config.board,
                scaling_seeds,
                config.scaling_simulations,
                config.rollout_limit,
                workers,
            )
        )
        metrics = _game_metrics(games, wall_seconds, cpu_seconds, cpu_count)
        rate = float(metrics["games_per_hour"])
        if workers == 1:
            sequential_rate = rate
        effective_workers = min(workers, config.scaling_games)
        metrics.update(
            workers=workers,
            effective_workers=effective_workers,
            speedup_vs_sequential=rate / sequential_rate,
            parallel_efficiency=rate / sequential_rate / effective_workers,
        )
        scaling.append(metrics)

    slowest_engine = max(
        engine["benchmarks"].items(),
        key=lambda item: item[1]["median_ns_per_call"],
    )
    slowest_search = max(searches, key=lambda item: item["move_latency_seconds"])
    fastest_search = min(searches, key=lambda item: item["move_latency_seconds"])
    move_application_seconds = (
        float(engine["benchmarks"]["move_application"]["median_ns_per_call"])
        / 1_000_000_000
    )
    best_scaling = max(scaling, key=lambda item: item["games_per_hour"])
    bottlenecks = [
        {
            "area": "engine",
            "finding": f"{slowest_engine[0]} is the slowest measured engine operation",
            "evidence": {
                "median_ns_per_call": slowest_engine[1]["median_ns_per_call"],
                "calls_per_second": slowest_engine[1]["calls_per_second"],
            },
        },
        {
            "area": "mcts",
            "finding": (
                "MCTS search dominates measured per-move engine application cost"
            ),
            "evidence": {
                "lowest_measured_search_budget": fastest_search["simulations"],
                "search_move_latency_seconds": fastest_search["move_latency_seconds"],
                "engine_move_application_seconds": move_application_seconds,
                "search_to_application_latency_ratio": (
                    float(fastest_search["move_latency_seconds"])
                    / move_application_seconds
                ),
                "highest_latency_budget": slowest_search["simulations"],
                "highest_move_latency_seconds": slowest_search["move_latency_seconds"],
            },
        },
        {
            "area": "selfplay_scaling",
            "finding": "best measured worker count selected by games/hour",
            "evidence": {
                "workers": best_scaling["workers"],
                "games_per_hour": best_scaling["games_per_hour"],
                "speedup_vs_sequential": best_scaling["speedup_vs_sequential"],
                "parallel_efficiency": best_scaling["parallel_efficiency"],
            },
        },
    ]

    return {
        "format": SELFPLAY_PERFORMANCE_FORMAT,
        "version": SELFPLAY_PERFORMANCE_VERSION,
        "config": config.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "available_cpus": cpu_count,
        },
        "nn_free_baseline": {
            "policy_value_network": False,
            "rollout_limit": config.rollout_limit,
            "engine": engine,
            "mcts_moves": searches,
            "full_games": full_games,
            "worker_scaling": scaling,
        },
        "identified_bottlenecks": bottlenecks,
    }


__all__ = [
    "SELFPLAY_PERFORMANCE_FORMAT",
    "SELFPLAY_PERFORMANCE_VERSION",
    "SelfPlayPerformanceConfig",
    "run_selfplay_performance_benchmark",
]
